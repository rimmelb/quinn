//! Cross-layer bandwidth estimator driven by radio-layer hints (CQI, MCS, PRB, RSRP/RSRQ, BLER).
//!
//! Motivation:
//! Traditional QUIC congestion control learns the available capacity from transport feedback
//! (ACK clocking). On mobile links the channel can fluctuate faster than the feedback loop can
//! converge, especially when the radio scheduler has additional knowledge (CQI, PRB allocation,
//! BLER) that the transport stack normally cannot see. This module exposes a light-weight helper
//! that ingests such hints and produces a filtered throughput prediction that can be fused with
//! the existing deliver-rate estimate.
//!
//! Behaviour in one line:
//!   1. Convert CQI/MCS/PRB (or SINR) into a raw PHY capacity estimate.
//!   2. Fold in BLER and measured PDSCH throughput to obtain an "effective" capacity.
//!   3. Smooth the values with EWMA and score them with a heuristic confidence value.
//!   4. Provide a blended result that can be combined with the ACK-based deliver rate.
//!
//! Soft-state: higher layers push measurements when available, no background timers are required.
//! Missing or stale signals mainly reduce the confidence, which down-weights the cross-layer term.

use std::fmt::Debug;

/// Nominal bandwidth (Hz) of a single Physical Resource Block for LTE/NR numerology 0.
/// Kept as f64 for convenience in downstream arithmetic.
const PRB_BW_HZ: f64 = 180_000.0;

/// Clamp range for the internal confidence score.
const CONFIDENCE_MIN: f64 = 0.05;
const CONFIDENCE_MAX: f64 = 1.0;

/// Default EWMA smoothing factors. Larger alpha reacts faster; smaller alpha damps noise more.
const ALPHA_PHY: f64 = 0.45;
const ALPHA_EFFECTIVE: f64 = 0.55;

/// Minimal/typical CQI table: CQI -> (MCS index, spectral efficiency [bit/s/Hz]).
/// Values are loosely aligned with 3GPP TS 36.213 Table 7.2.3-1 (rounded).
const CQI_TABLE: [CqiEntry; 16] = [
    CqiEntry::new(0, 0.1523),
    CqiEntry::new(0, 0.1523),
    CqiEntry::new(1, 0.2344),
    CqiEntry::new(2, 0.3770),
    CqiEntry::new(3, 0.6016),
    CqiEntry::new(4, 0.8770),
    CqiEntry::new(5, 1.1758),
    CqiEntry::new(6, 1.4766),
    CqiEntry::new(7, 1.9141),
    CqiEntry::new(9, 2.4063),
    CqiEntry::new(11, 2.7305),
    CqiEntry::new(13, 3.3223),
    CqiEntry::new(15, 3.9023),
    CqiEntry::new(17, 4.5234),
    CqiEntry::new(19, 5.1152),
    CqiEntry::new(27, 5.5547),
];

/// Spectral efficiency per MCS index (bit/s/Hz).
/// Based on 3GPP TS 38.214 Table 5.1.3.1-2 (rounded).
const MCS_EFFICIENCY: [f64; 28] = [
    0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023,
    4.5234, 5.1152, 5.5547, 6.2266, 6.9141, 7.4063, 7.9141, 8.4063, 8.8984, 9.3984, 9.8457,
    10.2734, 10.6953, 11.0352, 11.3789, 11.6953,
];

/// Coarse RSRQ thresholds for confidence heuristics.
/// (Tweak for your deployment if needed.)
const RSRQ_BAD_DB: f64 = -15.0;
const RSRQ_GOOD_DB: f64 = -9.0;

/// A compact CQI mapping entry. We carry the MCS index mostly for introspection/logging; the
/// estimator itself uses the spectral efficiency directly.
#[derive(Copy, Clone, Debug)]
struct CqiEntry {
    mcs_index: u8,
    spectral_efficiency: f64,
}

impl CqiEntry {
    const fn new(mcs_index: u8, spectral_efficiency: f64) -> Self {
        Self { mcs_index, spectral_efficiency }
    }
}

/// Collects radio-layer hints and turns them into a smoothed throughput prediction (bit/s).
#[derive(Debug, Clone)]
pub struct CrossLayerEstimator {
    // --- Configuration knobs / radio layout ---
    /// EWMA weight for PHY capacity smoothing; higher value reacts faster.
    alpha_phy: f64,
    /// EWMA weight for effective throughput smoothing.
    alpha_effective: f64,
    /// Bandwidth per PRB (Hz). Defaults to LTE/NR µ=0 = 180 kHz.
    prb_bw_hz: f64,
    /// Downlink fraction in TDD (% of time allocated to DL). Use 1.0 for FDD or DL-only tests.
    tdd_dl_ratio: f64,
    /// Whether the modem-reported PDSCH throughput is already goodput (post-retransmission).
    pdsch_is_goodput: bool,

    // --- Telemetry inputs / most recent readings ---
    /// Latest spectral efficiency (bit/s/Hz) inferred from CQI/MCS/SINR.
    spectral_efficiency: Option<f64>,
    /// Latest downlink PRB allocation for the UE.
    prb_count: Option<u32>,
    /// Last reported SINR (dB); used as a fallback when CQI/MCS is missing.
    sinr_db: Option<f64>,
    /// Modem-reported downlink throughput in bit/s, if available.
    throughput_bps: Option<f64>,
    /// Block error rate in [0,1]; higher values imply more retransmissions.
    bler: Option<f64>,

    // --- Derived / smoothed state ---
    /// Instantaneous PHY capacity (before BLER/PDSCH corrections).
    phy_cap_bps: Option<f64>,
    /// Smoothed PHY capacity after EWMA filtering.
    ewma_phy_bps: Option<f64>,
    /// Smoothed effective throughput after BLER/PDSCH corrections.
    ewma_effective_bps: Option<f64>,
    /// Confidence factor in [CONFIDENCE_MIN, CONFIDENCE_MAX]; influences blending weight.
    confidence: f64,

    // --- Bookkeeping for heuristics ---
    /// Last CQI/MCS values (used to infer stability).
    last_cqi: Option<u8>,
    last_mcs: Option<u8>,

    /// Recompute is needed when new telemetry arrives.
    dirty: bool,
}

impl Default for CrossLayerEstimator {
    fn default() -> Self { Self::new() }
}

impl CrossLayerEstimator {
    /// Create a fresh estimator with neutral confidence and default radio params.
    pub fn new() -> Self {
        Self {
            alpha_phy: ALPHA_PHY,
            alpha_effective: ALPHA_EFFECTIVE,
            prb_bw_hz: PRB_BW_HZ,
            tdd_dl_ratio: 1.0,
            pdsch_is_goodput: true,

            spectral_efficiency: None,
            prb_count: None,
            sinr_db: None,
            throughput_bps: None,
            bler: None,

            phy_cap_bps: None,
            ewma_phy_bps: None,
            ewma_effective_bps: None,
            confidence: 0.5,

            last_cqi: None,
            last_mcs: None,

            dirty: false,
        }
    }

    /// Override EWMA smoothing parameters.
    pub fn with_smoothing(mut self, alpha_phy: f64, alpha_effective: f64) -> Self {
        self.alpha_phy = alpha_phy.clamp(0.01, 0.99);
        self.alpha_effective = alpha_effective.clamp(0.01, 0.99);
        self
    }

    /// Override radio layout parameters (PRB bandwidth and TDD DL ratio).
    pub fn with_radio_params(mut self, prb_bw_hz: f64, tdd_dl_ratio: f64) -> Self {
        self.prb_bw_hz = prb_bw_hz.max(1.0);          // avoid zero/negative
        self.tdd_dl_ratio = tdd_dl_ratio.clamp(0.0, 1.0);
        self
    }

    /// Mark whether PDSCH throughput is already a goodput measurement.
    pub fn with_pdsch_is_goodput(mut self, is_goodput: bool) -> Self {
        self.pdsch_is_goodput = is_goodput;
        self
    }

    /// Push a new CQI reading (0-15) and refresh the spectral efficiency accordingly.
    pub fn update_cqi(&mut self, cqi: u8) {
        let cqi = cqi.min((CQI_TABLE.len() - 1) as u8);

        // Compute delta *before* overwriting last_cqi to preserve the previous value.
        let prev_cqi = self.last_cqi;
        let diff = prev_cqi.map(|p| p.abs_diff(cqi)).unwrap_or(0);

        let entry = CQI_TABLE[cqi as usize];
        self.spectral_efficiency = Some(entry.spectral_efficiency);
        self.last_cqi = Some(cqi);
        self.last_mcs = Some(entry.mcs_index);

        if diff <= 2 {
            // Small variations imply stability -> trust the radio prediction a bit more.
            self.bump_confidence(0.05);
        } else {
            // Rapid swings hint that radio-side prediction could be noisy.
            self.bump_confidence(-0.1);
        }
        self.dirty = true;
    }

    /// Record the current MCS index. Falls back to CQI inference if index is out of range.
    pub fn update_mcs(&mut self, mcs: u8) {
        if let Some(eff) = MCS_EFFICIENCY.get(mcs as usize).copied() {
            self.spectral_efficiency = Some(eff);
            self.last_mcs = Some(mcs);
            self.bump_confidence(0.05);
        } else {
            // Unknown MCS index: penalise confidence but keep previous value.
            self.bump_confidence(-0.1);
        }
        self.dirty = true;
    }

    /// Refresh the number of PRBs scheduled for the UE in the latest TTI.
    pub fn update_prb_allocation(&mut self, prbs: u32) {
        self.prb_count = Some(prbs);
        if prbs == 0 {
            self.bump_confidence(-0.2);
        } else {
            self.bump_confidence(0.02);
        }
        self.dirty = true;
    }

    /// Update SINR (dB). Used as a fallback when no CQI/MCS is available.
    pub fn update_sinr(&mut self, sinr_db: f64) {
        self.sinr_db = Some(sinr_db);
        if sinr_db < 0.0 {
            // Poor SNR often means a too-optimistic PHY estimate.
            self.bump_confidence(-0.05);
        } else {
            self.bump_confidence(0.03);
        }
        self.dirty = true;
    }

    /// Feed RSRP/RSRQ readings to adjust confidence without changing the capacity estimate.
    pub fn update_radio_quality(&mut self, rsrp_dbm: f64, rsrq_db: f64) {
        if rsrp_dbm < -110.0 {
            // Very weak received power: edge-like conditions -> be cautious.
            self.bump_confidence(-0.1);
        } else if rsrp_dbm > -95.0 {
            // Strong signal: CQI-driven capacity tends to be reliable.
            self.bump_confidence(0.04);
        }

        if rsrq_db < RSRQ_BAD_DB {
            // Heavy interference / cell load tends to increase retransmissions.
            self.bump_confidence(-0.15);
        } else if rsrq_db > RSRQ_GOOD_DB {
            self.bump_confidence(0.05);
        }
        // Not marking dirty: confidence only affects blending, not the core estimate.
    }

    /// Provide the measured PDSCH throughput (bit/s) — used to cap optimistic PHY estimates.
    pub fn update_pdsch_throughput(&mut self, throughput_bps: f64) {
        if throughput_bps.is_finite() && throughput_bps > 0.0 {
            self.throughput_bps = Some(throughput_bps);
            self.bump_confidence(0.08);
        } else {
            self.bump_confidence(-0.1);
        }
        self.dirty = true;
    }

    /// Provide the current BLER (0..1) to discount retransmissions.
    pub fn update_bler(&mut self, bler: f64) {
        if bler.is_finite() {
            let bler = bler.clamp(0.0, 1.0);
            self.bler = Some(bler);
            if bler > 0.2 {
                // High BLER means many retransmissions, so lower confidence markedly.
                self.bump_confidence(-0.2);
            } else if bler < 0.05 {
                // Low BLER: PHY prediction aligns well with reality.
                self.bump_confidence(0.05);
            }
        }
        self.dirty = true;
    }

    /// Current confidence score between `CONFIDENCE_MIN` and `CONFIDENCE_MAX`.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Return the smoothed *effective* throughput prediction in bit/s, if available.
    pub fn predict_effective_bps(&mut self) -> Option<f64> {
        if self.dirty {
            // Lazily recompute when new telemetry has arrived.
            self.recompute();
        }
        self.ewma_effective_bps
    }

    /// Blend the cross-layer prediction (if any) with an ACK-based deliver rate (bytes/s).
    /// The internal `confidence` acts as the mixing weight for the cross-layer term.
    pub fn blend_with_deliver_rate(&mut self, deliver_rate_bytes: u64) -> u64 {
        let deliver_rate_bps = (deliver_rate_bytes as f64) * 8.0;
        let Some(cross_layer_bps) = self.predict_effective_bps() else {
            // No cross-layer signal: fall back to the transport-only rate.
            return deliver_rate_bytes;
        };
        let weight = self.confidence.clamp(0.0, 1.0);
        let blended_bps = weight * cross_layer_bps + (1.0 - weight) * deliver_rate_bps;
        (blended_bps / 8.0).max(0.0) as u64
    }

    /// Recompute cached PHY and effective capacities after new telemetry.
    fn recompute(&mut self) {
        // Stage 1: refresh instantaneous PHY capacity and smooth it (EWMA).
        if let Some(raw_phy) = self.estimate_phy_capacity() {
            self.phy_cap_bps = Some(match self.ewma_phy_bps {
                Some(prev) => self.alpha_phy * raw_phy + (1.0 - self.alpha_phy) * prev,
                None => raw_phy,
            });
        }

        // Stage 2: derive effective capacity (apply BLER/PDSCH) and smooth it.
        if let Some(raw_effective) = self.estimate_effective_capacity() {
            self.ewma_effective_bps = Some(match self.ewma_effective_bps {
                Some(prev) => self.alpha_effective * raw_effective + (1.0 - self.alpha_effective) * prev,
                None => raw_effective,
            });
        } else {
            self.ewma_effective_bps = None;
        }

        self.dirty = false;
    }

    /// Estimate instantaneous PHY capacity from spectral efficiency and PRB allocation.
    /// Falls back to a Shannon-like efficiency approximation when only SINR is available.
    fn estimate_phy_capacity(&self) -> Option<f64> {
        let eff = if let Some(e) = self.spectral_efficiency {
            e
        } else if let Some(sinr_db) = self.sinr_db {
            let sinr_linear = 10f64.powf(sinr_db / 10.0);
            // Shannon-approximation fallback: log2(1 + SINR).
            // The small floor avoids fully collapsing the EWMA on brief outages.
            (1.0 + sinr_linear).log2().max(0.1)
        } else {
            return None;
        };

        let prb = self.prb_count?;
        let bandwidth_hz = prb as f64 * self.prb_bw_hz * self.tdd_dl_ratio;
        Some(eff * bandwidth_hz)
    }

    /// Apply BLER discount and PDSCH cap to convert PHY capacity into an "effective" one.
    fn estimate_effective_capacity(&self) -> Option<f64> {
        let phy = self.phy_cap_bps.or_else(|| self.estimate_phy_capacity())?;
        let mut effective = phy;

        if let Some(bler) = self.bler {
            let success_ratio = (1.0 - bler).clamp(0.0, 1.0);
            // Keep a small floor to avoid dropping to zero during transient spikes.
            effective *= success_ratio.max(0.05);
        }

        if let Some(actual) = self.throughput_bps {
            if self.pdsch_is_goodput {
                // Already the actual delivered (goodput) rate: treat as an upper bound.
                effective = effective.min(actual);
            } else {
                // If this is a pre-retransmission PHY figure, cap by the expected successful part.
                let bler = self.bler.unwrap_or(0.0);
                effective = effective.min(actual * (1.0 - bler).clamp(0.0, 1.0));
            }
        }

        Some(effective)
    }

    /// Adjust the internal confidence score within safe bounds.
    fn bump_confidence(&mut self, delta: f64) {
        self.confidence = (self.confidence + delta).clamp(CONFIDENCE_MIN, CONFIDENCE_MAX);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_prediction_uses_cqi_and_prb() {
        let mut est = CrossLayerEstimator::new();
        est.update_cqi(12);
        est.update_prb_allocation(50);
        est.update_bler(0.05);
        let predicted = est.predict_effective_bps().unwrap();
        // Should land near efficiency 3.3223 * 50 * 180k * (1 - 0.05).
        let expected = 3.3223 * 50.0 * PRB_BW_HZ * 0.95;
        assert!((predicted - expected).abs() / expected < 0.25);
    }

    #[test]
    fn high_bler_reduces_prediction() {
        let mut est = CrossLayerEstimator::new();
        est.update_cqi(14);
        est.update_prb_allocation(30);
        est.update_bler(0.5);
        let predicted = est.predict_effective_bps().unwrap();
        let baseline = 5.1152 * 30.0 * PRB_BW_HZ;
        assert!(predicted < baseline * 0.7);
    }

    #[test]
    fn blend_falls_back_to_deliver_rate() {
        let mut est = CrossLayerEstimator::new();
        let deliver = 2_000_000u64;
        assert_eq!(est.blend_with_deliver_rate(deliver), deliver);
    }
}
