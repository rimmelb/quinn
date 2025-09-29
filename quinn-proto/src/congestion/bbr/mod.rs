use std::any::Any;
use std::fmt::Debug;

use std::sync::{Arc, Mutex};


use rand::{Rng, SeedableRng};

use crate::congestion::ControllerMetrics;
use crate::congestion::bbr::bw_estimation::BandwidthEstimation;
use crate::congestion::bbr::min_max::MinMax;
use crate::connection::RttEstimator;
use crate::{Duration, Instant};

use super::{BASE_DATAGRAM_SIZE, Controller, ControllerFactory};

mod bw_estimation;
mod min_max;

/// Experimental! Use at your own risk.
///
/// Aims for reduced buffer bloat and improved performance over high bandwidth-delay product networks.
/// Based on google's quiche implementation <https://source.chromium.org/chromium/chromium/src/+/master:net/third_party/quiche/src/quic/core/congestion_control/bbr_sender.cc>
/// of BBR <https://datatracker.ietf.org/doc/html/draft-cardwell-iccrg-bbr-congestion-control>.
/// More discussion and links at <https://groups.google.com/g/bbr-dev>.
#[derive(Debug, Clone)]
pub struct Bbr {
    config: Arc<BbrConfig>,
    current_mtu: u64,
    max_bandwidth: BandwidthEstimation,
    acked_bytes: u64,
    mode: Mode,
    loss_state: LossState,
    recovery_state: RecoveryState,
    recovery_window: u64,
    is_at_full_bandwidth: bool,
    pacing_gain: f32,
    high_gain: f32,
    drain_gain: f32,
    cwnd_gain: f32,
    high_cwnd_gain: f32,
    last_cycle_start: Option<Instant>,
    current_cycle_offset: u8,
    init_cwnd: u64,
    min_cwnd: u64,
    prev_in_flight_count: u64,
    exit_probe_rtt_at: Option<Instant>,
    probe_rtt_last_started_at: Option<Instant>,
    min_rtt: Duration,
    exiting_quiescence: bool,
    pacing_rate: u64,
    max_acked_packet_number: u64,
    max_sent_packet_number: u64,
    end_recovery_at_packet_number: u64,
    cwnd: u64,
    current_round_trip_end_packet_number: u64,
    round_count: u64,
    bw_at_last_round: u64,
    round_wo_bw_gain: u64,
    ack_aggregation: AckAggregationState,
    random_number_generator: rand::rngs::StdRng,

    /// Deadline scheduler configuration
    deadline_config: Option<DeadlineConfig>,

    // --- NEW: single-path deadline scheduler state (virt. queue in packets) ---
    deadline_state: Arc<Mutex<DeadlineState>>,

}

#[derive(Debug, Clone)]
pub struct DeadlineConfig {
    pub enabled: bool,
    pub beta: f64,       // Conservative factor for pps estimation
    pub guard_ms: u64,   // Guard time for jitter
    pub default_mss: u32, // Fallback MSS
}

impl Default for DeadlineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            beta: 0.8,
            guard_ms: 10,
            default_mss: 1200,
        }
    }
}

#[derive(Debug, Clone)]
struct DeadlineState {
    q_pkts: f64,
    last: Option<Instant>,
}

impl Default for DeadlineState {
    fn default() -> Self {
         Self { q_pkts: 0.0, last: None }
     }
 }

impl Bbr {
    /// Construct a state using the given `config` and current time `now`
    pub fn new(config: Arc<BbrConfig>, current_mtu: u16) -> Self {
        let initial_window = config.initial_window;
        let deadline_config = config.deadline.clone();
        if let Some(dc) = &deadline_config {
            tracing::debug!(
                target: "bbr.deadline",
                enabled = dc.enabled,
                beta = dc.beta,
                guard_ms = dc.guard_ms,
                default_mss = dc.default_mss,
                "BBR constructed with DeadlineConfig"
            );
        } else {
            tracing::debug!(target: "bbr.deadline", "BBR constructed without DeadlineConfig");
        }

        Self {
            config,
            current_mtu: current_mtu as u64,
            max_bandwidth: BandwidthEstimation::default(),
            acked_bytes: 0,
            mode: Mode::Startup,
            loss_state: Default::default(),
            recovery_state: RecoveryState::NotInRecovery,
            recovery_window: 0,
            is_at_full_bandwidth: false,
            pacing_gain: K_DEFAULT_HIGH_GAIN,
            high_gain: K_DEFAULT_HIGH_GAIN,
            drain_gain: 1.0 / K_DEFAULT_HIGH_GAIN,
            cwnd_gain: K_DEFAULT_HIGH_GAIN,
            high_cwnd_gain: K_DEFAULT_HIGH_GAIN,
            last_cycle_start: None,
            current_cycle_offset: 0,
            init_cwnd: initial_window,
            min_cwnd: calculate_min_window(current_mtu as u64),
            prev_in_flight_count: 0,
            exit_probe_rtt_at: None,
            probe_rtt_last_started_at: None,
            min_rtt: Default::default(),
            exiting_quiescence: false,
            pacing_rate: 0,
            max_acked_packet_number: 0,
            max_sent_packet_number: 0,
            end_recovery_at_packet_number: 0,
            cwnd: initial_window,
            current_round_trip_end_packet_number: 0,
            round_count: 0,
            bw_at_last_round: 0,
            round_wo_bw_gain: 0,
            ack_aggregation: AckAggregationState::default(),
            deadline_config, // <-- FIX: take from BbrConfig
            random_number_generator: rand::rngs::StdRng::from_os_rng(),

            // NEW:
            deadline_state: Arc::new(Mutex::new(DeadlineState::default())),
        }
    }

    #[inline]
    fn deadline_decay_queue(&self, now: Instant, pps: f64) {
    if pps <= 0.0 { return; }
    let mut st = self.deadline_state.lock().unwrap();
    let last = st.last.unwrap_or(now);
    let dt = now.saturating_duration_since(last).as_secs_f64();
    if dt > 0.0 {
            st.q_pkts = (st.q_pkts - pps * dt).max(0.0);
            st.last = Some(now);
        }
    }

    fn enter_startup_mode(&mut self) {
        if self.config.fixed_pacing_bps.is_some() {
        // FIXED PACING: ne legyen “Startup-boost”, álljunk be cruisera
        self.mode = Mode::ProbeBw;
        self.pacing_gain = 1.0;
        self.cwnd_gain = 1.0;
        return;
        }
        self.mode = Mode::Startup;
        self.pacing_gain = self.high_gain;
        self.cwnd_gain = self.high_cwnd_gain;
    }

    fn enter_probe_bandwidth_mode(&mut self, now: Instant) {
        self.mode = Mode::ProbeBw;
        self.cwnd_gain = K_DERIVED_HIGH_CWNDGAIN;
        self.last_cycle_start = Some(now);
        // Pick a random offset for the gain cycle out of {0, 2..7} range. 1 is
        // excluded because in that case increased gain and decreased gain would not
        // follow each other.
        let mut rand_index = self
            .random_number_generator
            .random_range(0..K_PACING_GAIN.len() as u8 - 1);
        if rand_index >= 1 {
            rand_index += 1;
        }
        self.current_cycle_offset = rand_index;
        self.pacing_gain = K_PACING_GAIN[rand_index as usize];
    }

    fn update_recovery_state(&mut self, is_round_start: bool) {
        // Exit recovery when there are no losses for a round.
        if self.loss_state.has_losses() {
            self.end_recovery_at_packet_number = self.max_sent_packet_number;
        }
        match self.recovery_state {
            // Enter conservation on the first loss.
            RecoveryState::NotInRecovery if self.loss_state.has_losses() => {
                self.recovery_state = RecoveryState::Conservation;
                // This will cause the |recovery_window| to be set to the
                // correct value in CalculateRecoveryWindow().
                self.recovery_window = 0;
                // Since the conservation phase is meant to be lasting for a whole
                // round, extend the current round as if it were started right now.
                self.current_round_trip_end_packet_number = self.max_sent_packet_number;
            }
            RecoveryState::Growth | RecoveryState::Conservation => {
                if self.recovery_state == RecoveryState::Conservation && is_round_start {
                    self.recovery_state = RecoveryState::Growth;
                }
                // Exit recovery if appropriate.
                if !self.loss_state.has_losses()
                    && self.max_acked_packet_number > self.end_recovery_at_packet_number
                {
                    self.recovery_state = RecoveryState::NotInRecovery;
                }
            }
            _ => {}
        }
    }

    fn update_gain_cycle_phase(&mut self, now: Instant, in_flight: u64) {

        // FIXED PACING: ne ciklussal “hintázzon” a gain — maradjon 1.0
        if self.config.fixed_pacing_bps.is_some() {
        self.pacing_gain = 1.0;
        // de állapotgépet nem piszkáljuk, csak nem léptetünk ciklust
        return;
        }
        // In most cases, the cycle is advanced after an RTT passes.
        let mut should_advance_gain_cycling = self
            .last_cycle_start
            .map(|last_cycle_start| now.duration_since(last_cycle_start) > self.min_rtt)
            .unwrap_or(false);
        // If the pacing gain is above 1.0, the connection is trying to probe the
        // bandwidth by increasing the number of bytes in flight to at least
        // pacing_gain * BDP.  Make sure that it actually reaches the target, as
        // long as there are no losses suggesting that the buffers are not able to
        // hold that much.
        if self.pacing_gain > 1.0
            && !self.loss_state.has_losses()
            && self.prev_in_flight_count < self.get_target_cwnd(self.pacing_gain)
        {
            should_advance_gain_cycling = false;
        }

        // If pacing gain is below 1.0, the connection is trying to drain the extra
        // queue which could have been incurred by probing prior to it.  If the
        // number of bytes in flight falls down to the estimated BDP value earlier,
        // conclude that the queue has been successfully drained and exit this cycle
        // early.
        if self.pacing_gain < 1.0 && in_flight <= self.get_target_cwnd(1.0) {
            should_advance_gain_cycling = true;
        }

        if should_advance_gain_cycling {
            self.current_cycle_offset = (self.current_cycle_offset + 1) % K_PACING_GAIN.len() as u8;
            self.last_cycle_start = Some(now);
            // Stay in low gain mode until the target BDP is hit.  Low gain mode
            // will be exited immediately when the target BDP is achieved.
            if DRAIN_TO_TARGET
                && self.pacing_gain < 1.0
                && (K_PACING_GAIN[self.current_cycle_offset as usize] - 1.0).abs() < f32::EPSILON
                && in_flight > self.get_target_cwnd(1.0)
            {
                return;
            }
            self.pacing_gain = K_PACING_GAIN[self.current_cycle_offset as usize];
        }
    }

    fn maybe_exit_startup_or_drain(&mut self, now: Instant, in_flight: u64) {
        if self.mode == Mode::Startup && self.is_at_full_bandwidth {
            self.mode = Mode::Drain;
            self.pacing_gain = self.drain_gain;
            self.cwnd_gain = self.high_cwnd_gain;
        }
        if self.mode == Mode::Drain && in_flight <= self.get_target_cwnd(1.0) {
            self.enter_probe_bandwidth_mode(now);
        }
    }

    fn is_min_rtt_expired(&self, now: Instant, app_limited: bool) -> bool {
        !app_limited
            && self
                .probe_rtt_last_started_at
                .map(|last| now.saturating_duration_since(last) > Duration::from_secs(10))
                .unwrap_or(true)
    }

    fn maybe_enter_or_exit_probe_rtt(
        &mut self,
        now: Instant,
        is_round_start: bool,
        bytes_in_flight: u64,
        app_limited: bool,
    ) {
        let min_rtt_expired = self.is_min_rtt_expired(now, app_limited);
        if min_rtt_expired && !self.exiting_quiescence && self.mode != Mode::ProbeRtt {
            self.mode = Mode::ProbeRtt;
            self.pacing_gain = 1.0;
            // Do not decide on the time to exit ProbeRtt until the
            // |bytes_in_flight| is at the target small value.
            self.exit_probe_rtt_at = None;
            self.probe_rtt_last_started_at = Some(now);
        }

        if self.mode == Mode::ProbeRtt {
            if self.exit_probe_rtt_at.is_none() {
                // If the window has reached the appropriate size, schedule exiting
                // ProbeRtt.  The CWND during ProbeRtt is
                // kMinimumCongestionWindow, but we allow an extra packet since QUIC
                // checks CWND before sending a packet.
                if bytes_in_flight < self.get_probe_rtt_cwnd() + self.current_mtu {
                    const K_PROBE_RTT_TIME: Duration = Duration::from_millis(200);
                    self.exit_probe_rtt_at = Some(now + K_PROBE_RTT_TIME);
                }
            } else if is_round_start && now >= self.exit_probe_rtt_at.unwrap() {
                if !self.is_at_full_bandwidth {
                    self.enter_startup_mode();
                } else {
                    self.enter_probe_bandwidth_mode(now);
                }
            }
        }

        self.exiting_quiescence = false;
    }

    fn get_target_cwnd(&self, gain: f32) -> u64 {
        let bw = self.max_bandwidth.get_estimate();
        let bdp = self.min_rtt.as_micros() as u64 * bw;
        let bdpf = bdp as f64;
        let cwnd = ((gain as f64 * bdpf) / 1_000_000f64) as u64;
        // BDP estimate will be zero if no bandwidth samples are available yet.
        if cwnd == 0 {
            return self.init_cwnd;
        }
        cwnd.max(self.min_cwnd)
    }

    fn get_probe_rtt_cwnd(&self) -> u64 {
        const K_MODERATE_PROBE_RTT_MULTIPLIER: f32 = 0.75;
        if PROBE_RTT_BASED_ON_BDP {
            return self.get_target_cwnd(K_MODERATE_PROBE_RTT_MULTIPLIER);
        }
        self.min_cwnd
    }

fn calculate_pacing_rate(&mut self) {
    // FIXED PACING: ha be van állítva, közvetlenül ebből számolunk byte/s tempót
    if let Some(bps) = self.config.fixed_pacing_bps {
        // byte/s (metrics() majd *8-cal visszaadja bit/s-ban)
        self.pacing_rate = (bps / 8).max(1);
        // ne moduláljunk gain-nel fix módban
        self.pacing_gain = 1.0;
        return;
    }

    // --- Eredeti BBR logika (fallback) ---
    let bw = self.max_bandwidth.get_estimate();
    if bw == 0 {
        return;
    }
    let target_rate = (bw as f64 * self.pacing_gain as f64) as u64;
    if self.is_at_full_bandwidth {
        self.pacing_rate = target_rate;
        return;
    }

    // Pace: initial_window / RTT, amint van RTT
    if self.pacing_rate == 0 && self.min_rtt.as_nanos() != 0 {
        self.pacing_rate =
            BandwidthEstimation::bw_from_delta(self.init_cwnd, self.min_rtt).unwrap();
        return;
    }

    // Startupban ne csökkentsünk pacinget
    if self.pacing_rate < target_rate {
        self.pacing_rate = target_rate;
    }

    // Alsó korlát, ha be van állítva
    if let Some(floor_bps) = Some(self.config.min_pacing_bps) {
        if floor_bps > 0 && self.min_rtt.as_nanos() != 0 {
            let win_bytes = self.window();
            let rate_cwnd = ((win_bytes as u128 * 8_000_000u128)
                / (self.min_rtt.as_micros().max(1) as u128)) as u64;

            let loss_blocking = self.loss_state.has_losses() || self.recovery_state.in_recovery();
            if !loss_blocking {
                let desired = floor_bps;
                let capped = desired.min(rate_cwnd.max(1));
                if capped > self.pacing_rate {
                    self.pacing_rate = capped;
                }
            }
        }
    }
}


    fn calculate_cwnd(&mut self, bytes_acked: u64, excess_acked: u64) {
        if self.mode == Mode::ProbeRtt {
            return;
        }
        if let Some(bps) = self.config.fixed_pacing_bps {
        if self.min_rtt.as_nanos() != 0 {
            // cap-hez illesztett BDP: bytes = bps * RTT / 8
            let win_bytes = ((bps as f64) * self.min_rtt.as_secs_f64() / 8.0) as u64;
            self.cwnd = win_bytes.max(self.min_cwnd);
        } else {
            // amíg nincs RTT, tartsd kicsiben (pl. min_cwnd), hogy ne burstöljön
            self.cwnd = self.min_cwnd;
        }
        return; // ne növeljük tovább ACK-re
        }

        let mut target_window = self.get_target_cwnd(self.cwnd_gain);
        if self.is_at_full_bandwidth {
            // Add the max recently measured ack aggregation to CWND.
            target_window += self.ack_aggregation.max_ack_height.get();
        } else {
            // Add the most recent excess acked.  Because CWND never decreases in
            // STARTUP, this will automatically create a very localized max filter.
            target_window += excess_acked;
        }
        // Instead of immediately setting the target CWND as the new one, BBR grows
        // the CWND towards |target_window| by only increasing it |bytes_acked| at a
        // time.
        if self.is_at_full_bandwidth {
            self.cwnd = target_window.min(self.cwnd + bytes_acked);
        } else if (self.cwnd_gain < target_window as f32) || (self.acked_bytes < self.init_cwnd) {
            // If the connection is not yet out of startup phase, do not decrease
            // the window.
            self.cwnd += bytes_acked;
        }

        // Enforce the limits on the congestion window.
        if self.cwnd < self.min_cwnd {
            self.cwnd = self.min_cwnd;
        }
    }

    fn calculate_recovery_window(&mut self, bytes_acked: u64, bytes_lost: u64, in_flight: u64) {
        if !self.recovery_state.in_recovery() {
            return;
        }
        // Set up the initial recovery window.
        if self.recovery_window == 0 {
            self.recovery_window = self.min_cwnd.max(in_flight + bytes_acked);
            return;
        }

        // Remove losses from the recovery window, while accounting for a potential
        // integer underflow.
        if self.recovery_window >= bytes_lost {
            self.recovery_window -= bytes_lost;
        } else {
            // k_max_segment_size = current_mtu
            self.recovery_window = self.current_mtu;
        }
        // In CONSERVATION mode, just subtracting losses is sufficient.  In GROWTH,
        // release additional |bytes_acked| to achieve a slow-start-like behavior.
        if self.recovery_state == RecoveryState::Growth {
            self.recovery_window += bytes_acked;
        }

        // Sanity checks.  Ensure that we always allow to send at least an MSS or
        // |bytes_acked| in response, whichever is larger.
        self.recovery_window = self
            .recovery_window
            .max(in_flight + bytes_acked)
            .max(self.min_cwnd);
    }

    /// <https://datatracker.ietf.org/doc/html/draft-cardwell-iccrg-bbr-congestion-control#section-4.3.2.2>
    fn check_if_full_bw_reached(&mut self, app_limited: bool) {
        if app_limited {
            return;
        }
        let target = (self.bw_at_last_round as f64 * K_STARTUP_GROWTH_TARGET as f64) as u64;
        let bw = self.max_bandwidth.get_estimate();
        if bw >= target {
            self.bw_at_last_round = bw;
            self.round_wo_bw_gain = 0;
            self.ack_aggregation.max_ack_height.reset();
            return;
        }

        self.round_wo_bw_gain += 1;
        if self.round_wo_bw_gain >= K_ROUND_TRIPS_WITHOUT_GROWTH_BEFORE_EXITING_STARTUP as u64
            || (self.recovery_state.in_recovery())
        {
            self.is_at_full_bandwidth = true;
        }
    }
}

impl Controller for Bbr {
    fn on_sent(&mut self, now: Instant, bytes: u64, last_packet_number: u64) {
        self.max_sent_packet_number = last_packet_number;
        self.max_bandwidth.on_sent(now, bytes);
    }

    fn on_ack(
        &mut self,
        now: Instant,
        sent: Instant,
        bytes: u64,
        app_limited: bool,
        rtt: &RttEstimator,
    ) {
        self.max_bandwidth
            .on_ack(now, sent, bytes, self.round_count, app_limited);
        self.acked_bytes += bytes;
        if self.is_min_rtt_expired(now, app_limited) || self.min_rtt > rtt.min() {
            self.min_rtt = rtt.min();
        }
    }

    fn on_end_acks(
        &mut self,
        now: Instant,
        in_flight: u64,
        app_limited: bool,
        largest_packet_num_acked: Option<u64>,
    ) {
        let bytes_acked = self.max_bandwidth.bytes_acked_this_window();
        let excess_acked = self.ack_aggregation.update_ack_aggregation_bytes(
            bytes_acked,
            now,
            self.round_count,
            self.max_bandwidth.get_estimate(),
        );
        self.max_bandwidth.end_acks(self.round_count, app_limited);
        if let Some(largest_acked_packet) = largest_packet_num_acked {
            self.max_acked_packet_number = largest_acked_packet;
        }

        let mut is_round_start = false;
        if bytes_acked > 0 {
            is_round_start =
                self.max_acked_packet_number > self.current_round_trip_end_packet_number;
            if is_round_start {
                self.current_round_trip_end_packet_number = self.max_sent_packet_number;
                self.round_count += 1;
            }
        }

        self.update_recovery_state(is_round_start);

        if self.mode == Mode::ProbeBw {
            self.update_gain_cycle_phase(now, in_flight);
        }

        if is_round_start && !self.is_at_full_bandwidth {
            self.check_if_full_bw_reached(app_limited);
        }

        self.maybe_exit_startup_or_drain(now, in_flight);

        self.maybe_enter_or_exit_probe_rtt(now, is_round_start, in_flight, app_limited);

        // After the model is updated, recalculate the pacing rate and congestion window.
        self.calculate_pacing_rate();
        self.calculate_cwnd(bytes_acked, excess_acked);
        self.calculate_recovery_window(bytes_acked, self.loss_state.lost_bytes, in_flight);

        self.prev_in_flight_count = in_flight;
        self.loss_state.reset();
    }

    fn on_congestion_event(
        &mut self,
        _now: Instant,
        _sent: Instant,
        _is_persistent_congestion: bool,
        lost_bytes: u64,
    ) {
        self.loss_state.lost_bytes += lost_bytes;
    }

    fn on_mtu_update(&mut self, new_mtu: u16) {
        self.current_mtu = new_mtu as u64;
        self.min_cwnd = calculate_min_window(self.current_mtu);
        self.init_cwnd = self.config.initial_window.max(self.min_cwnd);
        self.cwnd = self.cwnd.max(self.min_cwnd);
    }

    fn window(&self) -> u64 {
    if self.mode == Mode::ProbeRtt {
        return self.get_probe_rtt_cwnd();
    }
    if let Some(bps) = self.config.fixed_pacing_bps {
        if self.min_rtt.as_nanos() != 0 {
            let win_bytes = ((bps as f64) * self.min_rtt.as_secs_f64() / 8.0) as u64;
            return win_bytes.max(self.min_cwnd);
        }
        return self.min_cwnd;
    }
    if self.recovery_state.in_recovery() && self.mode != Mode::Startup {
        return self.cwnd.min(self.recovery_window);
    }
    self.cwnd
    }


    fn metrics(&self) -> ControllerMetrics {
        ControllerMetrics {
            congestion_window: self.window(),
            ssthresh: None,
            pacing_rate: Some(self.pacing_rate * 8),
        }
    }

    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(self.clone())
    }

    fn initial_window(&self) -> u64 {
        self.config.initial_window
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

fn can_admit_object(
    &self,
    object_size: u64,
    deadline: Instant,
    now: Instant,
    rtt_hint: Duration,
) -> bool {
    // Config értékek kimásolása (ne tartsunk kölcsönt a későbbi lockig)
    let (enabled, beta, guard_ms, default_mss) = match self.deadline_config.as_ref() {
        Some(dc) => (dc.enabled, dc.beta, dc.guard_ms, dc.default_mss),
        None => (false, 0.0, 0, 1200),
    };
    if !enabled {
        return true;
    }

    // RTT padlóval
    let mut use_rtt = if self.min_rtt.as_nanos() != 0 { self.min_rtt } else { rtt_hint };
    let rtt_floor = Duration::from_millis(5);
    if use_rtt < rtt_floor { use_rtt = rtt_floor; }
    if use_rtt.as_nanos() == 0 {
        tracing::debug!(
            target: "bbr.deadline",
            object_size,
            min_rtt = ?self.min_rtt,
            rtt_hint = ?rtt_hint,
            "admit: no usable RTT (both zero) -> false"
        );
        return false;
    }

    // Effektív bps (a szűk komponens)
    let app_bps  = (self.max_bandwidth.get_estimate() as f64) * 8.0;
    let cwnd_bps = (self.cwnd as f64 * 8.0) / use_rtt.as_secs_f64();
    let pace_bps = (self.pacing_rate as f64) * 8.0;

    let mut effective_bps = f64::INFINITY;
    if cwnd_bps.is_finite() && cwnd_bps > 0.0 { effective_bps = effective_bps.min(cwnd_bps); }
    if app_bps.is_finite()  && app_bps  > 0.0 { effective_bps = effective_bps.min(app_bps); }
    if pace_bps.is_finite() && pace_bps > 0.0 { effective_bps = effective_bps.min(pace_bps); }

    if !effective_bps.is_finite() || effective_bps <= 0.0 {
        tracing::debug!(
            target: "bbr.deadline",
            object_size,
            bw_estimate_bytes_per_s = self.max_bandwidth.get_estimate(),
            cwnd_bytes = self.cwnd,
            min_rtt = ?self.min_rtt,
            use_rtt = ?use_rtt,
            %app_bps, %cwnd_bps, %pace_bps,
            "admit: no positive capacity component -> false"
        );
        return false;
    }

    // MPR paraméterek
    let mss = default_mss as f64;
    let pps = (effective_bps / 8.0 / mss).max(1.0) * beta; // konzervatív
    let pkt_num = ((object_size + default_mss as u64 - 1) / default_mss as u64).max(1) as f64;

    // Sor öregítése
    self.deadline_decay_queue(now, pps);

    // q állapot kiolvasása (rövid lock)
    let virt_q_before = {
        let st = self.deadline_state.lock().unwrap();
        st.q_pkts
    };

    // trans_time = RTT/2 + (q + pktNum)/pps
    let trans_time = use_rtt / 2 + Duration::from_secs_f64((virt_q_before + pkt_num) / pps);

    let guard = Duration::from_millis(guard_ms);
    let admit = now + trans_time + guard <= deadline;

    if admit {
        // vízbetöltés: q += pktNum (rövid lock)
        let mut st = self.deadline_state.lock().unwrap();
        st.q_pkts = virt_q_before + pkt_num;
        st.last = Some(now);
    }

    // Diagnosztika – virt_q_after kiolvasása külön (rövid lock)
    let virt_q_after = {
        let st = self.deadline_state.lock().unwrap();
        st.q_pkts
    };

    tracing::debug!(
        target: "bbr.deadline",
        object_size,
        bw_estimate_bytes_per_s = self.max_bandwidth.get_estimate(),
        cwnd_bytes = self.cwnd,
        min_rtt = ?self.min_rtt,
        use_rtt = ?use_rtt,
        app_bps = %app_bps,
        cwnd_bps = %cwnd_bps,
        pace_bps = %pace_bps,
        effective_bps = %effective_bps,
        mss = %mss,
        beta = %beta,
        pps = %pps,
        pkt_num = %pkt_num,
        virt_q_before = %virt_q_before,
        virt_q_after  = %virt_q_after,
        trans_time = ?trans_time,
        guard_ms = guard_ms,
        now = ?now,
        deadline = ?deadline,
        admit,
        "BBR single-path deadline admission (water-filling)"
    );

    admit
}

    
    fn suggest_priority(&self, 
        object_size: u64, 
        deadline: Instant, 
        now: Instant,
        rtt: Duration
    ) -> i32 {
        let Some(cfg) = self.deadline_config.as_ref().filter(|c| c.enabled) else {
            return 0;
        };
        
        // Slack számítás
        let app_bps = self.max_bandwidth.get_estimate() as f64;
        let cwnd_bps = (self.cwnd * 8) as f64 / rtt.as_secs_f64();
        let effective_bps = app_bps.min(cwnd_bps);

        let mss = cfg.default_mss as f64;
        let pps = (effective_bps / 8.0 / mss * cfg.beta).max(1.0);
        let pkt_count = ((object_size + cfg.default_mss as u64 - 1) / cfg.default_mss as u64).max(1);
        let guard = Duration::from_millis(cfg.guard_ms);
        
        let slack = (deadline - (now + rtt / 2)).as_secs_f64() 
                   - (pkt_count as f64) / pps 
                   - guard.as_secs_f64();
        
        slack_to_priority(slack * 1000.0) // Convert to ms
    }
}

fn slack_to_priority(slack_ms: f64) -> i32 {
    if !slack_ms.is_finite() { return 127; }
    if slack_ms <= 0.0 { return 0; }
    if slack_ms < 50.0 { return 8; }
    if slack_ms < 100.0 { return 16; }
    if slack_ms < 250.0 { return 32; }
    if slack_ms < 500.0 { return 64; }
    if slack_ms < 1000.0 { return 96; }
    127
}

/// Configuration for the [`Bbr`] congestion controller
#[derive(Debug, Clone)]
pub struct BbrConfig {
    initial_window: u64,
    min_pacing_bps: u64,
    deadline: Option<DeadlineConfig>,
    // FIX: optional fixed pacing in bits/s
    fixed_pacing_bps: Option<u64>,
}

impl BbrConfig {
    /// Default limit on the amount of outstanding data in bytes.
    ///
    /// Recommended value: `min(10 * max_datagram_size, max(2 * max_datagram_size, 14720))`
    pub fn initial_window(&mut self, value: u64) -> &mut Self {
        self.initial_window = value;
        self
    }
   
    /// For testing purposes only. If set to a non-zero value, this will
    /// enforce a minimum pacing rate in bits per second.
    pub fn min_pacing_bps(&mut self, v: u64) -> &mut Self { self.min_pacing_bps = v; self }

    pub fn fixed_pacing_bps(mut self, bps: u64) -> Self {
        self.fixed_pacing_bps = Some(bps);
        self
    }

    pub fn enable_deadline_scheduler(mut self, enabled: bool) -> Self {
        let mut d = self.deadline.unwrap_or_default();
        d.enabled = enabled;
        self.deadline = Some(d);
        self
    }
    pub fn guard_ms(mut self, ms: u64) -> Self {
        let mut d = self.deadline.unwrap_or_default();
        d.guard_ms = ms;
        self.deadline = Some(d);
        self
    }
    pub fn beta(mut self, beta: f64) -> Self {
        let mut d = self.deadline.unwrap_or_default();
        d.beta = beta;
        self.deadline = Some(d);
        self
    }
    pub fn default_mss(mut self, mss: u32) -> Self {
        let mut d = self.deadline.unwrap_or_default();
        d.default_mss = mss;
        self.deadline = Some(d);
        self
    }
}

impl Default for BbrConfig {
    fn default() -> Self {
        Self {
            initial_window: K_MAX_INITIAL_CONGESTION_WINDOW * BASE_DATAGRAM_SIZE,
            min_pacing_bps: 0,
            deadline: None,
            // FIX: default disabled
            fixed_pacing_bps: None,
        }
    }
}

impl ControllerFactory for BbrConfig {
    fn build(self: Arc<Self>, _now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        Box::new(Bbr::new(self, current_mtu))
    }
}

#[derive(Debug, Default, Copy, Clone)]
struct AckAggregationState {
    max_ack_height: MinMax,
    aggregation_epoch_start_time: Option<Instant>,
    aggregation_epoch_bytes: u64,
}

impl AckAggregationState {
    fn update_ack_aggregation_bytes(
        &mut self,
        newly_acked_bytes: u64,
        now: Instant,
        round: u64,
        max_bandwidth: u64,
    ) -> u64 {
        // Compute how many bytes are expected to be delivered, assuming max
        // bandwidth is correct.
        let expected_bytes_acked = max_bandwidth
            * now
                .saturating_duration_since(self.aggregation_epoch_start_time.unwrap_or(now))
                .as_micros() as u64
            / 1_000_000;

        // Reset the current aggregation epoch as soon as the ack arrival rate is
        // less than or equal to the max bandwidth.
        if self.aggregation_epoch_bytes <= expected_bytes_acked {
            // Reset to start measuring a new aggregation epoch.
            self.aggregation_epoch_bytes = newly_acked_bytes;
            self.aggregation_epoch_start_time = Some(now);
            return 0;
        }

        // Compute how many extra bytes were delivered vs max bandwidth.
        // Include the bytes most recently acknowledged to account for stretch acks.
        self.aggregation_epoch_bytes += newly_acked_bytes;
        let diff = self.aggregation_epoch_bytes - expected_bytes_acked;
        self.max_ack_height.update_max(round, diff);
        diff
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Mode {
    // Startup phase of the connection.
    Startup,
    // After achieving the highest possible bandwidth during the startup, lower
    // the pacing rate in order to drain the queue.
    Drain,
    // Cruising mode.
    ProbeBw,
    // Temporarily slow down sending in order to empty the buffer and measure
    // the real minimum RTT.
    ProbeRtt,
}

// Indicates how the congestion control limits the amount of bytes in flight.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum RecoveryState {
    // Do not limit.
    NotInRecovery,
    // Allow an extra outstanding byte for each byte acknowledged.
    Conservation,
    // Allow two extra outstanding bytes for each byte acknowledged (slow
    // start).
    Growth,
}

impl RecoveryState {
    pub(super) fn in_recovery(&self) -> bool {
        !matches!(self, Self::NotInRecovery)
    }
}

#[derive(Debug, Clone, Default)]
struct LossState {
    lost_bytes: u64,
}

impl LossState {
    pub(super) fn reset(&mut self) {
        self.lost_bytes = 0;
    }

    pub(super) fn has_losses(&self) -> bool {
        self.lost_bytes != 0
    }
}

fn calculate_min_window(current_mtu: u64) -> u64 {
    4 * current_mtu
}

// The gain used for the STARTUP, equal to 2/ln(2).
const K_DEFAULT_HIGH_GAIN: f32 = 2.885;
// The newly derived CWND gain for STARTUP, 2.
const K_DERIVED_HIGH_CWNDGAIN: f32 = 2.0;
// The cycle of gains used during the ProbeBw stage.
const K_PACING_GAIN: [f32; 8] = [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

const K_STARTUP_GROWTH_TARGET: f32 = 1.25;
const K_ROUND_TRIPS_WITHOUT_GROWTH_BEFORE_EXITING_STARTUP: u8 = 3;

// Do not allow initial congestion window to be greater than 200 packets.
const K_MAX_INITIAL_CONGESTION_WINDOW: u64 = 200;

const PROBE_RTT_BASED_ON_BDP: bool = true;
const DRAIN_TO_TARGET: bool = true;