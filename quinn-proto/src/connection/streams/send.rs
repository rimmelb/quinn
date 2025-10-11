use bytes::Bytes;
use std::{collections::VecDeque, time::Instant};
use thiserror::Error;

use crate::{VarInt, connection::send_buffer::SendBuffer, frame};

#[derive(Debug, Clone)]
struct ChunkProgress {
    len: u64,
    written: u64,
    acked: u64,
}

impl ChunkProgress {
    fn new(len: u64) -> Self {
        Self {
            len,
            written: 0,
            acked: 0,
        }
    }

    fn apply_write(&mut self, bytes: u64) -> u64 {
        if bytes == 0 || self.written >= self.len {
            return bytes;
        }
        let need = self.len - self.written;
        let take = need.min(bytes);
        self.written += take;
        bytes - take
    }

    fn apply_ack(&mut self, bytes: u64) -> u64 {
        if bytes == 0 || self.acked >= self.len {
            return bytes;
        }
        let need = self.len - self.acked;
        let take = need.min(bytes);
        self.acked += take;
        bytes - take
    }

    fn is_ready(&self) -> bool {
        self.written >= self.len
    }

    fn outstanding(&self) -> u64 {
        self.len.saturating_sub(self.acked)
    }

    fn available(&self) -> u64 {
        self.written.saturating_sub(self.acked)
    }

    fn is_done(&self) -> bool {
        self.acked >= self.len
    }
}

#[derive(Debug, Clone)]
pub struct ObjectEntry {
    header: ChunkProgress,
    payload: ChunkProgress,
    deadline_ms: Option<u64>,
    admitted: bool,
    failed_attempts: u8,
    retry_at: Option<Instant>,
}

impl ObjectEntry {
    fn with_header(header_len: u64) -> Self {
        Self {
            header: ChunkProgress::new(header_len),
            payload: ChunkProgress::new(0),
            deadline_ms: None,
            admitted: false,
            failed_attempts: 0,
            retry_at: None,
        }
    }

    fn set_payload(&mut self, payload_len: u64, deadline_ms: Option<u64>) {
        self.payload = ChunkProgress::new(payload_len);
        self.deadline_ms = deadline_ms;
        self.failed_attempts = 0;
        self.retry_at = None;
    }

    fn apply_write(&mut self, bytes: u64) -> u64 {
        let bytes = self.header.apply_write(bytes);
        self.payload.apply_write(bytes)
    }

    fn apply_ack(&mut self, bytes: u64) -> u64 {
        let bytes = self.header.apply_ack(bytes);
        self.payload.apply_ack(bytes)
    }

    fn is_ready(&self) -> bool {
        if !self.header.is_ready() {
            return false;
        }
        if self.payload.len == 0 {
            return true;
        }
        self.payload.written >= self.payload.len
    }

    fn outstanding(&self) -> u64 {
        self.header.outstanding() + self.payload.outstanding()
    }

    fn available(&self) -> u64 {
        self.header.available() + self.payload.available()
    }

    fn total_len(&self) -> u64 {
        self.header.len + self.payload.len
    }

    fn is_done(&self) -> bool {
        self.header.is_done() && self.payload.is_done()
    }
}

#[derive(Debug, Clone)]
pub(super) struct ChunkStatus {
    pub ready: bool,
    pub outstanding: u64,
}

#[derive(Debug, Clone)]
pub(super) struct ObjectStatus {
    pub ready: bool,
    pub outstanding: u64,
    pub available: u64,
    pub total_len: u64,
    pub deadline_ms: Option<u64>,
    pub admitted: bool,
}

#[derive(Debug)]
pub(super) struct StreamHints {
    /// Subgroup header (opcionális)
    subgroup: Option<ChunkProgress>,
    /// Object lista: header + payload
    objects: VecDeque<ObjectEntry>,
}

impl StreamHints {
    pub(super) fn new() -> Self {
        Self {
            subgroup: None,
            objects: VecDeque::new(),
        }
    }

    pub fn append_subgroup_header_size(&mut self, subgroup_header_size: u64) {
        self.subgroup = Some(ChunkProgress::new(subgroup_header_size));
    }

    pub fn append_object_header_size(&mut self, object_header_size: u64) {
        self.objects
            .push_back(ObjectEntry::with_header(object_header_size));
    }

    pub fn append_object_size(&mut self, object_size: u64, deadline: Option<u64>) {
        if let Some(entry) = self.objects.back_mut() {
            if entry.payload.len == 0 {
                entry.set_payload(object_size, deadline);
                return;
            }
        }
        let mut entry = ObjectEntry::with_header(0);
        entry.set_payload(object_size, deadline);
        self.objects.push_back(entry);
    }

    /// Subgroup header kész?
    pub(super) fn subgroup_status(&self) -> Option<ChunkStatus> {
        self.subgroup.as_ref().map(|chunk| ChunkStatus {
            ready: chunk.is_ready(),
            outstanding: chunk.outstanding(),
        })
    }

    /// Első object státusza
    pub(super) fn current_object_status(&self) -> Option<ObjectStatus> {
        let entry = self.objects.front()?;
        Some(ObjectStatus {
            ready: entry.is_ready(),
            outstanding: entry.outstanding(),
            available: entry.available(),
            total_len: entry.total_len(),
            deadline_ms: entry.deadline_ms,
            admitted: entry.admitted,
        })
    }

    /// Mark current object as admitted
    pub(super) fn mark_current_object_admitted(&mut self) {
        if let Some(entry) = self.objects.front_mut() {
            entry.admitted = true;
        }
    }

    /// Discard current object
    pub(super) fn discard_current_object(&mut self) -> Option<u64> {
        self.objects.pop_front().map(|entry| entry.total_len())
    }

    /// Van-e részleges object (header kész, payload nem)
    pub fn has_partial_object(&self) -> bool {
        self.current_object_status()
            .map(|s| s.total_len > 0 && !s.ready)
            .unwrap_or(false)
    }

    pub(super) fn on_bytes_written(&mut self, mut bytes: u64) {
        if bytes == 0 {
            return;
        }
        if let Some(sub) = &mut self.subgroup {
            bytes = sub.apply_write(bytes);
        }
        for entry in self.objects.iter_mut() {
            if bytes == 0 {
                break;
            }
            bytes = entry.apply_write(bytes);
        }
    }

    pub(super) fn on_bytes_acked(&mut self, mut bytes: u64) {
        if bytes == 0 {
            return;
        }
        if let Some(sub) = &mut self.subgroup {
            bytes = sub.apply_ack(bytes);
            if sub.is_done() {
                self.subgroup = None;
            }
        }
        while bytes > 0 {
            match self.objects.front_mut() {
                Some(entry) => {
                    let before = bytes;
                    bytes = entry.apply_ack(bytes);
                    if entry.is_done() {
                        self.objects.pop_front();
                    } else if before == bytes {
                        break;
                    }
                }
                None => break,
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct Send {
    pub(super) max_data: u64,
    pub(super) state: SendState,
    pub(super) pending: SendBuffer,
    pub(super) priority: i32,
    // Deadline alapú ütemezéshez opcionális abszolút határidő
    pub(super) deadline: Option<u64>,
    pub(super) slack_ms: Option<f64>,
    // Priority frissült-e úgy, hogy a pending queue entry-t frissíteni kell
    pub(super) priority_dirty: bool,
    /// Whether a frame containing a FIN bit must be transmitted, even if we don't have any new data
    pub(super) fin_pending: bool,
    /// Whether this stream is in the `connection_blocked` list of `Streams`
    pub(super) connection_blocked: bool,
    /// The reason the peer wants us to stop, if `STOP_SENDING` was received
    pub(super) stop_reason: Option<VarInt>,
    /// pending if object drop
    pub(super) stream_pending: bool,
    ///Size of the objects including the size of the subgroupheader
    pub(super) object_sizes: Option<StreamHints>,
}

impl Send {
    pub(super) fn new(max_data: VarInt) -> Box<Self> {
        Box::new(Self {
            max_data: max_data.into(),
            state: SendState::Ready,
            pending: SendBuffer::new(),
            priority: 0,
            deadline: None,
            slack_ms: None,
            priority_dirty: false,
            fin_pending: false,
            connection_blocked: false,
            stop_reason: None,
            object_sizes: None,
            stream_pending: false,
        })
    }

    /// Whether the stream has been reset
    pub(super) fn is_reset(&self) -> bool {
        matches!(self.state, SendState::ResetSent)
    }

    pub(super) fn finish(&mut self) -> Result<(), FinishError> {
        if let Some(error_code) = self.stop_reason {
            Err(FinishError::Stopped(error_code))
        } else if self.state == SendState::Ready {
            self.state = SendState::DataSent {
                finish_acked: false,
            };
            self.fin_pending = true;
            Ok(())
        } else {
            Err(FinishError::ClosedStream)
        }
    }

    pub(super) fn write<S: BytesSource>(
        &mut self,
        source: &mut S,
        limit: u64,
    ) -> Result<Written, WriteError> {
        if !self.is_writable() {
            return Err(WriteError::ClosedStream);
        }
        if let Some(error_code) = self.stop_reason {
            return Err(WriteError::Stopped(error_code));
        }
        let budget = self.max_data - self.pending.offset();
        if budget == 0 {
            return Err(WriteError::Blocked);
        }
        let mut limit = limit.min(budget) as usize;

        let mut result = Written::default();
        loop {
            let (chunk, chunks_consumed) = source.pop_chunk(limit);
            result.chunks += chunks_consumed;
            result.bytes += chunk.len();

            if chunk.is_empty() {
                break;
            }

            limit -= chunk.len();
            let chunk_len = chunk.len() as u64;
            self.pending.write(chunk);
            if let Some(hints) = &mut self.object_sizes {
                hints.on_bytes_written(chunk_len);
            }
        }

        Ok(result)
    }

    /// Update stream state due to a reset sent by the local application
    pub(super) fn reset(&mut self) {
        use SendState::*;
        if let DataSent { .. } | Ready = self.state {
            self.state = ResetSent;
        }
        self.object_sizes = None;
    }

    /// Handle STOP_SENDING
    ///
    /// Returns true if the stream was stopped due to this frame, and false
    /// if it had been stopped before
    pub(super) fn try_stop(&mut self, error_code: VarInt) -> bool {
        if self.stop_reason.is_none() {
            self.stop_reason = Some(error_code);
            true
        } else {
            false
        }
    }

    /// Returns whether the stream has been finished and all data has been acknowledged by the peer
    pub(super) fn ack(&mut self, frame: frame::StreamMeta) -> bool {
        let offsets = frame.offsets.clone();
        self.pending.ack(offsets.clone());
        if let Some(hints) = &mut self.object_sizes {
            let acked = offsets.end.saturating_sub(offsets.start);
            hints.on_bytes_acked(acked);
        }
        match self.state {
            SendState::DataSent {
                ref mut finish_acked,
            } => {
                *finish_acked |= frame.fin;
                *finish_acked && self.pending.is_fully_acked()
            }
            _ => false,
        }
    }

    /// Handle increase to stream-level flow control limit
    ///
    /// Returns whether the stream was unblocked
    pub(super) fn increase_max_data(&mut self, offset: u64) -> bool {
        if offset <= self.max_data || self.state != SendState::Ready {
            return false;
        }
        let was_blocked = self.pending.offset() == self.max_data;
        self.max_data = offset;
        was_blocked
    }

    pub(super) fn offset(&self) -> u64 {
        self.pending.offset()
    }

    pub(super) fn is_pending(&self) -> bool {
        self.pending.unacked() > 0 || !self.fin_pending
    }

    pub(super) fn is_writable(&self) -> bool {
        matches!(self.state, SendState::Ready)
    }

    /// Belső API: stream deadline beállítása
    pub(super) fn set_deadline(&mut self, deadline: Option<u64>) {
        self.deadline = deadline;
        // A prioritást csak akkor frissítjük automatikusan, ha van slack számítás felsőbb rétegen.
        self.priority_dirty = true;
    }

    /// Internal API: update slack (ms) and map it to a discrete priority band
    pub(super) fn set_slack_ms(&mut self, slack_ms: f64) {
        self.slack_ms = Some(slack_ms);
        let new_prio = slack_to_priority(slack_ms);
        if new_prio != self.priority {
            self.priority = new_prio;
            self.priority_dirty = true;
        }
    }

    pub(super) fn append_subgroup_header_size(&mut self, subgroup_header_size: u64) {
        self.object_sizes = Some(StreamHints::new());
        if let Some(hints) = &mut self.object_sizes {
            hints.append_subgroup_header_size(subgroup_header_size);
        }
    }

    pub(super) fn append_object_header_size(&mut self, object_header_size: u64) {
        if self.object_sizes.is_none() {
            self.object_sizes = Some(StreamHints::new());
        }
        if let Some(hints) = &mut self.object_sizes {
            hints.append_object_header_size(object_header_size);
        }
    }

    pub(super) fn append_object_size(&mut self, object_size: u64, deadline: Option<u64>) {
        if self.object_sizes.is_none() {
            self.object_sizes = Some(StreamHints::new());
        }
        if let Some(hints) = &mut self.object_sizes {
            hints.append_object_size(object_size, deadline);
        }
    }
}

/// Alkalmazásban korábban használt threshold mapping integrálása.
#[inline]
pub(super) fn slack_to_priority(slack_ms: f64) -> i32 {
    if !slack_ms.is_finite() {
        return 127;
    }
    if slack_ms <= 0.0 {
        return 0;
    }
    if slack_ms < 50.0 {
        return 8;
    }
    if slack_ms < 100.0 {
        return 16;
    }
    if slack_ms < 250.0 {
        return 32;
    }
    if slack_ms < 500.0 {
        return 64;
    }
    if slack_ms < 1000.0 {
        return 96;
    }
    127
}

/// A [`BytesSource`] implementation for `&'a mut [Bytes]`
///
/// The type allows to dequeue [`Bytes`] chunks from an array of chunks, up to
/// a configured limit.
pub(crate) struct BytesArray<'a> {
    /// The wrapped slice of `Bytes`
    chunks: &'a mut [Bytes],
    /// The amount of chunks consumed from this source
    consumed: usize,
}

impl<'a> BytesArray<'a> {
    pub(crate) fn from_chunks(chunks: &'a mut [Bytes]) -> Self {
        Self {
            chunks,
            consumed: 0,
        }
    }
}

impl BytesSource for BytesArray<'_> {
    fn pop_chunk(&mut self, limit: usize) -> (Bytes, usize) {
        // The loop exists to skip empty chunks while still marking them as
        // consumed
        let mut chunks_consumed = 0;

        while self.consumed < self.chunks.len() {
            let chunk = &mut self.chunks[self.consumed];

            if chunk.len() <= limit {
                let chunk = std::mem::take(chunk);
                self.consumed += 1;
                chunks_consumed += 1;
                if chunk.is_empty() {
                    continue;
                }
                return (chunk, chunks_consumed);
            } else if limit > 0 {
                let chunk = chunk.split_to(limit);
                return (chunk, chunks_consumed);
            } else {
                break;
            }
        }

        (Bytes::new(), chunks_consumed)
    }
}

/// A [`BytesSource`] implementation for `&[u8]`
///
/// The type allows to dequeue a single [`Bytes`] chunk, which will be lazily
/// created from a reference. This allows to defer the allocation until it is
/// known how much data needs to be copied.
pub(crate) struct ByteSlice<'a> {
    /// The wrapped byte slice
    data: &'a [u8],
}

impl<'a> ByteSlice<'a> {
    pub(crate) fn from_slice(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl BytesSource for ByteSlice<'_> {
    fn pop_chunk(&mut self, limit: usize) -> (Bytes, usize) {
        let limit = limit.min(self.data.len());
        if limit == 0 {
            return (Bytes::new(), 0);
        }

        let chunk = Bytes::from(self.data[..limit].to_owned());
        self.data = &self.data[chunk.len()..];

        let chunks_consumed = usize::from(self.data.is_empty());
        (chunk, chunks_consumed)
    }
}

/// A source of one or more buffers which can be converted into `Bytes` buffers on demand
///
/// The purpose of this data type is to defer conversion as long as possible,
/// so that no heap allocation is required in case no data is writable.
pub(super) trait BytesSource {
    /// Returns the next chunk from the source of owned chunks.
    ///
    /// This method will consume parts of the source.
    /// Calling it will yield `Bytes` elements up to the configured `limit`.
    ///
    /// The method returns a tuple:
    /// - The first item is the yielded `Bytes` element. The element will be
    ///   empty if the limit is zero or no more data is available.
    /// - The second item returns how many complete chunks inside the source had
    ///   had been consumed. This can be less than 1, if a chunk inside the
    ///   source had been truncated in order to adhere to the limit. It can also
    ///   be more than 1, if zero-length chunks had been skipped.
    fn pop_chunk(&mut self, limit: usize) -> (Bytes, usize);
}

/// Indicates how many bytes and chunks had been transferred in a write operation
#[derive(Debug, Default, PartialEq, Eq, Clone, Copy)]
pub struct Written {
    /// The amount of bytes which had been written
    pub bytes: usize,
    /// The amount of full chunks which had been written
    ///
    /// If a chunk was only partially written, it will not be counted by this field.
    pub chunks: usize,
}

/// Errors triggered while writing to a send stream
#[derive(Debug, Error, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum WriteError {
    /// The peer is not able to accept additional data, or the connection is congested.
    ///
    /// If the peer issues additional flow control credit, a [`StreamEvent::Writable`] event will
    /// be generated, indicating that retrying the write might succeed.
    ///
    /// [`StreamEvent::Writable`]: crate::StreamEvent::Writable
    #[error("unable to accept further writes")]
    Blocked,
    /// The peer is no longer accepting data on this stream, and it has been implicitly reset. The
    /// stream cannot be finished or further written to.
    ///
    /// Carries an application-defined error code.
    ///
    /// [`StreamEvent::Finished`]: crate::StreamEvent::Finished
    #[error("stopped by peer: code {0}")]
    Stopped(VarInt),
    /// The stream has not been opened or has already been finished or reset
    #[error("closed stream")]
    ClosedStream,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(super) enum SendState {
    /// Sending new data
    Ready,
    /// Stream was finished; now sending retransmits only
    DataSent { finish_acked: bool },
    /// Sent RESET
    ResetSent,
}

/// Reasons why attempting to finish a stream might fail
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FinishError {
    /// The peer is no longer accepting data on this stream. No
    /// [`StreamEvent::Finished`] event will be emitted for this stream.
    ///
    /// Carries an application-defined error code.
    ///
    /// [`StreamEvent::Finished`]: crate::StreamEvent::Finished
    #[error("stopped by peer: code {0}")]
    Stopped(VarInt),
    /// The stream has not been opened or was already finished or reset
    #[error("closed stream")]
    ClosedStream,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn bytes_array() {
        let full = b"Hello World 123456789 ABCDEFGHJIJKLMNOPQRSTUVWXYZ".to_owned();
        for limit in 0..full.len() {
            let mut chunks = [
                Bytes::from_static(b""),
                Bytes::from_static(b"Hello "),
                Bytes::from_static(b"Wo"),
                Bytes::from_static(b""),
                Bytes::from_static(b"r"),
                Bytes::from_static(b"ld"),
                Bytes::from_static(b""),
                Bytes::from_static(b" 12345678"),
                Bytes::from_static(b"9 ABCDE"),
                Bytes::from_static(b"F"),
                Bytes::from_static(b"GHJIJKLMNOPQRSTUVWXYZ"),
            ];
            let num_chunks = chunks.len();
            let last_chunk_len = chunks[chunks.len() - 1].len();

            let mut array = BytesArray::from_chunks(&mut chunks);

            let mut buf = Vec::new();
            let mut chunks_popped = 0;
            let mut chunks_consumed = 0;
            let mut remaining = limit;
            loop {
                let (chunk, consumed) = array.pop_chunk(remaining);
                chunks_consumed += consumed;

                if !chunk.is_empty() {
                    buf.extend_from_slice(&chunk);
                    remaining -= chunk.len();
                    chunks_popped += 1;
                } else {
                    break;
                }
            }

            assert_eq!(&buf[..], &full[..limit]);

            if limit == full.len() {
                // Full consumption of the last chunk
                assert_eq!(chunks_consumed, num_chunks);
                // Since there are empty chunks, we consume more than there are popped
                assert_eq!(chunks_consumed, chunks_popped + 3);
            } else if limit > full.len() - last_chunk_len {
                // Partial consumption of the last chunk
                assert_eq!(chunks_consumed, num_chunks - 1);
                assert_eq!(chunks_consumed, chunks_popped + 2);
            }
        }
    }

    #[test]
    fn byte_slice() {
        let full = b"Hello World 123456789 ABCDEFGHJIJKLMNOPQRSTUVWXYZ".to_owned();
        for limit in 0..full.len() {
            let mut array = ByteSlice::from_slice(&full[..]);

            let mut buf = Vec::new();
            let mut chunks_popped = 0;
            let mut chunks_consumed = 0;
            let mut remaining = limit;
            loop {
                let (chunk, consumed) = array.pop_chunk(remaining);
                chunks_consumed += consumed;

                if !chunk.is_empty() {
                    buf.extend_from_slice(&chunk);
                    remaining -= chunk.len();
                    chunks_popped += 1;
                } else {
                    break;
                }
            }

            assert_eq!(&buf[..], &full[..limit]);
            if limit != 0 {
                assert_eq!(chunks_popped, 1);
            } else {
                assert_eq!(chunks_popped, 0);
            }

            if limit == full.len() {
                assert_eq!(chunks_consumed, 1);
            } else {
                assert_eq!(chunks_consumed, 0);
            }
        }
    }
}
