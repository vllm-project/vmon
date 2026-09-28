// SPDX-License-Identifier: Apache-2.0

//! ZMQ KV cache event subscriber and aggregator.
//!
//! vLLM publishes KV cache block lifecycle events (BlockStored / BlockRemoved /
//! AllBlocksCleared) over a ZMQ PUB socket when `--kv-events-config` is set.
//!
//! This module subscribes to those events via libzmq and aggregates them into
//! per-interval metrics that the scraper drains each tick.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

/// How long each blocking recv waits before re-checking the stop flag (ms).
/// Bounds both the shutdown latency and how long a dropped subscriber's
/// socket keeps its file descriptors open.
const STOP_POLL_MS: i32 = 500;

/// Process-wide ZMQ context shared by every subscriber socket.
///
/// Sharing a context bounds IO-thread, reaper-thread and file-descriptor
/// overhead. Socket creation failures are logged per endpoint.
fn shared_context() -> &'static zmq::Context {
    static CTX: OnceLock<zmq::Context> = OnceLock::new();
    CTX.get_or_init(zmq::Context::new)
}

/// Aggregated KV event metrics for a single scrape interval.
#[derive(Debug, Clone, Default)]
pub struct KVEventMetrics {
    /// Data-parallel rank this subscriber follows (per worker), when known.
    /// Stamped by the scraper from the `--zmq-port` expansion; `None` on
    /// replays of old recordings. Display code falls back to the vec index
    /// when absent.
    pub dp_rank: Option<u16>,
    /// Blocks cached this interval.
    pub blocks_stored: u64,
    /// Blocks evicted this interval.
    pub blocks_removed: u64,
    /// Tokens cached this interval (from BlockStored token_ids).
    pub tokens_stored: u64,
    /// Running count of active cache blocks (+store, -remove, 0 on clear).
    pub active_blocks: i64,
    /// Total events received this interval.
    pub total_events: u64,
    /// Sequence number gaps detected this interval.
    pub seq_gaps: u64,
}

/// Internal mutable state shared between the subscriber thread and drain().
struct InnerState {
    // Per-interval accumulators (reset on drain)
    blocks_stored: u64,
    blocks_removed: u64,
    tokens_stored: u64,
    total_events: u64,
    seq_gaps: u64,
    // Persistent across drains
    active_blocks: i64,
    last_seq: i64,
}

/// Per-endpoint ZMQ event subscriber handle.
///
/// Spawns a background thread that connects a SUB socket (on the shared
/// process-wide context) and continuously receives messages into a shared
/// buffer. The scraper calls [`drain()`] each tick to collect aggregated
/// metrics. Dropping the handle signals the thread to exit and close its
/// socket, releasing the socket's file descriptors within one
/// [`STOP_POLL_MS`] tick.
pub struct KVEventSubscriber {
    state: Arc<Mutex<InnerState>>,
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl KVEventSubscriber {
    /// Spawn a subscriber connecting to the given ZMQ PUB endpoint.
    ///
    /// `topic` is the ZMQ subscription prefix (empty string = all messages).
    pub fn spawn(endpoint: String, topic: String) -> Self {
        let state = Arc::new(Mutex::new(InnerState {
            blocks_stored: 0,
            blocks_removed: 0,
            tokens_stored: 0,
            total_events: 0,
            seq_gaps: 0,
            active_blocks: 0,
            last_seq: -1,
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_state = Arc::clone(&state);
        let thread_stop = Arc::clone(&stop);
        let handle = thread::Builder::new()
            .name(format!("zmq-sub-{endpoint}"))
            .spawn(move || {
                subscriber_loop(&endpoint, &topic, thread_state, &thread_stop);
            })
            .expect("failed to spawn ZMQ subscriber thread");
        Self {
            state,
            stop,
            thread: Some(handle),
        }
    }

    /// Non-blocking drain: return accumulated metrics and reset per-interval counters.
    pub fn drain(&self) -> KVEventMetrics {
        let mut s = self.state.lock().unwrap();
        let m = KVEventMetrics {
            dp_rank: None, // stamped by the scraper, which knows the port map
            blocks_stored: s.blocks_stored,
            blocks_removed: s.blocks_removed,
            tokens_stored: s.tokens_stored,
            active_blocks: s.active_blocks,
            total_events: s.total_events,
            seq_gaps: s.seq_gaps,
        };
        s.blocks_stored = 0;
        s.blocks_removed = 0;
        s.tokens_stored = 0;
        s.total_events = 0;
        s.seq_gaps = 0;
        m
    }

    /// Stop the subscriber thread and wait for it to exit (used by tests to
    /// observe deterministic shutdown; production drops detach instead).
    #[cfg(test)]
    fn shutdown_and_join(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            h.join().expect("subscriber thread panicked");
        }
    }
}

impl Drop for KVEventSubscriber {
    /// Signal the subscriber thread to stop, then detach it. The thread
    /// wakes within one [`STOP_POLL_MS`] recv tick, drops its socket
    /// (closing the fds), and exits. Detaching rather than joining keeps a
    /// reconcile that drops many subscribers at once from stalling the
    /// scrape loop.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.thread.take();
    }
}

/// Receive exactly one event envelope without accumulating arbitrary multipart frames.
/// None closes the subscriber on a malformed oversized envelope.
fn recv_event_frames(sub: &zmq::Socket) -> Result<Option<Vec<Vec<u8>>>, zmq::Error> {
    let mut frames = Vec::with_capacity(3);
    loop {
        frames.push(sub.recv_bytes(0)?);
        let more = sub.get_rcvmore()?;
        if !more {
            return Ok(Some(frames));
        }
        if frames.len() == 3 {
            return Ok(None);
        }
    }
}

fn subscriber_loop(endpoint: &str, topic: &str, state: Arc<Mutex<InnerState>>, stop: &AtomicBool) {
    let sub = match shared_context().socket(zmq::SUB) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(endpoint, "ZMQ socket creation failed: {e}");
            return;
        }
    };
    // Bounded recv lets the loop notice the stop flag; zero linger makes
    // the socket close immediately on drop.
    if let Err(e) = sub
        .set_rcvtimeo(STOP_POLL_MS)
        .and(sub.set_linger(0))
        .and(sub.set_maxmsgsize(4 * 1024 * 1024))
        .and(sub.set_rcvhwm(32))
    {
        tracing::warn!(endpoint, "ZMQ socket options failed: {e}");
        return;
    }
    if let Err(e) = sub.connect(endpoint) {
        tracing::warn!(endpoint, "ZMQ connect failed: {e}");
        return;
    }
    if let Err(e) = sub.set_subscribe(topic.as_bytes()) {
        tracing::warn!(topic, "ZMQ subscribe failed: {e}");
        return;
    }
    tracing::info!(endpoint, "ZMQ KV event subscriber connected");

    while !stop.load(Ordering::Relaxed) {
        let frames = match recv_event_frames(&sub) {
            Ok(Some(f)) => f,
            Ok(None) => {
                tracing::warn!(endpoint, "closing ZMQ subscriber: too many event frames");
                return;
            }
            Err(zmq::Error::EAGAIN) => continue, // recv timeout: re-check stop flag
            Err(e) => {
                tracing::debug!("ZMQ recv error: {e}");
                thread::sleep(std::time::Duration::from_secs(1));
                continue;
            }
        };

        // 3-frame multipart: [topic, seq_bytes(8B BE i64), payload(msgpack)]
        if frames.len() < 3 {
            continue;
        }

        let seq_frame = &frames[1];
        let payload = &frames[2];

        // Sequence number tracking
        if seq_frame.len() >= 8 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&seq_frame[..8]);
            let seq = i64::from_be_bytes(buf);
            let mut s = state.lock().unwrap();
            if s.last_seq >= 0 && seq > s.last_seq + 1 {
                s.seq_gaps += (seq - s.last_seq - 1) as u64;
            }
            s.last_seq = seq;
        }

        // Decode and accumulate events
        for ev in decode_event_batch(payload) {
            let mut s = state.lock().unwrap();
            s.total_events += 1;
            match ev {
                DecodedEvent::BlockStored {
                    block_count,
                    token_count,
                } => {
                    s.blocks_stored += block_count as u64;
                    s.tokens_stored += token_count as u64;
                    s.active_blocks += block_count as i64;
                }
                DecodedEvent::BlockRemoved { block_count } => {
                    s.blocks_removed += block_count as u64;
                    s.active_blocks -= block_count as i64;
                }
                DecodedEvent::AllBlocksCleared => {
                    s.active_blocks = 0;
                }
            }
        }
    }
    tracing::debug!(endpoint, "ZMQ KV event subscriber stopped");
}

// ── msgpack decoding ──

enum DecodedEvent {
    BlockStored {
        block_count: usize,
        token_count: usize,
    },
    BlockRemoved {
        block_count: usize,
    },
    AllBlocksCleared,
}

/// Decode a msgpack KVEventBatch payload.
///
/// Two wire formats exist, both `KVEventBatch = [ts: f64, events: [...],
/// data_parallel_rank?: int]` at the top level:
///
/// Array-encoded events (msgspec `array_like=True, tag=True`):
/// ```text
/// BlockStored  = ["BlockStored",  [hashes], parent_hash, [token_ids], block_size, ...]
/// BlockRemoved = ["BlockRemoved", [hashes], medium]
/// AllBlocksCleared = ["AllBlocksCleared"]
/// ```
///
/// Map-encoded events use a `type` tag:
/// ```text
/// {"type": "BlockStored", "block_hashes": [...], "token_ids": [...],
///  "block_size": 256, "medium": "GPU", "group_idx": 0, ...}
/// ```
fn decode_event_batch(payload: &[u8]) -> Vec<DecodedEvent> {
    let mut cursor = payload;
    let val = match rmpv::decode::read_value_with_max_depth(&mut cursor, 64) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let Some(arr) = val.as_array() else {
        return Vec::new();
    };
    if arr.len() < 2 {
        return Vec::new();
    }
    let Some(events_arr) = arr[1].as_array() else {
        return Vec::new();
    };

    let mut out = Vec::with_capacity(events_arr.len());
    for ev in events_arr {
        let decoded = if let Some(ev_arr) = ev.as_array() {
            decode_array_event(ev_arr)
        } else if let Some(ev_map) = ev.as_map() {
            decode_map_event(ev_map)
        } else {
            None
        };
        if let Some(d) = decoded {
            out.push(d);
        }
    }
    out
}

/// Older array-shaped event (positional fields after the tag).
fn decode_array_event(ev_arr: &[rmpv::Value]) -> Option<DecodedEvent> {
    let tag = ev_arr.first()?.as_str()?;
    match tag {
        "BlockStored" if ev_arr.len() >= 5 => Some(DecodedEvent::BlockStored {
            block_count: ev_arr[1].as_array().map_or(0, |a| a.len()),
            token_count: ev_arr[3].as_array().map_or(0, |a| a.len()),
        }),
        "BlockRemoved" if ev_arr.len() >= 2 => Some(DecodedEvent::BlockRemoved {
            block_count: ev_arr[1].as_array().map_or(0, |a| a.len()),
        }),
        "AllBlocksCleared" => Some(DecodedEvent::AllBlocksCleared),
        _ => None, // ignore unknown event types
    }
}

/// Newer map-shaped event (`type` key carries the tag).
fn decode_map_event(ev_map: &[(rmpv::Value, rmpv::Value)]) -> Option<DecodedEvent> {
    let get = |key: &str| -> Option<&rmpv::Value> {
        ev_map.iter().find(|(k, _)| k.as_str() == Some(key)).map(|(_, v)| v)
    };
    let list_len =
        |key: &str| -> usize { get(key).and_then(|v| v.as_array()).map_or(0, |a| a.len()) };
    match get("type")?.as_str()? {
        "BlockStored" => Some(DecodedEvent::BlockStored {
            block_count: list_len("block_hashes"),
            token_count: list_len("token_ids"),
        }),
        "BlockRemoved" => Some(DecodedEvent::BlockRemoved {
            block_count: list_len("block_hashes"),
        }),
        "AllBlocksCleared" => Some(DecodedEvent::AllBlocksCleared),
        _ => None, // ignore unknown event types
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal msgpack KVEventBatch payload using rmpv.
    fn make_batch(events: Vec<rmpv::Value>) -> Vec<u8> {
        let batch = rmpv::Value::Array(vec![
            rmpv::Value::F64(1234567890.0), // ts
            rmpv::Value::Array(events),     // events
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &batch).unwrap();
        buf
    }

    fn hash_val(n: u64) -> rmpv::Value {
        rmpv::Value::Integer(n.into())
    }

    #[test]
    fn decode_block_stored() {
        let ev = rmpv::Value::Array(vec![
            rmpv::Value::String("BlockStored".into()),
            rmpv::Value::Array(vec![hash_val(1), hash_val(2)]), // 2 block hashes
            rmpv::Value::Nil,                                   // parent_hash
            rmpv::Value::Array(vec![
                // token_ids
                rmpv::Value::Integer(100.into()),
                rmpv::Value::Integer(101.into()),
                rmpv::Value::Integer(102.into()),
            ]),
            rmpv::Value::Integer(16.into()),   // block_size
            rmpv::Value::Nil,                  // lora_id
            rmpv::Value::String("GPU".into()), // medium
            rmpv::Value::Nil,                  // lora_name
        ]);
        let payload = make_batch(vec![ev]);
        let decoded = decode_event_batch(&payload);
        assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            DecodedEvent::BlockStored {
                block_count,
                token_count,
            } => {
                assert_eq!(*block_count, 2);
                assert_eq!(*token_count, 3);
            }
            _ => panic!("expected BlockStored"),
        }
    }

    #[test]
    fn decode_block_removed() {
        let ev = rmpv::Value::Array(vec![
            rmpv::Value::String("BlockRemoved".into()),
            rmpv::Value::Array(vec![hash_val(1)]), // 1 hash
            rmpv::Value::String("GPU".into()),     // medium
        ]);
        let payload = make_batch(vec![ev]);
        let decoded = decode_event_batch(&payload);
        assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            DecodedEvent::BlockRemoved { block_count } => assert_eq!(*block_count, 1),
            _ => panic!("expected BlockRemoved"),
        }
    }

    #[test]
    fn decode_all_blocks_cleared() {
        let ev = rmpv::Value::Array(vec![rmpv::Value::String("AllBlocksCleared".into())]);
        let payload = make_batch(vec![ev]);
        let decoded = decode_event_batch(&payload);
        assert_eq!(decoded.len(), 1);
        assert!(matches!(decoded[0], DecodedEvent::AllBlocksCleared));
    }

    #[test]
    fn decode_mixed_batch() {
        let events = vec![
            rmpv::Value::Array(vec![
                rmpv::Value::String("BlockStored".into()),
                rmpv::Value::Array(vec![hash_val(1)]),
                rmpv::Value::Nil,
                rmpv::Value::Array(vec![rmpv::Value::Integer(42.into())]),
                rmpv::Value::Integer(16.into()),
                rmpv::Value::Nil,
                rmpv::Value::Nil,
                rmpv::Value::Nil,
            ]),
            rmpv::Value::Array(vec![
                rmpv::Value::String("BlockRemoved".into()),
                rmpv::Value::Array(vec![hash_val(2), hash_val(3)]),
                rmpv::Value::Nil,
            ]),
            rmpv::Value::Array(vec![rmpv::Value::String("AllBlocksCleared".into())]),
            // Unknown event type — should be skipped
            rmpv::Value::Array(vec![rmpv::Value::String("FutureEvent".into())]),
        ];
        let payload = make_batch(events);
        let decoded = decode_event_batch(&payload);
        assert_eq!(decoded.len(), 3);
    }

    /// Decode map-shaped events with a `type` tag and named fields.
    #[test]
    fn decode_map_shaped_events() {
        let s = |v: &str| rmpv::Value::String(v.into());
        let stored = rmpv::Value::Map(vec![
            (s("type"), s("BlockStored")),
            (
                s("block_hashes"),
                rmpv::Value::Array(vec![hash_val(1), hash_val(2)]),
            ),
            (s("parent_block_hash"), rmpv::Value::Nil),
            (
                s("token_ids"),
                rmpv::Value::Array(vec![
                    rmpv::Value::Integer(100.into()),
                    rmpv::Value::Integer(101.into()),
                    rmpv::Value::Integer(102.into()),
                ]),
            ),
            (s("block_size"), rmpv::Value::Integer(256.into())),
            (s("medium"), s("GPU")),
            (s("group_idx"), rmpv::Value::Integer(0.into())),
            (s("kv_cache_spec_kind"), s("mla_attention")),
        ]);
        let removed = rmpv::Value::Map(vec![
            (s("type"), s("BlockRemoved")),
            (s("block_hashes"), rmpv::Value::Array(vec![hash_val(9)])),
            (s("medium"), s("GPU")),
        ]);
        let cleared = rmpv::Value::Map(vec![(s("type"), s("AllBlocksCleared"))]);
        let unknown = rmpv::Value::Map(vec![(s("type"), s("FutureEvent"))]);

        let payload = make_batch(vec![stored, removed, cleared, unknown]);
        let decoded = decode_event_batch(&payload);
        assert_eq!(decoded.len(), 3);
        match &decoded[0] {
            DecodedEvent::BlockStored {
                block_count,
                token_count,
            } => {
                assert_eq!(*block_count, 2);
                assert_eq!(*token_count, 3);
            }
            _ => panic!("expected BlockStored"),
        }
        match &decoded[1] {
            DecodedEvent::BlockRemoved { block_count } => assert_eq!(*block_count, 1),
            _ => panic!("expected BlockRemoved"),
        }
        assert!(matches!(decoded[2], DecodedEvent::AllBlocksCleared));
    }

    #[test]
    fn rejects_extra_multipart_frames() {
        let context = zmq::Context::new();
        let sender = context.socket(zmq::PAIR).unwrap();
        let receiver = context.socket(zmq::PAIR).unwrap();
        sender.set_linger(0).unwrap();
        receiver.set_linger(0).unwrap();
        receiver.set_rcvtimeo(2000).unwrap();
        sender.bind("inproc://bounded-event-test").unwrap();
        receiver.connect("inproc://bounded-event-test").unwrap();
        sender
            .send_multipart([b"one", b"two", b"end"].map(|s| s.as_slice()), 0)
            .unwrap();
        assert_eq!(recv_event_frames(&receiver).unwrap().unwrap().len(), 3);
        sender
            .send_multipart([b"one", b"two", b"bad", b"end"].map(|s| s.as_slice()), 0)
            .unwrap();
        assert!(recv_event_frames(&receiver).unwrap().is_none());
    }

    #[test]
    fn decode_empty_payload() {
        let decoded = decode_event_batch(&[]);
        assert!(decoded.is_empty());
    }

    #[test]
    fn decode_malformed_payload() {
        let decoded = decode_event_batch(&[0xff, 0x00]);
        assert!(decoded.is_empty());
    }

    /// End-to-end over a real PUB socket: the subscriber receives a batch,
    /// and shutdown_and_join returns (i.e. the stop flag actually breaks the
    /// recv loop — a hang here means dropped subscribers leak again).
    #[test]
    fn subscriber_receives_and_shuts_down() {
        let publisher = shared_context().socket(zmq::PUB).expect("pub socket");
        publisher.set_linger(0).expect("linger");
        publisher.bind("tcp://127.0.0.1:0").expect("bind");
        let endpoint =
            publisher.get_last_endpoint().expect("last_endpoint").expect("valid endpoint");
        let sub = KVEventSubscriber::spawn(endpoint, String::new());

        let payload = make_batch(vec![rmpv::Value::Array(vec![rmpv::Value::String(
            "AllBlocksCleared".into(),
        )])]);
        let seq = 1i64.to_be_bytes().to_vec();
        // PUB/SUB joins are async — republish until the subscriber sees one.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let mut got = 0;
        while got == 0 && std::time::Instant::now() < deadline {
            publisher
                .send_multipart([b"kv".to_vec(), seq.clone(), payload.clone()], 0)
                .expect("publish");
            thread::sleep(std::time::Duration::from_millis(20));
            got += sub.drain().total_events;
        }
        assert!(got > 0, "subscriber never received a published event");
        sub.shutdown_and_join();
    }

    /// Regression test for the fd leak that aborted vmon on large clusters:
    /// dropping subscribers must release their sockets' file descriptors.
    /// Spawns a batch against unreachable endpoints (zmq connects lazily),
    /// drops them, and polls /proc/self/fd until the count returns to
    /// baseline.
    #[cfg(target_os = "linux")]
    #[test]
    fn dropped_subscribers_release_fds() {
        let fd_count = || std::fs::read_dir("/proc/self/fd").unwrap().count();

        // Materialize the shared context (io + reaper threads) before the
        // baseline so its one-time fds don't count against the margin.
        let _ = shared_context();
        let baseline = fd_count();

        let subs: Vec<_> = (0..64)
            .map(|i| {
                KVEventSubscriber::spawn(format!("tcp://127.0.0.1:{}", 49152 + i), String::new())
            })
            .collect();
        // Sockets are created inside each subscriber thread, so the fd rise
        // is asynchronous — poll for it.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while fd_count() <= baseline + 32 && std::time::Instant::now() < deadline {
            thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(
            fd_count() > baseline + 32,
            "expected 64 live subscriber sockets to hold noticeably more fds"
        );
        drop(subs);

        // Threads exit within one STOP_POLL_MS tick; allow generous margin
        // for parallel tests touching fds.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let margin = 16;
        while fd_count() > baseline + margin && std::time::Instant::now() < deadline {
            thread::sleep(std::time::Duration::from_millis(50));
        }
        let after = fd_count();
        assert!(
            after <= baseline + margin,
            "fd count did not return to baseline: before={baseline}, after={after}"
        );
    }
}
