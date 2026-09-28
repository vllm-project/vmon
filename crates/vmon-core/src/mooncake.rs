// SPDX-License-Identifier: Apache-2.0

//! Mooncake Store metrics: centralized KV cache pool exposed on a leader
//! address (e.g. `:9003/metrics` + `/health`). Cluster-wide — one leader, not
//! per-vLLM-node — so the resulting `MooncakeMetrics` lives on `ClusterState`.

use tokio::time::Instant;

use crate::parser::{MetricFamily, Sample};
use crate::rate::RateCalc;

/// Raw values from one `/metrics` scrape. Cumulative counters not yet rated.
#[derive(Debug, Clone, Default)]
pub struct MooncakeScrape {
    // Capacity (gauges)
    pub mem_allocated_bytes: u64,
    pub mem_total_bytes: u64,

    // State (gauges)
    pub key_count: u64,
    pub soft_pin_key_count: u64,
    pub active_clients: u64,

    // Per-segment fill
    pub segments: Vec<MooncakeSegment>,

    // Request totals (curated subset — sums of the corresponding
    // *_requests_total families)
    pub get_replica_list_requests_total: u64,
    pub put_start_requests_total: u64,
    pub put_end_requests_total: u64,
    pub exist_key_requests_total: u64,
    pub remove_requests_total: u64,
    pub ping_requests_total: u64,
    pub batch_get_replica_list_requests_total: u64,
    pub batch_put_start_requests_total: u64,
    pub batch_put_end_requests_total: u64,
    pub batch_exist_key_requests_total: u64,

    /// Sum of `master_*_failures_total` across user ops only (GET / PUT_start /
    /// EXIST / REMOVE and their batch variants). Excludes ping_failures and
    /// put_end_failures so the "fail/s" displayed alongside Requests matches
    /// the by-op total.
    pub total_failures: u64,
    /// Sum of `master_*_requests_total` across user ops only. Excludes
    /// `master_ping_requests` (health-check, usually the dominant counter and
    /// not a real op) and `master_put_end_requests` / `master_batch_put_end_requests`
    /// (PUT is 2-phase — start+end double-counts the same operation).
    pub total_requests: u64,

    // Eviction
    pub successful_evictions_total: u64,
    pub attempted_evictions_total: u64,
    pub evicted_size_bytes_total: u64,

    // HA
    pub ha_oplog_standby_lag: i64,
    pub ha_oplog_pending_entries: i64,
    pub ha_standby_state: u8,
}

#[derive(Debug, Clone, Default)]
pub struct MooncakeSegment {
    pub segment: String,
    pub allocated_bytes: u64,
    pub total_bytes: u64,
}

impl MooncakeSegment {
    pub fn fill(&self) -> Option<f64> {
        if self.total_bytes == 0 {
            None
        } else {
            Some(self.allocated_bytes as f64 / self.total_bytes as f64)
        }
    }
}

/// `/health` JSON.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MooncakeHealth {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub ha_state: String,
    #[serde(default)]
    pub service_ready: bool,
}

/// Computed metrics including per-second rates. Stored on `ClusterState`.
#[derive(Debug, Clone, Default)]
pub struct MooncakeMetrics {
    pub addr: String,
    pub health: Option<MooncakeHealth>,

    pub mem_allocated_bytes: u64,
    pub mem_total_bytes: u64,
    pub mem_util: f64,

    pub key_count: u64,
    pub soft_pin_key_count: u64,
    pub active_clients: u64,

    pub total_request_rate: f64,
    pub failure_rate: f64,
    /// GET ops/s, includes both `master_get_replica_list_*` and its
    /// `batch_get_replica_list_*` counterpart. Deployments may use exclusively
    /// the batch path, so splitting batch out leaves
    /// the singular counters at 0 — fold them together for an at-a-glance
    /// "GETs per second regardless of transport" signal.
    pub get_rate: f64,
    /// PUT ops/s (put_start + batch_put_start).
    pub put_rate: f64,
    /// EXIST ops/s (exist_key + batch_exist_key).
    pub exist_rate: f64,
    /// REMOVE ops/s (remove + batch_remove), usually 0 outside of admin work.
    pub remove_rate: f64,
    /// PING ops/s — health-check traffic from clients, not real work. Surfaced
    /// separately so operators can sanity-check client connectivity without
    /// inflating the Requests total.
    pub ping_rate: f64,
    pub eviction_rate: f64,
    pub eviction_bytes_rate: f64,

    pub segments: Vec<MooncakeSegment>,
    pub segment_count: usize,
    pub segment_fill_min: f64,
    pub segment_fill_max: f64,

    pub ha_oplog_standby_lag: i64,
    pub ha_oplog_pending_entries: i64,
    pub ha_standby_state: u8,
}

/// Operation-name predicate used to decide what counts as a "user op" when
/// summing `total_requests` / `total_failures`. Match the part between
/// `master_` and `_requests`/`_failures`. Excludes `ping` (health check) and
/// the `_put_end` halves of the 2-phase PUT protocol.
fn is_user_op(op: &str) -> bool {
    matches!(
        op,
        "get_replica_list"
            | "put_start"
            | "exist_key"
            | "remove"
            | "batch_get_replica_list"
            | "batch_put_start"
            | "batch_exist_key"
            | "batch_remove"
    )
}

fn f64_to_u64_clamped(v: f64) -> u64 {
    if v <= 0.0 || !v.is_finite() {
        0
    } else {
        v.round() as u64
    }
}

fn f64_to_i64_clamped(v: f64) -> i64 {
    if !v.is_finite() { 0 } else { v.round() as i64 }
}

fn sum_family(fam: &MetricFamily) -> u64 {
    fam.samples.iter().map(|s| f64_to_u64_clamped(s.value)).sum()
}

fn first_value(fam: &MetricFamily) -> f64 {
    fam.samples.first().map(|s| s.value).unwrap_or(0.0)
}

fn segment_label(s: &Sample) -> Option<String> {
    s.label("segment").or_else(|| s.label("segment_name")).map(|v| v.to_string())
}

/// Extract Mooncake Store metrics from parsed Prometheus families.
///
/// The parser strips `_total` from counter family names, so this matches the
/// base names (e.g. `master_get_replica_list_requests`, not
/// `master_get_replica_list_requests_total`).
pub fn parse_mooncake_metrics(families: &[MetricFamily]) -> MooncakeScrape {
    let mut s = MooncakeScrape::default();
    let mut segments: std::collections::HashMap<String, MooncakeSegment> =
        std::collections::HashMap::new();

    for fam in families {
        let name = fam.name.as_str();

        // Roll up user-operation counters only. Exclude `ping` health checks.
        // The `*_put_end_*` families would double-count PUTs that already
        // increment `*_put_start_*`. Both are excluded from `total_requests`
        // / `total_failures` so that Requests ≈ get + put + exist + remove.
        if let Some(stripped) = name.strip_prefix("master_") {
            let is_user_op = match stripped.strip_suffix("_requests") {
                Some(op) => is_user_op(op),
                None => match stripped.strip_suffix("_failures") {
                    Some(op) => is_user_op(op),
                    None => false,
                },
            };
            if is_user_op {
                if stripped.ends_with("_requests") {
                    s.total_requests = s.total_requests.saturating_add(sum_family(fam));
                } else {
                    s.total_failures = s.total_failures.saturating_add(sum_family(fam));
                }
            }
        }

        match name {
            // Capacity gauges
            "master_allocated_bytes" => {
                s.mem_allocated_bytes = f64_to_u64_clamped(first_value(fam))
            }
            "master_total_capacity_bytes" => {
                s.mem_total_bytes = f64_to_u64_clamped(first_value(fam))
            }

            // State gauges
            "master_key_count" => s.key_count = f64_to_u64_clamped(first_value(fam)),
            "master_soft_pin_key_count" => {
                s.soft_pin_key_count = f64_to_u64_clamped(first_value(fam))
            }
            "master_active_clients" => s.active_clients = f64_to_u64_clamped(first_value(fam)),

            // Curated request counter families
            "master_get_replica_list_requests" => {
                s.get_replica_list_requests_total = sum_family(fam)
            }
            "master_put_start_requests" => s.put_start_requests_total = sum_family(fam),
            "master_put_end_requests" => s.put_end_requests_total = sum_family(fam),
            "master_exist_key_requests" => s.exist_key_requests_total = sum_family(fam),
            "master_batch_get_replica_list_requests" => {
                s.batch_get_replica_list_requests_total = sum_family(fam)
            }
            "master_batch_put_start_requests" => s.batch_put_start_requests_total = sum_family(fam),
            "master_batch_put_end_requests" => s.batch_put_end_requests_total = sum_family(fam),
            "master_batch_exist_key_requests" => s.batch_exist_key_requests_total = sum_family(fam),
            "master_remove_requests" => s.remove_requests_total = sum_family(fam),
            "master_ping_requests" => s.ping_requests_total = sum_family(fam),

            // Eviction
            "master_successful_evictions" => s.successful_evictions_total = sum_family(fam),
            "master_attempted_evictions" => s.attempted_evictions_total = sum_family(fam),
            "master_evicted_size_bytes" => s.evicted_size_bytes_total = sum_family(fam),

            // HA gauges
            "ha_oplog_standby_lag" => s.ha_oplog_standby_lag = f64_to_i64_clamped(first_value(fam)),
            "ha_oplog_pending_entries" => {
                s.ha_oplog_pending_entries = f64_to_i64_clamped(first_value(fam))
            }
            "ha_standby_state" => {
                let v = f64_to_u64_clamped(first_value(fam));
                s.ha_standby_state = v.min(255) as u8;
            }

            // Per-segment fill — accumulate by segment label.
            "segment_allocated_bytes" => {
                for sample in &fam.samples {
                    if let Some(seg) = segment_label(sample) {
                        let entry =
                            segments.entry(seg.clone()).or_insert_with(|| MooncakeSegment {
                                segment: seg,
                                ..Default::default()
                            });
                        entry.allocated_bytes = f64_to_u64_clamped(sample.value);
                    }
                }
            }
            "segment_capacity_bytes" | "segment_total_bytes" => {
                for sample in &fam.samples {
                    if let Some(seg) = segment_label(sample) {
                        let entry =
                            segments.entry(seg.clone()).or_insert_with(|| MooncakeSegment {
                                segment: seg,
                                ..Default::default()
                            });
                        entry.total_bytes = f64_to_u64_clamped(sample.value);
                    }
                }
            }
            _ => {}
        }
    }

    s.segments = segments.into_values().collect();
    s.segments.sort_by(|a, b| a.segment.cmp(&b.segment));

    // If per-segment capacity isn't exposed, fall back to dividing the total
    // capacity evenly across the segments we did see — gives a usable fill
    // approximation without bombing on the divide-by-zero path.
    if !s.segments.is_empty() && s.segments.iter().all(|seg| seg.total_bytes == 0) {
        let per_seg = s.mem_total_bytes / s.segments.len() as u64;
        for seg in s.segments.iter_mut() {
            seg.total_bytes = per_seg;
        }
    }

    s
}

#[derive(Debug, Default)]
pub struct MooncakeState {
    addr: String,
    rate_total: RateCalc,
    rate_fail: RateCalc,
    rate_get: RateCalc,
    rate_put: RateCalc,
    rate_exist: RateCalc,
    rate_remove: RateCalc,
    rate_ping: RateCalc,
    rate_eviction: RateCalc,
    rate_eviction_bytes: RateCalc,
}

impl MooncakeState {
    pub fn new(addr: String) -> Self {
        Self {
            addr,
            ..Default::default()
        }
    }

    /// Feed a fresh scrape + optional health JSON and produce computed metrics.
    pub fn update(
        &mut self,
        scrape: MooncakeScrape,
        health: Option<MooncakeHealth>,
        now: Instant,
    ) -> MooncakeMetrics {
        // Fold batch counters into their per-op totals. PutStart represents
        // puts initiated; pair with PutEnd if you ever need "completed puts"
        // — for the at-a-glance signal, start is fine.
        let get_total = scrape
            .get_replica_list_requests_total
            .saturating_add(scrape.batch_get_replica_list_requests_total);
        let put_total = scrape
            .put_start_requests_total
            .saturating_add(scrape.batch_put_start_requests_total);
        let exist_total = scrape
            .exist_key_requests_total
            .saturating_add(scrape.batch_exist_key_requests_total);

        let failure_rate = self.rate_fail.update(scrape.total_failures, now).unwrap_or(0.0);
        let get_rate = self.rate_get.update(get_total, now).unwrap_or(0.0);
        let put_rate = self.rate_put.update(put_total, now).unwrap_or(0.0);
        let exist_rate = self.rate_exist.update(exist_total, now).unwrap_or(0.0);
        let remove_rate = self.rate_remove.update(scrape.remove_requests_total, now).unwrap_or(0.0);
        let ping_rate = self.rate_ping.update(scrape.ping_requests_total, now).unwrap_or(0.0);
        // Drive Requests off the curated user-op total so the math closes:
        // Requests == get + put + exist + remove. `rate_total` is still kept
        // around as a sanity-check sink (scrape.total_requests already excludes
        // ping and put_end at parse time).
        let total_request_rate = self.rate_total.update(scrape.total_requests, now).unwrap_or(0.0);
        let eviction_rate =
            self.rate_eviction.update(scrape.successful_evictions_total, now).unwrap_or(0.0);
        let eviction_bytes_rate = self
            .rate_eviction_bytes
            .update(scrape.evicted_size_bytes_total, now)
            .unwrap_or(0.0);

        let mem_util = if scrape.mem_total_bytes == 0 {
            0.0
        } else {
            scrape.mem_allocated_bytes as f64 / scrape.mem_total_bytes as f64
        };

        let (segment_fill_min, segment_fill_max) = if scrape.segments.is_empty() {
            (0.0, 0.0)
        } else {
            let fills: Vec<f64> = scrape.segments.iter().filter_map(|s| s.fill()).collect();
            if fills.is_empty() {
                (0.0, 0.0)
            } else {
                let lo = fills.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = fills.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                (lo, hi)
            }
        };

        let segment_count = scrape.segments.len();

        MooncakeMetrics {
            addr: self.addr.clone(),
            health,
            mem_allocated_bytes: scrape.mem_allocated_bytes,
            mem_total_bytes: scrape.mem_total_bytes,
            mem_util,
            key_count: scrape.key_count,
            soft_pin_key_count: scrape.soft_pin_key_count,
            active_clients: scrape.active_clients,
            total_request_rate,
            failure_rate,
            get_rate,
            put_rate,
            exist_rate,
            remove_rate,
            ping_rate,
            eviction_rate,
            eviction_bytes_rate,
            segments: scrape.segments,
            segment_count,
            segment_fill_min,
            segment_fill_max,
            ha_oplog_standby_lag: scrape.ha_oplog_standby_lag,
            ha_oplog_pending_entries: scrape.ha_oplog_pending_entries,
            ha_standby_state: scrape.ha_standby_state,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_prometheus_text;
    use std::time::Duration;

    const SAMPLE: &str = r#"# TYPE master_allocated_bytes gauge
master_allocated_bytes 2220000000000
# TYPE master_total_capacity_bytes gauge
master_total_capacity_bytes 2500000000000
# TYPE master_key_count gauge
master_key_count 1085737
# TYPE master_soft_pin_key_count gauge
master_soft_pin_key_count 0
# TYPE master_active_clients gauge
master_active_clients 16
# TYPE master_get_replica_list_requests_total counter
master_get_replica_list_requests_total 1000
# TYPE master_put_start_requests_total counter
master_put_start_requests_total 500
# TYPE master_exist_key_requests_total counter
master_exist_key_requests_total 200
# TYPE master_ping_requests_total counter
master_ping_requests_total 1234
# TYPE master_get_replica_list_failures_total counter
master_get_replica_list_failures_total 3
# TYPE master_put_start_failures_total counter
master_put_start_failures_total 2
# TYPE master_successful_evictions_total counter
master_successful_evictions_total 7
# TYPE master_evicted_size_bytes_total counter
master_evicted_size_bytes_total 100000
# TYPE segment_allocated_bytes gauge
segment_allocated_bytes{segment="192.0.2.1:12666"} 800
segment_allocated_bytes{segment="192.0.2.2:12666"} 900
segment_allocated_bytes{segment="192.0.2.3:12666"} 1000
# TYPE ha_standby_state gauge
ha_standby_state 0
# TYPE ha_oplog_standby_lag gauge
ha_oplog_standby_lag 0
# TYPE ha_oplog_pending_entries gauge
ha_oplog_pending_entries 0
"#;

    #[test]
    fn parse_extracts_capacity_and_segments() {
        let fams = parse_prometheus_text(SAMPLE).unwrap();
        let s = parse_mooncake_metrics(&fams);
        assert_eq!(s.mem_allocated_bytes, 2_220_000_000_000);
        assert_eq!(s.mem_total_bytes, 2_500_000_000_000);
        assert_eq!(s.key_count, 1085737);
        assert_eq!(s.active_clients, 16);
        assert_eq!(s.segments.len(), 3);
        // sorted by name
        assert_eq!(s.segments[0].segment, "192.0.2.1:12666");
        assert_eq!(s.segments[0].allocated_bytes, 800);
        // fallback: total_bytes filled from master_total_capacity_bytes / 3
        assert!(s.segments[0].total_bytes > 0);
    }

    #[test]
    fn put_end_and_ping_excluded_from_total_requests() {
        // Synthetic fixture with both phases of a 2-phase PUT and a heavy ping
        // load — total_requests must count put_start but not put_end, and must
        // not count ping at all, even when pings dominate the raw counters.
        let txt = r#"# TYPE master_put_start_requests_total counter
master_put_start_requests_total 100
# TYPE master_put_end_requests_total counter
master_put_end_requests_total 99
# TYPE master_batch_put_start_requests_total counter
master_batch_put_start_requests_total 50
# TYPE master_batch_put_end_requests_total counter
master_batch_put_end_requests_total 48
# TYPE master_ping_requests_total counter
master_ping_requests_total 60000
"#;
        let fams = parse_prometheus_text(txt).unwrap();
        let s = parse_mooncake_metrics(&fams);
        // Only put_start counts (100), not put_end. Same for batch.
        assert_eq!(s.total_requests, 100 + 50);
        assert_eq!(s.ping_requests_total, 60000);
        assert_eq!(s.put_end_requests_total, 99);
    }

    #[test]
    fn failures_roll_up_across_counters() {
        let fams = parse_prometheus_text(SAMPLE).unwrap();
        let s = parse_mooncake_metrics(&fams);
        // 3 + 2 = 5 across get_replica_list_failures and put_start_failures
        assert_eq!(s.total_failures, 5);
        // Only user-op families count: get(1000) + put_start(500) + exist(200) = 1700.
        // master_ping_requests (1234) is health-check noise and is excluded.
        assert_eq!(s.total_requests, 1000 + 500 + 200);
        // ping_rate captured separately from total.
        assert_eq!(s.ping_requests_total, 1234);
    }

    #[test]
    fn rate_calc_yields_get_rate() {
        let mut st = MooncakeState::new("h:9003".into());
        let t0 = Instant::now();
        let scrape0 = MooncakeScrape {
            get_replica_list_requests_total: 100,
            total_requests: 100,
            ..Default::default()
        };
        let _ = st.update(scrape0, None, t0);

        let t1 = t0 + Duration::from_secs(1);
        let scrape1 = MooncakeScrape {
            get_replica_list_requests_total: 110,
            total_requests: 110,
            ..Default::default()
        };
        let m = st.update(scrape1, None, t1);
        assert!((m.get_rate - 10.0).abs() < 0.001);
        assert!((m.total_request_rate - 10.0).abs() < 0.001);
    }

    #[test]
    fn per_op_rates_fold_batch_counters() {
        // Only batch_* counters increase; non-batch counters remain zero.
        let mut st = MooncakeState::new("h:9003".into());
        let t0 = Instant::now();
        let scrape0 = MooncakeScrape {
            batch_get_replica_list_requests_total: 1000,
            batch_put_start_requests_total: 500,
            batch_exist_key_requests_total: 8000,
            ..Default::default()
        };
        let _ = st.update(scrape0, None, t0);

        let t1 = t0 + Duration::from_secs(1);
        let scrape1 = MooncakeScrape {
            batch_get_replica_list_requests_total: 1005,
            batch_put_start_requests_total: 503,
            batch_exist_key_requests_total: 8200,
            // Throw in a tiny non-batch tick to verify the totals fold.
            get_replica_list_requests_total: 2,
            ..Default::default()
        };
        let m = st.update(scrape1, None, t1);
        assert!((m.get_rate - 7.0).abs() < 0.001, "get_rate={}", m.get_rate);
        assert!((m.put_rate - 3.0).abs() < 0.001, "put_rate={}", m.put_rate);
        assert!(
            (m.exist_rate - 200.0).abs() < 0.001,
            "exist_rate={}",
            m.exist_rate
        );
    }

    #[test]
    fn segment_fill_range_computed() {
        let scrape = MooncakeScrape {
            mem_total_bytes: 0, // disable fallback
            segments: vec![
                MooncakeSegment {
                    segment: "a".into(),
                    allocated_bytes: 200,
                    total_bytes: 1000,
                },
                MooncakeSegment {
                    segment: "b".into(),
                    allocated_bytes: 800,
                    total_bytes: 1000,
                },
                MooncakeSegment {
                    segment: "c".into(),
                    allocated_bytes: 500,
                    total_bytes: 1000,
                },
            ],
            ..Default::default()
        };
        let mut st = MooncakeState::new("h:9003".into());
        let m = st.update(scrape, None, Instant::now());
        assert!((m.segment_fill_min - 0.2).abs() < 1e-9);
        assert!((m.segment_fill_max - 0.8).abs() < 1e-9);
        assert_eq!(m.segment_count, 3);
    }

    #[test]
    fn mem_util_zero_when_total_zero() {
        let mut st = MooncakeState::new("h:9003".into());
        let m = st.update(MooncakeScrape::default(), None, Instant::now());
        assert_eq!(m.mem_util, 0.0);
    }
}
