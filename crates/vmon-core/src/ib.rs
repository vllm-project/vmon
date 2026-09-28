// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use tokio::time::Instant;

use crate::parser::{MetricFamily, Sample};
use crate::rate::RateCalc;

/// IB port state value that means "link active and carrying traffic".
/// (See `node_infiniband_state_id` HELP: 0=no change, 1=down, 2=init, 3=armed,
/// 4=active, 5=act defer.)
pub const IB_STATE_ACTIVE: u8 = 4;

/// Per-(device, port) InfiniBand metrics from a single scrape.
#[derive(Debug, Clone, Default)]
pub struct IbDevice {
    pub device: String, // e.g. "mlx5_0"
    pub port: u32,      // typically 1

    // Cumulative counters (already in bytes — node_exporter exposes the
    // de-octet'd value, no need to multiply by 4 again).
    pub tx_bytes_total: u64,
    pub rx_bytes_total: u64,
    pub tx_packets_total: u64,
    pub rx_packets_total: u64,

    // Gauges
    pub state_id: u8,
    /// Max signal transfer rate of the link in bytes/sec
    /// (`node_infiniband_rate_bytes_per_second`).
    pub rate_bytes_per_sec: u64,

    // Computed by `IbRateTracker::apply`. Zero until the second scrape arrives.
    pub tx_gbps: f64,
    pub rx_gbps: f64,
}

impl IbDevice {
    pub fn is_active(&self) -> bool {
        self.state_id == IB_STATE_ACTIVE
    }

    /// Theoretical one-direction peak link speed in Gbps. None if not reported.
    pub fn link_gbps(&self) -> Option<f64> {
        if self.rate_bytes_per_sec == 0 {
            None
        } else {
            Some(self.rate_bytes_per_sec as f64 * 8.0 / 1e9)
        }
    }

    /// TX throughput as a fraction of one-direction link peak (0.0–1.0).
    pub fn tx_utilization(&self) -> Option<f64> {
        self.link_gbps().map(|peak| self.tx_gbps / peak)
    }

    /// RX throughput as a fraction of one-direction link peak (0.0–1.0).
    pub fn rx_utilization(&self) -> Option<f64> {
        self.link_gbps().map(|peak| self.rx_gbps / peak)
    }
}

/// All IB devices on one host.
#[derive(Debug, Clone, Default)]
pub struct IbScrape {
    pub devices: Vec<IbDevice>,
}

impl IbScrape {
    pub fn active(&self) -> impl Iterator<Item = &IbDevice> {
        self.devices.iter().filter(|d| d.is_active())
    }

    /// Sum of TX Gbps across active ports.
    pub fn total_tx_gbps(&self) -> f64 {
        self.active().map(|d| d.tx_gbps).sum()
    }

    /// Sum of RX Gbps across active ports.
    pub fn total_rx_gbps(&self) -> f64 {
        self.active().map(|d| d.rx_gbps).sum()
    }

    pub fn active_count(&self) -> usize {
        self.active().count()
    }

    /// Sum of theoretical one-direction peak Gbps across active ports.
    pub fn total_link_gbps(&self) -> f64 {
        self.active().filter_map(|d| d.link_gbps()).sum()
    }
}

fn key_of(s: &Sample) -> Option<(String, u32)> {
    let dev = s.label("device")?.to_string();
    let port: u32 = s.label("port").and_then(|v| v.parse().ok()).unwrap_or(1);
    Some((dev, port))
}

fn f64_to_u64_clamped(v: f64) -> u64 {
    if v <= 0.0 || !v.is_finite() {
        0
    } else {
        v.round() as u64
    }
}

/// Extract IB metrics from parsed Prometheus families (typically node_exporter).
/// Returns an `IbScrape` with cumulative counters populated; per-second rates
/// are filled later by `IbRateTracker::apply`.
///
/// Only families whose name starts with `node_infiniband_` are considered.
/// node_exporter emits other families (filesystem, disk) that share the
/// `device` + `port` label names, and those must not pollute the device list.
pub fn extract_ib_metrics(families: &[MetricFamily]) -> IbScrape {
    let mut by_key: HashMap<(String, u32), IbDevice> = HashMap::new();

    for fam in families {
        // The parser strips `_total` from counter family names, so we match
        // the base names here.
        let name = fam.name.as_str();
        if !name.starts_with("node_infiniband_") {
            continue;
        }
        for s in &fam.samples {
            let Some(k) = key_of(s) else { continue };
            let entry = by_key.entry(k.clone()).or_insert_with(|| IbDevice {
                device: k.0,
                port: k.1,
                ..Default::default()
            });
            match name {
                "node_infiniband_port_data_transmitted_bytes" => {
                    entry.tx_bytes_total = f64_to_u64_clamped(s.value);
                }
                "node_infiniband_port_data_received_bytes" => {
                    entry.rx_bytes_total = f64_to_u64_clamped(s.value);
                }
                "node_infiniband_port_packets_transmitted" => {
                    entry.tx_packets_total = f64_to_u64_clamped(s.value);
                }
                "node_infiniband_port_packets_received" => {
                    entry.rx_packets_total = f64_to_u64_clamped(s.value);
                }
                "node_infiniband_state_id" => {
                    entry.state_id = f64_to_u64_clamped(s.value).min(255) as u8;
                }
                "node_infiniband_rate_bytes_per_second" => {
                    entry.rate_bytes_per_sec = f64_to_u64_clamped(s.value);
                }
                _ => {}
            }
        }
    }

    let mut devices: Vec<IbDevice> = by_key.into_values().collect();
    devices.sort_by(|a, b| a.device.cmp(&b.device).then(a.port.cmp(&b.port)));
    IbScrape { devices }
}

/// Per-(host, device, port) rate state. Lives in the scraper task and is
/// fed cumulative byte counters each tick to compute Gbps.
#[derive(Debug, Default)]
pub struct IbRateTracker {
    by_host: HashMap<String, HashMap<(String, u32), DevRate>>,
}

#[derive(Debug, Default)]
struct DevRate {
    tx: RateCalc,
    rx: RateCalc,
}

impl IbRateTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a host's `IbScrape` and fill `tx_gbps` / `rx_gbps` in-place.
    pub fn apply(&mut self, host: &str, scrape: &mut IbScrape, now: Instant) {
        let host_state = self.by_host.entry(host.to_string()).or_default();
        for d in scrape.devices.iter_mut() {
            let dr = host_state.entry((d.device.clone(), d.port)).or_default();
            if let Some(bps) = dr.tx.update(d.tx_bytes_total, now) {
                d.tx_gbps = bps * 8.0 / 1e9;
            }
            if let Some(bps) = dr.rx.update(d.rx_bytes_total, now) {
                d.rx_gbps = bps * 8.0 / 1e9;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_prometheus_text;
    use std::time::Duration;

    const SAMPLE: &str = r#"# HELP node_infiniband_port_data_transmitted_bytes_total Number of data octets transmitted on all links
# TYPE node_infiniband_port_data_transmitted_bytes_total counter
node_infiniband_port_data_transmitted_bytes_total{device="mlx5_0",port="1"} 1000
node_infiniband_port_data_transmitted_bytes_total{device="mlx5_1",port="1"} 2000
# HELP node_infiniband_port_data_received_bytes_total Number of data octets received on all links
# TYPE node_infiniband_port_data_received_bytes_total counter
node_infiniband_port_data_received_bytes_total{device="mlx5_0",port="1"} 500
node_infiniband_port_data_received_bytes_total{device="mlx5_1",port="1"} 1500
# HELP node_infiniband_state_id State of the InfiniBand port
# TYPE node_infiniband_state_id gauge
node_infiniband_state_id{device="mlx5_0",port="1"} 4
node_infiniband_state_id{device="mlx5_1",port="1"} 1
# HELP node_infiniband_rate_bytes_per_second Maximum signal transfer rate
# TYPE node_infiniband_rate_bytes_per_second gauge
node_infiniband_rate_bytes_per_second{device="mlx5_0",port="1"} 5e+10
node_infiniband_rate_bytes_per_second{device="mlx5_1",port="1"} 2.5e+10
"#;

    #[test]
    fn test_extract_ib_basic() {
        let fams = parse_prometheus_text(SAMPLE).unwrap();
        let scrape = extract_ib_metrics(&fams);
        assert_eq!(scrape.devices.len(), 2);

        let d0 = &scrape.devices[0];
        assert_eq!(d0.device, "mlx5_0");
        assert_eq!(d0.port, 1);
        assert_eq!(d0.tx_bytes_total, 1000);
        assert_eq!(d0.rx_bytes_total, 500);
        assert_eq!(d0.state_id, 4);
        assert_eq!(d0.rate_bytes_per_sec, 50_000_000_000);
        assert!(d0.is_active());
        assert_eq!(d0.link_gbps(), Some(400.0)); // 5e10 * 8 / 1e9 = 400

        let d1 = &scrape.devices[1];
        assert!(!d1.is_active());

        assert_eq!(scrape.active_count(), 1);
    }

    #[test]
    fn test_extract_ib_ignores_non_ib_families() {
        // node_exporter exposes many families with `device` labels — make sure
        // only `node_infiniband_*` contribute to the IB device list.
        let s = r#"# TYPE node_infiniband_state_id gauge
node_infiniband_state_id{device="mlx5_0",port="1"} 4
# TYPE node_filesystem_size_bytes gauge
node_filesystem_size_bytes{device="/dev/md0",mountpoint="/"} 1099511627776
# TYPE node_disk_read_bytes_total counter
node_disk_read_bytes_total{device="sda"} 1234567
"#;
        let fams = parse_prometheus_text(s).unwrap();
        let scrape = extract_ib_metrics(&fams);
        assert_eq!(scrape.devices.len(), 1);
        assert_eq!(scrape.devices[0].device, "mlx5_0");
    }

    #[test]
    fn test_extract_ib_handles_scientific_notation() {
        // node_exporter emits values like 6.0318507835764e+13.
        let s = r#"# TYPE node_infiniband_port_data_transmitted_bytes_total counter
node_infiniband_port_data_transmitted_bytes_total{device="mlx5_0",port="1"} 6.0318507835764e+13
# TYPE node_infiniband_state_id gauge
node_infiniband_state_id{device="mlx5_0",port="1"} 4
"#;
        let fams = parse_prometheus_text(s).unwrap();
        let scrape = extract_ib_metrics(&fams);
        assert_eq!(scrape.devices.len(), 1);
        assert_eq!(scrape.devices[0].tx_bytes_total, 60_318_507_835_764);
    }

    #[test]
    fn test_rate_tracker_basic() {
        let mut tracker = IbRateTracker::new();
        let t0 = Instant::now();

        let mut s0 = IbScrape {
            devices: vec![IbDevice {
                device: "mlx5_0".into(),
                port: 1,
                tx_bytes_total: 1_000_000_000,
                rx_bytes_total: 0,
                state_id: 4,
                rate_bytes_per_sec: 50_000_000_000,
                ..Default::default()
            }],
        };
        tracker.apply("h1", &mut s0, t0);
        assert_eq!(s0.devices[0].tx_gbps, 0.0); // first sample → no rate yet

        let t1 = t0 + Duration::from_secs(1);
        let mut s1 = IbScrape {
            devices: vec![IbDevice {
                device: "mlx5_0".into(),
                port: 1,
                tx_bytes_total: 2_250_000_000, // +1.25 GB in 1s = 10 Gbps
                rx_bytes_total: 125_000_000,   // +125 MB in 1s = 1 Gbps
                state_id: 4,
                rate_bytes_per_sec: 50_000_000_000,
                ..Default::default()
            }],
        };
        tracker.apply("h1", &mut s1, t1);
        assert!((s1.devices[0].tx_gbps - 10.0).abs() < 0.001);
        assert!((s1.devices[0].rx_gbps - 1.0).abs() < 0.001);
        assert_eq!(s1.total_tx_gbps(), s1.devices[0].tx_gbps); // active, summed
    }

    #[test]
    fn test_rate_tracker_isolates_hosts() {
        // Same device name on two hosts must not share rate state.
        let mut tracker = IbRateTracker::new();
        let t0 = Instant::now();
        let dev = || IbDevice {
            device: "mlx5_0".into(),
            port: 1,
            tx_bytes_total: 1_000_000_000,
            state_id: 4,
            ..Default::default()
        };
        let mut a0 = IbScrape {
            devices: vec![dev()],
        };
        let mut b0 = IbScrape {
            devices: vec![dev()],
        };
        tracker.apply("hA", &mut a0, t0);
        tracker.apply("hB", &mut b0, t0);

        let t1 = t0 + Duration::from_secs(1);
        let mut a1 = IbScrape {
            devices: vec![IbDevice {
                tx_bytes_total: 2_000_000_000, // +1 GB → 8 Gbps
                ..dev()
            }],
        };
        tracker.apply("hA", &mut a1, t1);
        // hB never got a second sample, so even though we *would* compute a
        // rate, hA's prev value is independent.
        assert!((a1.devices[0].tx_gbps - 8.0).abs() < 0.001);
    }
}
