// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use serde::Serialize;
use serde_json::Value;

use crate::collector::TimeSeriesCollector;

/// Coerce a non-finite f64 (NaN, +Inf, -Inf) to JSON null. serde_json rejects
/// non-finite numbers at serialization time, so callers building `Value::from`
/// directly must funnel f64s through this helper to avoid an error that
/// erases unrelated data.
fn finite_or_null(v: f64) -> Value {
    if v.is_finite() {
        Value::from(v)
    } else {
        Value::Null
    }
}

#[derive(Serialize)]
struct JsonReport {
    meta: Meta,
    samples: Vec<Value>,
}

#[derive(Serialize)]
struct Meta {
    duration_secs: f64,
    sample_count: usize,
    nodes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics_filter: Option<Vec<String>>,
}

/// Identity fields on `GpuSample` that are always preserved when any
/// `gpus.*` filter is active, so per-GPU rows stay distinguishable.
const GPU_IDENTITY_FIELDS: &[&str] = &["index", "uuid", "name"];

/// Identity fields on `IbSample` that are always preserved when any
/// `ibs.*` filter is active, so per-device rows stay distinguishable.
const IB_IDENTITY_FIELDS: &[&str] = &["device", "port"];

type FilterSet<'a> = Option<HashSet<&'a str>>;

pub fn render(collector: &TimeSeriesCollector, metrics: Option<&[String]>) -> String {
    let duration = collector.samples.last().map(|s| s.elapsed_secs).unwrap_or(0.0);

    // Split the filter into top-level keys and nested per-row (`gpus.<f>` /
    // `ibs.<f>`) keys.
    let (top_filter, gpu_filter, ib_filter): (FilterSet, FilterSet, FilterSet) =
        match metrics.filter(|m| !m.is_empty()) {
            None => (None, None, None),
            Some(m) => {
                let mut top: HashSet<&str> = HashSet::new();
                let mut gpu: HashSet<&str> = HashSet::new();
                let mut ib: HashSet<&str> = HashSet::new();
                for entry in m {
                    if let Some(field) = entry.strip_prefix("gpus.") {
                        gpu.insert(field);
                    } else if let Some(field) = entry.strip_prefix("ibs.") {
                        ib.insert(field);
                    } else {
                        top.insert(entry.as_str());
                    }
                }
                // If the user specified any nested field, keep its container in
                // the top output even though it isn't in the raw top filter set.
                if !gpu.is_empty() {
                    top.insert("gpus");
                }
                if !ib.is_empty() {
                    top.insert("ibs");
                }
                (
                    Some(top),
                    if gpu.is_empty() { None } else { Some(gpu) },
                    if ib.is_empty() { None } else { Some(ib) },
                )
            }
        };

    // Whether to emit the cluster-wide `mooncake` field. With no filter,
    // always emit. With a filter, only emit if "mooncake" was named.
    let include_mooncake = top_filter.as_ref().map(|t| t.contains("mooncake")).unwrap_or(true);

    let samples: Vec<Value> = collector
        .samples
        .iter()
        .map(|s| {
            let mut obj = serde_json::Map::new();
            obj.insert("elapsed_secs".to_string(), finite_or_null(s.elapsed_secs));
            obj.insert("timestamp_ms".to_string(), Value::from(s.timestamp_ms));
            let mut nodes_obj = serde_json::Map::new();
            for (addr, node_sample) in &s.nodes {
                // Normalize non-finite values without changing the collected
                // sample or discarding its other fields.
                let mut sanitized = node_sample.clone();
                sanitized.sanitize();
                let node_val = serde_json::to_value(&sanitized).unwrap_or(Value::Null);
                let filtered = filter_node(
                    node_val,
                    top_filter.as_ref(),
                    gpu_filter.as_ref(),
                    ib_filter.as_ref(),
                );
                nodes_obj.insert(addr.clone(), filtered);
            }
            obj.insert("nodes".to_string(), Value::Object(nodes_obj));
            if include_mooncake {
                if let Some(mc) = s.mooncake.as_ref() {
                    obj.insert(
                        "mooncake".to_string(),
                        serde_json::to_value(mc).unwrap_or(Value::Null),
                    );
                }
                if !s.mooncakes.is_empty() {
                    obj.insert(
                        "mooncakes".to_string(),
                        serde_json::to_value(&s.mooncakes).unwrap_or(Value::Null),
                    );
                }
            }
            Value::Object(obj)
        })
        .collect();

    let report = JsonReport {
        meta: Meta {
            duration_secs: duration,
            sample_count: collector.samples.len(),
            nodes: collector.node_addrs.clone(),
            metrics_filter: metrics.map(|m| m.to_vec()),
        },
        samples,
    };

    serde_json::to_string_pretty(&report).unwrap_or_default()
}

/// Apply the `--metrics` filter to a single node's JSON object. Top-level
/// keys are filtered by `top_filter`; nested arrays (`gpus`, `ibs`) are
/// sub-filtered to their identity fields ∪ the corresponding nested filter.
fn filter_node(
    node: Value,
    top_filter: Option<&HashSet<&str>>,
    gpu_filter: Option<&HashSet<&str>>,
    ib_filter: Option<&HashSet<&str>>,
) -> Value {
    let Value::Object(mut map) = node else {
        return node;
    };

    if let Some(top) = top_filter {
        map.retain(|k, _| top.contains(k.as_str()));
    }

    if let Some(gpu_keep) = gpu_filter {
        sub_filter_array(&mut map, "gpus", GPU_IDENTITY_FIELDS, gpu_keep);
    }
    if let Some(ib_keep) = ib_filter {
        sub_filter_array(&mut map, "ibs", IB_IDENTITY_FIELDS, ib_keep);
    }

    Value::Object(map)
}

fn sub_filter_array(
    map: &mut serde_json::Map<String, Value>,
    key: &str,
    identity_fields: &[&str],
    keep: &HashSet<&str>,
) {
    if let Some(Value::Array(arr)) = map.remove(key) {
        let filtered: Vec<Value> = arr
            .into_iter()
            .map(|g| match g {
                Value::Object(fields) => {
                    let kept: serde_json::Map<String, Value> = fields
                        .into_iter()
                        .filter(|(k, _)| {
                            identity_fields.contains(&k.as_str()) || keep.contains(k.as_str())
                        })
                        .collect();
                    Value::Object(kept)
                }
                other => other,
            })
            .collect();
        map.insert(key.to_string(), Value::Array(filtered));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collector::{GpuSample, IbSample, NodeSample, TimeSample, TimeSeriesCollector};
    use std::collections::HashMap;

    fn sample_node() -> NodeSample {
        NodeSample {
            kv_cache: 0.42,
            running: 1.0,
            generation_tps: 2100.0,
            ttft_p50_ms: 50.0,
            ttft_p99_ms: 120.0,
            gpu_utilization: Some(80.0),
            gpu_mem_utilization: Some(50.0),
            gpu_power_watts: Some(322.4),
            gpu_temperature: Some(25.0),
            gpu_vram_used_bytes: Some(0),
            gpu_vram_total_bytes: Some(0),
            gpu_nvlink_tx_kbps: Some(0),
            gpu_nvlink_rx_kbps: Some(0),
            gpus: vec![
                GpuSample {
                    index: 0,
                    uuid: "GPU-u0".to_string(),
                    name: "Example GPU".to_string(),
                    dram_active: Some(0.52),
                    tensor_active: Some(0.41),
                    power_watts: 161.2,
                    ..Default::default()
                },
                GpuSample {
                    index: 1,
                    uuid: "GPU-u1".to_string(),
                    name: "Example GPU".to_string(),
                    dram_active: Some(0.47),
                    tensor_active: Some(0.38),
                    power_watts: 163.7,
                    ..Default::default()
                },
            ],
            ib_total_tx_gbps: Some(120.0),
            ib_total_rx_gbps: Some(80.0),
            ib_active_count: Some(2),
            ib_total_link_gbps: Some(800.0),
            ibs: vec![
                IbSample {
                    device: "mlx5_0".to_string(),
                    port: 1,
                    tx_gbps: 60.0,
                    rx_gbps: 40.0,
                    state_id: 4,
                    rate_bytes_per_sec: 50_000_000_000,
                    link_gbps: Some(400.0),
                    ..Default::default()
                },
                IbSample {
                    device: "mlx5_1".to_string(),
                    port: 1,
                    tx_gbps: 60.0,
                    rx_gbps: 40.0,
                    state_id: 4,
                    rate_bytes_per_sec: 50_000_000_000,
                    link_gbps: Some(400.0),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    fn collector_with_one_sample() -> TimeSeriesCollector {
        let mut c = TimeSeriesCollector::new(vec!["host:8000".to_string()]);
        let mut nodes = HashMap::new();
        nodes.insert("host:8000".to_string(), sample_node());
        c.samples.push(TimeSample {
            elapsed_secs: 4.02,
            timestamp_ms: 1_761_134_567_890,
            nodes,
            mooncake: None,
            mooncakes: Vec::new(),
        });
        c
    }

    #[test]
    fn no_filter_passes_all_fields() {
        let c = collector_with_one_sample();
        let out = render(&c, None);
        let v: Value = serde_json::from_str(&out).unwrap();
        let node = &v["samples"][0]["nodes"]["host:8000"];
        assert_eq!(node["kv_cache"].as_f64().unwrap(), 0.42);
        assert_eq!(node["gpus"].as_array().unwrap().len(), 2);
        assert_eq!(node["gpus"][0]["uuid"].as_str().unwrap(), "GPU-u0");
        assert_eq!(node["gpus"][0]["dram_active"].as_f64().unwrap(), 0.52);
        assert_eq!(
            v["samples"][0]["timestamp_ms"].as_u64().unwrap(),
            1_761_134_567_890
        );
    }

    #[test]
    fn gpus_dot_filter_drills_into_per_gpu_fields() {
        let c = collector_with_one_sample();
        let filter = vec![
            "ttft_p99_ms".to_string(),
            "gpus.dram_active".to_string(),
            "gpus.tensor_active".to_string(),
        ];
        let out = render(&c, Some(&filter));
        let v: Value = serde_json::from_str(&out).unwrap();
        let node = &v["samples"][0]["nodes"]["host:8000"];
        // Top-level: only ttft_p99_ms and gpus survive.
        let node_obj = node.as_object().unwrap();
        let keys: HashSet<&str> = node_obj.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, HashSet::from(["ttft_p99_ms", "gpus"]));
        // Per-GPU: identity fields always kept plus the two requested fields.
        let gpu0 = &node["gpus"][0].as_object().unwrap();
        let gpu_keys: HashSet<&str> = gpu0.keys().map(|s| s.as_str()).collect();
        assert_eq!(
            gpu_keys,
            HashSet::from(["index", "uuid", "name", "dram_active", "tensor_active"])
        );
        assert_eq!(gpu0["dram_active"].as_f64().unwrap(), 0.52);
    }

    #[test]
    fn top_only_filter_keeps_gpus_out() {
        let c = collector_with_one_sample();
        let filter = vec!["ttft_p99_ms".to_string()];
        let out = render(&c, Some(&filter));
        let v: Value = serde_json::from_str(&out).unwrap();
        let node_obj = v["samples"][0]["nodes"]["host:8000"].as_object().unwrap();
        let keys: HashSet<&str> = node_obj.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, HashSet::from(["ttft_p99_ms"]));
    }

    #[test]
    fn ibs_dot_filter_drills_into_per_device_fields() {
        let c = collector_with_one_sample();
        let filter = vec![
            "ttft_p99_ms".to_string(),
            "ibs.tx_gbps".to_string(),
            "ibs.rx_gbps".to_string(),
        ];
        let out = render(&c, Some(&filter));
        let v: Value = serde_json::from_str(&out).unwrap();
        let node = &v["samples"][0]["nodes"]["host:8000"];
        let node_obj = node.as_object().unwrap();
        let keys: HashSet<&str> = node_obj.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, HashSet::from(["ttft_p99_ms", "ibs"]));
        let ib0 = &node["ibs"][0].as_object().unwrap();
        let ib_keys: HashSet<&str> = ib0.keys().map(|s| s.as_str()).collect();
        assert_eq!(
            ib_keys,
            HashSet::from(["device", "port", "tx_gbps", "rx_gbps"])
        );
        assert_eq!(ib0["device"].as_str().unwrap(), "mlx5_0");
        assert_eq!(ib0["tx_gbps"].as_f64().unwrap(), 60.0);
    }

    #[test]
    fn gpus_only_filter_keeps_gpus_and_drops_scalars() {
        let c = collector_with_one_sample();
        let filter = vec!["gpus.dram_active".to_string()];
        let out = render(&c, Some(&filter));
        let v: Value = serde_json::from_str(&out).unwrap();
        let node_obj = v["samples"][0]["nodes"]["host:8000"].as_object().unwrap();
        let keys: HashSet<&str> = node_obj.keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, HashSet::from(["gpus"]));
    }
}
