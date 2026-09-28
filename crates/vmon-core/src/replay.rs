// SPDX-License-Identifier: Apache-2.0

//! Replay file format: the on-disk JSON produced by `vmon collect`, loaded by
//! `vmon replay`.
//!
//! The per-tick sample types (`NodeSample`, `GpuSample`, `IbSample`,
//! `TimeSample`) live in [`crate::sample`] and are shared with the writer side
//! in `vmon-report`. This module only carries the top-level file envelope
//! (`ReplayFile` + `ReplayMeta`).

use serde::Deserialize;

use crate::sample::TimeSample;

/// Deserialized replay file structure matching the JSON report format.
#[derive(Debug, Deserialize)]
pub struct ReplayFile {
    pub meta: ReplayMeta,
    pub samples: Vec<TimeSample>,
}

#[derive(Debug, Deserialize)]
pub struct ReplayMeta {
    pub duration_secs: f64,
    pub sample_count: usize,
    pub nodes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pre-v0.6 JSON (no `gpus`, no `timestamp_ms`) still deserializes cleanly
    /// via `#[serde(default)]` — the new fields take their defaults and no
    /// error is returned. Guards the upgrade path for existing replay files.
    #[test]
    fn legacy_json_without_gpus_or_timestamp_deserializes() {
        let legacy = r#"{
            "meta": { "duration_secs": 10.0, "sample_count": 1, "nodes": ["host:8000"] },
            "samples": [
                {
                    "elapsed_secs": 2.0,
                    "nodes": {
                        "host:8000": {
                            "kv_cache": 0.42,
                            "running": 1.0,
                            "generation_tps": 2100.0,
                            "ttft_p99_ms": 120.0,
                            "mfu_percent": null,
                            "server_load": null,
                            "gpu_utilization": 80.0
                        }
                    }
                }
            ]
        }"#;
        let file: ReplayFile = serde_json::from_str(legacy).expect("parse legacy json");
        assert_eq!(file.samples.len(), 1);
        let sample = &file.samples[0];
        assert_eq!(sample.timestamp_ms, 0);
        let node = &sample.nodes["host:8000"];
        assert!(node.gpus.is_empty());
        assert!(node.ibs.is_empty());
        assert_eq!(node.ttft_p99_ms, 120.0);
    }

    /// New-format JSON with per-GPU series + timestamp + IB devices loads and
    /// every field surfaces correctly.
    #[test]
    fn new_json_with_gpus_ibs_and_timestamp_round_trips() {
        let modern = r#"{
            "meta": { "duration_secs": 10.0, "sample_count": 1, "nodes": ["host:8000"] },
            "samples": [
                {
                    "elapsed_secs": 2.0,
                    "timestamp_ms": 1761134567890,
                    "nodes": {
                        "host:8000": {
                            "kv_cache": 0.42,
                            "running": 1.0,
                            "generation_tps": 2100.0,
                            "mfu_percent": null,
                            "server_load": null,
                            "gpus": [
                                { "index": 0, "uuid": "GPU-u0", "dram_active": 0.52, "tensor_active": 0.41 },
                                { "index": 1, "uuid": "GPU-u1", "dram_active": 0.47 }
                            ],
                            "ibs": [
                                { "device": "mlx5_0", "port": 1, "tx_gbps": 60.0, "state_id": 4, "link_gbps": 400.0 }
                            ]
                        }
                    }
                }
            ]
        }"#;
        let file: ReplayFile = serde_json::from_str(modern).expect("parse new json");
        let sample = &file.samples[0];
        assert_eq!(sample.timestamp_ms, 1_761_134_567_890);
        let node = &sample.nodes["host:8000"];
        assert_eq!(node.gpus.len(), 2);
        assert_eq!(node.gpus[0].uuid, "GPU-u0");
        assert!((node.gpus[0].dram_active.unwrap() - 0.52).abs() < 1e-6);
        assert!((node.gpus[1].dram_active.unwrap() - 0.47).abs() < 1e-6);
        // Fields not present in JSON take defaults.
        assert_eq!(node.gpus[0].xid_errors, 0);
        // IB devices round-trip too.
        assert_eq!(node.ibs.len(), 1);
        assert_eq!(node.ibs[0].device, "mlx5_0");
        assert!((node.ibs[0].tx_gbps - 60.0).abs() < 1e-6);
        assert_eq!(node.ibs[0].link_gbps, Some(400.0));
    }
}
