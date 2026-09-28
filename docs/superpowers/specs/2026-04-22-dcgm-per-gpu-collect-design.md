# Per-GPU DCGM Metrics in `vmon collect` — Design

**Date:** 2026-04-22
**Status:** Draft — awaiting review

## Problem

`vmon collect` already scrapes a DCGM exporter (or `vmon agent`'s `/metrics`) on
each host's `--gpu-port` and parses most DCGM fields in `vmon-core/src/gpu.rs`.
The data even survives into `NodeMetrics.gpu_scrape.gpus: Vec<GpuMetrics>` as
per-GPU detail. But at the report boundary — `NodeSample` in
`vmon-report/src/collector.rs` — the per-GPU structure is collapsed to a handful
of node-aggregated scalars (`gpu_utilization`, `gpu_power_watts`, etc.). So the
JSON and HTML reports produced by `vmon collect` cannot show per-GPU behavior
over time, even though the scrape path has all of it in memory.

Separately, a few DCGM fields the exporter emits today are not parsed at all
(`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`, `DCGM_FI_DEV_MEMORY_TEMP`,
`DCGM_FI_DEV_XID_ERRORS`, etc.) and are silently discarded.

An alternative approach spawns `dcgmi dmon` as a subprocess
(SLURM-aware via `srun --overlap`) and writes a separate JSON sidecar.
That approach solves a different problem (no DCGM exporter deployed) and is
out of scope here — when `dcgm-exporter` is already running on
port 9400 with profile metrics enabled.

## Goal

Make `vmon collect`'s own JSON report contain per-GPU time series for every
DCGM metric the exporter emits, with each tick's vLLM-side and DCGM-side
numbers sharing a single wall-clock timestamp so cross-system correlation is
explicit.

## Non-Goals

- Spawning `dcgmi dmon` subprocesses, in any form. No SLURM awareness in
  `vmon collect`.
- Deploying or configuring DCGM exporter — assume it is reachable on
  `--gpu-port` (or via `vmon agent --forward`).
- Parsing `DCGM_FI_PROF_SM_ACTIVE` — not in the default exporter counters
  config. Can be added later if/when the counters CSV is customized.
- Deriving rates from counter metrics (`TOTAL_ENERGY_CONSUMPTION`,
  `PCIE_REPLAY_COUNTER`, `XID_ERRORS`, remapped rows). We store raw counter
  values; downstream consumers diff consecutive samples.
- Per-GPU charts in the HTML report. The JSON carries full per-GPU detail in
  this change; HTML visualization can follow once the data is flowing and we
  know what views are actually useful.
- New CLI flags on `vmon collect`. The change takes effect transparently
  whenever `--gpu-port` is reachable.
- `vmon watch` (TUI) or `vmon agent` — untouched.

## Background: current data flow

```
DCGM exporter (port 9400)
   │  Prometheus text
   ▼
parser::parse_prometheus_text     ──► MetricFamily[]
   │
   ▼
gpu::extract_gpu_metrics           ──► GpuScrape { gpus: Vec<GpuMetrics> }  ◄── per-GPU present
   │
   ▼
scraper::Scraper::run_loop         ──► ClusterState (watch channel)  ◄── vLLM+GPU joined here
   │                                    via tokio::join!(scrape_futs, gpu_futs)
   ▼
report::TimeSeriesCollector::record
   │                               ──► TimeSample { elapsed_secs, nodes: HashMap<_, NodeSample> }
   ▼                                    ◄── per-GPU FLATTENED here today
report::json::render / html::render
```

Timeline alignment of vLLM and DCGM is already achieved by the `tokio::join!`
in `scraper.rs:328` — both scrapes fire in the same tick and their results are
folded into the same `ClusterState`. This design makes that invariant explicit
by adding a wall-clock timestamp to each `TimeSample`.

## Design

### 1. `GpuMetrics` in `vmon-core/src/gpu.rs` — widen

Add the identity field:

- `uuid: String` — parsed from the `UUID` label on DCGM samples; empty when
  absent.

Add 15 new per-GPU metric fields covering every `DCGM_FI_*` family that
`dcgm-exporter` emits today but the code ignores:

| Field | Type | DCGM source | Unit |
|---|---|---|---|
| `dram_active` | `f64` | `DCGM_FI_PROF_DRAM_ACTIVE` | fraction 0–1 |
| `gr_engine_active` | `f64` | `DCGM_FI_PROF_GR_ENGINE_ACTIVE` | fraction 0–1 |
| `tensor_active` | `f64` | `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | fraction 0–1 |
| `enc_utilization` | `f64` | `DCGM_FI_DEV_ENC_UTIL` | % |
| `dec_utilization` | `f64` | `DCGM_FI_DEV_DEC_UTIL` | % |
| `mem_temperature` | `f64` | `DCGM_FI_DEV_MEMORY_TEMP` | °C |
| `total_energy_mj` | `u64` | `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION` | mJ, cumulative |
| `pcie_replay_count` | `u64` | `DCGM_FI_DEV_PCIE_REPLAY_COUNTER` | count, cumulative |
| `xid_errors` | `u64` | `DCGM_FI_DEV_XID_ERRORS` | count, cumulative |
| `remapped_rows_correctable` | `u64` | `DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS` | count |
| `remapped_rows_uncorrectable` | `u64` | `DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS` | count |
| `row_remap_failure` | `u64` | `DCGM_FI_DEV_ROW_REMAP_FAILURE` | 0/1 |
| `vgpu_license_status` | `u64` | `DCGM_FI_DEV_VGPU_LICENSE_STATUS` | enum |
| `pcie_prof_tx_bytes_per_sec` | `u64` | `DCGM_FI_PROF_PCIE_TX_BYTES` | bytes/s |
| `pcie_prof_rx_bytes_per_sec` | `u64` | `DCGM_FI_PROF_PCIE_RX_BYTES` | bytes/s |

**Existing fallbacks kept intact.** The current code populates
`mem_utilization` from `DCGM_FI_PROF_DRAM_ACTIVE * 100` when
`DCGM_FI_DEV_MEM_COPY_UTIL` reports 0, and similarly for `utilization` from
`DCGM_FI_PROF_GR_ENGINE_ACTIVE`. Those fallbacks stay so existing HTML charts
and external consumers continue to show meaningful values. In parallel, the
new `dram_active` / `gr_engine_active` / `tensor_active` fields are always
populated directly from the profile metrics, so per-GPU detail is exact.

### 2. `vmon-core/src/gpu.rs::extract_gpu_metrics` — parser additions

Add one `match` arm per new metric in the DCGM branch. Also pick up the
`UUID` label once per GPU, like `name` / `modelName` are picked up today
(`gpu.rs:314-321`). No structural changes to the function.

### 3. `NodeSample` in `vmon-report/src/collector.rs` — add `gpus`

```rust
pub struct NodeSample {
    // … all existing fields preserved unchanged …
    pub gpus: Vec<GpuSample>,
}
```

All existing aggregated fields (`gpu_utilization`, `gpu_mem_utilization`,
`gpu_power_watts`, `gpu_temperature`, `gpu_vram_used_bytes`,
`gpu_vram_total_bytes`, `gpu_nvlink_tx_kbps`, `gpu_nvlink_rx_kbps`) stay in
place — backward compat for existing HTML charts and any external consumers.

### 4. New `GpuSample` type in `vmon-report/src/collector.rs`

Mirrors `GpuMetrics` 1:1 over the fields listed in sections 1 + the existing
fields already in `GpuMetrics`. `Serialize + Deserialize` via serde. Populated
from `NodeMetrics.gpu_scrape.gpus[i]` in `NodeSample::from`. When
`gpu_scrape` is `None` or `gpus` is empty, `NodeSample.gpus` is an empty `Vec`
— the JSON shape is always `"gpus": [...]` (possibly empty), never absent.

### 5. `TimeSample` — add wall-clock timestamp

```rust
pub struct TimeSample {
    pub elapsed_secs: f64,
    pub timestamp_ms: u64,  // NEW — Unix millis at record() time
    pub nodes: HashMap<String, NodeSample>,
}
```

Set in `TimeSeriesCollector::record` via
`SystemTime::now().duration_since(UNIX_EPOCH)`. Both the vLLM-side and
GPU-side numbers in this tick share the same `timestamp_ms`, directly encoding
the "aligned on the timeline" property and enabling correlation with external
systems (Grafana, nsys, separate DCGM scrapers).

### 6. `METRIC_NAMES` and `--metrics` filter — per-GPU support

`METRIC_NAMES` in `collector.rs` is extended with entries `gpus.<field>` for
every field on `GpuSample`, so `vmon collect --list-metrics` prints the full
menu.

`json::render` gains filter handling for `gpus.<field>` entries:

- Entries without a dot apply to the top-level `NodeSample` scalars — unchanged
  behavior.
- Entries with prefix `gpus.` apply to each `GpuSample` inside `gpus[]`. The
  `gpus` array is included when any such entry is present, and each element is
  sub-filtered to the specified fields plus always-included identity fields
  `index` and `uuid` (so GPUs stay distinguishable in the output).

Example:

```
vmon collect --metrics ttft_p99_ms,gpus.dram_active,gpus.tensor_active ...
```

yields, per sample:

```json
{ "elapsed_secs": 4.02, "timestamp_ms": 1761134567890,
  "nodes": {
    "host:8000": {
      "ttft_p99_ms": 120.0,
      "gpus": [
        { "index": 0, "uuid": "GPU-…", "dram_active": 0.52, "tensor_active": 0.41 },
        { "index": 1, "uuid": "GPU-…", "dram_active": 0.47, "tensor_active": 0.38 },
        …
      ]
    }
  }
}
```

### 7. Replay compatibility in `vmon-core/src/replay.rs`

`replay.rs` intentionally mirrors `NodeSample` / `TimeSample` with its own
`ReplaySample` / `ReplayTimeSample` types to avoid a circular dependency
between `vmon-core` and `vmon-report`. That mirror must track the new fields:

- Add a `ReplayGpuSample` struct mirroring `GpuSample` (same field list, all
  `#[serde(default)]`).
- Add `pub gpus: Vec<ReplayGpuSample>` with `#[serde(default)]` to
  `ReplaySample`.
- Add `pub timestamp_ms: u64` with `#[serde(default)]` to `ReplayTimeSample`.

`#[serde(default)]` on the new fields keeps pre-change JSON files
deserializable — they yield an empty `gpus` array and `timestamp_ms: 0`. No
logic change elsewhere.

### 8. HTML report (`vmon-report/src/html.rs`)

No rendering change in this design. Existing aggregated GPU charts continue to
work from the preserved aggregated fields. The per-GPU array is embedded in
the JSON blob the HTML page already ships, so users wanting per-GPU views can
extract it manually or read the separate `.json` output. A first-class
per-GPU HTML view is a follow-up once we know which views are actually useful
in practice.

## Error handling

This change is almost entirely additive (new parse arms, new struct fields),
so the error surface is minimal:

- Missing DCGM fields (e.g. edited `default-counters.csv`) leave the
  corresponding `GpuMetrics` / `GpuSample` field at its `Default` zero. No
  panic, no log spam. Matches current behavior for fields already parsed.
- `UUID` label missing → `uuid: String` stays empty.
- `gpu` label parse failure already skips the sample (`gpu.rs:305-308`) —
  unchanged.
- GPU scrape failed entirely → `gpu_scrape: None` → `NodeSample.gpus` is an
  empty `Vec`. JSON shape stays consistent.
- Counter resets / wraparound are not handled here — downstream consumers that
  diff counter samples handle them.

## Backward compatibility

- **Writers (`vmon collect`):** adding fields is serde-additive. External
  consumers that ignore unknown fields see the same data plus two new fields
  they can ignore.
- **Readers (`vmon replay`):** `#[serde(default)]` on new fields keeps
  pre-change JSON files loadable.
- **HTML charts:** depend only on the preserved aggregated fields — no visual
  change.
- **`--metrics` filter:** old filter strings keep working (no dot → top-level,
  same as today). New `gpus.<field>` syntax is additive.
- **`--list-metrics`:** output grows; consumers grep'ing specific names are
  unaffected.

## Testing

### Existing tests that must not break
`gpu.rs` has 12 unit tests, including `test_extract_gpu_metrics`,
`test_extract_dcgm_metrics`, `test_extract_dcgm_prof_metrics`. All should pass
unchanged — the fallback populations of `utilization` / `mem_utilization` are
preserved.

### New tests

1. **`gpu.rs` — full DCGM field coverage.** One Prometheus text fixture
   containing all 15 new metric families with 2 GPUs. Assert each of the 15
   new `GpuMetrics` fields populates correctly on both GPUs.
2. **`gpu.rs` — profile-vs-dev independence.** When `DCGM_FI_DEV_GPU_UTIL`
   and `DCGM_FI_PROF_GR_ENGINE_ACTIVE` both appear, assert `utilization` gets
   the DEV value (exact), `gr_engine_active` gets the PROF value. Same for
   `mem_utilization` vs `dram_active`.
3. **`gpu.rs` — `uuid` label parsing** from a DCGM sample.
4. **`collector.rs` — `NodeSample::from` round-trip.** Build a `NodeMetrics`
   with a `GpuScrape` containing 4 `GpuMetrics`; call
   `TimeSeriesCollector::record()`; serialize the last sample to JSON;
   deserialize; assert `gpus.len() == 4`, field values match, and
   `timestamp_ms > 0`.
5. **`json.rs` — filter drilling.** With filter
   `["ttft_p99_ms", "gpus.dram_active", "gpus.tensor_active"]`, assert the
   node object has only `ttft_p99_ms`, and each `gpus[]` entry has only
   `index`, `uuid`, `dram_active`, `tensor_active`.
6. **`json.rs` — no filter, full pass-through.** Filter `None` → all fields
   present, `gpus` array has N entries per node.
7. **`replay.rs` — legacy JSON compatibility.** A small inline JSON fixture
   without `gpus` / `timestamp_ms` deserializes into the replay types with
   defaults, no error.

### Manual smoke test (documented in plan)

- Run `vmon collect <host>:8000 -d 10s -o /tmp/r.json` on a host with the
  DCGM exporter live on port 9400. Confirm `gpus[]` is populated with the
  expected number of entries per tick and `timestamp_ms` is monotonic.
- Run `vmon replay /tmp/r.json` — confirm no deserialization errors and the
  TUI behaves normally (per-GPU detail shown in the detail panel as today,
  since `NodeSample.gpus` feeds into the same downstream code path).

## Files touched

| Path | Change |
|---|---|
| `crates/vmon-core/src/gpu.rs` | Add 15 new fields to `GpuMetrics`, add `uuid`; add parse arms for new DCGM metrics; parse `UUID` label; update existing tests; add new tests. |
| `crates/vmon-core/src/replay.rs` | Add `ReplayGpuSample`; add `gpus` to `ReplaySample`; add `timestamp_ms` to `ReplayTimeSample`; all with `#[serde(default)]`; legacy-JSON test. |
| `crates/vmon-report/src/collector.rs` | Add `GpuSample` type; add `gpus` to `NodeSample`; add `timestamp_ms` to `TimeSample`; populate both in `record()` / `NodeSample::from`; extend `METRIC_NAMES` with `gpus.*`; add round-trip test. |
| `crates/vmon-report/src/json.rs` | Emit `timestamp_ms`; handle `gpus.<field>` filter syntax; add tests. |

No changes to `vmon-cli`, `vmon-tui`, `vmon-core/src/scraper.rs`,
`vmon-core/src/node.rs`, or `vmon-report/src/html.rs`.

## Open questions

None at draft time. Two plausible follow-ups, deliberately scoped out of this
change:

1. Per-GPU HTML visualization (small-multiples or overlay, default-visible
   fields, interactive toggles).
2. A `vmon collect --dcgmi` mode that spawns `dcgmi dmon` as a fallback for
   SLURM hosts without an exporter. Needs its own design.
