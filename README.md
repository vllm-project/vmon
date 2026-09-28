# vmon

Lightweight terminal dashboard for monitoring vLLM inference clusters. Monitor one server or a Slurm cluster from a single binary.

## Features

- **Real-time TUI** — throughput sparklines, latency percentiles (mean / p50 / p90 / p99), KV cache, queue depth
- **Multi-node** — monitor multiple vLLM instances side-by-side
- **GPU monitoring** — per-GPU utilization, memory bandwidth, power, temperature, VRAM via built-in agent or DCGM exporter
- **InfiniBand / RDMA** — per-device tx/rx Gbps and link rate via node_exporter
- **Mooncake Store** — centralized KV cache pool: capacity, segment fill, request/eviction rates, HA state
- **NVIDIA Dynamo** — per-rank fan-out across globally-incrementing ports, auto-probed if unset
- **DP-aware** — per-engine metrics with TP-aware GPU mapping in sub-rows
- **Windowed metrics** — delta histograms show recent latency and token distribution, not all-time averages
- **Dev mode actions** — sleep/wake engines, reset prefix cache (single node or cluster-wide)
- **Report export** — collect time-series data and generate HTML (with interactive SVG charts) or JSON reports
- **Replay** — replay collected JSON reports in the TUI with pause/speed controls
- **Config diff** — compare vLLM config between nodes side-by-side in the TUI
- **Slurm integration** — auto-discovers nodes from `SLURM_JOB_NODELIST` or `squeue`
- **Detail pane** — drill into per-node config, system env, and server info with search (`/`)
- **No OpenSSL required** — HTTP clients use rustls; GPU metrics use the installed NVIDIA driver

## Install

### One-command installation

The installer builds vmon from source and installs it to `~/.local/bin` without
sudo. Install **Rust 1.96 or newer**, **git**, a **C/C++ compiler**, and
**pkg-config** first. Downloading the script also requires **curl**.

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vmon/main/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
vmon --version
```

Add the `export PATH` line to your shell profile to keep it for future sessions.
The first build can take several minutes. Run the same command again to update.

Custom installation directory:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vmon/main/install.sh | bash -s -- --to "$HOME/bin"
```

To select a published tag or another branch, append `--ref <tag-or-branch>`.
Use `--help` to see installer options.

The anonymous download command requires this repository to be public. While it
is private, use an authenticated Git checkout as shown below. Prebuilt release
binaries are not yet available; these commands compile from source.

### Install from a checkout

With the same build prerequisites:

```bash
git clone https://github.com/vllm-project/vmon.git
cd vmon
./install.sh
export PATH="$HOME/.local/bin:$PATH"
vmon --version
```

`./install.sh` uses the current checkout; `./install.sh --to "$HOME/bin"` changes
the destination. Remove the installed `vmon` file to uninstall.

## Quick start

Start your vLLM server, then point vmon at its host and API port:

```bash
# Monitor a local vLLM server (default API port: 8000)
vmon watch localhost:8000

# Monitor remote servers; replace these example names with your hosts
vmon watch gpu-node01:8000 gpu-node02:8000

# Collect a one-minute HTML report, then open report.html in a browser
vmon collect localhost:8000 -d 1m -o report.html

# Save JSON for later replay
vmon collect localhost:8000 -d 1m -o metrics.json
vmon replay metrics.json
```

Press `q` to quit the dashboard and `Enter` to inspect the selected node.
For optional GPU metrics, run `vmon agent` on the same host as the monitored
server; see [GPU metrics agent](#vmon-agent--gpu-metrics-agent) for remote access.
Use `vmon --help` or `vmon watch --help` for all options.

## Usage

### `vmon watch` — Real-time dashboard

```bash
# Auto-detect nodes (reads SLURM_JOB_NODELIST or queries squeue)
vmon watch

# Single node (default port 8000)
vmon watch 192.0.2.1

# Multiple nodes
vmon watch 192.0.2.1 192.0.2.2 192.0.2.3

# Custom port and interval
vmon watch 192.0.2.1:8001 192.0.2.2:8001 -i 1s

# Slurm: specific job(s), with per-job port override
vmon watch -j 12345
vmon watch -j 12345:8001,67890:8002

# With GPU monitoring (default port 9400, set 0 to disable)
vmon watch 192.0.2.1 192.0.2.2 --gpu-port 9400

# With InfiniBand monitoring via node_exporter (default port 9100, set 0 to disable)
vmon watch 192.0.2.1 192.0.2.2 --ib-port 9100

# With Mooncake Store (centralized KV cache pool — show in detail pane)
vmon watch 192.0.2.1 --mooncake=192.0.2.1:9003

# With ZMQ KV cache event monitoring
vmon watch 192.0.2.1 --zmq-port 5557

# Multiple DP ranks (subscribes to port 5557, 5558, ...)
vmon watch 192.0.2.1 --zmq-port 5557 --zmq-dp-ranks 4

# NVIDIA Dynamo cluster: auto-probes per-rank ports on each host
vmon watch gpu-node[01-04] --dynamo

# Dynamo with explicit base port (force fixed expansion)
vmon watch gpu-node[01-04] --dynamo --dynamo-base-port 8081 --dynamo-ranks-per-host 8
```

Dynamo discovery uses HTTP `/metrics` probes, explicit `host:port` addresses,
or `--dynamo-base-port` with `--dynamo-ranks-per-host`. It does not read job log
directories or launcher configuration files. To subscribe to KV events in the
dashboard, set `--zmq-port` and, when needed, `--zmq-dp-ranks` explicitly.

**Node auto-discovery:** When no nodes are specified, vmon tries (in order):
1. `SLURM_JOB_NODELIST` env var (inside a Slurm job)
2. `squeue -u $USER` (on a Slurm login node)
3. `localhost` (fallback)

**Keybindings:**

| Key | Action |
|-----|--------|
| `j/k` or `Up/Down` | Select node |
| `Enter` | Toggle detail pane |
| `Space` | Expand per-GPU / per-engine sub-rows |
| `h/l` or `Tab/Shift-Tab` | Switch detail tab (Overview / KV / Hardware / Info) |
| `[` / `]` | Scroll detail pane |
| `/` | Search in detail pane |
| `d` | Config diff between two nodes (Info tab) |
| `g` | Toggle throughput graph view |
| `Esc` | Exit search / diff mode |
| `s` | Sleep/wake engine (dev mode) |
| `R` | Reset prefix cache (single node or cluster-wide) |
| `q` / `Ctrl-C` | Quit |

### `vmon agent` — GPU metrics agent

Run on each GPU node to expose hardware metrics via HTTP. The default listener is
loopback-only; use `--bind <management-IP>` for remote scraping and restrict access
to that network. The endpoint has no authentication or TLS. See [SECURITY.md](SECURITY.md).

 Reads from NVML (NVIDIA Management Library), which is part of the NVIDIA driver — no extra installation needed.

```bash
# Local monitoring (127.0.0.1:9400, 1s refresh)
vmon agent

# Remote monitoring on an access-controlled management network
vmon agent --bind 192.0.2.10

# Daemon mode (private PID/log files in ~/.local/state/vmon)
vmon agent -d

# Custom port and interval
vmon agent --port 9402 --interval 2s

# Forward/proxy an external Prometheus endpoint (e.g. DCGM exporter)
vmon agent --port 9401 --forward http://localhost:9400/metrics

# Daemon with custom log/pid paths
vmon agent -d --log-file /var/log/vmon-agent.log --pid-file /var/run/vmon-agent.pid

# Stop daemon
kill "$(cat ~/.local/state/vmon/agent.pid)"
```

The agent exposes Prometheus-format metrics at `http://<host>:9400/metrics`:

| Metric | Description |
|--------|-------------|
| `vmon_gpu_utilization` | GPU compute utilization (0-100%) |
| `vmon_gpu_memory_utilization` | Memory bandwidth utilization (0-100%) |
| `vmon_gpu_power_watts` | Power draw in watts |
| `vmon_gpu_temperature_celsius` | Temperature in °C |
| `vmon_gpu_memory_used_bytes` | VRAM used |
| `vmon_gpu_memory_total_bytes` | VRAM total |
| `vmon_gpu_clock_mhz` | SM clock speed in MHz |
| `vmon_cpu_usage_percent` | Host CPU utilization (0-100%) |
| `vmon_memory_used_bytes` | Host memory used |
| `vmon_memory_total_bytes` | Host memory total |

NVML is optional — if no NVIDIA driver is present, the agent still serves CPU/memory metrics.

Typical deployment: `scp vmon gpu-node: && ssh gpu-node './vmon agent --bind 192.0.2.10 -d'`

**DCGM exporter** is also supported out of the box. If you already have [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) running, vmon auto-detects the metric format. Recognized DCGM metrics: `DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_POWER_USAGE`, `DCGM_FI_DEV_GPU_TEMP`, `DCGM_FI_DEV_FB_USED`/`FB_FREE`, `DCGM_FI_DEV_SM_CLOCK`, `DCGM_FI_DEV_MEM_COPY_UTIL`.

To get both GPU (from DCGM) and host CPU/memory metrics, use `--forward` to proxy DCGM through the vmon agent:

```bash
# On each node: use its management-network address
vmon agent --bind 192.0.2.10 --port 9401 --forward http://localhost:9400/metrics

# On your workstation
vmon watch 192.0.2.1 192.0.2.2 --gpu-port 9401
```

Same-host deduplication: if multiple vLLM instances share a host (e.g. `192.0.2.1:8000` and `192.0.2.1:8001`), vmon only scrapes the GPU agent once per host.

### `vmon collect` — Generate reports

```bash
# Collect 5 minutes of data, output HTML report
vmon collect 192.0.2.1 192.0.2.2 -d 5m

# Auto-detect nodes from Slurm
vmon collect -d 5m

# JSON output, custom interval
vmon collect 192.0.2.1 -d 1h -i 5s -o metrics.json

# Include Mooncake Store samples in the JSON output
vmon collect 192.0.2.1 -d 5m --mooncake=192.0.2.1:9003 -o metrics.json

# Restrict to a subset of fields (see `--list-metrics` for all names)
vmon collect 192.0.2.1 -d 5m --metrics generation_tps,itl_p99_ms,kv_cache -o metrics.json

# Cap in-memory samples for very long collections (FIFO when exceeded)
vmon collect 192.0.2.1 -d 24h -i 10s --max-samples 5000 -o metrics.json

# Capture raw Prometheus metric families to JSONL files via config
cat > raw-captures.toml <<'EOF'
[[capture]]
name = "mooncake"
output = "mooncake-raw.jsonl"
mode = "prefix"
patterns = ["vllm:mooncake_store_"]

[[capture]]
name = "router"
output = "router-raw.jsonl"
mode = "regex"
patterns = ["^vllm:router_.*"]
EOF

vmon collect 192.0.2.1 -d 5m -o metrics.json \
  --raw-capture-config raw-captures.toml
```

The HTML report is a self-contained single file with interactive SVG charts that work offline, without external JavaScript libraries. Click a legend entry to toggle a series; hover over a chart or use the arrow keys to inspect samples.
When `--raw-capture-config` is set, vmon also writes one JSON object per scrape per node and capture rule containing the matched raw Prometheus metric families, without parsing or aggregating them into report metrics. Each JSONL entry includes `unix_secs`, the scraped `node` target, the capture `name`, and the rendered Prometheus text in `metrics`.
`--mooncake-raw-output` has been removed; migrate to a one-rule `raw-captures.toml` file using the `mooncake` example above.

### `vmon replay` — Replay collected data

```bash
# Replay a JSON report in the TUI
vmon replay metrics.json

# 5x speed
vmon replay metrics.json --speed 5
```

| Key | Action |
|-----|--------|
| `Space` | Pause / resume |
| `>` / `<` | Speed up / slow down (1x, 2x, 5x, 10x) |
| `n` | Step forward one sample (when paused) |

## vLLM Configuration

### Server Info (Info Tab)

For full functionality (config and system env inspection in the detail pane), start vLLM with:

```bash
VLLM_SERVER_DEV_MODE=1 vllm serve <model> ...
```

Use dev mode only on a trusted, access-controlled network. Common credential
fields are redacted in vmon, but configuration and reports still need review
before sharing.

This enables the `/server_info` endpoint which exposes `vllm_config`, `vllm_env`, and `system_env`, and also enables engine control endpoints (`/sleep`, `/wake_up`, `/reset_prefix_cache`).

Without dev mode, vmon still works — metrics and latency are unaffected, but the Info tab will be empty and sleep/wake/reset actions will be unavailable.

### KV Cache Residency Metrics

To see KV block lifecycle metrics (useful for PD disaggregated deployment):

```bash
vllm serve <model> --kv-cache-metrics
```

This exposes `vllm:kv_block_lifetime_seconds`, `vllm:kv_block_idle_before_evict_seconds`, and `vllm:kv_block_reuse_gap_seconds` sampled histograms. vmon shows windowed p50/p99 in a "KV Block Residency" section in the detail pane. Use `--kv-cache-metrics-sample` to control sampling rate.

### MFU / Performance Metrics

To see per-GPU FLOPs and memory bandwidth in the detail pane, enable MFU metrics on the vLLM side:

```bash
vllm serve <model> --enable-mfu-metrics
```

This exposes `vllm:estimated_flops_per_gpu_total`, `vllm:estimated_read_bytes_per_gpu_total`, and `vllm:estimated_write_bytes_per_gpu_total` counters. vmon computes the per-second rates and displays them in a "Performance (per GPU)" section.

Speculative decoding metrics (`vllm:spec_decode_*`) are automatically available when vLLM is running with speculative decoding enabled — no extra flag needed.

### ZMQ KV Cache Events

vmon can subscribe to vLLM's ZMQ PUB socket to monitor KV cache block lifecycle events in real-time. This requires enabling KV cache events on the vLLM side:

```bash
vllm serve <model> --kv-events-config '{"enable_kv_cache_events": true}'
```

By default, vLLM publishes events on `tcp://*:5557` (one port per DP rank: 5557, 5558, ...). Then point vmon at that port:

```bash
vmon watch 192.0.2.1 --zmq-port 5557
```

| Flag | Default | Description |
|------|---------|-------------|
| `--zmq-port` | *(disabled)* | ZMQ PUB port for KV cache events. Omit to disable ZMQ. |
| `--zmq-topic` | `""` (all) | ZMQ topic filter. Empty string subscribes to all messages. |
| `--zmq-dp-ranks` | `1` | Number of DP ranks to subscribe. Connects to port, port+1, ..., port+N-1. |

**Displayed metrics:**

| Metric | Description |
|--------|-------------|
| Blk Stored | Blocks cached this interval |
| Blk Evicted | Blocks evicted this interval |
| Active Blks | Running count of live cache blocks |
| Tok Cached | Tokens stored in new blocks this interval |
| Churn | Total store + evict events (cache turbulence) |
| Seq Gaps | Sequence number gaps (indicates dropped events) |

With multiple DP ranks (`--zmq-dp-ranks N`), each rank's metrics are shown side-by-side for comparison, with aggregate sparklines below.

## Metrics

vmon reads from the vLLM `/metrics` Prometheus endpoint and computes:

| Category | Metrics |
|----------|---------|
| Throughput | Generation tokens/s, Prefill tokens/s, Requests/s |
| Runtime | Requests running/waiting, KV cache usage, Preemptions |
| Latency | TTFT, ITL, E2E, Queue time, Prefill time, Decode time, Inference time, TPOT (mean / p50 / p90 / p99, windowed + cumulative) |
| Cache | Prefix cache hit rate, External cache hit rate, Multi-modal cache hit rate (External hit gets a top-table `Ext%` sparkline alongside `Cache%` in NIXL / PD-disaggregated deployments) |
| KV Residency | Block lifetime, idle-before-evict, reuse gap (p50/p99) (requires `--kv-cache-metrics` on vLLM) |
| Spec Decode | Acceptance rate, drafts/s, draft/accepted token totals (auto-detected when speculative decoding is enabled) |
| Performance | MFU%, estimated FLOPs/s, memory read/write bandwidth per GPU (requires `--enable-mfu-metrics` on vLLM) |
| NIXL Transfers | Transfer/post latency (p50/p99), avg bytes/descriptors, failed transfers/notifications, expired KV (auto-detected with NIXL connector) |
| HTTP | QPS, Error rate, Status code breakdown |
| Requests | Avg prompt/generation tokens, token distribution p50/p99, finished reason breakdown |
| KV Events | Block store/evict rate, active blocks, token cache rate, churn, seq gaps (requires `--zmq-port`) |
| GPU Hardware | Utilization, memory bandwidth, power, temperature, VRAM per GPU (requires `vmon agent` or DCGM) |
| InfiniBand | Per-device tx/rx Gbps, active device count, total link Gbps (requires node_exporter on `--ib-port`) |
| Mooncake Store | Memory util, key count, request rate by op (get / put / exist / remove with batch folded in), ping rate, failure rate, eviction rate, segment fill range, HA state (requires `--mooncake <addr>`) |

## Architecture

```
vmon/
├── vmon-core     # Prometheus parser, histogram math, scraper, data model
├── vmon-tui      # ratatui terminal UI
├── vmon-report   # HTML/JSON report generation
└── vmon-cli      # clap CLI entry point
```

`vmon-core` has no UI dependencies. `vmon-report` depends on `core`;
`vmon-tui` uses both for live display and recording.

## License

[Apache License 2.0](LICENSE). Third-party components retain their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

For development conventions and validation commands, see [CONTRIBUTING.md](CONTRIBUTING.md).
