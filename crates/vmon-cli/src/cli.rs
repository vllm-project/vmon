// SPDX-License-Identifier: Apache-2.0

//! Command-line interface and argument parsing.
use clap::{Parser, Subcommand};
use std::net::IpAddr;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "vmon", about = "Monitor vLLM inference clusters", version)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
    /// Real-time TUI dashboard
    Watch {
        /// Node addresses (e.g. 192.0.2.1 or 192.0.2.1:8001).
        /// If omitted, auto-detects from SLURM_JOB_NODELIST env var.
        nodes: Vec<String>,

        /// Slurm job ID(s) to discover nodes from
        #[arg(short = 'j', long, value_delimiter = ',')]
        slurm_job: Vec<String>,

        /// Slurm job name pattern to discover nodes from
        #[arg(short = 'J', long, conflicts_with = "slurm_job")]
        slurm_job_name: Option<String>,

        /// Default vLLM port
        #[arg(short = 'p', long, default_value = "8000")]
        port: u16,

        /// Scrape interval
        #[arg(short = 'i', long, default_value = "2s", value_parser = parse_duration)]
        interval: Duration,

        /// ZMQ port for KV cache events (uniform base: every host is
        /// subscribed at port, port+1, … per --zmq-dp-ranks).
        #[arg(long)]
        zmq_port: Option<u16>,

        /// ZMQ topic filter
        #[arg(long, default_value = "")]
        zmq_topic: String,

        /// Number of DP ranks for ZMQ
        #[arg(long, default_value = "1")]
        zmq_dp_ranks: u16,

        /// GPU agent port (0 to disable)
        #[arg(long, default_value = "9400")]
        gpu_port: u16,

        /// node_exporter port for InfiniBand metrics (0 to disable)
        #[arg(long, default_value = "9100")]
        ib_port: u16,

        /// Mooncake Store master. Scrapes the centralized KV cache pool
        /// /metrics + /health and shows them in the Details panel.
        ///
        /// Omit the flag to disable. Pass bare `--mooncake` to auto-target the
        /// task's first node on port 8702, or
        /// `--mooncake=host:port` for an explicit address.
        #[arg(long, num_args = 0..=1, require_equals = true, value_name = "HOST:PORT")]
        mooncake: Option<Option<String>>,

        /// NVIDIA Dynamo mode: skip /version, /server_info, /is_sleeping and
        /// fan out per-host across globally-incrementing rank ports starting
        /// at `--dynamo-base-port`.
        #[arg(long)]
        dynamo: bool,

        /// Base port for Dynamo per-rank metrics (rank 0 of the first host).
        /// Subsequent ranks increment globally across all hosts.
        ///
        /// By default, probe candidate ports for dynamo_* metrics.
        /// Set this flag to use fixed port expansion instead.
        #[arg(long)]
        dynamo_base_port: Option<u16>,

        /// Ranks per host (overrides SLURM_GPUS_ON_NODE / scontrol detection).
        #[arg(long)]
        dynamo_ranks_per_host: Option<u16>,
    },

    /// Replay a collected JSON report in the TUI
    Replay {
        /// Path to a JSON report file from `vmon collect`
        file: PathBuf,

        /// Playback speed multiplier
        #[arg(long, default_value = "1")]
        speed: f64,
    },

    /// Run metrics agent (exposes host CPU/MEM + GPU metrics via HTTP)
    Agent {
        /// Listen address. Use a management-network IP for remote monitoring.
        #[arg(long, default_value = "127.0.0.1")]
        bind: IpAddr,

        /// Listen port
        #[arg(long, default_value = "9400")]
        port: u16,

        /// Metrics refresh interval
        #[arg(long, default_value = "1s", value_parser = parse_duration)]
        interval: Duration,

        /// Forward (proxy) metrics from another Prometheus endpoint (e.g. DCGM exporter).
        /// The forwarded metrics are merged with the agent's own CPU/MEM/GPU metrics.
        #[arg(long)]
        forward: Option<String>,

        /// Run as background daemon
        #[arg(long, short = 'd')]
        daemon: bool,

        /// PID file path (default: ~/.local/state/vmon/agent.pid)
        #[arg(long)]
        pid_file: Option<PathBuf>,

        /// Log file path (default: ~/.local/state/vmon/agent.log)
        #[arg(long)]
        log_file: Option<PathBuf>,
    },

    /// Collect metrics and generate a report
    Collect {
        /// Node addresses.
        /// If omitted, auto-detects from SLURM_JOB_NODELIST env var.
        nodes: Vec<String>,

        /// Slurm job ID(s) to discover nodes from
        #[arg(short = 'j', long, value_delimiter = ',')]
        slurm_job: Vec<String>,

        /// Slurm job name pattern to discover nodes from
        #[arg(short = 'J', long, conflicts_with = "slurm_job")]
        slurm_job_name: Option<String>,

        /// Collection duration (e.g. 30s, 5m, 1h)
        #[arg(short = 'd', long, required_unless_present = "list_metrics", value_parser = parse_duration, default_value = "1s")]
        duration: Duration,

        /// Scrape interval
        #[arg(short = 'i', long, default_value = "2s", value_parser = parse_duration)]
        interval: Duration,

        /// Default vLLM port
        #[arg(short = 'p', long, default_value = "8000")]
        port: u16,

        /// Output file path (.html or .json)
        #[arg(short = 'o', long, default_value = "report.html")]
        output: PathBuf,

        /// GPU agent port (0 to disable)
        #[arg(long, default_value = "9400")]
        gpu_port: u16,

        /// node_exporter port for InfiniBand metrics (0 to disable)
        #[arg(long, default_value = "9100")]
        ib_port: u16,

        /// Mooncake Store master. Samples include the centralized KV cache
        /// pool's metrics.
        ///
        /// Omit the flag to disable. Pass bare `--mooncake` to auto-target the
        /// task's first node on port 8702, or
        /// `--mooncake=host:port` for an explicit address.
        #[arg(long, num_args = 0..=1, require_equals = true, value_name = "HOST:PORT")]
        mooncake: Option<Option<String>>,

        /// Only include these metrics (comma-separated field names).
        /// Example: --metrics generation_tps,itl_p99_ms,kv_cache
        #[arg(long, value_delimiter = ',')]
        metrics: Option<Vec<String>>,

        /// List all available metric field names and exit.
        #[arg(long)]
        list_metrics: bool,

        /// NVIDIA Dynamo mode: skip /version, /server_info, /is_sleeping and
        /// fan out per-host across globally-incrementing rank ports starting
        /// at `--dynamo-base-port`.
        #[arg(long)]
        dynamo: bool,

        /// Base port for Dynamo per-rank metrics (rank 0 of the first host).
        /// Subsequent ranks increment globally across all hosts.
        ///
        /// By default, probe candidate ports for dynamo_* metrics.
        /// Set this flag to use fixed port expansion instead.
        #[arg(long)]
        dynamo_base_port: Option<u16>,

        /// Ranks per host (overrides SLURM_GPUS_ON_NODE / scontrol detection).
        #[arg(long)]
        dynamo_ranks_per_host: Option<u16>,

        /// Capture raw Prometheus metric families to JSONL files using a TOML config.
        #[arg(long)]
        raw_capture_config: Option<PathBuf>,

        /// Cap the number of in-memory samples. When exceeded the oldest
        /// samples are dropped (FIFO). Useful for very long collections to
        /// bound RAM. Default unbounded.
        #[arg(long)]
        max_samples: Option<usize>,

        /// Periodically (re)write --output during collection so a hard kill
        /// (SIGKILL, crash, power loss) still leaves the latest checkpoint on
        /// disk. Ctrl-C already saves on exit, independent of this. The write
        /// is atomic (temp file + rename), so readers never see a torn file.
        /// Set 0 to disable and only write once at the end.
        #[arg(long, default_value = "30s", value_parser = parse_duration)]
        flush_interval: Duration,
    },
}

fn parse_duration(s: &str) -> Result<Duration, humantime::DurationError> {
    humantime::parse_duration(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_listener_is_loopback_unless_explicitly_overridden() {
        let Cli {
            command:
                Command::Agent {
                    bind,
                    pid_file,
                    log_file,
                    ..
                },
        } = Cli::parse_from(["vmon", "agent"])
        else {
            panic!("expected agent")
        };
        assert!(bind.is_loopback());
        assert!(pid_file.is_none() && log_file.is_none());
        let Cli {
            command: Command::Agent { bind, .. },
        } = Cli::parse_from(["vmon", "agent", "--bind", "::1"])
        else {
            panic!("expected agent")
        };
        assert_eq!(bind, "::1".parse::<IpAddr>().unwrap());
    }
}
