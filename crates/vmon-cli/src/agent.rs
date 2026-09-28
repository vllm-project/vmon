// SPDX-License-Identifier: Apache-2.0

//! Host metrics agent and its private daemon state.
use std::collections::HashMap;
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Reject shared or symlinked default state directories. Explicit paths still
/// use no-follow opens and a held PID-file lock to prevent duplicate daemons.
fn agent_state_dir() -> std::io::Result<PathBuf> {
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
    let home = std::env::var_os("HOME")
        .ok_or_else(|| std::io::Error::other("HOME is unset; provide --pid-file and --log-file"))?;
    let dir = PathBuf::from(home).join(".local/state/vmon");
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(&dir)?;
    let meta = std::fs::symlink_metadata(&dir)?;
    // SAFETY: geteuid has no preconditions and does not access Rust memory.
    let uid = unsafe { libc::geteuid() };
    if !meta.is_dir() || meta.uid() != uid || meta.permissions().mode() & 0o077 != 0 {
        return Err(std::io::Error::other(
            "agent state directory must be owned by this user with mode 0700",
        ));
    }
    Ok(dir)
}

fn open_agent_file(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    // SAFETY: geteuid has no preconditions.
    let uid = unsafe { libc::geteuid() };
    if !meta.is_file()
        || meta.uid() != uid
        || meta.nlink() != 1
        || meta.permissions().mode() & 0o077 != 0
    {
        return Err(std::io::Error::other(
            "agent file must be a private, single-link regular file owned by this user",
        ));
    }
    Ok(file)
}

pub(crate) fn daemonize(
    bind: IpAddr,
    port: u16,
    interval: Duration,
    forward: Option<&str>,
    pid_file: Option<&Path>,
    log_file: Option<&Path>,
) -> std::io::Result<()> {
    let dir = if pid_file.is_none() || log_file.is_none() {
        agent_state_dir()?
    } else {
        PathBuf::new()
    };
    let default_pid = dir.join("agent.pid");
    let default_log = dir.join("agent.log");
    let pid_file = pid_file.unwrap_or(&default_pid);
    let log_file = log_file.unwrap_or(&default_log);
    let mut pid_lock = open_agent_file(pid_file)?;
    pid_lock
        .try_lock()
        .map_err(|e| std::io::Error::other(format!("agent PID file is already locked: {e}")))?;
    let log = open_agent_file(log_file)?;
    use std::os::unix::fs::MetadataExt;
    if (log.metadata()?.dev(), log.metadata()?.ino())
        == (pid_lock.metadata()?.dev(), pid_lock.metadata()?.ino())
    {
        return Err(std::io::Error::other("PID and log files must be different"));
    }
    log.set_len(0)?;
    // The child holds the locked PID inode through stdin for its entire lifetime.
    let mut args = vec![
        "agent".to_string(),
        "--bind".to_string(),
        bind.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--interval".to_string(),
        format!("{}ns", interval.as_nanos()),
    ];
    if let Some(url) = forward {
        args.extend(["--forward".into(), url.into()]);
    }
    #[allow(clippy::zombie_processes)]
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args(&args)
        .stdout(log.try_clone()?)
        .stderr(log)
        .stdin(pid_lock.try_clone()?)
        .spawn()?;
    let pid = child.id();
    if let Err(e) = (|| {
        pid_lock.set_len(0)?;
        writeln!(pid_lock, "{pid}")?;
        pid_lock.sync_all()
    })() {
        let _ = child.kill();
        let _ = child.wait();
        return Err(e);
    }
    eprintln!("vmon agent daemonized (PID: {pid})");
    eprintln!("  log: {}", log_file.display());
    eprintln!("  pid: {}", pid_file.display());
    Ok(())
}

pub(crate) async fn run_agent(
    bind: IpAddr,
    port: u16,
    interval: Duration,
    forward: Option<String>,
) {
    let (nvml, device_count) = match nvml_wrapper::Nvml::init() {
        Ok(nvml) => match nvml.device_count() {
            Ok(c) => {
                eprintln!("vmon agent: {c} GPU(s) detected");
                (Some(nvml), c)
            }
            Err(e) => {
                eprintln!("Warning: failed to enumerate GPUs: {e}");
                (None, 0)
            }
        },
        Err(e) => {
            eprintln!("Warning: NVML not available ({e}), GPU metrics disabled");
            (None, 0)
        }
    };

    if forward.is_some() {
        eprintln!("vmon agent: forwarding upstream metrics");
    }
    eprintln!("vmon agent: interval {interval:?}, port {port}");

    let fwd_client = forward.as_ref().map(|_| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("failed to build HTTP client")
    });

    // Collect initial snapshot
    let mut cpu_state = CpuState::new();
    let mut nvlink_state = NvlinkState::new();
    let snapshot = collect_snapshot(
        nvml.as_ref(),
        device_count,
        &mut cpu_state,
        &mut nvlink_state,
    );
    let fwd_snapshot = fetch_forward(fwd_client.as_ref(), forward.as_deref()).await;
    let initial = merge_snapshots(&snapshot, &fwd_snapshot);
    let state: Arc<RwLock<String>> = Arc::new(RwLock::new(initial));

    // Background task to refresh metrics
    let bg_state = state.clone();
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(interval);
        loop {
            tick.tick().await;
            let snap = collect_snapshot(
                nvml.as_ref(),
                device_count,
                &mut cpu_state,
                &mut nvlink_state,
            );
            let fwd = fetch_forward(fwd_client.as_ref(), forward.as_deref()).await;
            *bg_state.write().await = merge_snapshots(&snap, &fwd);
        }
    });

    // HTTP server
    let app = axum::Router::new().route(
        "/metrics",
        axum::routing::get(move || {
            let state = state.clone();
            async move { state.read().await.clone() }
        }),
    );

    let listener =
        tokio::net::TcpListener::bind(SocketAddr::new(bind, port))
            .await
            .unwrap_or_else(|e| {
                eprintln!("Failed to bind port {port}: {e}");
                std::process::exit(1);
            });
    eprintln!(
        "vmon agent listening on http://{}/metrics",
        SocketAddr::new(bind, port)
    );

    axum::serve(listener, app).await.unwrap_or_else(|e| {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    });
}

/// Read the raw NVLink bandwidth cumulative counter (in KiB) from NVML.
/// C0 = TX (transmit), C1 = RX (receive).
fn read_nvlink_counter(device: &nvml_wrapper::Device, is_tx: bool) -> Option<u64> {
    use nvml_wrapper::enums::device::SampleValue;
    use nvml_wrapper::structs::device::FieldId;
    use nvml_wrapper::sys_exports::field_id;

    let field_id_value = if is_tx {
        field_id::NVML_FI_DEV_NVLINK_BANDWIDTH_C0_TOTAL
    } else {
        field_id::NVML_FI_DEV_NVLINK_BANDWIDTH_C1_TOTAL
    };

    let values = device.field_values_for(&[FieldId(field_id_value)]).ok()?;
    if let Some(Ok(sample)) = values.first() {
        if let Ok(value) = &sample.value {
            let counter = match value {
                SampleValue::U64(v) => *v,
                SampleValue::U32(v) => *v as u64,
                SampleValue::I64(v) => *v as u64,
                SampleValue::F64(v) => *v as u64,
            };
            return Some(counter);
        }
    }

    None
}

/// Tracks per-GPU NVLink cumulative counters to compute bandwidth rates (KB/s).
struct NvlinkState {
    /// (gpu_index) -> (prev_tx_counter_kib, prev_rx_counter_kib, timestamp)
    prev: HashMap<u32, (u64, u64, Instant)>,
}

impl NvlinkState {
    fn new() -> Self {
        Self {
            prev: HashMap::new(),
        }
    }

    /// Compute NVLink bandwidth rates in KB/s from cumulative KiB counters.
    /// Returns (tx_kbps, rx_kbps) for the given GPU.
    fn update(&mut self, gpu_idx: u32, tx_counter: u64, rx_counter: u64) -> (u64, u64) {
        let now = Instant::now();
        let result = if let Some(&(prev_tx, prev_rx, prev_time)) = self.prev.get(&gpu_idx) {
            let dt = now.duration_since(prev_time).as_secs_f64();
            if dt > 0.0 {
                // Counters are cumulative KiB; delta/dt gives KiB/s ≈ KB/s
                let tx_kbps = (tx_counter.saturating_sub(prev_tx) as f64 / dt) as u64;
                let rx_kbps = (rx_counter.saturating_sub(prev_rx) as f64 / dt) as u64;
                (tx_kbps, rx_kbps)
            } else {
                (0, 0)
            }
        } else {
            (0, 0) // First reading, no rate yet
        };
        self.prev.insert(gpu_idx, (tx_counter, rx_counter, now));
        result
    }
}

fn collect_snapshot(
    nvml: Option<&nvml_wrapper::Nvml>,
    device_count: u32,
    cpu_state: &mut CpuState,
    nvlink_state: &mut NvlinkState,
) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    // Pre-compute NVLink bandwidth rates from cumulative counters
    let mut nvlink_rates: HashMap<u32, (u64, u64)> = HashMap::new();
    if let Some(nvml) = nvml {
        for i in 0..device_count {
            if let Ok(device) = nvml.device_by_index(i) {
                let tx = read_nvlink_counter(&device, true);
                let rx = read_nvlink_counter(&device, false);
                if let (Some(tx_val), Some(rx_val)) = (tx, rx) {
                    nvlink_rates.insert(i, nvlink_state.update(i, tx_val, rx_val));
                }
            }
        }
    }

    // ── Host CPU metrics (from /proc/stat) ──
    if let Some(cpu) = cpu_state.update() {
        writeln!(
            out,
            "# HELP vmon_cpu_usage_percent Host CPU utilization percentage"
        )
        .unwrap();
        writeln!(out, "# TYPE vmon_cpu_usage_percent gauge").unwrap();
        writeln!(out, "vmon_cpu_usage_percent {:.1}", cpu).unwrap();
    }

    // ── Host memory metrics (from /proc/meminfo) ──
    if let Some((used, total)) = read_memory_info() {
        writeln!(
            out,
            "# HELP vmon_memory_used_bytes Host memory used in bytes"
        )
        .unwrap();
        writeln!(out, "# TYPE vmon_memory_used_bytes gauge").unwrap();
        writeln!(out, "vmon_memory_used_bytes {used}").unwrap();
        writeln!(
            out,
            "# HELP vmon_memory_total_bytes Host memory total in bytes"
        )
        .unwrap();
        writeln!(out, "# TYPE vmon_memory_total_bytes gauge").unwrap();
        writeln!(out, "vmon_memory_total_bytes {total}").unwrap();
    }

    // ── GPU metrics (from NVML) ──
    if let Some(nvml) = nvml {
        let metrics = [
            (
                "vmon_gpu_utilization",
                "GPU compute utilization percentage",
                "gauge",
            ),
            (
                "vmon_gpu_memory_utilization",
                "GPU memory bandwidth utilization percentage",
                "gauge",
            ),
            ("vmon_gpu_power_watts", "GPU power draw in watts", "gauge"),
            (
                "vmon_gpu_temperature_celsius",
                "GPU temperature in degrees Celsius",
                "gauge",
            ),
            (
                "vmon_gpu_memory_used_bytes",
                "GPU memory used in bytes",
                "gauge",
            ),
            (
                "vmon_gpu_memory_total_bytes",
                "GPU memory total in bytes",
                "gauge",
            ),
            ("vmon_gpu_clock_mhz", "GPU SM clock speed in MHz", "gauge"),
            (
                "vmon_gpu_mem_clock_mhz",
                "GPU memory clock speed in MHz",
                "gauge",
            ),
            (
                "vmon_gpu_pcie_tx_kbps",
                "GPU PCIe TX throughput in KB/s",
                "gauge",
            ),
            (
                "vmon_gpu_pcie_rx_kbps",
                "GPU PCIe RX throughput in KB/s",
                "gauge",
            ),
            (
                "vmon_gpu_power_limit_watts",
                "GPU power management limit in watts",
                "gauge",
            ),
            (
                "vmon_gpu_throttle_reasons",
                "GPU clock throttle reasons bitmask",
                "gauge",
            ),
            (
                "vmon_gpu_ecc_sbe_volatile",
                "GPU corrected (single-bit) ECC errors since boot",
                "gauge",
            ),
            (
                "vmon_gpu_ecc_dbe_volatile",
                "GPU uncorrected (double-bit) ECC errors since boot",
                "gauge",
            ),
            (
                "vmon_gpu_nvlink_tx_kbps",
                "GPU NVLink TX throughput in KB/s",
                "gauge",
            ),
            (
                "vmon_gpu_nvlink_rx_kbps",
                "GPU NVLink RX throughput in KB/s",
                "gauge",
            ),
        ];

        for (name, help, mtype) in &metrics {
            writeln!(out, "# HELP {name} {help}").unwrap();
            writeln!(out, "# TYPE {name} {mtype}").unwrap();
            for i in 0..device_count {
                if let Ok(device) = nvml.device_by_index(i) {
                    let gpu_name = device.name().unwrap_or_default();
                    let val: String = match *name {
                        "vmon_gpu_utilization" => device
                            .utilization_rates()
                            .map(|u| u.gpu.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_memory_utilization" => device
                            .utilization_rates()
                            .map(|u| u.memory.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_power_watts" => device
                            .power_usage()
                            .map(|mw| format!("{:.1}", mw as f64 / 1000.0))
                            .unwrap_or_default(),
                        "vmon_gpu_temperature_celsius" => device
                            .temperature(
                                nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                            )
                            .map(|t| t.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_memory_used_bytes" => {
                            device.memory_info().map(|m| m.used.to_string()).unwrap_or_default()
                        }
                        "vmon_gpu_memory_total_bytes" => {
                            device.memory_info().map(|m| m.total.to_string()).unwrap_or_default()
                        }
                        "vmon_gpu_clock_mhz" => device
                            .clock_info(nvml_wrapper::enum_wrappers::device::Clock::SM)
                            .map(|c| c.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_mem_clock_mhz" => device
                            .clock_info(nvml_wrapper::enum_wrappers::device::Clock::Memory)
                            .map(|c| c.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_pcie_tx_kbps" => device
                            .pcie_throughput(
                                nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Send,
                            )
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_pcie_rx_kbps" => device
                            .pcie_throughput(
                                nvml_wrapper::enum_wrappers::device::PcieUtilCounter::Receive,
                            )
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_power_limit_watts" => device
                            .power_management_limit()
                            .map(|mw| format!("{:.1}", mw as f64 / 1000.0))
                            .unwrap_or_default(),
                        "vmon_gpu_throttle_reasons" => device
                            .current_throttle_reasons()
                            .map(|r| r.bits().to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_ecc_sbe_volatile" => device
                            .total_ecc_errors(
                                nvml_wrapper::enum_wrappers::device::MemoryError::Corrected,
                                nvml_wrapper::enum_wrappers::device::EccCounter::Volatile,
                            )
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_ecc_dbe_volatile" => device
                            .total_ecc_errors(
                                nvml_wrapper::enum_wrappers::device::MemoryError::Uncorrected,
                                nvml_wrapper::enum_wrappers::device::EccCounter::Volatile,
                            )
                            .map(|v| v.to_string())
                            .unwrap_or_default(),
                        "vmon_gpu_nvlink_tx_kbps" => {
                            nvlink_rates.get(&i).map(|(tx, _)| tx.to_string()).unwrap_or_default()
                        }
                        "vmon_gpu_nvlink_rx_kbps" => {
                            nvlink_rates.get(&i).map(|(_, rx)| rx.to_string()).unwrap_or_default()
                        }
                        _ => String::new(),
                    };
                    if !val.is_empty() {
                        if gpu_name.is_empty() {
                            writeln!(out, "{name}{{gpu=\"{i}\"}} {val}").unwrap();
                        } else {
                            let safe_name = gpu_name
                                .replace('\\', "\\\\")
                                .replace('"', "\\\"")
                                .replace('\n', "\\n");
                            writeln!(out, "{name}{{gpu=\"{i}\",name=\"{safe_name}\"}} {val}")
                                .unwrap();
                        }
                    }
                }
            }
        }
    }

    out
}

/// Stateful CPU usage tracker. Computes delta between consecutive reads of /proc/stat.
struct CpuState {
    prev_total: u64,
    prev_idle: u64,
}

impl CpuState {
    fn new() -> Self {
        let (total, idle) = Self::read_cpu_times().unwrap_or((0, 0));
        Self {
            prev_total: total,
            prev_idle: idle,
        }
    }

    fn read_cpu_times() -> Option<(u64, u64)> {
        let stat = std::fs::read_to_string("/proc/stat").ok()?;
        let line = stat.lines().next()?;
        if !line.starts_with("cpu ") {
            return None;
        }
        let vals: Vec<u64> =
            line.split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect();
        if vals.len() < 4 {
            return None;
        }
        // user + nice + system + idle + iowait + irq + softirq + steal
        let total: u64 = vals.iter().sum();
        let idle = vals[3] + vals.get(4).copied().unwrap_or(0); // idle + iowait
        Some((total, idle))
    }

    /// Read current CPU times and compute utilization since last call.
    fn update(&mut self) -> Option<f64> {
        let (total, idle) = Self::read_cpu_times()?;
        let total_delta = total.saturating_sub(self.prev_total);
        let idle_delta = idle.saturating_sub(self.prev_idle);
        self.prev_total = total;
        self.prev_idle = idle;
        if total_delta == 0 {
            return Some(0.0);
        }
        Some((1.0 - idle_delta as f64 / total_delta as f64) * 100.0)
    }
}

/// Read memory info from /proc/meminfo.
/// Returns (used_bytes, total_bytes).
fn read_memory_info() -> Option<(u64, u64)> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb: Option<u64> = None;
    let mut available_kb: Option<u64> = None;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kb = rest.trim().strip_suffix("kB").and_then(|v| v.trim().parse().ok());
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kb = rest.trim().strip_suffix("kB").and_then(|v| v.trim().parse().ok());
        }
        if total_kb.is_some() && available_kb.is_some() {
            break;
        }
    }
    let total = total_kb? * 1024;
    let available = available_kb? * 1024;
    Some((total.saturating_sub(available), total))
}

/// Fetch metrics from a forwarded (proxied) Prometheus endpoint.
async fn fetch_forward(client: Option<&reqwest::Client>, url: Option<&str>) -> String {
    let (Some(client), Some(url)) = (client, url) else {
        return String::new();
    };
    let resp = match client.get(url).send().await {
        Ok(r) if r.status().is_success() => r,
        _ => return String::new(),
    };
    vmon_core::http::text(resp).await.unwrap_or_default()
}

/// Merge agent's own metrics with forwarded metrics.
/// Agent metrics take priority (appear first); forwarded metrics are appended.
fn merge_snapshots(agent: &str, forwarded: &str) -> String {
    if forwarded.is_empty() {
        return agent.to_string();
    }
    let mut out = agent.to_string();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(forwarded);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};

    #[test]
    fn daemon_files_reject_symlinks_hardlinks_and_shared_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("target");
        let file = open_agent_file(&target).unwrap();
        let link = dir.path().join("link");
        symlink(&target, &link).unwrap();
        assert!(open_agent_file(&link).is_err());
        std::fs::remove_file(&link).unwrap();
        std::fs::hard_link(&target, &link).unwrap();
        assert!(open_agent_file(&target).is_err());
        std::fs::remove_file(&link).unwrap();
        file.set_permissions(std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(open_agent_file(&target).is_err());
    }

    #[test]
    fn daemon_pid_lock_excludes_another_instance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("agent.pid");
        let first = open_agent_file(&path).unwrap();
        first.try_lock().unwrap();
        let second = open_agent_file(&path).unwrap();
        assert!(second.try_lock().is_err());
        drop(first);
        second.try_lock().unwrap();
    }
}
