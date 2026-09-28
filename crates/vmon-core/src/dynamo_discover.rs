// SPDX-License-Identifier: Apache-2.0

//! Probe-based discovery of NVIDIA Dynamo `/metrics` endpoints.
//!
//! Dynamo's etcd registration only stores NATS transport subjects, not the
//! `system_status_server` HTTP host:port. So we can't ask etcd "where do I
//! scrape?" — we have to discover by probing a known set of candidate ports
//! per host. A port counts as a Dynamo metrics endpoint when:
//!
//! 1. `GET /metrics` returns 2xx within a short timeout
//! 2. The body contains `dynamo_` (filters out unrelated HTTP servers)

use futures::future::join_all;
use futures::stream::{self, StreamExt};
use std::time::Duration;

/// Probe span for globally numbered worker ports: up to 20 hosts × 8 ranks.
const SYS_PORT_SPAN: u16 = 160;

/// Candidate frontend and worker ports. Include alternate worker ranges
/// for deployments that use a different base port.
fn candidate_ports() -> impl Iterator<Item = u16> {
    std::iter::once(8180)
        .chain(7500..7500 + SYS_PORT_SPAN)
        .chain(8200..=8231)
        .chain(8081..=8127)
}

const PROBE_TIMEOUT: Duration = Duration::from_millis(1500);

/// Limit concurrent connections per host to avoid bursts that cause probe
/// timeouts or exhaust local ephemeral ports.
const MAX_CONCURRENT_PROBES_PER_HOST: usize = 16;

/// Fallback port used as a placeholder when discovery finds no live
/// `/metrics` endpoints on a host (e.g., dynamo workers still loading
/// the model). Subsequent discovery passes replace it with live endpoints.
pub const PLACEHOLDER_PORT: u16 = 7500;

/// Whether `addr` is a placeholder address emitted by `discover_addrs` when
/// no live `/metrics` endpoint was found. Used by the follow loop to re-probe
/// such hosts on subsequent ticks — without this, a host that was still
/// loading its model at startup would stay stuck on the placeholder forever
/// even after its workers come up on real ports.
pub fn is_placeholder_addr(addr: &str) -> bool {
    matches!(addr.rsplit_once(':'), Some((_, p)) if p.parse::<u16>() == Ok(PLACEHOLDER_PORT))
}

/// One attempt at probing host:port. Returns:
/// - `Ok(Some(port))`  endpoint serves dynamo metrics
/// - `Ok(None)`        endpoint reachable but not dynamo (don't retry)
/// - `Err(())`         transient failure (timeout / connect error) — caller may retry
async fn probe_port_once(
    client: &reqwest::Client,
    host: &str,
    port: u16,
) -> Result<Option<u16>, ()> {
    let url = format!("http://{host}:{port}/metrics");
    let resp = match client.get(&url).timeout(PROBE_TIMEOUT).send().await {
        Ok(r) => r,
        Err(_) => return Err(()),
    };
    if !resp.status().is_success() {
        return Ok(None);
    }
    let body = match crate::http::text(resp).await {
        Ok(b) => b,
        Err(_) => return Err(()),
    };
    Ok(if body.contains("dynamo_") {
        Some(port)
    } else {
        None
    })
}

async fn probe_port(client: &reqwest::Client, host: &str, port: u16) -> Option<u16> {
    match probe_port_once(client, host, port).await {
        Ok(opt) => opt,
        // Retry once on transient failure — under heavy probe load some SYNs
        // get dropped and the first attempt times out even though the worker
        // is up.
        Err(()) => probe_port_once(client, host, port).await.ok().flatten(),
    }
}

/// Probe a single host across the candidate port list and return the live ones, sorted.
pub async fn discover_host_ports(client: &reqwest::Client, host: &str) -> Vec<u16> {
    let mut alive: Vec<u16> = stream::iter(candidate_ports())
        .map(|p| probe_port(client, host, p))
        .buffer_unordered(MAX_CONCURRENT_PROBES_PER_HOST)
        .filter_map(|r| async move { r })
        .collect()
        .await;
    alive.sort_unstable();
    alive
}

/// Discover Dynamo `/metrics` addresses across multiple hosts.
///
/// Hosts may be bare names (e.g. `node01`) or `host:port` (passed through as-is
/// without probing). Returns a flat list of `host:port` strings, ordered by
/// input host order then ascending port.
pub async fn discover_addrs(hosts: &[String]) -> Vec<String> {
    let client = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .build()
        .expect("failed to build HTTP client");

    let bare: Vec<String> = hosts
        .iter()
        .filter(|h| {
            !h.contains(':') || h.rsplit_once(':').is_none_or(|(_, p)| p.parse::<u16>().is_err())
        })
        .cloned()
        .collect();
    let explicit: Vec<String> = hosts.iter().filter(|h| !bare.contains(*h)).cloned().collect();

    let tasks = bare.iter().map(|h| {
        let client = client.clone();
        let host = h.clone();
        async move {
            let ports = discover_host_ports(&client, &host).await;
            (host, ports)
        }
    });
    let results = join_all(tasks).await;

    let mut addrs = Vec::new();
    addrs.extend(explicit);
    for (host, ports) in results {
        if ports.is_empty() {
            // Worker probably still loading the model (or never will register
            // a metrics endpoint). Emit a placeholder addr so the host still
            // appears in the TUI for GPU / IB / SLURM data, and /metrics
            // scraping starts working automatically once a worker eventually
            // binds to the placeholder port (7500).
            tracing::info!(
                host = %host,
                placeholder_port = PLACEHOLDER_PORT,
                "Dynamo discover: no /metrics endpoints (worker not ready yet?); keeping host with placeholder"
            );
            addrs.push(format!("{host}:{PLACEHOLDER_PORT}"));
            continue;
        }
        tracing::info!(
            host = %host,
            count = ports.len(),
            ports = ?ports,
            "Dynamo discover: found endpoint(s)"
        );
        for p in ports {
            addrs.push(format!("{host}:{p}"));
        }
    }
    addrs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    /// Spin up a tiny HTTP server on a random port that returns the given body
    /// on `/metrics` and 404 elsewhere. Returns the bound port.
    async fn fake_server(body: &'static str) -> u16 {
        let listener = TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(x) => x,
                    Err(_) => continue,
                };
                let body = body.to_string();
                tokio::spawn(async move {
                    let mut buf = [0u8; 1024];
                    let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
                    let req = String::from_utf8_lossy(&buf);
                    let response = if req.starts_with("GET /metrics") {
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                            body.len()
                        )
                    } else {
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_string()
                    };
                    let _ = sock.write_all(response.as_bytes()).await;
                });
            }
        });
        port
    }

    #[test]
    fn candidate_ports_cover_large_worker_groups() {
        let ports: Vec<u16> = candidate_ports().collect();
        // Cover worker ports beyond the first 32 ranks and alternate bases.
        for p in [8180, 7500, 7535, 7571, 7595, 8200, 8231, 8081, 8127] {
            assert!(ports.contains(&p), "port {p} should be probed");
        }
    }

    #[tokio::test]
    async fn probe_accepts_dynamo_body() {
        let port = fake_server(
            "# HELP dynamo_component_uptime_seconds ...\ndynamo_component_uptime_seconds 1.0\n",
        )
        .await;
        let client = reqwest::Client::new();
        let result = probe_port(&client, "127.0.0.1", port).await;
        assert_eq!(result, Some(port));
    }

    #[tokio::test]
    async fn probe_rejects_non_dynamo_body() {
        let port = fake_server("# HELP go_threads Number of OS threads\ngo_threads 5\n").await;
        let client = reqwest::Client::new();
        let result = probe_port(&client, "127.0.0.1", port).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn probe_returns_none_on_unreachable_port() {
        // Port 1 is reserved and almost certainly closed
        let client =
            reqwest::Client::builder().timeout(Duration::from_millis(200)).build().unwrap();
        let result = probe_port(&client, "127.0.0.1", 1).await;
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn discover_addrs_passes_through_explicit_hostport() {
        // Explicit host:port stays as-is, even if the port isn't reachable.
        let addrs = discover_addrs(&["unreachable-host-xyz:9999".to_string()]).await;
        assert_eq!(addrs, vec!["unreachable-host-xyz:9999"]);
    }

    #[tokio::test]
    async fn discover_addrs_keeps_host_with_placeholder_when_no_endpoints() {
        // Bare hostname with no live /metrics endpoint should still appear in
        // the addr list so GPU / IB / SLURM data continues flowing for the
        // host, and a worker that comes up later on the placeholder port is
        // picked up automatically.
        let addrs = discover_addrs(&["nonexistent-host-for-test-only".to_string()]).await;
        assert_eq!(
            addrs,
            vec![format!("nonexistent-host-for-test-only:{PLACEHOLDER_PORT}")]
        );
    }
}
