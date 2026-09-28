# Security

Please report vulnerabilities privately to the repository maintainers using
GitHub's **Security → Report a vulnerability** feature when available. If it is
unavailable, request a private contact channel from a maintainer without posting
exploit details, credentials or collected reports in a public issue.

The current release line receives security fixes. Use the committed Cargo.lock
with `--locked` and check dependency advisories before distributing a build.

## Deployment

`vmon agent` binds to `127.0.0.1` by default. Remote monitoring requires an explicit
`--bind <management-IP>` (or `--bind 0.0.0.0` on an access-controlled network).
The metrics endpoint has no built-in authentication or TLS. Restrict it with a
firewall, tunnel or authenticated TLS proxy; do not expose it to the Internet.
Forwarded metrics are exposed to the same clients as the agent's own metrics.

vLLM dev mode exposes configuration and engine-control endpoints. Enable it only
on trusted, access-controlled networks. vmon redacts common credential field
names in server information before storing it in UI state, including nested JSON.
This is best-effort filtering, not a guarantee that arbitrary configuration is
safe to publish. Reports and raw captures can retain addresses, GPU UUIDs, model
names and metric labels. Review and redact them before sharing.

HTTP responses are limited to 16 MiB. ZMQ frames and receive queues are bounded.
Only subscribe to trusted publishers; ZeroMQ does not authenticate peers here.

Daemon files default to `~/.local/state/vmon/agent.{pid,log}`. The state directory
must be private (0700); custom PID/log files must be owned by the invoking user,
regular, single-link files with no group/other permissions. A held PID-file lock
prevents two agents from using the same daemon state. Use separate files for
multiple agents. Reports are atomically written with private permissions (0600).
