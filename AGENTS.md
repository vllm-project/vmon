# Repository conventions

- Target Unix-like platforms. Keep dependencies and package metadata in the root Cargo workspace; member crates inherit them.
- Prefer small modules with a clear responsibility. Keep CLI parsing and orchestration separate from agent, transport and filesystem code.
- Preserve applicable comments and documentation verbatim when moving code. Document public APIs and security boundaries concisely.
- Propagate typed errors and preserve their source. Avoid flattening errors with `to_string()`. Use structured fields for context. Do not add panics to recoverable input or I/O paths.
- Bound data read from remote endpoints. Escape each output context separately. Treat node names, labels and server configuration as untrusted.
- Do not include credentials, real cluster identifiers or collected reports in fixtures. Use `example` domains and documentation IP ranges.
- Use deterministic synchronization for async tests. Test externally visible behavior; prefer snapshots for large structured outputs, and focused assertions for security invariants.
- Run `cargo fmt --all -- --check` and `cargo clippy --locked --workspace --all-targets --all-features -- -D warnings`. Run `cargo nextest run --locked --workspace` when installed; otherwise `cargo test --locked --workspace`.
- Run `cargo deny check` before changing the dependency lockfile for release. Keep third-party licenses with packaged artifacts.
- Never rewrite existing Git history or publish a release as part of ordinary code cleanup. Public source exports contain only reviewed source files, without `.git` or runtime data.
