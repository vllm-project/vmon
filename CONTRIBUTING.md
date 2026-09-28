# Contributing

Use Rust 1.96 or newer. Package metadata, third-party dependencies and lints live
in the workspace Cargo.toml. The code targets Unix-like systems. See AGENTS.md
for module, error-handling, documentation and test conventions.

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace
cargo deny check
python3 -B -m unittest discover -s scripts/tests -v
```

If cargo-nextest is installed, use `cargo nextest run --locked --workspace` for
unit/integration tests and `cargo test --locked --workspace --doc` for doctests.
The optional `.githooks/pre-commit` runs read-only format and lint checks; enable
it with `git config core.hooksPath .githooks`.

Use small modules, preserve error sources and existing documentation, and add
regression tests for security fixes. Keep raw captures in `reports/`. Never add
real credentials, cluster identifiers or personal paths to example data.
Use synthetic node names, `example` model names, and documentation IP ranges in
fixtures.

For releases, build with `--locked`, generate dependency license materials with
`scripts/license_bundle.py`, and include those materials and checksums alongside
the binaries. See SECURITY.md for runtime trust boundaries and private reporting.
