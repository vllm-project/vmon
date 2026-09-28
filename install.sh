#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Install vmon from a local checkout or fetch source when run through a pipe.

set -euo pipefail

usage() {
    cat <<'HELP'
Usage: install.sh [--to DIR] [--ref REF]
  --to DIR   Install directory (default: ~/.local/bin)
  --ref REF  Fetch a branch or tag instead of using local source (default: main)

Requires Rust 1.96+, a C/C++ compiler, and pkg-config.
Fetching source also requires git and access to vllm-project/vmon.
HELP
}

die() {
    echo "Error: $*" >&2
    exit 1
}

cleanup() {
    if [ -n "${vmon_install_stage:-}" ]; then rm -f -- "$vmon_install_stage"; fi
    if [ -n "${vmon_install_source:-}" ]; then rm -rf -- "$vmon_install_source"; fi
}

main() {
    local install_dir="${HOME}/.local/bin" source_ref="main" fetch_source=false
    local root="" script_path="${BASH_SOURCE[0]:-}"
    vmon_install_source=""
    vmon_install_stage=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --to|--ref)
                [ "$#" -ge 2 ] && [ -n "$2" ] || die "$1 requires a value"
                if [ "$1" = --to ]; then
                    install_dir="$2"
                else
                    source_ref="$2"
                    fetch_source=true
                fi
                shift 2 ;;
            --help|-h) usage; return ;;
            *) die "Unknown option: $1 (see --help)" ;;
        esac
    done
    for tool in cargo rustc; do
        command -v "$tool" >/dev/null 2>&1 || die "$tool not found; install Rust 1.96+ first: https://rustup.rs"
    done
    if [ -n "$script_path" ] && [ -f "$script_path" ] && [ "$fetch_source" = false ]; then
        root="$(cd -- "$(dirname -- "$script_path")" && pwd)"
        [ -f "$root/Cargo.toml" ] || root=""
    fi

    trap cleanup EXIT
    if [ -z "$root" ]; then
        command -v git >/dev/null 2>&1 || die "git not found; install git to fetch the source"
        vmon_install_source="$(mktemp -d "${TMPDIR:-/tmp}/vmon-source.XXXXXXXX")"
        echo "Fetching vmon source ($source_ref)..."
        git clone --depth 1 --branch "$source_ref" -- \
            https://github.com/vllm-project/vmon.git "$vmon_install_source/source"
        root="$vmon_install_source/source"
    fi

    local target_dir="${CARGO_TARGET_DIR:-${root}/target}" host
    host="$(rustc -vV | sed -n 's/^host: //p')"
    [ -n "$host" ] || die "could not determine the Rust host target"
    echo "Building vmon (release)..."
    cargo build --locked --release --manifest-path "$root/Cargo.toml" \
        --target-dir "$target_dir" --target "$host" -p vmon-cli

    mkdir -p -- "$install_dir"
    install_dir="$(cd -- "$install_dir" && pwd)"
    # Rename a staged file so replacing a running binary remains atomic.
    vmon_install_stage="$(mktemp "${install_dir}/.vmon.install.XXXXXXXX")"
    cp -- "$target_dir/$host/release/vmon" "$vmon_install_stage"
    chmod 0755 "$vmon_install_stage"
    mv -f -- "$vmon_install_stage" "$install_dir/vmon"
    local installed_version
    installed_version="$("$install_dir/vmon" --version)"
    echo "Installed $installed_version to $install_dir/vmon"
    case ":${PATH}:" in
        *":${install_dir}:"*) ;;
        *) printf '\nAdd the install directory to PATH:\n  export PATH=%q:"$PATH"\n' "$install_dir" ;;
    esac
    cleanup
    trap - EXIT
}

# Read the complete script before starting commands that may consume stdin.
main "$@"
