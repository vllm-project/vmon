# Third-party components

vmon source is licensed under Apache-2.0. Dependencies retain their own licenses.

| Embedded component | Version | Source | License material |
| --- | --- | --- | --- |
| libzmq (through zeromq-src) | 4.3.4 | https://github.com/zeromq/libzmq/tree/v4.3.4 | licenses/libzmq-COPYING.txt, licenses/libzmq-COPYING.LESSER.txt, licenses/libzmq-exception.txt |

The libzmq source uses LGPL-3.0-or-later with its documented independent-module
linking exception. The Rust wrapper's package license does not replace this
license. `scripts/license_bundle.py` includes the exact bundled native source in
release license materials, alongside the upstream license texts and exception.

The binary release archives include a `third-party` directory generated from the
locked Cargo dependency metadata. It records each dependency's version, declared
license, source/repository and license/notice files available in its package.
Its native source archive also includes libzmq's bundled third-party notices.

For crates that omit license texts from their published packages,
`licenses/cargo/index.json` maps exact package versions to shared license files
and commit-pinned upstream sources. Only byte-identical copies are shared;
distinct copyright notices are retained. The license bundle script expands
these files and their source URLs into each dependency's directory.
