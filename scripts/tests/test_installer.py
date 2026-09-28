# SPDX-License-Identifier: Apache-2.0
"""Exercise local and piped installation without fetching or building dependencies."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


def executable(path, body):
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.mock = self.base / "bin"
        self.mock.mkdir()
        self.work = self.base / "tmp"
        self.work.mkdir()
        self.dest = self.base / "install ' $(touch INJECTED)"
        self.env = dict(os.environ, PATH=str(self.mock) + ":" + os.environ["PATH"],
                        TMPDIR=str(self.work), TEST_GIT_ARGS=str(self.base / "git-args"))
        self.env.pop("CARGO_TARGET_DIR", None)
        executable(self.mock / "rustc", "echo 'host: aarch64-unknown-linux-gnu'\n")
        executable(self.mock / "git", '''printf '%s\\n' "$@" > "$TEST_GIT_ARGS"
dest="${@: -1}"
mkdir -p "$dest"
touch "$dest/Cargo.toml"
''')
        executable(self.mock / "cargo", '''if [[ "${TEST_BUILD_FAIL:-}" == 1 ]]; then exit 9; fi
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-dir) target_dir="$2"; shift 2 ;;
        --target) target="$2"; shift 2 ;;
        *) shift ;;
    esac
done
mkdir -p "$target_dir/$target/release"
printf '#!/usr/bin/env bash\\necho "vmon 0.1.0"\\n' > "$target_dir/$target/release/vmon"
''')

    def piped(self, *args):
        return subprocess.run(["bash", "-s", "--", *args], input=(ROOT / "install.sh").read_bytes(),
                              cwd=self.base, env=self.env, capture_output=True)

    def test_piped_install_fetches_selected_ref_and_cleans_up(self):
        result = self.piped("--to", str(self.dest), "--ref", "v0.1.0")
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        args = (self.base / "git-args").read_text().splitlines()
        self.assertEqual(args[:7], ["clone", "--depth", "1", "--branch", "v0.1.0", "--",
                                   "https://github.com/vllm-project/vmon.git"])
        self.assertTrue(os.access(self.dest / "vmon", os.X_OK))
        self.assertEqual(list(self.work.iterdir()), [])
        self.assertFalse((self.base / "INJECTED").exists())
        self.assertEqual(subprocess.check_output([str(self.dest / "vmon"), "--version"], text=True),
                         "vmon 0.1.0\n")

    def test_local_checkout_uses_custom_target_directory_without_fetching(self):
        repo = self.base / "checkout with spaces"
        repo.mkdir()
        (repo / "Cargo.toml").touch()
        shutil.copyfile(ROOT / "install.sh", repo / "install.sh")
        self.env["CARGO_TARGET_DIR"] = str(self.base / "custom target")
        result = subprocess.run(["bash", str(repo / "install.sh"), "--to", str(self.dest)],
                                cwd=self.base, env=self.env, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertTrue((self.dest / "vmon").is_file())
        self.assertFalse((self.base / "git-args").exists())

    def test_build_failure_keeps_existing_install_and_removes_download(self):
        self.dest.mkdir()
        (self.dest / "vmon").write_text("existing binary")
        self.env["TEST_BUILD_FAIL"] = "1"
        result = self.piped("--to", str(self.dest))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.dest / "vmon").read_text(), "existing binary")
        self.assertEqual(list(self.work.iterdir()), [])

    def test_missing_option_value_is_reported(self):
        for option in ("--to", "--ref"):
            with self.subTest(option=option):
                result = self.piped(option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires a value", result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
