# SPDX-License-Identifier: Apache-2.0
"""Check that shared fallback texts expand without losing package notices."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("license_bundle", ROOT / "scripts/license_bundle.py")
license_bundle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(license_bundle)


class LicenseBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "licenses/cargo"
        self.shared = self.base / "example"
        self.shared.mkdir(parents=True)
        self.text = b"MIT license\nCopyright Example contributors\n"
        (self.shared / "LICENSE-MIT").write_bytes(self.text)
        self.url = "https://example.com/revision/LICENSE-MIT"
        self.index = {"example": {
            "packages": ["example-1.0.0", "example-sys-1.0.0"],
            "sources": {"LICENSE-MIT": self.url},
        }}
        self.write_index()

    def write_index(self):
        (self.base / "index.json").write_text(json.dumps(self.index))

    def destination(self, name):
        dest = self.root / "release" / name
        dest.mkdir(parents=True)
        return dest

    def test_shared_text_and_sources_expand_for_each_package(self):
        for package in self.index["example"]["packages"]:
            with self.subTest(package=package):
                dest = self.destination(package)
                license_bundle.copy_fallback_licenses(self.root, package, dest)
                self.assertEqual((dest / "LICENSE-MIT").read_bytes(), self.text)
                self.assertEqual((dest / "SOURCE.txt").read_text(), self.url + "\n")

    def test_unreviewed_version_has_no_fallback(self):
        with self.assertRaises(RuntimeError):
            license_bundle.copy_fallback_licenses(
                self.root, "example-2.0.0", self.destination("unknown"))

    def test_missing_text_fails_instead_of_omitting_the_notice(self):
        (self.shared / "LICENSE-MIT").unlink()
        with self.assertRaises(FileNotFoundError):
            license_bundle.copy_fallback_licenses(
                self.root, "example-1.0.0", self.destination("missing"))

    def test_ambiguous_package_mapping_is_rejected(self):
        self.index["duplicate"] = self.index["example"]
        self.write_index()
        with self.assertRaises(RuntimeError):
            license_bundle.copy_fallback_licenses(
                self.root, "example-1.0.0", self.destination("ambiguous"))

    def test_fallback_cannot_read_outside_license_directory(self):
        self.index["example"]["sources"] = {"../../../outside": self.url}
        self.write_index()
        with self.assertRaises(RuntimeError):
            license_bundle.copy_fallback_licenses(
                self.root, "example-1.0.0", self.destination("outside"))


if __name__ == "__main__":
    unittest.main()
