#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect locked dependency licenses and bundled native source for a release."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def copy_fallback_licenses(root, package, dest):
    """Expand shared, reviewed texts into a package's release directory."""
    base = root / "licenses" / "cargo"
    index = json.loads((base / "index.json").read_text())
    matches = [(group, entry) for group, entry in index.items()
               if package in entry["packages"]]
    if len(matches) != 1 or not matches[0][1]["sources"]:
        raise RuntimeError(f"No unique license fallback found for {package}")
    group, entry = matches[0]
    for filename in entry["sources"]:
        source = base / group / filename
        if (Path(filename).name != filename or filename in {".", ".."}
                or not source.resolve().is_relative_to(base.resolve())):
            raise RuntimeError(f"License outside fallback directory: {source}")
        shutil.copyfile(source, dest / filename)
    (dest / "SOURCE.txt").write_text("\n".join(entry["sources"].values()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    subprocess.run(["cargo", "fetch", "--locked"], cwd=root, check=True)
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--offline", "--format-version", "1"], cwd=root
    ))
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for package in sorted(metadata["packages"], key=lambda p: (p["name"], p["version"])):
        if package["source"] is None:
            continue
        source = Path(package["manifest_path"]).parent
        dest = args.output / f'{package["name"]}-{package["version"]}'
        dest.mkdir()
        files = set()
        if package.get("license_file"):
            files.add(source / package["license_file"])
        for candidate in source.iterdir():
            if candidate.is_file() and candidate.name.upper().startswith(
                ("LICENSE", "LICENCE", "COPYING", "NOTICE", "UNLICENSE", "COPYRIGHT", "AUTHORS")
            ):
                files.add(candidate)
        if not files:
            # Some published crates omit license texts. Reviewed upstream texts
            # and their commit-pinned source URLs are kept in the repository.
            copy_fallback_licenses(root, dest.name, dest)
        for file in sorted(files):
            if not file.resolve().is_relative_to(source.resolve()):
                raise RuntimeError(f"License outside package: {file}")
            shutil.copyfile(file, dest / file.name)
        if package["name"] == "zeromq-src":
            with tarfile.open(dest / "libzmq-source.tar.gz", "w:gz") as archive:
                archive.add(source / "vendor", arcname="libzmq-source")
        records.append({key: package.get(key) for key in
                        ("name", "version", "license", "source", "repository")})
    (args.output / "dependencies.json").write_text(json.dumps(records, indent=2) + "\n")
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(root / name, args.output / name)
    shutil.copytree(root / "licenses", args.output / "licenses")


if __name__ == "__main__":
    main()
