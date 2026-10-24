#!/usr/bin/env python3
"""Verify every tracked file against CHECKSUMS.sha256.

Usage:  python verify_checksums.py [--quiet]
Exit code 0 iff all files match and no file is missing.
"""
import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "CHECKSUMS.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    entries = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, rel = line.split(None, 1)
        entries.append((digest, rel.strip()))

    bad, missing = [], []
    for digest, rel in entries:
        p = ROOT / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if sha256(p) != digest:
            bad.append(rel)

    total = len(entries)
    print(f"checked {total} file(s)")
    if missing:
        print(f"MISSING ({len(missing)}):")
        for r in missing[:20]:
            print("  -", r)
    if bad:
        print(f"MISMATCH ({len(bad)}):")
        for r in bad[:20]:
            print("  ~", r)
    if bad or missing:
        return 1
    print("OK: all files match CHECKSUMS.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())
