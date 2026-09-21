#!/usr/bin/env python3
"""Verify a model directory against reproduce/model-manifest.json.

Usage: python3 check-model-manifest.py <model-dir> [manifest]

Checks, per manifest entry with a sha256 (the LFS files, including all 48
shards): file exists, size matches, streaming SHA-256 matches. Entries
without a hash (small text files) get an existence check. Non-empty
mismatch summary; exit 0 clean, 2 on any failure. Hashing ~428 GiB of
shards takes minutes on NVMe.
"""
import hashlib
import json
import os
import sys


def main() -> int:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    manifest_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model-manifest.json")
    if not model_dir or not os.path.isdir(model_dir):
        print(f"usage: check-model-manifest.py <model-dir> [manifest]\nno model dir: {model_dir!r}", file=sys.stderr)
        return 2
    with open(manifest_path) as f:
        man = json.load(f)
    base = os.path.abspath(model_dir)
    fails, checked, hashed = [], 0, 0
    for ent in man["files"]:
        rel = ent["path"]
        p = os.path.abspath(os.path.join(base, rel))
        if os.path.commonpath([base, p]) != base:
            fails.append(f"{rel}: path escapes model dir")
            continue
        if not os.path.isfile(p):
            fails.append(f"{rel}: missing")
            continue
        checked += 1
        if ent["size"] is not None and os.path.getsize(p) != ent["size"]:
            fails.append(f"{rel}: size {os.path.getsize(p)} != {ent['size']}")
            continue
        want = ent["sha256"]
        if not want:
            continue
        hashed += 1
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        if got != want:
            fails.append(f"{rel}: sha256 {got} != {want}")
        else:
            print(f"ok {rel}")
    print(f"manifest rev {man['revision'][:12]}: {checked} files checked, {hashed} hashed, {len(fails)} failures")
    for line in fails[:20]:
        print("FAIL", line)
    return 2 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
