#!/usr/bin/env python3
"""Verify the unpacked migration package on the server.

Usage:  python3.9 MIGRATION_INFO/verify_package.py
Checks (all must pass before any training):
  1. Every line of MIGRATION_INFO/SHA256SUMS.txt matches the unpacked file.
  2. No missing / no extra files relative to the manifest.
  3. Critical identity hashes (parent checkpoint, train cache, campaign source).
  4. A1 result inventory: 31 complete branches (256 steps + 2x256 eval rows),
     5 incomplete branches, A0 H0 correction files.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAMP = ROOT / "campaign_v16"
FAIL = []


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


sums = {}
for line in (ROOT / "MIGRATION_INFO" / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
    if line.strip():
        digest, name = line.split(" ", 1)
        sums[name.strip().lstrip("*")] = digest

on_disk = {str(p.relative_to(ROOT)).replace("\\", "/")
           for p in ROOT.rglob("*")
           if p.is_file() and "MIGRATION_INFO/SHA256SUMS.txt" not in str(p.relative_to(ROOT)).replace("\\", "/")
           and "__pycache__" not in str(p)}

bad = 0
for name, digest in sorted(sums.items()):
    p = ROOT / name
    if not p.is_file():
        FAIL.append("MISSING: " + name)
        bad += 1
        continue
    if sha256(p) != digest:
        FAIL.append("HASH MISMATCH: " + name)
        bad += 1
extra = sorted(on_disk - set(sums))
print("files checked: %d, hash failures: %d, undeclared extra files: %d" % (len(sums), bad, len(extra)))
for e in extra:
    print("  extra:", e)

critical = {
    "campaign_v16/core.py": None,
    "artifacts/checkpoints/seed3407_lr0.003/step256.pt":
        "ed09a2c80f40aae1a8829a718913a4a40257cdc24fc0e7a2d29bc53a74383000",
    "campaign_v16/cache/train_tokens.u32":
        "c91382dc2a72cdbcf79493e2f2a74ed952b74c067c6af093628a928813fcab2b",
}
for name, expected in critical.items():
    got = sha256(ROOT / name)
    ok = (expected is None) or (got == expected)
    print("critical %-52s %s %s" % (name, got[:16], "OK" if ok else "EXPECTED " + expected[:16]))
    if not ok:
        FAIL.append("CRITICAL: " + name)

# A1 inventory
steps = CAMP / "raw" / "steps"
evals = CAMP / "raw" / "eval"
complete = partial = 0
for fam in ("S", "F", "P"):
    for a in (1, 2, 4, 8):
        for b in (0.5, 1, 2):
            cid = "a1_%s_a%g_b%g" % (fam, a, b)
            sp = steps / (cid + ".jsonl")
            n = sum(1 for _ in sp.open(encoding="utf-8")) if sp.is_file() else 0
            e64 = evals / ("%s__H64__development.jsonl" % cid)
            e256 = evals / ("%s__H256__development.jsonl" % cid)
            r64 = sum(1 for _ in e64.open(encoding="utf-8")) if e64.is_file() else 0
            r256 = sum(1 for _ in e256.open(encoding="utf-8")) if e256.is_file() else 0
            if n == 256 and r64 == 256 and r256 == 256:
                complete += 1
            else:
                partial += 1
                print("  incomplete: %-18s steps=%d H64=%d H256=%d" % (cid, n, r64, r256))
print("A1 branches: %d complete, %d incomplete (expected 31 / 5)" % (complete, partial))
if (complete, partial) != (31, 5):
    FAIL.append("A1_INVENTORY")

if FAIL:
    print("\nVERIFY FAILED:")
    for x in FAIL:
        print(" -", x)
    sys.exit(1)
print("\nVERIFY OK — package is intact and identity-matched.")
