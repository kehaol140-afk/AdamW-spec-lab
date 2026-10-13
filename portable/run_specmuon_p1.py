#!/usr/bin/env python3
"""Portable wrapper: SpecMuon P1 pilot with dynamic path mapping.

Frozen source src/specmuon/run_pilot.py (io_utils.LAB_ROOT absolute) resolves external assets via io_utils.LAB_ROOT → muon_transition + msa parent states. This wrapper
verifies assets and prints the frozen plan; module-constant patching hooks
are documented in docs/PORTABLE_VS_FROZEN_DIFF.md. Frozen file never modified.
Default DRY-RUN; execution requires --execute AND ADAMWSPEC_ALLOW_GPU=1
(not authorized in this archive round).
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from portable.paths import load_paths, describe  # noqa: E402

ASSETS = [('MSA parent step512', 'CKPT_ROOT', 'msa/outputs/p0_p1_v1_1/parent/parent_state_step512.pt'), ('MSA parent step1536', 'CKPT_ROOT', 'msa/outputs/p0_p1_v1_1/parent/parent_state_step1536.pt'), ('muon routing report', 'MUON_TRANSITION_ROOT', 'outputs/m0/routing_report.json')]

def main():
    import argparse, os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.config)
    print("== portable SpecMuon P1 pilot =="); print(describe(paths))
    ok = True
    for name, key, rel in ASSETS:
        root = paths.get(key, ""); p = Path(root) / rel if root else None
        hit = bool(p and p.exists()); ok &= hit
        print(f"  [{'x' if hit else ' '}] {name:24s} {p}")
    print("\nfrozen entry : python src/specmuon/run_pilot.py (from experiments/specmuon-pilot/)")
    print("frozen config sha16: pilot_frozen_v1.json (FROZEN_BEFORE_ANY_P1_RUN)")
    print("plan: 14 branches x 256 steps from s512/s1536 restore; ~20min/branch GPU history")
    if not ok:
        print("RESULT: BLOCKED_ASSET — restore per docs/ASSET_RESTORATION_GUIDE.md"); sys.exit(1)
    if not (a.execute and os.environ.get("ADAMWSPEC_ALLOW_GPU") == "1"):
        print("RESULT: DRY-RUN OK (no GPU side effect)."); sys.exit(0)
    print("RESULT: execution not implemented in this archive round — use frozen entry."); sys.exit(3)

if __name__ == "__main__":
    main()
