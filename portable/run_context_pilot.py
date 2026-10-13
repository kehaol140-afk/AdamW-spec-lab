#!/usr/bin/env python3
"""Portable wrapper: Context-Aware Muon 100M pilot with dynamic path mapping.

Frozen source scripts/run_pilot_driver.py (ROOT-relative; DEV→V16 eval_sets absolute) resolves external assets via run_pilot_arm.DEV / mint_pilot_cache CORPUS/V16_CACHE/S30M_CACHE/TOKDIR / model_100m MINIMIND_MODEL. This wrapper
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

ASSETS = [('pilot cache 33.7M', 'CKPT_ROOT', 'pilot_train_tokens_33p7m.u32'), ('V16 eval_sets', 'V16_ROOT', 'campaign_v16/eval_sets.json'), ('corpus', 'DATA_ROOT', 'pretrain_t2t_mini.jsonl'), ('minimind model', 'MINIMIND_ROOT', 'model/model_minimind.py')]

def main():
    import argparse, os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.config)
    print("== portable Context-Aware Muon 100M pilot =="); print(describe(paths))
    ok = True
    for name, key, rel in ASSETS:
        root = paths.get(key, ""); p = Path(root) / rel if root else None
        hit = bool(p and p.exists()); ok &= hit
        print(f"  [{'x' if hit else ' '}] {name:24s} {p}")
    print("\nfrozen entry : python scripts/run_pilot_driver.py (assets minted first; run_pilot_arm.py per arm)")
    print("frozen config sha16: 8149bcbbdf41ff33 (configs/pilot_arms_v1.json)")
    print("plan: 15 arms x 4096 steps, seed 3471, 33.55M tok/arm; history 4.55h GPU")
    if not ok:
        print("RESULT: BLOCKED_ASSET — restore per docs/ASSET_RESTORATION_GUIDE.md"); sys.exit(1)
    if not (a.execute and os.environ.get("ADAMWSPEC_ALLOW_GPU") == "1"):
        print("RESULT: DRY-RUN OK (no GPU side effect)."); sys.exit(0)
    print("RESULT: execution not implemented in this archive round — use frozen entry."); sys.exit(3)

if __name__ == "__main__":
    main()
