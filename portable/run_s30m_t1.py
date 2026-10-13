#!/usr/bin/env python3
"""Portable wrapper: 30M scale-transfer T1 with dynamic path mapping.

Frozen source src/s30m_runner.py (LAB-relative; MINIMIND_BASE=LAB/minimind_base) resolves external assets via s30m_core.LAB → minimind_base; s30m_mint_cache CORPUS/V16_CACHE/TOKDIR. This wrapper
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

ASSETS = [('s30m cache', 'CKPT_ROOT', 'train_tokens_s30m.u32'), ('V16 cache (prefix identity)', 'V16_ROOT', 'campaign_v16/cache/train_tokens.u32'), ('corpus', 'DATA_ROOT', 'pretrain_t2t_mini.jsonl'), ('minimind model', 'MINIMIND_ROOT', 'model/model_minimind.py')]

def main():
    import argparse, os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.config)
    print("== portable 30M scale-transfer T1 =="); print(describe(paths))
    ok = True
    for name, key, rel in ASSETS:
        root = paths.get(key, ""); p = Path(root) / rel if root else None
        hit = bool(p and p.exists()); ok &= hit
        print(f"  [{'x' if hit else ' '}] {name:24s} {p}")
    print("\nfrozen entry : python src/s30m_runner.py (mint cache first: src/s30m_mint_cache.py)")
    print("frozen config sha16: 0156149803ff3310 (configs/frozen_30m_t1.json)")
    print("plan: 13 runs x 2048 steps, seed 3421, 16.78M tok/run; history 1.45h GPU")
    if not ok:
        print("RESULT: BLOCKED_ASSET — restore per docs/ASSET_RESTORATION_GUIDE.md"); sys.exit(1)
    if not (a.execute and os.environ.get("ADAMWSPEC_ALLOW_GPU") == "1"):
        print("RESULT: DRY-RUN OK (no GPU side effect)."); sys.exit(0)
    print("RESULT: execution not implemented in this archive round — use frozen entry."); sys.exit(3)

if __name__ == "__main__":
    main()
