#!/usr/bin/env python3
"""Portable wrapper: V16 campaign (A1/A2/B/C/D queue) with dynamic path mapping.

Frozen source campaign_v16/run.py resolves external assets via ROOT derived from __file__ (already relative); needs token cache + parent checkpoint under campaign_v16/. This wrapper
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

ASSETS = [('token cache', 'V16_ROOT', 'campaign_v16/cache/train_tokens.u32'), ('parent ckpt step256', 'V16_ROOT', 'artifacts/checkpoints/seed3407_lr0.003/step256.pt'), ('campaign_config', 'V16_ROOT', 'campaign_v16/campaign_config.json')]

def main():
    import argparse, os
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.config)
    print("== portable V16 campaign (A1/A2/B/C/D queue) =="); print(describe(paths))
    ok = True
    for name, key, rel in ASSETS:
        root = paths.get(key, ""); p = Path(root) / rel if root else None
        hit = bool(p and p.exists()); ok &= hit
        print(f"  [{'x' if hit else ' '}] {name:24s} {p}")
    print("\nfrozen entry : python -m campaign_v16.run (from experiments/adamw-spectral-v13-v16/)")
    print("frozen config sha16: 96f9b9f2ee54cb92 (PLAN_FROZEN.md)")
    print("plan: queue-driven: 71 items history; budget container 120k attempts/400M forwards/24h device")
    if not ok:
        print("RESULT: BLOCKED_ASSET — restore per docs/ASSET_RESTORATION_GUIDE.md"); sys.exit(1)
    if not (a.execute and os.environ.get("ADAMWSPEC_ALLOW_GPU") == "1"):
        print("RESULT: DRY-RUN OK (no GPU side effect)."); sys.exit(0)
    print("RESULT: execution not implemented in this archive round — use frozen entry."); sys.exit(3)

if __name__ == "__main__":
    main()
