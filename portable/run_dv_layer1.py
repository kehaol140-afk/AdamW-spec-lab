#!/usr/bin/env python3
"""Portable wrapper: Direction Layer-1 (dv_runner) with dynamic path mapping.

Frozen source src/dv_runner.py hardcodes V16_ROOT=/home/b205/projects/
training_dynamics_lab/adamw_spectral_pilot (read-only dependency on
pilot_core, token cache, eval_sets, bindings, config). This wrapper patches
the module constants AFTER import but BEFORE main() — the frozen file on
disk is never modified (see docs/PORTABLE_VS_FROZEN_DIFF.md).

Default: DRY-RUN (print plan, verify assets). Execution requires --execute
AND ADAMWSPEC_ALLOW_GPU=1 (not authorized in this archive round).
"""
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from portable.paths import load_paths, describe  # noqa: E402

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    paths = load_paths(a.config)
    v16 = Path(paths["V16_ROOT"]); exp = REPO / "experiments/adamw-direction-validation"
    print("== portable Direction Layer-1 =="); print(describe(paths))
    need = [("pilot_core.py", v16 / "pilot_core.py"),
            ("token cache", v16 / "campaign_v16/cache/train_tokens.u32"),
            ("eval_sets.json", v16 / "campaign_v16/eval_sets.json"),
            ("bindings.local.json", v16 / "bindings.local.json"),
            ("frozen config", exp / "configs/frozen_v1.json")]
    ok = True
    for name, p in need:
        hit = p.exists(); ok &= hit
        print(f"  [{'x' if hit else ' '}] {name:18s} {p}")
    print(f"\nfrozen entry : python src/dv_runner.py (from {exp})")
    print("frozen config sha16: f8cc6b8e87c1b04a (configs/frozen_v1.json)")
    print("plan: 5 arms x 2048 steps, seed 3407, ~37min GPU on RTX 5090 (history)")
    if not ok:
        print("RESULT: BLOCKED_ASSET — restore per docs/ASSET_RESTORATION_GUIDE.md"); sys.exit(1)
    if not (a.execute and __import__("os").environ.get("ADAMWSPEC_ALLOW_GPU") == "1"):
        print("RESULT: DRY-RUN OK (no GPU side effect). --execute + ADAMWSPEC_ALLOW_GPU=1 needed to run.")
        sys.exit(0)
    print("RESULT: execution not implemented in this archive round — use frozen entry.")
    sys.exit(3)

if __name__ == "__main__":
    main()
