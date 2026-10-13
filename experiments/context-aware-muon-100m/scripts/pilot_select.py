"""Selection steps for the frozen pilot (LOCK doc sec.2 queue).

--stage B_muon : best lr among B_muon_lr* by dev H4096 NLL
                 ties within 1e-4 -> smaller lr (conservative end);
                 best at grid edge -> RANGE_UNRESOLVED flag (recorded, honest).
--stage C_A    : best lam among C_A_lam* by dev H4096 NLL
                 ties within 1e-4 -> lam 0.65 (closer to anchor);
                 edge -> RANGE_UNRESOLVED flag.

Merges into results/pilot/selection.json (never removes prior keys).
Selection consumes development only (dev-grade, allowed); sealed sets untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "pilot"
H = "4096"


def dev4096(arm_id: str) -> float:
    r = json.load(open(PILOT / arm_id / "run.json"))
    return r["dev_nll_by_horizon"][H]


def main():
    stage = sys.argv[sys.argv.index("--stage") + 1]
    sel_path = PILOT / "selection.json"
    sel = json.load(open(sel_path)) if sel_path.exists() else {}

    if stage == "B_muon":
        grid = [("B_muon_lr5e-3", 0.005), ("B_muon_lr1e-2", 0.01), ("B_muon_lr2e-2", 0.02)]
        vals = {aid: dev4096(aid) for aid, _ in grid}
        best_aid = min(vals, key=lambda a: vals[a])
        # tie within 1e-4 -> smaller lr (grid is ascending in lr; pick the smaller-lr arm)
        cands = [a for a in vals if vals[a] <= vals[best_aid] + 1e-4]
        best_aid = min(cands, key=lambda a: dict(grid)[a])
        best_lr = dict(grid)[best_aid]
        flags = []
        if best_aid in ("B_muon_lr5e-3", "B_muon_lr2e-2"):
            flags.append("RANGE_UNRESOLVED_B_MUON (best at grid edge; single extension needs new authorization)")
        sel["B_muon_best_lr"] = best_lr
        sel["B_muon_best_arm"] = best_aid
        sel["B_muon_dev4096_all"] = {a: vals[a] for a in vals}
        sel["flags"] = sel.get("flags", []) + flags
    elif stage == "C_A":
        grid = [("C_A_lam0p65", 0.65), ("C_A_lam0p85", 0.85)]
        vals = {aid: dev4096(aid) for aid, _ in grid}
        best_aid = min(vals, key=lambda a: vals[a])
        cands = [a for a in vals if vals[a] <= vals[best_aid] + 1e-4]
        best_aid = min(cands, key=lambda a: dict(grid)[a])  # tie -> 0.65
        sel["C_A_best_lam"] = dict(grid)[best_aid]
        sel["C_A_best_arm"] = best_aid
        sel["C_A_dev4096_all"] = {a: vals[a] for a in vals}
    else:
        raise SystemExit(f"unknown stage {stage}")

    sel_path.write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
