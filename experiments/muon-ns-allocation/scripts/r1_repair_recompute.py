"""Stage A: recompute the 176 affected extension-calibration rows with the repaired
BatchContext (full-parameter restore), verify retained initial rows by spot check,
replay the ORIGINAL selection rules -> corrected_selection_lock, and list which check
points are missing for the corrected roles (to be evaluated, <=72)."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.r1_probe import BatchContext, evaluate_batch
from src.r1_selection import lock_roles
from src.state_restore import build_q_sets, access_guard
from src.recipe_registry import P1_STATES

OUT = ROOT / "outputs" / "r0_r1_v1_1"
CORR = ROOT / "outputs" / "r0_r1_v1_1_corrected"
CORR.mkdir(parents=True, exist_ok=True)
guard = access_guard({"select"})


def pid(a, b, s):
    return f"a{a}|b{b}|{s}"


def main():
    t0 = time.time()
    q = build_q_sets()
    ext = json.loads((OUT / "extension_lock.json").read_text())
    A_fin, B_fin = ext["A_final"], ext["B_final"]
    A0, B0 = ext["A0"], ext["B0"]
    ext_pts = [{"id": pid(a, b, st), "a": a, "b": b, "structure": st}
               for a in A_fin for b in B_fin for st in ("NS5", "NS3_MATCHED")
               if not (a in A0 and b in B0)]
    spot_pts = [{"id": pid(0.75, 1.25, "NS5"), "a": 0.75, "b": 1.25, "structure": "NS5"},
                {"id": pid(0.75, 1.25, "NS3_MATCHED"), "a": 0.75, "b": 1.25, "structure": "NS3_MATCHED"}]

    old_cal = list(csv.DictReader(open(OUT / "calibration.csv")))
    rows_out, diffs = [], []
    cal_by_state = {str(s): [] for s in P1_STATES}
    for s in P1_STATES:
        payload = torch.load(ROOT / "outputs" / "p0_p1_v1_1" / "parent" / f"parent_state_step{s}.pt",
                             map_location="cpu", weights_only=False)
        agg_ext, agg_spot = {}, {}
        spot_ids = {p["id"] for p in spot_pts}
        for bi in range(4):
            ctx = BatchContext(payload, s, (s + 8 + bi, s + 9 + bi))
            rr, nchk = evaluate_batch(ctx, q, "select", ext_pts + spot_pts)
            guard("select")
            for r in rr:
                r.update({"state": s, "batch": bi + 1, "split": "select"})
                if r["point_id"] in spot_ids:
                    agg_spot.setdefault(r["point_id"], []).append(r["V"])
                else:
                    agg_ext.setdefault(r["point_id"], []).append(r["V"])
            rows_out.extend([r for r in rr if r["point_id"] not in spot_ids])
            print(f"[{time.time()-t0:.0f}s] s{s} b{bi+1}: {len(rr)} ext+spot pts, c={ctx.c:.4f}", flush=True)
            del ctx
            torch.cuda.empty_cache()
        # spot-check retained initial rows (must be bitwise identical: they ran before P)
        for pidk, vs in agg_spot.items():
            old = [float(r["V"]) for r in old_cal if r["state"] == str(s) and r["point_id"] == pidk]
            assert len(old) == 4, pidk
            for a, b in zip(old, vs):
                if a != b:
                    diffs.append({"state": s, "point": pidk, "old": a, "new": b})
        # corrected calibration aggregate: retained initial + corrected extension + native
        for r in old_cal:
            if r["state"] == str(s) and r.get("split") == "select" and "::repeat" not in r["point_id"] \
                    and r["point_id"] not in ("Z", "P"):
                cal_by_state[str(s)].append(r)
        for pidk, vs in agg_ext.items():
            r0 = next(r for r in rows_out if r["state"] == s and r["point_id"] == pidk)
            cal_by_state[str(s)].append({"point_id": pidk, "a": r0.get("a"), "b": r0.get("b"),
                                         "structure": r0.get("structure"),
                                         "V": sum(vs) / len(vs)})
    # build per-state means
    means = {str(s): [] for s in P1_STATES}
    for s in P1_STATES:
        agg = {}
        for r in cal_by_state[str(s)]:
            key = r.get("point_id")
            agg.setdefault(key, []).append(float(r["V"]) if "V" in r else r["V_mean"])
        # native point from old rows (a1.0|b1.0|NS5)
        nat = [float(r["V"]) for r in old_cal if r["state"] == str(s)
               and r["point_id"] == "a1.0|b1.0|NS5"]
        agg["a1.0|b1.0|NS5"] = nat
        for pidk, vs in agg.items():
            r0 = next(r for r in cal_by_state[str(s)] if r.get("point_id") == pidk)
            a_v = r0.get("a")
            b_v = r0.get("b")
            means[str(s)].append({"id": pidk,
                                  "a": float(a_v) if a_v not in (None, "", "None") else 1.0,
                                  "b": float(b_v) if b_v not in (None, "", "None") else 1.0,
                                  "structure": r0.get("structure") or "NS5",
                                  "V_mean": sum(vs) / len(vs)})
    old_lock = json.loads((OUT / "selection_lock.json").read_text())
    new_lock = lock_roles(means, A_fin, B_fin)
    # attach corrected V means + mark differences from the historical lock
    changed = []
    for s in ("512", "1536"):
        for role in ("G", "A", "S", "S_at_A", "A_at_S", "G_at_A"):
            old_id = old_lock["roles"][s][role]["id"] if old_lock["roles"][s].get(role) else None
            new_id = new_lock[s][role]["id"] if new_lock[s].get(role) else None
            if old_id != new_id:
                changed.append({"state": s, "role": role, "old": old_id, "new": new_id})
    json.dump({"roles": new_lock, "A_final": A_fin, "B_final": B_fin,
               "note": "CORRECTED lock: replayed original near-best/tie rules on "
                       "initial(161, retained, spot-verified) + recomputed extension(176, "
                       "repaired aux baseline); produced AFTER viewing old check results "
                       "(erratum, not a new pre-registration)",
               "role_changes_vs_historical": changed,
               "spot_check_diffs": diffs},
              open(CORR / "corrected_selection_lock.json", "w"), indent=1, default=str)
    allf = []
    for r in rows_out:
        for k in r:
            if k not in allf:
                allf.append(k)
    with open(CORR / "extension_calibration_corrected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allf, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)
    # compare corrected vs old extension rows
    cmp_rows = []
    for r in rows_out:
        old = [x for x in old_cal if x["state"] == str(r["state"]) and x["point_id"] == r["point_id"]
               and x.get("split") == "select" and x.get("a") not in (None, "", "None")]
        if old:
            ob = [float(x["V"]) for x in old if int(x["batch"]) == int(r["batch"])]
            if ob:
                cmp_rows.append({"state": r["state"], "batch": r["batch"], "point": r["point_id"],
                                 "old_V": ob[0], "new_V": r["V"], "diff": r["V"] - ob[0]})
    with open(CORR / "extension_old_vs_corrected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state", "batch", "point", "old_V", "new_V", "diff"])
        w.writeheader()
        w.writerows(cmp_rows)
    import numpy as np
    d = np.array([c["diff"] for c in cmp_rows])
    print(f"corrected ext rows: {len(rows_out)}; spot-check diffs (retained initial): {len(diffs)}; "
          f"old-vs-new ext V diff: mean={d.mean():.3e} max|d|={np.abs(d).max():.3e}")
    print("role changes vs historical:", changed)


if __name__ == "__main__":
    main()
