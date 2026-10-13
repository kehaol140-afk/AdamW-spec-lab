"""R1 main: initial calibration -> one-shot extension -> fill-in calibration ->
selection_lock -> check-side evaluation. Single process; contexts cached."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.r1_probe import BatchContext, evaluate_batch, O_NAMES
from src.r1_selection import compute_extension, lock_roles, select_family
from src.state_restore import build_q_sets, access_guard
from src.recipe_registry import P1_STATES
import src.recipe_registry as rr

OUT = ROOT / "outputs" / "r0_r1_v1_1"
A0, B0 = [0.5, 0.75, 1.0], [0.75, 1.0, 1.25]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pid(a, b, s):
    return f"a{a}|b{b}|{s}"


def main():
    t0 = time.time()
    guard_cal = access_guard({"select"})
    guard_chk = access_guard({"P1_check"})
    q = build_q_sets()
    payloads, ctxs = {}, {}
    cal_rows, norm_rows, meta_rows = [], [], []
    for s in P1_STATES:
        payloads[s] = torch.load(ROOT / "outputs" / "p0_p1_v1_1" / "parent" / f"parent_state_step{s}.pt",
                                 map_location="cpu", weights_only=False)
    def get_ctx(s, bi):  # bi in 0..3 -> calibration batches [s+8+bi, s+12); check [s+12+bi, s+16)
        return ctxs.setdefault((s, bi), None)

    def run_points(s, bi, side, points, tag):
        key = (s, bi, side)
        if key not in ctxs:
            ctxs[key] = BatchContext(payloads[s], s, (s + 8 + bi + (0 if side == "select" else 4),
                                                       s + 9 + bi + (0 if side == "select" else 4)))
        ctx = ctxs[key]
        rows, nchk = evaluate_batch(ctx, q, side, points,
                                    repeat_native=(side == "select" and bi == 0 and s == 512))
        for r in rows:
            r.update({"state": s, "batch": bi + 1, "split": side})
        for nc in nchk:
            nc.update({"state": s, "batch": bi + 1, "c": ctx.c,
                       "nu5": ctx.nu5, "nu3": ctx.nu3})
        return rows, nchk, ctx

    # ---- initial calibration grid ----
    pts0 = ([{"id": "a1.0|b1.0|NS5", "kind": "native"}]
            + [{"id": pid(a, b, st), "a": a, "b": b, "structure": st}
               for a in A0 for b in B0 for st in ("NS5", "NS3_MATCHED")
               if not (a == 1.0 and b == 1.0 and st == "NS5")]
            + [{"id": "Z", "kind": "Z"}, {"id": "P", "kind": "P"}])
    cal_by_state = {str(s): [] for s in P1_STATES}
    for s in P1_STATES:
        agg = {}
        for bi in range(4):
            rows, nchk, ctx = run_points(s, bi, "select", pts0, "init")
            guard_cal("select")
            cal_rows.extend(rows)
            norm_rows.extend(nchk)
            meta_rows.append({"state": s, "batch": bi + 1, "lr": ctx.lr, "c": ctx.c,
                              "nu5": ctx.nu5, "nu3": ctx.nu3, "loss_mean": ctx.loss_mean,
                              "preclip": ctx.preclip, "targets": ctx.n_targets})
            for r in rows:
                if "a" in r and "::repeat" not in r["point_id"]:
                    agg.setdefault(r["point_id"], []).append(r["V"])
            print(f"[{time.time()-t0:.0f}s] cal s{s} b{bi+1}: {len(rows)} pts, c={ctx.c:.4f}, "
                  f"native V={rows[0]['V']:.6f}", flush=True)
        for pidk, vs in agg.items():
            r0 = next(r for r in cal_rows if r["state"] == s and r["point_id"] == pidk)
            cal_by_state[str(s)].append({"id": pidk, "a": r0.get("a"), "b": r0.get("b"),
                                         "structure": r0.get("structure"),
                                         "V_mean": sum(vs) / len(vs)})
        cal_by_state[str(s)].append({"id": "a1.0|b1.0|NS5", "a": 1.0, "b": 1.0,
                                     "structure": "NS5",
                                     "V_mean": sum(v for k, v in
                                                   [(r["point_id"], r["V"]) for r in cal_rows
                                                    if r["state"] == s and r["point_id"] == "a1.0|b1.0|NS5"]) / 4})
    e_repeat = 0.0
    nat_rows = [r for r in cal_rows if "::repeat" in r["point_id"]]
    for r in nat_rows:
        orig = next(x for x in cal_rows if x["state"] == r["state"] and x["point_id"] == r["point_id"].replace("::repeat", ""))
        e_repeat = max(e_repeat, abs(orig["V"] - r["V"]))

    # ---- extension ----
    A_fin, B_fin, trig, union = compute_extension(cal_by_state, A0, B0)
    json.dump({"A0": A0, "B0": B0, "triggers_per_state_family": trig,
               "touch_union": union, "A_final": A_fin, "B_final": B_fin,
               "computed_from": "initial calibration only, before any check-side access"},
              open(OUT / "extension_lock.json", "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] extension: union={union} A_final={A_fin} B_final={B_fin}", flush=True)

    # ---- fill-in calibration for extended grid ----
    new_pts = [{"id": pid(a, b, st), "a": a, "b": b, "structure": st}
               for a in A_fin for b in B_fin for st in ("NS5", "NS3_MATCHED")
               if not (a in A0 and b in B0)]
    if new_pts:
        for s in P1_STATES:
            agg = {}
            for bi in range(4):
                rows, nchk, ctx = run_points(s, bi, "select", new_pts, "ext")
                guard_cal("select")
                cal_rows.extend(rows)
                norm_rows.extend(nchk)
                for r in rows:
                    if "a" in r:
                        agg.setdefault(r["point_id"], []).append(r["V"])
                print(f"[{time.time()-t0:.0f}s] cal-ext s{s} b{bi+1}: {len(rows)} pts", flush=True)
            for pidk, vs in agg.items():
                r0 = next(r for r in cal_rows if r["state"] == s and r["point_id"] == pidk)
                cal_by_state[str(s)].append({"id": pidk, "a": r0.get("a"), "b": r0.get("b"),
                                             "structure": r0.get("structure"),
                                             "V_mean": sum(vs) / len(vs)})

    # ---- selection lock ----
    lock = lock_roles(cal_by_state, A_fin, B_fin)
    code_sha = {f: sha256_file(ROOT / f) for f in
                ("src/r1_probe.py", "src/r1_selection.py", "scripts/run_r1.py")}
    json.dump({"roles": lock, "A_final": A_fin, "B_final": B_fin,
               "e_repeat": e_repeat, "delta_practical_step": 1e-5,
               "numeric_resolution_limit": 1e-6, "tie_tol": 1e-6,
               "code_sha256": code_sha,
               "note": "locked after ALL calibration (incl. fill-in), before Q_P1_check"},
              open(OUT / "selection_lock.json", "w"), indent=1, default=str)
    print(f"[{time.time()-t0:.0f}s] selection locked: "
          f"{ {s: {k: v['id'] if isinstance(v, dict) else v for k, v in d.items()} for s, d in lock.items()} }", flush=True)

    # ---- check side ----
    chk_rows = []
    for s in P1_STATES:
        roles = lock[str(s)]
        seen, pts = set(), []
        for role in ("N", "G", "A", "S_at_A", "S", "A_at_S", "G_at_A", "Z", "P"):
            r = roles.get(role)
            if r is None:
                continue
            key = r["id"]
            if key in seen:
                continue
            seen.add(key)
            if r.get("kind") in ("Z", "P"):
                pts.append({"id": r["id"], "kind": r["kind"]})
            elif role == "N":
                pts.append({"id": "a1.0|b1.0|NS5", "kind": "native"})
            else:
                pts.append({"id": r["id"], "a": r["a"], "b": r["b"], "structure": r["structure"]})
        for bi in range(4):
            rows, nchk, ctx = run_points(s, bi, "P1_check", pts, "chk")
            guard_chk("P1_check")
            for r in rows:
                r["role_point"] = r["point_id"]
            rows = [dict(r, role=next((role for role in ("N","G","A","S_at_A","S","A_at_S","G_at_A","Z","P")
                                        if roles.get(role, {}).get("id") == r["point_id"]), "shared")) for r in rows]
            chk_rows.extend(rows)
            norm_rows.extend(nchk)
            print(f"[{time.time()-t0:.0f}s] chk s{s} b{bi+1}: {len(rows)} pts", flush=True)

    allf = []
    for r in cal_rows + chk_rows:
        for k in r:
            if k not in allf:
                allf.append(k)
    with open(OUT / "calibration.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allf, extrasaction="ignore")
        w.writeheader(); w.writerows(cal_rows)
    with open(OUT / "check_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allf + ["role"], extrasaction="ignore")
        w.writeheader(); w.writerows(chk_rows)
    with open(OUT / "norm_match_checks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["state", "batch", "point_id", "a", "b",
                                          "realized_rel_diff", "c", "nu5", "nu3"])
        w.writeheader(); w.writerows(norm_rows)
    with open(OUT / "batch_meta.json", "w") as f:
        json.dump(meta_rows, f, indent=1, default=str)
    json.dump({"e_repeat": e_repeat, "n_cal_rows": len(cal_rows), "n_chk_rows": len(chk_rows),
               "wall_seconds": time.time() - t0}, open(OUT / "run_meta.json", "w"), indent=1)
    print(f"R1 done in {time.time()-t0:.0f}s: cal={len(cal_rows)} chk={len(chk_rows)} "
          f"e_repeat={e_repeat}")


if __name__ == "__main__":
    main()
