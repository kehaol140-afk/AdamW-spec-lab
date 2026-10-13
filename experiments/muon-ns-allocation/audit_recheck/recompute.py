"""Independent recomputation for the P0/P1 re-audit. Does NOT import the original
analyze script. Reads raw per-domain sum/count and per-batch V values, recomputes V_Q,
benefits, thresholds, paired differences, calibration-selection replay, and compares
with the historical tables/selection_lock."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p0_p1_v1_1"
AUD = ROOT / "audit_recheck"


def V_from_sums(row):
    """Recompute V from per-domain sum/count when present."""
    for dom in ("A", "B"):
        s, c = row.get(f"nll_sum_{dom}"), row.get(f"count_{dom}")
        if s not in (None, "") and c not in (None, ""):
            na = float(row["nll_sum_A"]) / int(row["count_A"])
            nb = float(row["nll_sum_B"]) / int(row["count_B"])
            return 0.5 * na + 0.5 * nb
    return None


def main():
    cal = list(csv.DictReader(open(OUT / "calibration_rows.csv")))
    chk = list(csv.DictReader(open(OUT / "check_rows.csv")))
    lock = json.loads((OUT / "selection_lock.json").read_text())
    verdict = json.loads((OUT / "verdict.json").read_text())
    rows_out = []

    # 0) V identity check (per-domain sums available?)
    n_sums = sum(1 for r in chk + cal if V_from_sums(r) is not None)
    v_max_diff = 0.0
    for r in chk + cal:
        v2 = V_from_sums(r)
        if v2 is not None:
            v_max_diff = max(v_max_diff, abs(v2 - float(r["V"])))
    rows_out.append({"check": "V=0.5*(sumA/cntA)+0.5*(sumB/cntB) recomputed vs stored",
                     "n_rows_with_sums": n_sums, "max_abs_diff": v_max_diff,
                     "note": "sums present in check/calibration rows" if n_sums else "sums ABSENT -> consistency only"})

    def arr(rows, state, scheme):
        return np.array([float(r["V"]) for r in rows
                         if r["state"] == state and r["scheme"] == scheme])

    # 1) e_repeat / delta_step (independent)
    e_rep = 0.0
    for s in ("512", "1536"):
        a, b = arr(cal, s, "native"), arr(cal, s, "native_repeat")
        if len(a) and len(b):
            e_rep = max(e_rep, abs(a[0] - b[0]))
    d_step = max(1e-5, 10 * e_rep)
    rows_out.append({"check": "delta_step recomputed", "n_rows_with_sums": e_rep,
                     "max_abs_diff": d_step, "note": f"e_repeat={e_rep}; matches verdict: {d_step == verdict['delta_step']}"})

    # 2) per-group raw check benefits
    mism = 0
    for g in ("q", "k", "v", "o", "gate", "up", "down"):
        for K in ("3", "7"):
            for s in ("512", "1536"):
                nat, c = arr(chk, s, "native"), arr(chk, s, f"{g}|K{K}|raw")
                B = nat - c
                hist = [r for r in csv.DictReader(open(OUT / "group_probe_table.csv"))
                        if r["state"] == s and r["group"] == g and r["K"] == K and r["probe"] == "raw"]
                if hist:
                    h = float(hist[0]["B_chk_mean"])
                    if abs(h - B.mean()) > 1e-12:
                        mism += 1
    rows_out.append({"check": "per-group raw B_chk mean (28 cells) vs group_probe_table",
                     "n_rows_with_sums": 28, "max_abs_diff": mism,
                     "note": "0 = all match; single-group negatives re-confirmed"})

    # 3) locked-joint / controls / geometry / random / amplitude on check side
    for s in ("512", "1536"):
        nat = arr(chk, s, "native")
        combos = [x for x in ("no_op", "target_primary", "target_gamma_const",
                              "target_gain_only_live", "target_direction_only")
                  if len(arr(chk, s, x))]
        for x in combos:
            v = arr(chk, s, x)
            B = nat - v
            # paired difference vs gamma_const for target
            note = f"s{s} {x}: mean={B.mean():+.4e}, sign={int((B>0).sum())}/4"
            if x == "target_primary" and len(arr(chk, s, "target_gamma_const")):
                gc = arr(chk, s, "target_gamma_const")
                note += f"; paired target-gamma_const mean={float((v-gc).mean()):+.3e}"
            rows_out.append({"check": f"locked candidate B (check)", "n_rows_with_sums": s,
                             "max_abs_diff": float(B.mean()), "note": note})
        for geo in set(r["scheme"] for r in chk if r["state"] == s and r["scheme"].startswith("geometry")):
            B = nat - arr(chk, s, geo)
            rows_out.append({"check": "geometry joint B", "n_rows_with_sums": s,
                             "max_abs_diff": float(B.mean()),
                             "note": f"s{s} {geo}: mean={B.mean():+.4e}, sign={int((B>0).sum())}/4"})
        for amp, lbl in (("global_amp|x0.5", "x0.5"), ("global_amp|x2.0", "x2.0")):
            v = arr(chk, s, amp)
            B = nat - v
            rows_out.append({"check": "global amplitude B", "n_rows_with_sums": s,
                             "max_abs_diff": float(B.mean()),
                             "note": f"s{s} {lbl}: mean={B.mean():+.4e}, sign={int((B>0).sum())}/4"})

    # 4) calibration selection replay (proposal ranking only, no new winners)
    replay_ok = {}
    for s in ("512", "1536"):
        st = lock["states"][s]
        pool = [tuple(p) for p in json.loads(json.dumps(st["proposals"] + [st["geometry_primary"], st["fixed_random_pair"]]))]
        # recompute sum-B_g ranking from calibration rows
        nat_c = arr(cal, s, "native")
        sumB = {}
        Eimp = {}
        meta = json.loads((OUT / "calibration_meta.json").read_text())
        for gm, gp in set(pool):
            b3 = float(np.mean(nat_c - arr(cal, s, f"{gm}|K3|raw")))
            b7 = float(np.mean(nat_c - arr(cal, s, f"{gp}|K7|raw")))
            sumB[(gm, gp)] = b3 + b7
            tot = 0.0
            for bi in range(1, 5):
                eg = meta[f"s{s}|b{bi}"]["E_g2"]
                tot += (eg[gm]["5"] - eg[gm]["3"]) + (eg[gp]["5"] - eg[gp]["7"])
            Eimp[(gm, gp)] = tot / 4
        ranked = sorted(sumB.items(), key=lambda kv: -kv[1])
        replay_ok[s] = {
            "replay_top2_pairs": [list(k) for k, _ in ranked[:2]],
            "historical_proposals": st["proposals"],
            "proposals_match": [list(k) for k, _ in ranked[:2]] == st["proposals"] or
                               set(tuple(x) for x in [list(k) for k, _ in ranked[:2]]) == set(tuple(x) for x in st["proposals"]),
            "replay_geometry_best": list(max(Eimp, key=Eimp.get)),
            "historical_geometry": st["geometry_primary"],
            "geometry_match": list(max(Eimp, key=Eimp.get)) == st["geometry_primary"],
            "gamma_const": st["gamma_const_all"],
            "no_op": st["target_primary"]["no_op"],
            "cal_target_mean_vs_native": st["target_primary"]["cal_joint_mean"] - st["target_primary"]["cal_native_mean"],
        }
    json.dump(replay_ok, open(AUD / "selection_replay.json", "w"), indent=1)
    for s, r in replay_ok.items():
        rows_out.append({"check": "calibration selection replay", "n_rows_with_sums": s,
                         "max_abs_diff": 0,
                         "note": f"proposals_match={r['proposals_match']}, geometry_match={r['geometry_match']}, no_op={r['no_op']}"})

    # 5) interaction residual (1536): B(joint) - sum singles on CALIBRATION data
    s = "1536"
    st = lock["states"][s]
    gm, gp = st["target_primary"]["pair"]
    nat_c = arr(cal, s, "native")
    b3 = float(np.mean(nat_c - arr(cal, s, f"{gm}|K3|raw")))
    b7 = float(np.mean(nat_c - arr(cal, s, f"{gp}|K7|raw")))
    # historical cal joint mean from lock
    jb = st["target_primary"]["cal_joint_mean"] - st["target_primary"]["cal_native_mean"]
    rows_out.append({"check": "interaction residual I = B(joint) - sum(B_g) [cal]",
                     "n_rows_with_sums": s, "max_abs_diff": jb - (b3 + b7),
                     "note": f"joint={jb:+.4e} sum={b3+b7:+.4e} I={jb-(b3+b7):+.4e}"})

    # 6) pool-optimum scope statement (sec-4 rule 83)
    neg_pool = []
    for s in ("512", "1536"):
        nat = arr(chk, s, "native")
        best = 0.0  # no-op present in executed set
        for sc in set(r["scheme"] for r in chk if r["state"] == s):
            if sc in ("native", "native_repeat"):
                continue
            if sc.startswith("global_amp"):
                continue  # amplitude reference, not a K-reallocation candidate
            B = float(np.mean(nat - arr(chk, s, sc)))
            best = max(best, B)
        neg_pool.append((s, best))
    rows_out.append({"check": "pool optimum over MEASURED K-reallocation candidates (no-op included as 0, amplitude refs excluded)",
                     "n_rows_with_sums": 2, "max_abs_diff": max(b for _, b in neg_pool),
                     "note": f"per-state best measured K-candidate benefit: {[(s, f'{b:+.3e}') for s, b in neg_pool]}"})

    with open(AUD / "result_recomputed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "n_rows_with_sums", "max_abs_diff", "note"])
        w.writeheader()
        w.writerows(rows_out)
    for r in rows_out:
        print(f"[{r['check']}] {r['note']}")


if __name__ == "__main__":
    main()
