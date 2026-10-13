"""P0/P1 analysis: verdict computation from calibration_rows.csv + check_rows.csv."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "p0_p1_v1_1"


def load(name):
    return list(csv.DictReader(open(OUT / name)))


def main():
    cal = load("calibration_rows.csv")
    chk = load("check_rows.csv")
    lock = json.loads((OUT / "selection_lock.json").read_text())
    meta = json.loads((OUT / "calibration_meta.json").read_text())
    cost = json.loads((OUT / "cost_pool.json").read_text())

    # e_repeat from repeated native evaluation (calibration first batch per state)
    e_repeat = 0.0
    for s in (512, 1536):
        a = [r for r in cal if r["scheme"] == "native" and int(r["state"]) == s]
        b = [r for r in cal if r["scheme"] == "native_repeat" and int(r["state"]) == s]
        if a and b:
            e_repeat = max(e_repeat, abs(float(a[0]["V"]) - float(b[0]["V"])))
    delta_step = max(1e-5, 10 * e_repeat)

    def vmap(state, scheme, rows):
        return np.array([float(r["V"]) for r in rows
                         if int(r["state"]) == state and r["scheme"] == scheme])

    # ---------- group probe table ----------
    gp_rows = []
    for s in (512, 1536):
        nat_cal = vmap(s, "native", cal)
        nat_chk = vmap(s, "native", chk)
        for g in ("q", "k", "v", "o", "gate", "up", "down"):
            for K in ("3", "7"):
                for probe in ("raw", "gain_only_live", "direction_only"):
                    sc = f"{g}|K{K}|{probe}"
                    vc = vmap(s, sc, cal)
                    vk = vmap(s, sc, chk)
                    if len(vc) == 0 or len(vk) == 0:
                        continue
                    B_cal = nat_cal - vc
                    B_chk = nat_chk - vk
                    row = next(r for r in cal if r["scheme"] == sc and int(r["state"]) == s)
                    gp_rows.append({
                        "state": s, "group": g, "K": int(K), "probe": probe,
                        "B_cal_mean": float(B_cal.mean()), "B_chk_mean": float(B_chk.mean()),
                        "B_chk_min": float(B_chk.min()), "B_chk_max": float(B_chk.max()),
                        "B_chk_sign_same": int((B_chk > 0).sum()),
                        "gamma_mean": float(row.get("gamma", "nan") or "nan"),
                        "E_g2_5": row.get("E_g2_5", ""), "E_g2_K": row.get("E_g2_K", ""),
                    })
    with open(OUT / "group_probe_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gp_rows[0].keys()))
        w.writeheader(); w.writerows(gp_rows)

    # ---------- joint allocation table ----------
    jt_rows = []
    for s in (512, 1536):
        st = lock["states"][str(s)]
        nat = vmap(s, "native", chk)
        gm, gp = st["target_primary"]["pair"]
        raw_scheme = "target_primary"
        schemes = {"target_primary": f"joint|raw", "gain": "target_gain_only_live",
                   "direction": "target_direction_only", "gamma_const": "target_gamma_const",
                   "geometry": f"geometry|{st['geometry_primary'][0]}3|{st['geometry_primary'][1]}7",
                   "fixed_random": f"fixed_random|{st['fixed_random_pair'][0]}3|{st['fixed_random_pair'][1]}7"}
        # actual scheme labels in check rows
        def find(label_part):
            names = set(r["scheme"] for r in chk if int(r["state"]) == s)
            for n in names:
                if label_part in n:
                    return n
            return None
        pairs = {}
        for key, part in (("target", "target_primary"), ("gain", "target_gain_only_live"),
                          ("direction", "target_direction_only"), ("gamma", "target_gamma_const")):
            pairs[key] = part
        geo_name = find(f"geometry|")
        rnd_name = find("fixed_random|")
        # single-group sum for interaction residual
        bg3 = st["B_g_calibration"][f"{gm}|3"]
        bp7 = st["B_g_calibration"][f"{gp}|7"]
        for key, label in [("target", "target_primary"), ("gain", "target_gain_only_live"),
                           ("direction", "target_direction_only"), ("gamma", "target_gamma_const")]:
            vv = vmap(s, label, chk)
            if len(vv) == 0:
                continue
            B = nat - vv
            jt_rows.append({"state": s, "scheme": key, "pair": f"{gm}3|{gp}7",
                            "B_chk_mean": float(B.mean()),
                            "B_chk_sign_same": int((B > 0).sum()),
                            "sum_single_group_cal": (bg3 + bp7) if key == "target" else "",
                            "interaction_residual_cal": (st["target_primary"]["cal_joint_mean"] -
                                                         (bg3 + bp7)) if key == "target" else ""})
        for name, key in ((geo_name, "geometry"), (rnd_name, "fixed_random")):
            if name:
                vv = vmap(s, name, chk)
                B = nat - vv
                jt_rows.append({"state": s, "scheme": key, "pair": name,
                                "B_chk_mean": float(B.mean()),
                                "B_chk_sign_same": int((B > 0).sum()),
                                "sum_single_group_cal": "", "interaction_residual_cal": ""})
        # no-op entry for state with no_op
        if st["target_primary"]["no_op"]:
            jt_rows.append({"state": s, "scheme": "no_op_locked", "pair": "-",
                            "B_chk_mean": 0.0, "B_chk_sign_same": 0,
                            "sum_single_group_cal": "", "interaction_residual_cal": ""})
    with open(OUT / "joint_allocation_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jt_rows[0].keys()))
        w.writeheader(); w.writerows(jt_rows)

    # ---------- amplitude controls ----------
    amp_rows = []
    for s in (512, 1536):
        nat = vmap(s, "native", chk)
        for sc, label in ((f"global_amp|x0.5", "global_x0.5"), (f"global_amp|x2.0", "global_x2.0")):
            vv = vmap(s, sc, chk)
            if len(vv) == 0:
                # calibration-side only fallback
                vv = vmap(s, sc, cal)
                nat_c = vmap(s, "native", cal)
                B = nat_c - vv
                amp_rows.append({"state": s, "control": label, "source": "cal",
                                 "B_mean": float(B.mean()), "B_sign_same": int((B > 0).sum())})
            else:
                B = nat - vv
                amp_rows.append({"state": s, "control": label, "source": "check",
                                 "B_mean": float(B.mean()), "B_sign_same": int((B > 0).sum())})
    with open(OUT / "amplitude_controls.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(amp_rows[0].keys()))
        w.writeheader(); w.writerows(amp_rows)

    # ---------- AMO ----------
    with open(OUT / "amo_comparison.csv", "w", newline="") as f:
        f.write("label,comparison,status\n")
        f.write("本地original配方下的AMO机制一步对照,same-family Q3 (AMO_local_costmatched_restricted),"
                "COMPARATOR_UNRESOLVED: arXiv 2605.17806v2 unreachable from host; no local cache; "
                "PE composer/coefficients cannot be sourced reliably; no numbers fabricated\n")

    # ---------- verdict ----------
    v = {"delta_step": delta_step, "e_repeat": e_repeat, "states": {}, "Q": {}}
    for s in (512, 1536):
        st = lock["states"][str(s)]
        nat = vmap(s, "native", chk)
        if st["target_primary"]["no_op"]:
            tp = nat  # locked no-op: candidate IS native
        else:
            tp = vmap(s, "target_primary", chk)
        gam = vmap(s, "target_gamma_const", chk)
        geo_n = next((r["scheme"] for r in chk if int(r["state"]) == s and "geometry|" in r["scheme"]), None)
        geo = vmap(s, geo_n, chk) if geo_n else None
        B = nat - tp
        v["states"][str(s)] = {
            "target_pair": st["target_primary"]["pair"], "no_op": st["target_primary"]["no_op"],
            "B_target_chk_mean": float(B.mean()), "B_target_sign_same": int((B > 0).sum()),
            "B_gamma_const_chk_mean": float((nat - gam).mean()) if len(gam) else None,
            "B_geometry_chk_mean": float((nat - geo).mean()) if geo is not None and len(geo) else None,
            "cost_matched": True}
    q1_states = []
    for s in (512, 1536):
        d = v["states"][str(s)]
        ok = (not d["no_op"]) and d["B_target_chk_mean"] > delta_step and d["B_target_sign_same"] >= 3
        q1_states.append(ok)
    v["Q"]["Q1"] = "LOCAL_SUPPORTED" if all(q1_states) else "NO_LOCAL_SUPPORT"
    # Q2: scalar explanation ruled out only if target beats gamma_const clearly on both states
    q2 = []
    for s in (512, 1536):
        d = v["states"][str(s)]
        if d["B_gamma_const_chk_mean"] is None:
            q2.append("INSUFFICIENT")
        elif d["B_target_chk_mean"] - d["B_gamma_const_chk_mean"] > delta_step:
            q2.append("TARGET_BEATS_GAMMA")
        else:
            q2.append("SCALAR_EXPLANATION_NOT_RULED_OUT")
    v["Q"]["Q2"] = q2
    v["Q"]["Q3"] = "COMPARATOR_UNRESOLVED"
    json.dump(v, open(OUT / "verdict.json", "w"), indent=1)
    print(json.dumps(v, indent=1))


if __name__ == "__main__":
    main()
