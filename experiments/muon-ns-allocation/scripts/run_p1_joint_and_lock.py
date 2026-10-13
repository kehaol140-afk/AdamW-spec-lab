"""P1b joint selection on calibration data only; writes selection_lock.json.

Pool = measured same-budget pairs. Proposals by calibration mean of sum(B_g); joint
forward measured on the 4 calibration batches (Q_select); geometry primary by
calibration E_g^2 improvement (no Q); fixed random pair by pre-registered seed
20260929 drawn BEFORE any joint objective is read. No Q_P1_check access.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.p1_engine import evaluate_batch
from src.state_restore import build_q_sets, access_guard
from src.recipe_registry import GROUPS, P1_STATES

OUT = ROOT / "outputs" / "p0_p1_v1_1"
FIXED_RANDOM_SEED = 20260929  # pre-registered before reading any joint objective
guard = access_guard({"select"})


def main():
    t0 = time.time()
    q = build_q_sets()
    rows = list(csv.DictReader(open(OUT / "calibration_rows.csv")))
    meta = json.loads(open(OUT / "calibration_meta.json").read())
    cost = json.loads(open(OUT / "cost_pool.json").read())
    pool = cost["measured_pool"]
    pool_pairs = [tuple(r["pair"]) for r in pool]
    if not pool_pairs:
        json.dump({"status": "NO_EQUAL_COST_PAIR"}, open(OUT / "selection_lock.json", "w"), indent=1)
        raise SystemExit("NO_EQUAL_COST_PAIR")

    def cal_mean(state, scheme):
        vals = [float(r["V"]) for r in rows
                if int(r["state"]) == state and r["scheme"] == scheme and r["batch"] != "native"]
        vals = [float(r["V"]) for r in rows if int(r["state"]) == state and r["scheme"] == scheme]
        return sum(vals) / max(len(vals), 1)

    lock = {"fixed_random_seed": FIXED_RANDOM_SEED, "pool": pool_pairs, "states": {}}
    joint_schemes_all = []
    for s in P1_STATES:
        # per-group raw benefits (calibration means)
        B = {}
        for g in GROUPS:
            for K in (3, 7):
                B[(g, K)] = cal_mean(s, "native") - cal_mean(s, f"{g}|K{K}|raw")
        # proposals by sum of single-group benefits within pool
        scored = []
        for (gm, gp) in pool_pairs:
            scored.append({"pair": [gm, gp], "sumB": B[(gm, 3)] + B[(gp, 7)]})
        scored.sort(key=lambda r: -r["sumB"])
        proposals = scored[:2]
        # geometry primary: calibration E_g^2 improvement sum (mean over 4 batches)
        Eimp = {}
        for gm, gp in pool_pairs:
            tot = 0.0
            for bi in range(1, 5):
                eg = meta[f"s{s}|b{bi}"]["E_g2"]
                tot += (eg[gm]["5"] - eg[gm]["3"]) + (eg[gp]["5"] - eg[gp]["7"])
            Eimp[(gm, gp)] = tot / 4
        geom_pair = max(pool_pairs, key=lambda p: Eimp[p])
        # fixed random pair (pre-registered seed, uniform over pool, before any joint read)
        rng = random.Random(FIXED_RANDOM_SEED + s)
        fixed_pair = rng.choice(sorted(pool_pairs))
        # gamma_const per changed group: median calibration gamma
        gamma_const = {}
        for g, K in (("q", 3), ("k", 3), ("v", 3), ("o", 3), ("gate", 3), ("up", 3), ("down", 3),
                     ("q", 7), ("k", 7), ("v", 7), ("o", 7), ("gate", 7), ("up", 7), ("down", 7)):
            vals = [meta[f"s{s}|b{bi}"]["gamma"][f"{g}|{K}"] for bi in range(1, 5)]
            gamma_const[f"{g}|{K}"] = sorted(vals)[len(vals) // 2]

        # joint schemes to measure on calibration batches
        schemes = []
        for i, pr in enumerate(proposals):
            gm, gp = pr["pair"]
            schemes.append({"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "raw",
                            "label": f"proposal{i+1}|{gm}3|{gp}7"})
        gm, gp = geom_pair
        schemes.append({"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "raw",
                        "label": f"geometry|{gm}3|{gp}7"})
        gm, gp = fixed_pair
        schemes.append({"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "raw",
                        "label": f"fixed_random|{gm}3|{gp}7"})
        # controls attach to the TOP proposal (target candidate) — added after its
        # identity is known from the measured joint means below; first pass measures raw
        joint_schemes_all.append((s, schemes, proposals, geom_pair, fixed_pair, gamma_const,
                                  {f"{g}|{K}": B[(g, K)] for g in GROUPS for K in (3, 7)},
                                  {str(k): v for k, v in Eimp.items()}))

    # measure the raw joint schemes on the 4 calibration batches per state
    results = {}
    for s, schemes, proposals, geom_pair, fixed_pair, gamma_const, Bdict, Edict in joint_schemes_all:
        sp = OUT / "parent" / f"parent_state_step{s}.pt"
        per_scheme = {}
        for bi in range(4):
            rr, _ = evaluate_batch(sp, s, (s + bi, s + bi + 1), q, "select",
                                   schemes + [{"kind": "native"}])
            guard("select")
            for r in rr:
                per_scheme.setdefault(r["scheme"], []).append(r["V"])
        results[s] = {"per_scheme": per_scheme, "proposals": proposals,
                      "geom_pair": list(geom_pair), "fixed_pair": list(fixed_pair),
                      "gamma_const": gamma_const, "B_g": Bdict, "E_imp": Edict}
        print(f"state {s} joint (cal):", {k: round(sum(v) / len(v), 6) for k, v in per_scheme.items()},
              flush=True)

    # lock: target_primary = better proposal by measured joint mean (tie -> native-first rule)
    lock["states"] = {}
    final_schemes = {}
    for s in P1_STATES:
        r = results[s]
        p1 = [k for k in r["per_scheme"] if k.startswith("proposal1|")][0]
        p2 = [k for k in r["per_scheme"] if k.startswith("proposal2|")][0]
        m1 = sum(r["per_scheme"][p1]) / 4
        m2 = sum(r["per_scheme"][p2]) / 4
        nat = sum(r["per_scheme"]["native"]) / 4
        # benefit of joint vs native (V lower = better)
        best_label = p1 if (m1, p1) <= (m2, p2) else p2  # strict tuple tie -> deterministic
        best_mean = min(m1, m2)
        no_op = not (best_mean < nat - 1e-9)  # not strictly better -> lock no-op
        gm, gp = [x for x in best_label.split("|")[1:3]]
        gm = gm[:-1]
        gp = gp[:-1]
        target_scheme = {"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "raw",
                         "label": "target_primary"}
        controls = [
            {"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "gain_only_live",
             "label": "target_gain_only_live"},
            {"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "direction_only",
             "label": "target_direction_only"},
            {"kind": "joint", "K_vector": {gm: 3, gp: 7}, "probe": "gamma_const",
             "gamma_const": {gm: r["gamma_const"][f"{gm}|3"], gp: r["gamma_const"][f"{gp}|7"]},
             "label": "target_gamma_const"},
        ]
        final_schemes[s] = ([{"kind": "native"}, target_scheme] + controls) if not no_op \
            else [{"kind": "native"}, {"kind": "native", "label": "no_op"}]
        # always evaluate geometry_primary and fixed_random pairs on the check side too
        gpair = r["geom_pair"]
        rpair = r["fixed_pair"]
        final_schemes[s] += [
            {"kind": "joint", "K_vector": {gpair[0]: 3, gpair[1]: 7}, "probe": "raw",
             "label": f"geometry|{gpair[0]}3|{gpair[1]}7"},
            {"kind": "joint", "K_vector": {rpair[0]: 3, rpair[1]: 7}, "probe": "raw",
             "label": f"fixed_random|{rpair[0]}3|{rpair[1]}7"},
        ]
        lock["states"][str(s)] = {
            "target_primary": {"pair": [gm, gp], "cal_joint_mean": best_mean,
                               "cal_native_mean": nat, "no_op": no_op},
            "geometry_primary": r["geom_pair"], "fixed_random_pair": r["fixed_pair"],
            "proposals": [p["pair"] for p in r["proposals"]],
            "gamma_const_all": r["gamma_const"], "B_g_calibration": r["B_g"],
            "E_improvement_calibration": r["E_imp"],
            "cal_joint_means": {k: sum(v) / len(v) for k, v in r["per_scheme"].items()},
        }
    lock["locked_at_stage"] = "after calibration joint measurement; before ANY Q_P1_check"
    json.dump(lock, open(OUT / "selection_lock.json", "w"), indent=1, default=str)
    json.dump({str(k): v for k, v in final_schemes.items()},
              open(OUT / "final_schemes.json", "w"), indent=1, default=str)
    print(f"selection locked in {time.time()-t0:.0f}s ->", 
          {s: (d["target_primary"]["pair"], d["target_primary"]["no_op"]) for s, d in lock["states"].items()})


if __name__ == "__main__":
    main()
