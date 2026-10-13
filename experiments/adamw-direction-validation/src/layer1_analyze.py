#!/usr/bin/env python3
"""Layer-1 analysis: primary/secondary comparisons, replication gate, update-norm audit,
reversal rule, layer1_gate.json. Uses ONLY frozen statistics (doc_bootstrap 1000 reps,
seed 9071) and frozen thresholds from configs/frozen_v1.json."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "raw"
V16 = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")
sys.path.insert(0, str(HERE))
from src.dv_core import token_weighted, paired_bootstrap  # noqa: E402

ARMS = ["dv_S1", "dv_S8", "dv_F8", "dv_P8", "dv_S4"]
V16_REF = {"dv_S1": "b_S1_plain_lr0.001", "dv_F8": "b_F1_a8_lr0.001", "dv_P8": "b_P1_a8_lr0.001"}
STEP_KEYS = ["eta", "loss_sum", "token_count", "nll", "family_used", "A_norm", "D_norm", "q_t",
             "zero_norm_fallback"]
HORIZONS = [512, 1024, 2048]


def steps(arm):
    return [json.loads(l) for l in (RAW / "steps" / f"{arm}.jsonl").open(encoding="utf-8")]


def eval_rows(arm, h, split="development"):
    p = RAW / "eval" / f"{arm}__H{h}__{split}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8")]


def replication_gate():
    out = {}
    for arm, ref in V16_REF.items():
        dv, v16 = steps(arm), [json.loads(l) for l in (V16 / "campaign_v16" / "raw" / "steps" / f"{ref}.jsonl").open(encoding="utf-8")]
        n = min(len(dv), len(v16))
        mism = []
        for i in range(n):
            for k in STEP_KEYS:
                if dv[i][k] != v16[i][k]:
                    mism.append((i + 1, k, v16[i][k], dv[i][k]))
                    break
        out[arm] = {"ref": ref, "steps_compared": n, "mismatches": len(mism),
                    "first_mismatch": mism[0] if mism else None,
                    "bitwise": (len(mism) == 0 and n == len(v16) == len(dv))}
    return out


def nll_table():
    rows = []
    for arm in ARMS:
        row = {"arm": arm}
        for h in HORIZONS:
            row[f"dev_H{h}_nll"] = token_weighted(eval_rows(arm, h))
        rows.append(row)
    return rows


def main_comparisons(tbl):
    idx = {r["arm"]: r for r in tbl}
    out = {}
    for name, (x, y) in {"F8-S8": ("dv_F8", "dv_S8"), "P8-S8": ("dv_P8", "dv_S8"),
                         "F8-P8": ("dv_F8", "dv_P8"), "S8-S1": ("dv_S8", "dv_S1"),
                         "S4-S1": ("dv_S4", "dv_S1")}.items():
        per_h = {}
        for h in HORIZONS:
            b = paired_bootstrap(eval_rows(x, h), eval_rows(y, h), seed=9071, reps=1000)
            per_h[h] = b
        out[name] = per_h
    return out


def gate_eval(comp):
    f = comp["F8-S8"][2048]; p = comp["P8-S8"][2048]
    f_pass = (f["point"] <= -1e-3) and (f["ci95"][1] < 0)
    p_pass = (p["point"] <= -1e-3) and (p["ci95"][1] < 0)
    reversal = (abs(comp["F8-S8"][2048]["point"]) < 1e-3) and (abs(comp["P8-S8"][2048]["point"]) < 1e-3)
    return {"F8_beats_S8": f_pass, "P8_beats_S8": p_pass,
            "pass": bool(f_pass or p_pass),
            "reversal_DIRECTION_AND_SCALE_CONFOUNDED": bool(reversal),
            "thresholds": {"point": "-1e-3", "ci95_upper": "0 (entirely below 0)"},
            "frozen_rule": "layer1 gate: at least one primary comparison passes (frozen_v1.json)"}


def norm_audit():
    out = {}
    st = {a: steps(a) for a in ARMS}
    for a in ARMS:
        q = np.array([r["q_t"] for r in st[a]])
        An = np.array([r["A_norm"] for r in st[a]])
        out[a] = {
            "q_first": float(q[0]), "q_mean": float(q.mean()), "q_last100_mean": float(q[-100:].mean()),
            "q_min": float(q.min()), "q_max": float(q.max()),
            "A_first": float(An[0]), "A_mean": float(An.mean()), "A_last100_mean": float(An[-100:].mean()),
            "nonfinite_q": int((~np.isfinite(q)).sum()),
            "zero_norm_fallback_steps": int(sum(1 for r in st[a] if r.get("zero_norm_fallback"))),
            "nonfinite_nll_steps": int(sum(1 for r in st[a] if not math.isfinite(r["nll"]))),
            "opt_step_seconds_total": float(sum(r["opt_step_seconds"] for r in st[a])),
        }
    q1 = np.array([r["q_t"] for r in st["dv_S1"]]); A1 = np.array([r["A_norm"] for r in st["dv_S1"]])
    for a, k in (("dv_S8", 8), ("dv_S4", 4)):
        q = np.array([r["q_t"] for r in st[a]])
        ratio = q[: len(q1)] / q1[: len(q)]
        A = np.array([r["A_norm"] for r in st[a]])
        Ar = A[: len(A1)] / A1[: len(A)]
        out[f"{a}_vs_S1"] = {
            "first_step_q_ratio": float(ratio[0]), "first_step_exact": bool(
                st[a][0]["A_norm"] == st["dv_S1"][0]["A_norm"]),  # same init+first grad => identical A_t
            "q_ratio_step1": float(ratio[0]), "q_ratio_step2": float(ratio[1]) if len(ratio) > 1 else None,
            "q_ratio_step10": float(ratio[9]) if len(ratio) > 9 else None,
            "q_ratio_step100": float(ratio[99]) if len(ratio) > 99 else None,
            "q_ratio_step512": float(ratio[511]) if len(ratio) > 511 else None,
            "q_ratio_step2048": float(ratio[-1]),
            "A_ratio_step2048": float(Ar[-1]),
            "note": "ratio drifts from k as branches diverge (endogenous A_t trajectories)",
        }
    fp = out["dv_F8"]; sp8 = out["dv_S8"]; pp = out["dv_P8"]
    out["fp_s8_p8_norm_semantics"] = {
        "rule": "per-step target-block displacement norm q_t = eta*a*||A_t^{branch}||; F/P rescaled to ||A_t|| exactly by construction (dv_core.n_a fp64)",
        "F8_vs_S8_q_mean_ratio": fp["q_mean"] / sp8["q_mean"],
        "P8_vs_S8_q_mean_ratio": pp["q_mean"] / sp8["q_mean"],
        "F8_vs_S8_q_last100_ratio": fp["q_last100_mean"] / sp8["q_last100_mean"],
        "P8_vs_S8_q_last100_ratio": pp["q_last100_mean"] / sp8["q_last100_mean"],
    }
    return out


def main():
    rep = replication_gate()
    tbl = nll_table()
    comp = main_comparisons(tbl)
    gate = gate_eval(comp)
    audit = norm_audit()
    meta = json.load(open(RAW / "run_meta_layer1.json", encoding="utf-8")) if (RAW / "run_meta_layer1.json").exists() else []
    payload = {"replication_gate": rep, "nll_table": tbl, "comparisons": comp,
               "gate": gate, "norm_audit": audit, "run_meta": meta}
    (RAW / "layer1_analysis.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    json.dump(gate, open(HERE / "layer1_gate.json", "w"), indent=1)

    with open(HERE / "LAYER1_RESULTS.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "dev_H512_nll", "dev_H1024_nll", "dev_H2048_nll", "q_t_mean", "q_t_last100_mean", "wall_seconds"])
        walls = {m["id"]: m.get("wall_seconds") for m in meta}
        for r in tbl:
            a = r["arm"]
            w.writerow([a, f"{r['dev_H512_nll']:.6f}", f"{r['dev_H1024_nll']:.6f}", f"{r['dev_H2048_nll']:.6f}",
                        f"{audit[a]['q_mean']:.6f}", f"{audit[a]['q_last100_mean']:.6f}", walls.get(a, "")])
        w.writerow([])
        w.writerow(["comparison", "horizon", "delta_nll", "ci95_lo", "ci95_hi", "ci_entirely_below_0"])
        for name, per in comp.items():
            for h in HORIZONS:
                b = per[h]
                w.writerow([name, h, f"{b['point']:.6f}", f"{b['ci95'][0]:.6f}", f"{b['ci95'][1]:.6f}",
                            str(b["ci95"][1] < 0)])
    print(json.dumps({"gate": gate, "rep": {k: v["bitwise"] for k, v in rep.items()},
                      "nll2048": {r["arm"]: round(r["dev_H2048_nll"], 4) for r in tbl}}, indent=1))


if __name__ == "__main__":
    main()
