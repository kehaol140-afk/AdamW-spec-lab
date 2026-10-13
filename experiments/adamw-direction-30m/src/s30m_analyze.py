#!/usr/bin/env python3
"""T1 analysis: tables, gates, norm audit, CRN check, layer contrast, Muon reference.
Selection state comes from raw/stage_state.json (written by the runner at stage boundaries)."""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE / "src"))
from dv_core import token_weighted, paired_bootstrap  # noqa: E402

cfg = json.load(open(HERE / "configs" / "frozen_30m_t1.json", encoding="utf-8"))
state = json.load(open(RAW / "stage_state.json", encoding="utf-8"))
lr, a, wf = state["lr_star"], state["a_star"], state["winner_family"]
ARMS = [f"s30_S1_lr{l:g}_L1" for l in cfg["stages"]["lr_grid"]] + \
       [f"s30_S{a_}_lr{lr:g}_L1" for a_ in cfg["stages"]["a_grid"]] + \
       [f"s30_F{a}_lr{lr:g}_L1", f"s30_P{a}_lr{lr:g}_L1",
        f"s30_S{a}_lr{lr:g}_L2", f"s30_{wf}{a}_lr{lr:g}_L2"] + \
       [f"s30_M_lr{l:g}" for l in cfg["stages"]["muon_lr_grid"]]


def steps(arm):
    return [json.loads(l) for l in (RAW / "steps" / f"{arm}.jsonl").open(encoding="utf-8")]


def ev(arm, h):
    return [json.loads(l) for l in (RAW / "eval" / f"{arm}__H{h}__development.jsonl").open(encoding="utf-8")]


def main():
    out = {"lr_star": lr, "a_star": a, "winner_family": wf, "state": state}
    rows = []
    for arm in ARMS:
        if not (RAW / "eval" / f"{arm}__H2048__development.jsonl").exists():
            continue
        st = steps(arm)
        q = np.array([r["q_t"] for r in st if r["q_t"] is not None])
        row = {"arm": arm}
        for h in (512, 1024, 2048):
            row[f"dev_H{h}"] = token_weighted(ev(arm, h))
        row.update({
            "q_mean": float(q.mean()) if len(q) else None,
            "q_last100": float(q[-100:].mean()) if len(q) else None,
            "wall_s": st[-1]["wall_cum"],
            "nonfinite": sum(1 for r in st if not math.isfinite(r["nll"])),
            "fallback": sum(1 for r in st if r.get("zero_norm_fallback")),
            "train_nll_last": st[-1]["nll"],
            "first_step_loss_sum": st[0]["loss_sum"],
            "opt_step_s_total": sum(r["opt_step_seconds"] for r in st),
        })
        rows.append(row)
    out["table"] = rows
    idx = {r["arm"]: r for r in rows}

    def arm_of(pat):
        for r in rows:
            if pat in r["arm"]:
                return r["arm"]
        return None

    S1, Sa = arm_of(f"S1_lr{lr:g}_L1"), arm_of(f"S{a}_lr{lr:g}_L1")
    Fa, Pa = arm_of(f"F{a}_lr{lr:g}_L1"), arm_of(f"P{a}_lr{lr:g}_L1")
    SaL2, WfL2 = arm_of(f"S{a}_lr{lr:g}_L2"), arm_of(f"{wf}{a}_lr{lr:g}_L2")
    M_arms = sorted(r["arm"] for r in rows if "_M_lr" in r["arm"])
    comps = {}
    for name, (x, y) in {"F-S": (Fa, Sa), "P-S": (Pa, Sa), "F-P": (Fa, Pa),
                         "S-S1": (Sa, S1), "layer_S": (SaL2, Sa),
                         "layer_winner": (WfL2, arm_of(f"{wf}{a}_lr{lr:g}_L1")),
                         "winnerL2-S1": (WfL2, S1)}.items():
        if x and y:
            per = {}
            for h in (512, 1024, 2048):
                per[str(h)] = paired_bootstrap(ev(x, h), ev(y, h), seed=9071, reps=1000)
            comps[name] = {"x": x, "y": y, "per_horizon": per}  # keys are str(h)
    out["comparisons"] = comps
    f2048 = comps["F-S"]["per_horizon"]["2048"]; p2048 = comps["P-S"]["per_horizon"]["2048"]
    gate = {
        "F_beats_S": bool(f2048["point"] <= -1e-3 and f2048["ci95"][1] < 0),
        "P_beats_S": bool(p2048["point"] <= -1e-3 and p2048["ci95"][1] < 0),
        "pass": bool((f2048["point"] <= -1e-3 and f2048["ci95"][1] < 0) or
                     (p2048["point"] <= -1e-3 and p2048["ci95"][1] < 0)),
    }
    out["gate"] = gate
    out["muon"] = {m: idx[m]["dev_H2048"] for m in M_arms}
    best_m = min(M_arms, key=lambda m: idx[m]["dev_H2048"]) if M_arms else None
    out["muon_best"] = {"arm": best_m,
                        "boundary": bool(best_m and best_m.endswith("lr0.02"))}
    crn = len({r["first_step_loss_sum"] for r in rows}) == 1 if rows else None
    out["crn_first_step_identical"] = crn
    # norm audit
    audit = {}
    for r in rows:
        arm = r["arm"]
        audit[arm] = {"q_mean": r["q_mean"], "q_last100": r["q_last100"]}
    if Sa and S1:
        qSa = np.array([x["q_t"] for x in steps(Sa)]); qS1 = np.array([x["q_t"] for x in steps(S1)])
        n = min(len(qSa), len(qS1))
        audit["Sa_vs_S1"] = {"a": a, "first_ratio": float(qSa[0] / qS1[0]),
                             "mid_ratio_step1024": float(qSa[1023] / qS1[1023]),
                             "last_ratio": float(qSa[n - 1] / qS1[n - 1])}
    if Fa and Sa:
        qF = np.array([x["q_t"] for x in steps(Fa)]); qS = np.array([x["q_t"] for x in steps(Sa)])
        qP = np.array([x["q_t"] for x in steps(Pa)])
        audit["F_P_vs_S_matched"] = {"F_q_mean_ratio": float(qF.mean() / qS.mean()),
                                     "P_q_mean_ratio": float(qP.mean() / qS.mean()),
                                     "F_last100_ratio": float(qF[-100:].mean() / qS[-100:].mean()),
                                     "P_last100_ratio": float(qP[-100:].mean() / qS[-100:].mean())}
    out["norm_audit"] = audit
    (RAW / "t1_analysis.json").write_text(json.dumps(out, indent=1))

    with open(HERE / "30M_RESULTS.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "dev_H512", "dev_H1024", "dev_H2048", "q_t_mean", "q_t_last100",
                    "train_nll_last", "wall_s", "opt_step_s", "nonfinite", "fallback"])
        for r in rows:
            w.writerow([r["arm"], f"{r['dev_H512']:.6f}", f"{r['dev_H1024']:.6f}",
                        f"{r['dev_H2048']:.6f}",
                        *([""] if r["q_mean"] is None else [f"{r['q_mean']:.5f}", f"{r['q_last100']:.5f}"]),
                        f"{r['train_nll_last']:.5f}", f"{r['wall_s']:.0f}",
                        f"{r['opt_step_s_total']:.1f}", r["nonfinite"], r["fallback"]])
        w.writerow([])
        w.writerow(["selections", "lr_star", lr, "a_star", a, "winner_family", wf])
        w.writerow(["comparison", "x", "y", "delta_H2048", "ci95_lo", "ci95_hi", "below0"])
        for name, c in comps.items():
            b = c["per_horizon"]["2048"]
            w.writerow([name, c["x"], c["y"], f"{b['point']:+.6f}", f"{b['ci95'][0]:+.6f}",
                        f"{b['ci95'][1]:+.6f}", b["ci95"][1] < 0])
        w.writerow(["gate", json.dumps(gate)])
        w.writerow(["muon_best", best_m, "boundary_unresolved", out["muon_best"]["boundary"]])
        w.writerow(["crn_first_step_identical", crn])
    print(json.dumps({"gate": gate, "lr": lr, "a": a, "wf": wf, "muon": out["muon"],
                      "crn": crn, "nll2048": {r["arm"]: round(r["dev_H2048"], 4) for r in rows}},
                     indent=1))


if __name__ == "__main__":
    main()
