"""P1 analysis: results tables, cost table, determinism/norm checks, frozen decision.

Usage: PYTHONPATH=src python -m specmuon.analyze_p1
Reads results/p1/{branches/*/result.json, branches/*/train_metrics.jsonl,
init_eval_s*.json, bench_gpu.json, replay_gate.json}; writes P1_RESULTS.csv,
P1_RUNTIME_COST.csv, decision.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "p1"

ARM_ORDER = ["native_lr1.0", "native_lr0.5", "native_lr2.0", "exact_polar",
             "head_lam0.25", "bulk_lam0.25", "muon_p0.5"]
CANDIDATES = ["head_lam0.25", "bulk_lam0.25"]

TH = {"beats_native": 0.002, "beats_polar_margin": 0.001}


def load_metrics(run_id: str):
    rows = [json.loads(l) for l in
            open(OUT / "branches" / run_id / "train_metrics.jsonl")]
    return rows


def main() -> None:
    branches = sorted((OUT / "branches").iterdir())
    results, costs = [], []
    determinism_checks = {}
    norm_checks = []
    for bdir in branches:
        if not (bdir / "result.json").exists():
            continue
        r = json.load(open(bdir / "result.json"))
        run_id = r["run_id"]
        rows = load_metrics(run_id)
        # determinism: the s512 native branches were executed twice (relaunch after
        # a tool timeout); loss/preclip/lr must be bit-identical (timings differ).
        if len(rows) == 512 and run_id.startswith("s512_native"):
            num_fields = ["rel_step", "global_step", "lr", "train_loss", "preclip_norm"]
            same = all(all(a[k] == b[k] for k in num_fields)
                       for a, b in zip(rows[:256], rows[256:]))
            determinism_checks[run_id] = same
            rows = rows[:256]
        assert len(rows) == 256, f"{run_id}: {len(rows)} rows"
        op_rows = [x for x in rows if "op" in x]
        norm_err = max((x["op"]["max_rel_norm_err"] for x in op_rows), default=0.0)
        t_svd = sum(x["op"]["t_svd_s"] for x in op_rows)
        t_probe = sum(x["op"]["t_native_probe_s"] for x in op_rows)
        t_muon = sum(x["t_muon_opt_s"] for x in rows)
        t_aux = sum(x["t_aux_opt_s"] for x in rows)
        losses = [x["train_loss"] for x in rows]
        norm_checks.append({"run_id": run_id, "max_rel_norm_err": norm_err,
                            "nonfinite_loss": r["nonfinite_loss"],
                            "nonfinite_grad": r["nonfinite_grad"]})
        q = r["final_eval"]
        results.append({
            "run_id": run_id, "restore_step": r["restore_step"], "arm": r["arm"],
            "v_q_H256": q["v_q"], "nll_A": q["nll_A"], "nll_B": q["nll_B"],
            "init_v_q_H256": json.load(open(OUT / f"init_eval_s{r['restore_step']}.json"))["v_q"],
            "final_train_loss": losses[-1],
            "mean_train_loss_last32": sum(losses[-32:]) / 32,
            "wall_seconds": r["wall_seconds"],
            "gpu_peak_mib": r["gpu_peak_mib"],
            "max_rel_norm_err": norm_err,
            "nonfinite": int(r["nonfinite_loss"] or r["nonfinite_grad"] or r["resume_fail"]),
        })
        costs.append({
            "run_id": run_id, "restore_step": r["restore_step"], "arm": r["arm"],
            "wall_seconds": r["wall_seconds"],
            "mean_step_s": r["wall_seconds"] / 256,
            "mean_t_muon_opt_s": t_muon / 256,
            "mean_t_aux_opt_s": t_aux / 256,
            "mean_t_svd_bundle_s": (t_svd / 256) if op_rows else 0.0,
            "mean_t_native_probe_s": (t_probe / 256) if op_rows else 0.0,
            "svd_total_s": t_svd, "gpu_peak_mib": r["gpu_peak_mib"],
        })

    def v(step, arm):
        return next(x["v_q_H256"] for x in results
                    if x["restore_step"] == step and x["arm"] == arm)

    # frozen decision computation
    decision = {"per_point": {}, "determinism_relaunch_check": determinism_checks}
    for step in sorted({x["restore_step"] for x in results}):
        d = {}
        base = v(step, "native_lr1.0")
        best_lr = min(v(step, "native_lr0.5"), v(step, "native_lr2.0"), v(step, "native_lr1.0"))
        for arm in ARM_ORDER:
            d[arm] = {"v_q": v(step, arm), "delta_vs_native": base - v(step, arm)}
        polar_gain = base - v(step, "exact_polar")
        cand = {}
        for arm in CANDIDATES:
            delta_native = base - v(step, arm)
            delta_lr = best_lr - v(step, arm)  # >0: beats LR grid
            beyond_polar = delta_native - polar_gain
            cand[arm] = {
                "delta_vs_native": delta_native,
                "delta_vs_best_lr": delta_lr,
                "beyond_exact_polar": beyond_polar,
                "beats_native_by_2e-3": delta_native > TH["beats_native"],
                "beats_lr_grid": delta_lr > 0.0,
                "beats_polar_by_1e-3": beyond_polar > TH["beats_polar_margin"],
            }
        decision["per_point"][step] = {"arms": d, "best_lr_v_q": best_lr,
                                       "candidates": cand}

    points = sorted(decision["per_point"].keys())
    verdicts = {}
    for arm in CANDIDATES:
        c1 = decision["per_point"][points[0]]["candidates"][arm]
        c2 = decision["per_point"][points[-1]]["candidates"][arm]
        if c1["beats_native_by_2e-3"] and c2["beats_native_by_2e-3"]:
            if not (c1["beats_lr_grid"] and c2["beats_lr_grid"]):
                verdicts[arm] = "LR_DOMINATED"
            elif not (c1["beats_polar_by_1e-3"] and c2["beats_polar_by_1e-3"]):
                verdicts[arm] = "POLAR_ONLY"
            else:
                verdicts[arm] = "PROMISING"
        elif c1["beats_native_by_2e-3"] or c2["beats_native_by_2e-3"]:
            verdicts[arm] = "SINGLE_POINT"
        else:
            verdicts[arm] = "NO_EFFECT"
    decision["verdicts"] = verdicts

    with open(OUT / "P1_RESULTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(sorted(results, key=lambda x: (x["restore_step"],
                                                   ARM_ORDER.index(x["arm"]))))
    with open(OUT / "P1_RUNTIME_COST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(costs[0].keys()))
        w.writeheader()
        w.writerows(sorted(costs, key=lambda x: (x["restore_step"],
                                                 ARM_ORDER.index(x["arm"]))))
    json.dump({"decision": decision, "norm_checks": norm_checks},
              open(OUT / "decision.json", "w"), indent=1)
    print(json.dumps({"verdicts": verdicts,
                      "per_point": {str(s): {a: round(dd["v_q"], 6) for a, dd in p["arms"].items()}
                                    for s, p in decision["per_point"].items()},
                      "determinism_relaunch_check": determinism_checks}, indent=1))


if __name__ == "__main__":
    main()
