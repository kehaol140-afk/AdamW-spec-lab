"""Frozen-pilot analysis (Phase 4 adjudication).

Inputs: results/pilot/<arm>/{run.json, metrics.jsonl, eval_docs_H4096.jsonl},
        results/pilot/selection.json.
Outputs: results/pilot/analysis.json, PILOT_100M_RESULTS.csv.

Contrasts (task primary list):
  1. C_A(best lam) vs B_muon(best lr)
  2. C_A(best lam) vs best Muon anchor (any convention incl. match_rms)
  3. C_A(best lam) vs low-mu best
  4. C_B vs C_A(best lam)
  5. same-token endpoint + same-wall-clock competitiveness table
Gates (frozen): >=1e-3 dev H4096 gain, paired per-doc bootstrap CI95 (1000 reps,
seed 9071) entirely below 0, low-mu must NOT reproduce same-magnitude gain,
RANGE_UNRESOLVED honesty flags, dev-grade caveat (no confirmatory claim).
Also: metrics.jsonl integrity audit per arm (raced file disclosure).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "pilot"
H = "4096"
BOOT_REPS, BOOT_SEED = 1000, 9071
GAIN_THRESHOLD = 1e-3

ARMS = [
    "B_adamw_lr3e-4", "B_adamw_lr1e-3", "B_adamw_lr3e-3",
    "B_muon_lr5e-3", "B_muon_lr1e-2", "B_muon_lr2e-2",
    "B_muon_rms_lr1e-3", "B_muon_rms_lr3e-3", "B_muon_rms_lr1e-2",
    "B_lowmu_0p9", "B_lowmu_0p8",
    "C_A_lam0p65", "C_A_lam0p85", "C_A_nat", "C_B",
]


def run(arm):
    return json.load(open(PILOT / arm / "run.json"))


def dev(arm, h=H):
    return run(arm)["dev_nll_by_horizon"][str(h) if str(h) in run(arm)["dev_nll_by_horizon"] else h]


def docs(arm):
    return [json.loads(l) for l in (PILOT / arm / "eval_docs_H4096.jsonl").open()]


def paired_bootstrap(rows_a, rows_b, reps=BOOT_REPS, seed=BOOT_SEED):
    """H0: token-weighted mean per-token NLL difference (a-b).
    Paired docs: diff_i = loss_sum_a_i - loss_sum_b_i; weight w_i = token_count
    (same docs, same token counts). Estimate = sum(diff_i)/sum(w_i);
    bootstrap resamples doc indices jointly (diff and weight stay paired)."""
    da = {r["doc_id"]: (r["loss_sum"], r["token_count"]) for r in rows_a}
    db = {r["doc_id"]: (r["loss_sum"], r["token_count"]) for r in rows_b}
    ids = sorted(set(da) & set(db))
    diffs = [da[i][0] - db[i][0] for i in ids]
    wts = [da[i][1] for i in ids]
    mean = sum(diffs) / sum(wts)
    rng = random.Random(seed)
    n = len(ids)
    stats = []
    for _ in range(reps):
        num = 0.0
        den = 0
        for _ in range(n):
            j = rng.randrange(n)
            num += diffs[j]
            den += wts[j]
        stats.append(num / den)
    stats.sort()
    return mean, stats[int(0.025 * reps)], stats[int(0.975 * reps)]


def metrics_audit(arm):
    """Per-line JSON validity + step monotonicity; returns (n_lines, n_bad, n_dup)."""
    seen, bad, prev = 0, 0, -1
    last_step = -1
    n_lines = 0
    for line in (PILOT / arm / "metrics.jsonl").open():
        n_lines += 1
        try:
            d = json.loads(line)
            st = d["step"]
            if st <= last_step:
                bad += 1
            last_step = st
        except Exception:
            bad += 1
    return n_lines, bad


def main():
    sel = json.load(open(PILOT / "selection.json"))
    R = {a: run(a) for a in ARMS if (PILOT / a / "run.json").exists()}
    assert len(R) == 15, f"missing arms: {set(ARMS) - set(R)}"

    best_muon = min(["B_muon_lr5e-3", "B_muon_lr1e-2", "B_muon_lr2e-2"], key=lambda a: dev(a))
    anchor_cands = [a for a in ["B_muon_lr5e-3", "B_muon_lr1e-2", "B_muon_lr2e-2",
                                "B_muon_rms_lr1e-3", "B_muon_rms_lr3e-3", "B_muon_rms_lr1e-2"]]
    best_anchor = min(anchor_cands, key=lambda a: dev(a))
    lowmu_best = min(["B_lowmu_0p9", "B_lowmu_0p8"], key=lambda a: dev(a))
    c_a_best = "C_A_lam0p65" if abs(sel["C_A_best_lam"] - 0.65) < 1e-9 else "C_A_lam0p85"
    c_b = "C_B"

    contrasts = {}
    for name, a, b in [
        ("C_A_vs_B_muon", c_a_best, best_muon),
        ("C_A_vs_best_anchor", c_a_best, best_anchor),
        ("C_A_vs_lowmu_best", c_a_best, lowmu_best),
        ("C_B_vs_C_A", c_b, c_a_best),
        ("lowmu_best_vs_best_anchor", lowmu_best, best_anchor),
        ("C_A_065_vs_best_anchor", "C_A_lam0p65", best_anchor),
        ("C_A_085_vs_best_anchor", "C_A_lam0p85", best_anchor),
        ("C_A_nat_vs_best_anchor", "C_A_nat", best_anchor),
    ]:
        m, lo, hi = paired_bootstrap(docs(a), docs(b))
        contrasts[name] = {"a": a, "b": b, "mean_diff": m, "ci95": [lo, hi],
                           "ci_entirely_negative": hi < 0,
                           "gain_ge_1e-3": m <= -GAIN_THRESHOLD}

    # momentum-explanation check (frozen wording): candidate gain vs anchor >=1e-3
    # AND gain vs lowmu best <1e-3 -> MOMENTUM_EXPLAINED
    gain_anchor = -contrasts["C_A_vs_best_anchor"]["mean_diff"]
    gain_lowmu = -contrasts["C_A_vs_lowmu_best"]["mean_diff"]
    gate_pass = (contrasts["C_A_vs_best_anchor"]["ci_entirely_negative"]
                 and contrasts["C_A_vs_best_anchor"]["gain_ge_1e-3"]
                 and not (gain_lowmu < GAIN_THRESHOLD))
    cb_better = contrasts["C_B_vs_C_A"]["mean_diff"] < 0 and contrasts["C_B_vs_C_A"]["ci_entirely_negative"]

    # tokens-to-threshold (coarse horizons; threshold = best anchor endpoint)
    thr = dev(best_anchor)
    ttt = {}
    for a in [c_a_best, c_b, lowmu_best, best_muon, best_anchor]:
        cross = None
        for h in (1024, 2048, 4096):
            if dev(a, h) <= thr:
                cross = h
                break
        ttt[a] = {"threshold": thr, "first_horizon_at_or_below": cross,
                  "tokens_at_that_horizon": cross * 8192 if cross else None}

    # integrity audit
    audit = {a: metrics_audit(a) for a in ARMS}

    # honest boundary flags for EVERY family grid (frozen boundary rule)
    range_flags = []
    if best_muon == "B_muon_lr5e-3" or best_muon == "B_muon_lr2e-2":
        range_flags.append("RANGE_UNRESOLVED_B_MUON (original best at grid edge)")
    best_rms = min(["B_muon_rms_lr1e-3", "B_muon_rms_lr3e-3", "B_muon_rms_lr1e-2"], key=lambda a: dev(a))
    if best_rms in ("B_muon_rms_lr1e-3", "B_muon_rms_lr1e-2"):
        range_flags.append(f"RANGE_UNRESOLVED_B_MUON_RMS ({best_rms} at grid edge)")
    best_adamw = min(["B_adamw_lr3e-4", "B_adamw_lr1e-3", "B_adamw_lr3e-3"], key=lambda a: dev(a))
    if best_adamw in ("B_adamw_lr3e-4", "B_adamw_lr3e-3"):
        range_flags.append(f"RANGE_UNRESOLVED_ADAMW ({best_adamw} at grid edge)")

    out = {
        "selection": sel,
        "best_muon_original": best_muon,
        "best_anchor": best_anchor,
        "lowmu_best": lowmu_best,
        "c_a_best": c_a_best,
        "contrasts": contrasts,
        "gate": {
            "C_A_pass_vs_best_anchor": gate_pass,
            "momentum_explained": (gain_anchor >= GAIN_THRESHOLD and gain_lowmu < GAIN_THRESHOLD),
            "C_B_beats_C_A": cb_better,
            "range_flags": range_flags,
        },
        "tokens_to_threshold": ttt,
        "metrics_integrity": audit,
        "caveat": "dev-grade selection set; CIs are exploratory (same-dev selection), NOT confirmatory inference; confirm layer = decisive",
    }
    (PILOT / "analysis.json").write_text(json.dumps(out, indent=1))

    # CSV
    with (ROOT / "PILOT_100M_RESULTS.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "peak_lr", "lam", "dev_nll_H1024", "dev_nll_H2048", "dev_nll_H4096",
                    "wall_s", "tok_per_s", "vram_gib", "init_fp", "final_fp",
                    "tokens_consumed", "metrics_lines", "metrics_bad_lines"])
        for a in ARMS:
            r = R[a]
            arm_cfg = json.load(open(ROOT / "configs" / "pilot_arms_v1.json"))["arms"][a]
            n, bad = audit[a]
            w.writerow([a, r["peak_lr"], arm_cfg.get("lam", ""),
                        dev(a, 1024), dev(a, 2048), dev(a, 4096),
                        round(r["wall_s"], 1), round(r["throughput_tok_s"], 1),
                        round(r["vram_peak_gib"], 2), r["init_fingerprint"], r["final_fingerprint"],
                        r["tokens_consumed"], n, bad])
    print(json.dumps(out["gate"], indent=1))
    print("analysis + CSV written")


if __name__ == "__main__":
    main()
