"""P0-B: AdamW effective-update spectral diagnostics from the sealed v07 bank.

Usage:
  python -m specmuon.run_p0b --config configs/p0b_audit_v1.json --phase 1
  python -m specmuon.run_p0b --config configs/p0b_audit_v1.json --phase 2
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import torch

from .io_utils import V07_DIR, load_v07_state
from .metrics import cosine_matrices, energy_shares, participation_ratio


def module_class(name: str) -> str:
    if "embed_tokens" in name:
        return "embed"
    for tag in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if tag in name:
            return tag.replace("_proj", "")
    return "other"


def layer_idx(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


def analyze_state(st: dict) -> list[dict]:
    rows = []
    for name in st["names"]:
        A = st["A"][name]
        if A.ndim != 2:
            continue
        s = torch.linalg.svdvals(A.double())
        r = min(A.shape)
        e = energy_shares(s, ks=(1, max(1, int(0.1 * r))))
        rows.append({
            "seed": st["seed"], "step": st["step"], "name": name,
            "module": module_class(name), "layer": layer_idx(name),
            "shape0": A.shape[0], "shape1": A.shape[1], "r": r,
            "E1": e[1], "E10pct": e[max(1, int(0.1 * r))],
            "PR": participation_ratio(s),
            "PR_over_r": participation_ratio(s) / r,
            "sigma1_over_fro": float(s[0] / s.norm()),
            "froA": float(s.norm()),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--phase", type=int, default=1)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    torch.set_num_threads(cfg["threads"])
    t0 = time.perf_counter()

    if args.phase == 1:
        plan = [(s, t) for s in cfg["phase1_seeds"] for t in cfg["phase1_steps"]]
    else:
        plan = [(s, t) for s in cfg["all_seeds"] for t in cfg["all_steps"]]
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    A_cache = {}
    for seed, step in plan:
        path = V07_DIR / f"seed_{seed}" / "states" / f"state_step_{step:03d}.pt"
        t_load = time.perf_counter()
        st = load_v07_state(path)
        dt = time.perf_counter() - t_load
        st_rows = analyze_state(st)
        rows.extend(st_rows)
        A_cache[(seed, step)] = {
            n: st["A"][n] for n in st["names"] if st["A"][n].ndim == 2
        }
        print(f"[p0b] seed {seed} step {step}: {len(st_rows)} 2D modules ({dt:.1f}s)", flush=True)

    with open(out_dir / f"p0b_phase{args.phase}_per_matrix.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-name stability across states
    by_name = {}
    for r in rows:
        by_name.setdefault((r["module"], r["name"]), []).append(r)
    stab_rows = []
    for (mod, name), rs in sorted(by_name.items()):
        def stats(k):
            vals = [x[k] for x in rs]
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / max(1, len(vals) - 1)
            return mu, (var ** 0.5 / mu if mu > 0 else 0.0)
        mu_e1, cv_e1 = stats("E1")
        mu_pr, cv_pr = stats("PR_over_r")
        stab_rows.append({
            "module": mod, "name": name, "n_states": len(rs),
            "E1_mean": mu_e1, "E1_cv": cv_e1,
            "PR_over_r_mean": mu_pr, "PR_over_r_cv": cv_pr,
            "E10pct_mean": stats("E10pct")[0],
        })
    with open(out_dir / f"p0b_phase{args.phase}_stability.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stab_rows[0].keys()))
        w.writeheader()
        w.writerows(stab_rows)

    # cross-state cosine of A for same name (sampled pairs to bound CPU)
    keys = sorted(A_cache.keys())
    pair_rows = []
    names_sample = sorted({n for k in keys for n in A_cache[k]})[:40]
    for (k1, k2) in itertools.combinations(keys, 2):
        for name in names_sample:
            A1, A2 = A_cache[k1].get(name), A_cache[k2].get(name)
            if A1 is None or A2 is None:
                continue
            pair_rows.append({
                "seed1": k1[0], "step1": k1[1], "seed2": k2[0], "step2": k2[1],
                "module": module_class(name), "name": name,
                "cosA": cosine_matrices(A1, A2),
            })
    with open(out_dir / f"p0b_phase{args.phase}_pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    # free memory
    A_cache.clear()

    n_pairs_expected = len(keys) * (len(keys) - 1) // 2
    report = {
        "phase": args.phase, "n_states": len(keys), "n_matrix_rows": len(rows),
        "n_pairs": n_pairs_expected, "n_pair_rows": len(pair_rows),
        "wall_seconds": time.perf_counter() - t0,
    }
    with open(out_dir / f"p0b_phase{args.phase}_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[p0b] phase {args.phase} done in {report['wall_seconds']:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
