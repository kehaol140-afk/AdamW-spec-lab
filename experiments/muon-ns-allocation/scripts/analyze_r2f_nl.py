"""N->L endpoint evaluation + Delta_tail / G_NL + existing-curve summary (read-only)."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.state_restore import build_q_sets, eval_V
from muon_transition.model import build_model
from scripts.run_r2f import CORR

OUT = ROOT / "outputs" / "r2f_nl_tail_v1"


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    t0 = time.time()
    q = build_q_sets()
    runs = [f"s{s}_N_to_L" for s in (512, 1536)]
    json.dump({"runs": [{"run_id": r, "sha": sha(OUT / f"{r}_step256_full.pt")} for r in runs],
               "target": "Q_P2_endpoint (developmental, historically exposed)",
               "reused": ["s512_N_to_N", "s512_N_to_AH", "s1536_N_to_N", "s1536_N_to_AH"],
               "locked_before_evaluation": True},
              open(OUT / "evaluation_lock_nl.json", "w"), indent=1)
    model = build_model().to("cuda")
    rows = []
    for r in runs:
        blob = torch.load(OUT / f"{r}_step256_full.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        rows.append({"run_id": r, **eval_V(model, q, "P2_endpoint")})
    with open(OUT / "endpoint_values_nl.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    corr = {r["run_id"]: r for r in csv.DictReader(open(CORR / "endpoint_values_corrected.csv"))}
    res = {}
    for s in (512, 1536):
        J_NN = float(corr[f"s{s}_N_to_N"]["V"])
        J_NAH = float(corr[f"s{s}_N_to_AH"]["V"])
        J_NL = next(r["V"] for r in rows if r["run_id"] == f"s{s}_N_to_L")
        domA = {k: float(corr[f"s{s}_N_to_{k}"]["nll_A"]) for k in ("N", "AH")}
        domA["L"] = next(r["nll_A"] for r in rows if r["run_id"] == f"s{s}_N_to_L")
        domB = {k: float(corr[f"s{s}_N_to_{k}"]["nll_B"]) for k in ("N", "AH")}
        domB["L"] = next(r["nll_B"] for r in rows if r["run_id"] == f"s{s}_N_to_L")
        res[s] = {"J_NN": J_NN, "J_NAH": J_NAH, "J_NL": J_NL,
                  "Delta_tail=J(N,L)-J(N,AH)": J_NL - J_NAH,
                  "G_NL=J(N,N)-J(N,L)": J_NN - J_NL,
                  "dom_A": domA, "dom_B": domB}
    json.dump(res, open(OUT / "nl_effects.json", "w"), indent=1)
    for s, r in res.items():
        print(f"s{s}: J(N,N)={r['J_NN']:.6f} J(N,AH)={r['J_NAH']:.6f} J(N,L)={r['J_NL']:.6f}")
        print(f"   Delta_tail={r['Delta_tail=J(N,L)-J(N,AH)']:+.6e} (pos=AH better) "
              f"G_NL={r['G_NL=J(N,N)-J(N,L)']:+.6e} (pos=L better) | tau=1e-3")
        print(f"   domA N={r['dom_A']['N']:.6f} AH={r['dom_A']['AH']:.6f} L={r['dom_A']['L']:.6f}")
        print(f"   domB N={r['dom_B']['N']:.6f} AH={r['dom_A']['H'] if False else r['dom_B']['AH']:.6f} L={r['dom_B']['L']:.6f}")

    # ---- existing-curve summary (read-only from prior corrected artifacts) ----
    print("\n=== curve summary (Q_P2_endpoint diagnostic snapshots, from R2F tables) ===")
    r2f_rows = list(csv.DictReader(open(ROOT / "outputs" / "r2f_lrwd_retention_v1" / "endpoint_values.csv")))
    snap = {}
    for r in r2f_rows:
        if r.get("rel_step") in ("1", "16", "64") and "to" in r["run_id"]:
            snap[(r["run_id"], int(r["rel_step"]))] = float(r["V"])
    for s in (512, 1536):
        for pair in (("N_to_N", "AH_to_N"), ("N_to_AH", "AH_to_AH")):
            for k in (1, 16, 64):
                a = snap.get((f"s{s}_{pair[0]}", k))
                b = snap.get((f"s{s}_{pair[1]}", k))
                if a is not None and b is not None:
                    print(f"s{s} rel{k}: J*({pair[0]})={a:.6f} J*({pair[1]})={b:.6f} diff={a-b:+.4e}")
    json.dump({"wall_seconds": time.time() - t0}, open(OUT / "analysis_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
