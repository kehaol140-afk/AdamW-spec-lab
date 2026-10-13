"""Corrected R2F crossover analysis: evaluate the four corrected AH-source branches
on Q_P2_endpoint (new evaluation lock), reuse the unaffected N-source branches, and
recompute the 2x2 tables + old-vs-corrected deltas."""
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
from scripts.run_r2f import CORR, OUT as R2F

R2DIR = ROOT / "outputs" / "r2_amplitude_continuation_v1"


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
    corr_runs = [f"s{s}_AH_to_{d}_corrected" for s in (512, 1536) for d in ("N", "AH")]
    lock = {"runs": [{"run_id": r, "file": f"{r}_step256_full.pt",
                      "sha": sha(CORR / f"{r}_step256_full.pt"),
                      "role": "corrected_AH_source_crossover"} for r in corr_runs],
            "reused_unchanged": ["s512_N_to_N", "s512_N_to_AH", "s1536_N_to_N", "s1536_N_to_AH"],
            "target": "Q_P2_endpoint (developmental, historically exposed)",
            "locked_before_evaluation": True}
    json.dump(lock, open(CORR / "evaluation_lock_corrected.json", "w"), indent=1)

    model = build_model().to("cuda")
    rows = []
    for r in corr_runs:
        blob = torch.load(CORR / f"{r}_step256_full.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        res = eval_V(model, q, "P2_endpoint")
        rows.append({"run_id": r, **res})
    # reuse unaffected N-source endpoints from the original R2F table (identity holds:
    # same target/pool/precision; those branches never touched the bug)
    old_r2f = {r["run_id"]: r for r in csv.DictReader(open(R2F / "endpoint_values.csv"))
               if r.get("role") == "crossover_endpoint"}
    allf = ["run_id", "nll_sum_A", "count_A", "nll_A", "nll_sum_B", "count_B", "nll_B", "V"]
    with open(CORR / "endpoint_values_corrected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allf, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows + [{"run_id": k, **{kk: v[kk] for kk in
                                              ("nll_sum_A", "count_A", "nll_A", "nll_sum_B", "count_B", "nll_B", "V")}}
                            for k, v in old_r2f.items()])

    def V(rid):
        for r in rows:
            if r["run_id"] == rid:
                return r
        if rid in old_r2f:
            return old_r2f[rid]
        raise KeyError(rid)

    # old WRONG values for delta reporting
    wrong = {r["run_id"]: r for r in csv.DictReader(open(R2F / "endpoint_values.csv"))
             if r.get("role") == "crossover_endpoint" and "AH_to" in r["run_id"]}
    deltas = []
    row_by_id = {row["run_id"]: row for row in rows}
    for rid in corr_runs:
        w_rid = rid.replace("_corrected", "")
        rr = row_by_id[rid]
        d = {"run_id": rid, "old_wrong_V": float(wrong[w_rid]["V"]), "corrected_V": rr["V"],
             "delta=wrong-corrected": float(wrong[w_rid]["V"]) - rr["V"],
             "old_wrong_nll_A": float(wrong[w_rid]["nll_A"]), "corrected_nll_A": rr["nll_A"],
             "old_wrong_nll_B": float(wrong[w_rid]["nll_B"]), "corrected_nll_B": rr["nll_B"]}
        deltas.append(d)
    with open(CORR / "wrong_vs_corrected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0].keys()))
        w.writeheader(); w.writerows(deltas)

    cross = []
    for s in (512, 1536):
        J = {"N,N": V(f"s{s}_N_to_N"), "N,AH": V(f"s{s}_N_to_AH"),
             "AH,N": V(f"s{s}_AH_to_N_corrected"), "AH,AH": V(f"s{s}_AH_to_AH_corrected")}
        R_N = float(J["N,N"]["V"]) - float(J["AH,N"]["V"])
        R_AH = float(J["N,AH"]["V"]) - float(J["AH,AH"]["V"])
        T_N = float(J["N,N"]["V"]) - float(J["N,AH"]["V"])
        T_AH = float(J["AH,N"]["V"]) - float(J["AH,AH"]["V"])
        I_path = float(J["AH,AH"]["V"]) - float(J["AH,N"]["V"]) - float(J["N,AH"]["V"]) + float(J["N,N"]["V"])
        cross.append({"origin": s,
                      "J_NN": float(J["N,N"]["V"]), "J_NAH": float(J["N,AH"]["V"]),
                      "J_AHN": float(J["AH,N"]["V"]), "J_AHAH": float(J["AH,AH"]["V"]),
                      "R_N": R_N, "R_AH": R_AH, "T_N": T_N, "T_AH": T_AH, "I_path": I_path,
                      "dom_A": {k: float(J[k]["nll_A"]) for k in J},
                      "dom_B": {k: float(J[k]["nll_B"]) for k in J}})
    with open(CORR / "cross_path_effects_corrected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader(); w.writerows(cross)
    json.dump({"wall_seconds": time.time() - t0}, open(CORR / "analysis_meta.json", "w"), indent=1)
    for d in deltas:
        print(f"{d['run_id']}: wrong={d['old_wrong_V']:.6f} corrected={d['corrected_V']:.6f} "
              f"delta={d['delta=wrong-corrected']:+.4e}")
    for c in cross:
        print(f"s{c['origin']}: J(N,N)={c['J_NN']:.6f} J(N,AH)={c['J_NAH']:.6f} "
              f"J(AH,N)={c['J_AHN']:.6f} J(AH,AH)={c['J_AHAH']:.6f}")
        print(f"   R_N={c['R_N']:+.4e} R_AH={c['R_AH']:+.4e} T_N={c['T_N']:+.4e} "
              f"T_AH={c['T_AH']:+.4e} I_path={c['I_path']:+.4e}")


if __name__ == "__main__":
    main()
