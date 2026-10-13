"""R2F analysis: evaluation lock -> Q_P2_endpoint on L/crossover endpoints and
diagnostic snapshots -> LR-WD attribution + crossover path effects tables."""
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

OUT = ROOT / "outputs" / "r2f_lrwd_retention_v1"
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
    # ---- evaluation lock BEFORE any new endpoint evaluation ----
    runs = []
    for step in (512, 1536):
        runs.append({"run_id": f"s{step}_L256", "file": f"s{step}_L256_step256_full.pt",
                     "sha": sha(OUT / f"s{step}_L256_step256_full.pt"), "role": "L_endpoint"})
        for src in ("N", "AH"):
            for dst in ("N", "AH"):
                rid = f"s{step}_{src}_to_{dst}"
                runs.append({"run_id": rid, "file": f"{rid}_step256_full.pt",
                             "sha": sha(OUT / f"{rid}_step256_full.pt"), "role": "crossover_endpoint"})
    json.dump({"runs": runs, "target": "Q_P2_endpoint (developmental, historically exposed)",
               "V_def": "0.5*(sum_A/count_A)+0.5*(sum_B/count_B)",
               "D_TRIGGER": json.loads((OUT / "D_TRIGGER.json").read_text()),
               "locked_before_evaluation": True},
              open(OUT / "evaluation_lock.json", "w"), indent=1)

    model = build_model().to("cuda")
    rows = []
    for r in runs:
        blob = torch.load(OUT / r["file"], map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        res = eval_V(model, q, "P2_endpoint")
        rows.append({"run_id": r["run_id"], "role": r["role"], "t_total": 512
                     if "to" in r["run_id"] else 256, **res})
    # diagnostic snapshots (rel steps 1/16/64) for the crossover runs
    diag = []
    for step in (512, 1536):
        for src in ("N", "AH"):
            for dst in ("N", "AH"):
                for k in (1, 16, 64):
                    f = OUT / f"s{step}_{src}_to_{dst}_step{k}_model.pt"
                    blob = torch.load(f, map_location="cpu", weights_only=False)
                    missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
                    assert set(missing) <= {"lm_head.weight"} and not unexpected
                    res = eval_V(model, q, "P2_endpoint")
                    diag.append({"run_id": f"s{step}_{src}_to_{dst}", "rel_step": k, **res})
    # L snapshots too
    for step in (512, 1536):
        for k in (1, 16, 64):
            f = OUT / f"s{step}_L256_step{k}_model.pt"
            blob = torch.load(f, map_location="cpu", weights_only=False)
            missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
            assert set(missing) <= {"lm_head.weight"} and not unexpected
            res = eval_V(model, q, "P2_endpoint")
            diag.append({"run_id": f"s{step}_L256", "rel_step": k, **res})
    allf = []
    for r in rows + diag:
        for k in r:
            if k not in allf:
                allf.append(k)
    with open(OUT / "endpoint_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allf, extrasaction="ignore")
        w.writeheader(); w.writerows(rows + diag)

    def V(rid):
        r = [x for x in rows if x["run_id"] == rid]
        return r[0]

    # ---- LR-WD attribution ----
    old = {(int(r["restore_step"]), r["arm"]): r
           for r in csv.DictReader(open(R2DIR / "endpoint_values.csv"))
           if r["t"] == "256" and r["arm"] in ("N", "AH")}
    attr = []
    for step in (512, 1536):
        n, ah, l = old[(step, "N")], old[(step, "AH")], V(f"s{step}_L256")
        B_L = float(n["V"]) - l["V"]
        B_decay = l["V"] - float(ah["V"])
        attr.append({"origin": step,
                     "V_N": float(n["V"]), "V_AH": float(ah["V"]), "V_L": l["V"],
                     "B_L=V_N-V_L": B_L, "B_decay=V_L-V_AH": B_decay,
                     "A_L_nll_A": l["nll_A"], "A_L_nll_B": l["nll_B"],
                     "N_nll_A": float(n["nll_A"]), "N_nll_B": float(n["nll_B"]),
                     "AH_nll_A": float(ah["nll_A"]), "AH_nll_B": float(ah["nll_B"])})
    with open(OUT / "lrwd_attribution.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(attr[0].keys()))
        w.writeheader(); w.writerows(attr)

    # ---- crossover path effects ----
    cross = []
    for step in (512, 1536):
        J = {f"{p},{q_}": V(f"s{step}_{p}_to_{q_}") for p in ("N", "AH") for q_ in ("N", "AH")}
        R_N = J["N,N"]["V"] - J["AH,N"]["V"]
        R_AH = J["N,AH"]["V"] - J["AH,AH"]["V"]
        T_N = J["N,N"]["V"] - J["N,AH"]["V"]
        T_AH = J["AH,N"]["V"] - J["AH,AH"]["V"]
        I_path = J["AH,AH"]["V"] - J["AH,N"]["V"] - J["N,AH"]["V"] + J["N,N"]["V"]
        B0 = float(old[(step, "N")]["V"]) - float(old[(step, "AH")]["V"])
        cross.append({"origin": step,
                      "J_NN": J["N,N"]["V"], "J_NAH": J["N,AH"]["V"],
                      "J_AHN": J["AH,N"]["V"], "J_AHAH": J["AH,AH"]["V"],
                      "R_N=J(N,N)-J(AH,N)": R_N, "R_AH=J(N,AH)-J(AH,AH)": R_AH,
                      "T_N=J(N,N)-J(N,AH)": T_N, "T_AH=J(AH,N)-J(AH,AH)": T_AH,
                      "I_path": I_path, "B0=V_N(256)-V_AH(256)": B0,
                      "R_N_over_B0": R_N / B0 if B0 != 0 else None,
                      "R_AH_over_B0": R_AH / B0 if B0 != 0 else None,
                      "dom_A": {k: J[k]["nll_A"] for k in J},
                      "dom_B": {k: J[k]["nll_B"] for k in J}})
    with open(OUT / "cross_path_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader(); w.writerows(cross)
    json.dump({"wall_seconds": time.time() - t0}, open(OUT / "analysis_meta.json", "w"), indent=1)
    for a in attr:
        print(f"s{a['origin']}: V_N={a['V_N']:.6f} V_L={a['V_L']:.6f} V_AH={a['V_AH']:.6f} "
              f"| B_L={a['B_L=V_N-V_L']:+.4e} B_decay={a['B_decay=V_L-V_AH']:+.4e}")
    for c in cross:
        print(f"s{c['origin']}: J(N,N)={c['J_NN']:.6f} J(N,AH)={c['J_NAH']:.6f} "
              f"J(AH,N)={c['J_AHN']:.6f} J(AH,AH)={c['J_AHAH']:.6f}")
        print(f"        R_N={c['R_N=J(N,N)-J(AH,N)']:+.4e} R_AH={c['R_AH=J(N,AH)-J(AH,AH)']:+.4e} "
              f"T_N={c['T_N=J(N,N)-J(N,AH)']:+.4e} T_AH={c['T_AH=J(AH,N)-J(AH,AH)']:+.4e} "
              f"I_path={c['I_path']:+.4e} B0={c['B0=V_N(256)-V_AH(256)']:+.4e}")


if __name__ == "__main__":
    main()
