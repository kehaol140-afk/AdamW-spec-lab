"""R2 analysis: endpoint_lock -> Q_P2_endpoint evaluation (step0/1/256 per arm,
both restore points + repeat endpoint) -> paired effects."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.state_restore import build_q_sets, eval_V, access_guard
from src.r2_data_scale import ARMS

OUT = ROOT / "outputs" / "r2_amplitude_continuation_v1"


def main():
    t0 = time.time()
    q = build_q_sets()
    guard = access_guard({"P2_endpoint"})  # first-ever access to P2_endpoint
    # endpoint lock BEFORE any endpoint evaluation
    import hashlib
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    lock = {"runs": [], "target": "Q_P2_endpoint (P0P1_Q_v1 split; first access this round; "
                                  "old dev-pool historical exposure applies)",
            "V_def": "0.5*(sum_A/count_A)+0.5*(sum_B/count_B)",
            "locked_before_first_evaluation": True}
    for step in (512, 1536):
        for arm in ARMS:
            lock["runs"].append({"run_id": f"s{step}_{arm}", "restore_step": step, "arm": arm,
                                 "step1_file": sha(OUT / f"s{step}_{arm}_step1_model.pt"),
                                 "step256_file": sha(OUT / f"s{step}_{arm}_step256_full.pt")})
    lock["runs"].append({"run_id": "s512_N_repeat", "step256_file": sha(OUT / "s512_N_repeat_step256_full.pt")})
    json.dump(lock, open(OUT / "endpoint_lock.json", "w"), indent=1)

    model = None
    from muon_transition.model import build_model
    model = build_model().to("cuda")
    rows = []
    for step in (512, 1536):
        payload = torch.load(ROOT / "outputs" / "p0_p1_v1_1" / "parent" / f"parent_state_step{step}.pt",
                             map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        res = eval_V(model, q, "P2_endpoint")
        guard("P2_endpoint")
        rows.append({"restore_step": step, "arm": "INIT", "t": 0, **res})
        for arm in ARMS:
            for t, fname in ((1, f"s{step}_{arm}_step1_model.pt"), (256, f"s{step}_{arm}_step256_full.pt")):
                blob = torch.load(OUT / fname, map_location="cpu", weights_only=False)
                if t == 1:
                    # step1 snapshot stores named_parameters (tied lm_head deduped);
                    # loading embed_tokens re-binds the tied head automatically
                    missing, unexpected = model.load_state_dict(blob["model_state"], strict=False)
                    assert set(missing) <= {"lm_head.weight"} and not unexpected, (missing, unexpected)
                else:
                    model.load_state_dict(blob["model_state"])
                res = eval_V(model, q, "P2_endpoint")
                guard("P2_endpoint")
                rows.append({"restore_step": step, "arm": arm, "t": t, **res})
    blob = torch.load(OUT / "s512_N_repeat_step256_full.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model_state"])
    res = eval_V(model, q, "P2_endpoint")
    rows.append({"restore_step": 512, "arm": "N_repeat", "t": 256, **res})

    allf = []
    for r in rows:
        for k in r:
            if k not in allf:
                allf.append(k)
    with open(OUT / "endpoint_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=allf, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    # paired effects: reference minus candidate (positive = candidate better)
    def v(step, arm, t):
        r = [x for x in rows if x["restore_step"] == step and x["arm"] == arm and x["t"] == t]
        return r[0]
    eff = []
    for step in (512, 1536):
        for t in (1, 256):
            n = v(step, "N", t)
            for cand, label in (("MH", "G_MH=V_N-V_MH"), ("AH", "G_AH=V_N-V_AH")):
                c = v(step, cand, t)
                eff.append({"restore_step": step, "t": t, "comparison": label,
                            "value": n["V"] - c["V"],
                            "domain_A": n["nll_A"] - c["nll_A"],
                            "domain_B": n["nll_B"] - c["nll_B"]})
            mh = v(step, "MH", t)
            ah = v(step, "AH", t)
            eff.append({"restore_step": step, "t": t, "comparison": "G_ratio=V_AH-V_MH",
                        "value": ah["V"] - mh["V"],
                        "domain_A": ah["nll_A"] - mh["nll_A"],
                        "domain_B": ah["nll_B"] - mh["nll_B"]})
    with open(OUT / "paired_effects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["restore_step", "t", "comparison", "value",
                                          "domain_A", "domain_B"])
        w.writeheader(); w.writerows(eff)
    json.dump({"wall_seconds": time.time() - t0, "n_endpoint_points": len(rows)},
              open(OUT / "analysis_meta.json", "w"), indent=1)
    for r in rows:
        print(f"s{r['restore_step']} {r['arm']:>8} t={r['t']:>3}: V={r['V']:.6f} "
              f"A={r['nll_A']:.6f} B={r['nll_B']:.6f}")
    print()
    for e in eff:
        print(f"s{e['restore_step']} t={e['t']:>3} {e['comparison']:>16}: {e['value']:+.6e} "
              f"(A {e['domain_A']:+.2e} / B {e['domain_B']:+.2e})")


if __name__ == "__main__":
    main()
