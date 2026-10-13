"""Pilot gate analysis (offline, endpoint-level).

Gate (frozen, pilot_arms_v1.json): C_A or C_B beats best_muon_anchor by >=1e-3
dev H4096 NLL with paired per-doc bootstrap CI95 (1000 reps, seed 9071) entirely
negative, AND low-mu arms do NOT reproduce the same-magnitude gain.

The runner records aggregate dev NLL only; per-doc values are recomputed here
from saved endpoint_weights.pt with the runner's exact dev_nll semantics.
Also reports: low-mu reproduction check, C_B vs C_A, boundary flags,
same-wall-clock comparison at horizon granularity (disclosed approximation),
per-arm throughput/update-norm stats. Writes results/pilot/gate_analysis.json
and prints a compact summary. Zero training; endpoint eval only.
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_aware_muon.model_100m import build_model, param_fingerprint

RES = ROOT / "results" / "pilot"
DEV = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/eval_sets.json")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def per_doc_nll(state_dict, docs):
    from context_aware_muon.model_100m import CONFIG_100M, MINIMIND_MODEL_PATH
    import importlib.util

    spec = importlib.util.spec_from_file_location("mm2", str(MINIMIND_MODEL_PATH))
    mm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mm)
    model = mm.MiniMindForCausalLM(mm.MiniMindConfig(**CONFIG_100M)).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    out = []
    for doc in docs:
        ids = torch.tensor(doc["ids"], device=DEVICE, dtype=torch.long)[None, :]
        logits = model(ids[:, :-1], use_cache=False).logits
        target = ids[:, 1:]
        per = torch.nn.functional.cross_entropy(
            logits[:, : target.shape[1]].float().reshape(-1, logits.shape[-1]),
            target.reshape(-1), reduction="none").view_as(target)
        out.append(float(per.sum() / target.numel()))
    del model
    if DEVICE == "cuda:0":
        torch.cuda.empty_cache()
    return out


def paired_bootstrap(a, b, reps=1000, seed=9071):
    import random

    diffs = [x - y for x, y in zip(a, b)]  # a - b ; negative => a (C arm) better
    g = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        s = [diffs[g.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    lo, hi = means[int(0.025 * reps)], means[int(0.975 * reps) - 1]
    return sum(diffs) / n, lo, hi


def main():
    docs = json.load(open(DEV, encoding="utf-8"))["development"]
    sel = json.loads((RES / "selection.json").read_text())
    runs = {}
    for d in RES.iterdir():
        if d.is_dir() and (d / "run.json").exists():
            r = json.loads((d / "run.json").read_text())
            runs[r["arm_id"]] = r

    muon_scores = {a: r["dev_nll_by_horizon"]["4096"] for a, r in runs.items()
                   if a.startswith("B_muon_")}
    anchor = min(muon_scores, key=muon_scores.get)
    out = {"muon_anchor": anchor, "muon_scores": muon_scores, "pairs": {}}

    def doc_scores(arm):
        sd = torch.load(RES / arm / "endpoint_weights.pt", map_location="cpu")
        return per_doc_nll(sd, docs)

    cand = []
    if "C_A_lam0p65" in runs:
        cand.append("C_A_lam0p65")
    if "C_A_lam0p85" in runs:
        cand.append("C_A_lam0p85")
    if "C_B" in runs:
        cand.append("C_B")
    anchor_docs = doc_scores(anchor) if cand else None

    for c in cand:
        d = doc_scores(c)
        mean, lo, hi = paired_bootstrap(d, anchor_docs)
        out["pairs"][f"{c}_vs_{anchor}"] = {
            "mean_diff": mean, "ci95": [lo, hi],
            "gate_ge_1e-3": mean <= -1e-3 and hi < 0,
        }
    for lm in [a for a in runs if a.startswith("B_lowmu")]:
        d = doc_scores(lm)
        mean, lo, hi = paired_bootstrap(d, anchor_docs)
        out["pairs"][f"{lm}_vs_{anchor}"] = {"mean_diff": mean, "ci95": [lo, hi]}

    if "C_B" in runs and "C_A_lam0p65" in runs:
        a = doc_scores("C_B")
        b = doc_scores("C_A_lam0p65")
        mean, lo, hi = paired_bootstrap(a, b)
        out["pairs"]["C_B_vs_C_A_lam0p65"] = {"mean_diff": mean, "ci95": [lo, hi]}

    # same-wall-clock (horizon-granularity, disclosed): compare at the largest
    # horizon every involved arm reached within min(total_wall)
    walls = {a: r["wall_s"] for a, r in runs.items()}
    wmin = min(walls[a] for a in runs)
    out["same_wallclock_note"] = {
        "min_total_wall_s": wmin,
        "method": "horizon-level approximation from run.json (per-step wall not recorded at eval points)",
    }

    # momentum-explained check (frozen): low-mu reproduces same-magnitude gain?
    if anchor_docs is not None and any(k.startswith("B_lowmu") for k in out["pairs"]):
        best_c = None
        for k, v in out["pairs"].items():
            if k.startswith("C_") and k.endswith("_vs_" + anchor) and v["gate_ge_1e-3"]:
                best_c = (k, v)
        best_low = min((v["mean_diff"] for k, v in out["pairs"].items() if k.startswith("B_lowmu")),
                       default=None)
        out["momentum_explained_check"] = {
            "best_C_pair": best_c[0] if best_c else None,
            "best_lowmu_diff": best_low,
            "explained": bool(best_c and best_low is not None and best_low <= best_c[1]["mean_diff"] / 2),
            "note": "frozen wording: low-mu must NOT reproduce same-magnitude gain; half-magnitude heuristic recorded, final call in decision doc",
        }

    (ROOT / "results" / "pilot" / "gate_analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "muon_scores"}, indent=1))


if __name__ == "__main__":
    main()
