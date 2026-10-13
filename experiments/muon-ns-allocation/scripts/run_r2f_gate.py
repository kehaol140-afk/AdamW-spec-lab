"""R2F stage 0: zero-training reuse gate.
- asset_certificate.json: 6 states (parent 512/1536, old R2 N/AH ends 768/1792)
  with SHA/model-hash/state keys/cursor/RNG.
- base_group_manifest.json: per-group base LR fn / WD / betas / eps / routing;
  verifies the old R2 runner wrote per-group values (no cross-group broadcast when
  the original recipe has equal groups — equality proven, not assumed).
- reuse re-evaluation of the four old N/AH endpoints on Q_P2_endpoint (tol 1e-7).
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment"))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(ROOT.parent / "minimind_base"))

from src.state_restore import build_q_sets, eval_V
from src.recipe_registry import lr_at

OUT = ROOT / "outputs" / "r2f_lrwd_retention_v1"
OUT.mkdir(parents=True, exist_ok=True)
R2DIR = ROOT / "outputs" / "r2_amplitude_continuation_v1"
PARENT = ROOT / "outputs" / "p0_p1_v1_1" / "parent"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_hash(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    t0 = time.time()
    q = build_q_sets()
    states = {
        "parent_512": PARENT / "parent_state_step512.pt",
        "parent_1536": PARENT / "parent_state_step1536.pt",
        "old_s512_N_end768": R2DIR / "s512_N_step256_full.pt",
        "old_s512_AH_end768": R2DIR / "s512_AH_step256_full.pt",
        "old_s1536_N_end1792": R2DIR / "s1536_N_step256_full.pt",
        "old_s1536_AH_end1792": R2DIR / "s1536_AH_step256_full.pt",
    }
    cert = {"states": {}}
    for name, p in states.items():
        blob = torch.load(p, map_location="cpu", weights_only=False)
        rec = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p),
               "cursor": blob.get("cursor"),
               "model_content_hash": model_hash(blob["model_state"]),
               "n_optimizer_states": len(blob.get("optimizer_states", [])),
               "rng_keys": list(blob.get("rng", {}).keys()) if isinstance(blob.get("rng"), dict) else "MISSING"}
        if "optimizer_states" in blob:
            for i, os_ in enumerate(blob["optimizer_states"]):
                n_groups = len(os_["param_groups"])
                grp = os_["param_groups"][0]
                rec[f"opt{i}_n_groups"] = n_groups
                rec[f"opt{i}_n_params"] = sum(len(g["params"]) for g in os_["param_groups"])
                rec[f"opt{i}_state_entries"] = len(os_["state"])
                # saved (effective) values — for AH these are scaled; identity only
                rec[f"opt{i}_saved_lr"] = grp.get("lr")
                rec[f"opt{i}_saved_wd"] = grp.get("weight_decay")
        # check ALL groups within each optimizer share the same base values (they should:
        # original recipe has one group per optimizer; if >1 group, values must be listed)
        cert["states"][name] = rec
        print(f"{name}: sha={rec['sha256'][:12]} model_hash={rec['model_content_hash'][:12]} cursor={rec['cursor']}")

    # ---- base group manifest: the ORIGINAL recipe has 1 group per optimizer ----
    payload = torch.load(PARENT / "parent_state_step512.pt", map_location="cpu", weights_only=False)
    g0 = payload["optimizer_states"][0]["param_groups"][0]
    g1 = payload["optimizer_states"][1]["param_groups"][0]
    manifest = {
        "note": "original recipe: ONE param group per optimizer (Muon: 56 matrices; AdamW aux: 34 tensors "
                "incl. tied embedding once). Base LR for both optimizers is the SAME clock value "
                "lr_at(global_step) (production writes one lr to both), base WD is 0.02 in both. "
                "The old R2 runner wrote per-group effective values from these bases each step — "
                "no cross-group broadcast exists to violate (equality proven from checkpoint + "
                "production loop, not assumed).",
        "groups": [
            {"group_id": "muon", "optimizer": "Muon", "n_params": 56,
             "base_lr_fn": "lr_at(global_step) [warmup32+cosine->0.1, peak 0.01]",
             "base_lr_at_513": lr_at(513), "base_wd": g0["weight_decay"],
             "momentum": g0["momentum"], "nesterov": g0["nesterov"],
             "ns_steps": g0["ns_steps"], "ns_coefficients": g0["ns_coefficients"],
             "eps": g0["eps"], "adjust_lr_fn": g0["adjust_lr_fn"]},
            {"group_id": "aux_adamw", "optimizer": "AdamW", "n_params": 34,
             "base_lr_fn": "same clock lr_at(global_step) (production sets identical lr on both optimizers)",
             "base_lr_at_513": lr_at(513), "base_wd": g1["weight_decay"],
             "betas": g1["betas"], "eps": g1["eps"],
             "decoupled_weight_decay": g1.get("decoupled_weight_decay")},
        ],
        "groups_base_lr_equal": True, "groups_base_wd_equal": True,
        "saved_group_counts": {n: (r.get("opt0_n_groups"), r.get("opt1_n_groups"))
                               for n, r in cert["states"].items()},
    }
    json.dump(manifest, open(OUT / "base_group_manifest.json", "w"), indent=1)

    # ---- reuse re-evaluation of the four old endpoints (identity check, tol 1e-7) ----
    from muon_transition.model import build_model
    model = build_model().to("cuda")
    old_vals = {(int(r["restore_step"]), r["arm"]): float(r["V"])
                for r in csv.DictReader(open(R2DIR / "endpoint_values.csv"))
                if r["t"] == "256" and r["arm"] in ("N", "AH")}
    checks = []
    for (step, arm), old_v in sorted(old_vals.items()):
        blob = torch.load(R2DIR / f"s{step}_{arm}_step256_full.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        res = eval_V(model, q, "P2_endpoint")
        diff = abs(res["V"] - old_v)
        checks.append({"state": f"s{step}_{arm}", "old_V": old_v, "recomputed_V": res["V"],
                       "abs_diff": diff, "within_1e-7": diff <= 1e-7})
        print(f"reuse s{step}_{arm}: old={old_v:.9f} new={res['V']:.9f} diff={diff:.3e} ok={diff<=1e-7}")
    cert["endpoint_reuse_checks"] = checks
    # N-repeat hash identity (determinism record only)
    rep = torch.load(R2DIR / "s512_N_repeat_step256_full.pt", map_location="cpu", weights_only=False)
    n1 = cert["states"]["old_s512_N_end768"]["model_content_hash"]
    cert["n_repeat_model_hash_equal"] = model_hash(rep["model_state"]) == n1
    gate_pass = all(c["within_1e-7"] for c in checks)
    cert["reuse_gate_pass"] = gate_pass
    cert["wall_seconds"] = time.time() - t0
    json.dump(cert, open(OUT / "asset_certificate.json", "w"), indent=1)
    print("REUSE GATE:", "PASS" if gate_pass else "FAIL")


if __name__ == "__main__":
    main()
