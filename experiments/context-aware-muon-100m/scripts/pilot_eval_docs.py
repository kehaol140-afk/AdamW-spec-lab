"""Per-document endpoint evaluation for pilot arms (paired bootstrap input).

Loads results/pilot/<arm>/endpoint_weights.pt into a fresh 100M model,
evaluates V16 development 256 docs one-by-one (dv loss_sum_count semantics:
one label shift, explicit length mask, fp32 CE), writes
results/pilot/<arm>/eval_docs_H4096.jsonl with per-doc loss_sum/token_count.

Also verifies: endpoint fingerprint matches run.json final_fingerprint
(identity of evaluated artifact); aggregate equals run.json dev NLL to <1e-9
(same docs, same semantics).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_aware_muon.model_100m import build_model, param_fingerprint

DEV = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/eval_sets.json")


@torch.no_grad()
def main():
    arm = sys.argv[1]
    device = "cuda:0"
    out_dir = ROOT / "results" / "pilot" / arm
    run = json.load(open(out_dir / "run.json"))
    docs = json.load(open(DEV, encoding="utf-8"))["development"]

    model = build_model("100m", seed=0, device=device)  # weights overwritten below
    sd = torch.load(out_dir / "endpoint_weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    fp = param_fingerprint(model)
    assert fp == run["final_fingerprint"], f"endpoint identity mismatch {fp} != {run['final_fingerprint']}"

    t0 = time.perf_counter()
    rows = []
    with (out_dir / "eval_docs_H4096.jsonl").open("w") as f:
        for doc in docs:
            ids = torch.tensor(doc["ids"], device=device, dtype=torch.long)[None, :]
            logits = model(ids[:, :-1], use_cache=False).logits
            target = ids[:, 1:]
            per = torch.nn.functional.cross_entropy(
                logits[:, : target.shape[1]].float().reshape(-1, logits.shape[-1]),
                target.reshape(-1), reduction="none").view_as(target)
            row = {"doc_id": doc["doc_id"], "loss_sum": float(per.sum()),
                   "token_count": int(target.numel())}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
    agg = sum(r["loss_sum"] for r in rows) / sum(r["token_count"] for r in rows)
    delta = abs(agg - run["dev_nll_by_horizon"]["4096"])
    print(json.dumps({
        "arm": arm, "endpoint_fp": fp,
        "agg_dev_nll": agg, "run_json_dev_nll": run["dev_nll_by_horizon"]["4096"],
        "abs_delta": delta, "eval_seconds": round(time.perf_counter() - t0, 1),
        "n_docs": len(rows),
    }))
    assert delta < 1e-9, "aggregate mismatch vs run.json (semantics drift?)"


if __name__ == "__main__":
    main()
