"""Standalone cost/pool benchmark (rerunnable without redoing calibration)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment"))
sys.path.insert(0, str(ROOT.parent / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(ROOT.parent / "minimind_base"))

from src.recipe_registry import GROUPS, group_of_param
from src.state_restore import load_anchor
from src.p1_engine import momenta_from_payload
from src.group_probe import collect_G, compute_H
from src.cost_meter import group_step_flops, kv_flops_ratio, benchmark_K
from muon_transition.mixture import MixtureStream, build_step_manifest
from muon_transition.m1_data_cache import get_domain_windows

OUT = ROOT / "outputs" / "p0_p1_v1_1"


def main():
    t0 = time.time()
    payload = torch.load(OUT / "parent" / "parent_state_step512.pt",
                         map_location="cpu", weights_only=False)
    model, _ = load_anchor(payload)
    shapes = {}
    for name, p in model.named_parameters():
        g = group_of_param(name)
        if g:
            shapes.setdefault(g, []).append(tuple(p.shape))
    sf = group_step_flops(shapes)
    pairs = [(gm, gp) for gm in GROUPS for gp in GROUPS if gm != gp]
    flops_ok = []
    for gm, gp in pairs:
        r = kv_flops_ratio(sf, {gm: 3, gp: 7})
        if 0.98 <= r <= 1.02:
            flops_ok.append({"pair": [gm, gp], "flops_ratio": r})

    a_train, b_train = get_domain_windows()
    steps = build_step_manifest(a_train, b_train, 0.5, 513, seed=2030)["steps"][512:513]
    stream = MixtureStream({"steps": steps}, a_train, b_train)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    collect_G(model, stream, 0, autocast)
    momenta = momenta_from_payload(payload, model)
    H = compute_H(model, momenta)
    H_items = list(H.items())
    # interleaved blocks: ref, cand1, ref, cand2, ... recompute ref each time to
    # interleave clock conditions
    ref_ts, cand_ts = [], {i: [] for i in range(len(flops_ok))}
    order = []
    for i in range(len(flops_ok)):
        order += ["ref", i]
    if not order:
        order = ["ref", "ref"]
    for _ in range(24):
        for tag in order:
            if tag == "ref":
                ref_ts.append(benchmark_K(H_items, {}, n_blocks=1, warmup=1))
            else:
                gm, gp = flops_ok[tag]["pair"]
                cand_ts[tag].append(benchmark_K(H_items, {gm: 3, gp: 7}, n_blocks=1, warmup=1))
    import numpy as np
    ref_med = float(np.median(ref_ts))
    for i, rec in enumerate(flops_ok):
        rec["cand_ms"] = float(np.median(cand_ts[i])) * 1e3
        rec["ref_ms"] = ref_med * 1e3
        rec["time_ratio_median"] = rec["cand_ms"] / rec["ref_ms"]
        rec["budget_ok"] = 0.98 <= rec["time_ratio_median"] <= 1.02
    pool = [r for r in flops_ok if r["budget_ok"]]
    cost = {"group_step_flops": sf, "ref_benchmark_ms": ref_med * 1e3,
            "n_timing_blocks": 24, "analytic_feasible": flops_ok, "measured_pool": pool,
            "note": "pool = pairs with BOTH analytic FLOPs and measured median time within [0.98,1.02]"}
    json.dump(cost, open(OUT / "cost_pool.json", "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] analytic-feasible {len(flops_ok)}/{len(pairs)}; "
          f"measured pool {len(pool)}; ref={ref_med*1e3:.2f}ms")
    for r in flops_ok:
        print(f"  {r['pair']}: flops={r['flops_ratio']:.6f} time={r['time_ratio_median']:.4f} ok={r['budget_ok']}")


if __name__ == "__main__":
    main()
