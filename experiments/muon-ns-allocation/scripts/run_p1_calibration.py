"""P1 calibration: 2 states x 4 batches x 45 KJ points on Q_select (+ geometry, gamma,
nu, e_repeat, cost benchmark for the same-budget pool). No Q_P1_check access here."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.p1_engine import evaluate_batch, kj_schemes
from src.state_restore import build_q_sets, access_guard
from src.recipe_registry import GROUPS, P1_STATES
from src.cost_meter import (ns_flops_per_step, group_step_flops, kv_flops_ratio,
                            benchmark_K)
from src.group_probe import compute_H, ns_cache

OUT = ROOT / "outputs" / "p0_p1_v1_1"
guard = access_guard({"select"})


def main():
    t0 = time.time()
    q = build_q_sets(verbose=True)
    parent = OUT / "parent"
    rows, meta = [], {}
    for s in P1_STATES:
        sp = parent / f"parent_state_step{s}.pt"
        assert sp.exists(), f"missing parent state {s}"
        for bi in range(4):
            rr, mm = evaluate_batch(sp, s, (s + bi, s + bi + 1), q, "select",
                                    kj_schemes(), repeat_native=(bi == 0),
                                    independent_check=(bi == 0))
            guard("select")
            for r in rr:
                r.update({"state": s, "batch": bi + 1})
            rows.extend(rr)
            meta[f"s{s}|b{bi+1}"] = mm  # includes E_g2, gamma, nu5, grad info, probe lr
            print(f"[{time.time()-t0:.0f}s] state {s} cal batch {bi+1}: native V="
                  f"{[r['V'] for r in rr if r['scheme']=='native'][0]:.6f} "
                  f"({len(rr)} points)", flush=True)
    all_fields = []
    for r in rows:
        for k in r.keys():
            if k not in all_fields:
                all_fields.append(k)
    with open(OUT / "calibration_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    json.dump(meta, open(OUT / "calibration_meta.json", "w"), indent=1, default=str)

    # ---- cost: analytic FLOPs + measured same-budget pool ----
    shapes = {}
    payload = torch.load(parent / "parent_state_step512.pt", map_location="cpu", weights_only=False)
    from src.state_restore import load_anchor
    from src.p1_engine import momenta_from_payload
    model, _ = load_anchor(payload)
    for name, p in model.named_parameters():
        from src.recipe_registry import group_of_param
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
    # measured timing for analytically-feasible pairs (+ reference), alternating blocks
    from muon_transition.mixture import MixtureStream, build_step_manifest
    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    steps = build_step_manifest(a_train, b_train, 0.5, 513, seed=2030)["steps"][512:513]
    stream = MixtureStream({"steps": steps}, a_train, b_train)
    from src.group_probe import collect_G
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    import torch.nn.functional as F
    for p in model.parameters():
        p.grad = None
    for mb in stream.micro_batches(0):
        batch = mb.to("cuda")
        with autocast:
            logits = model(batch).logits[..., :-1, :].contiguous()
            tt = batch[..., 1:].contiguous()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   tt.reshape(-1), reduction="mean")
        (loss / 4).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
    momenta = momenta_from_payload(payload, model)
    H = compute_H(model, momenta)
    H_list = list(H.items())
    ref_t = benchmark_K(H_list, {})
    for rec in flops_ok:
        gm, gp = rec["pair"]
        cand_t = benchmark_K(H_list, {gm: 3, gp: 7})
        rec["time_ratio_median"] = cand_t / ref_t
        rec["ref_ms"] = ref_t * 1e3
        rec["cand_ms"] = cand_t * 1e3
        rec["budget_ok"] = 0.98 <= cand_t / ref_t <= 1.02
    pool = [r for r in flops_ok if r["budget_ok"]]
    cost = {"group_step_flops": sf, "ref_benchmark_ms": ref_t * 1e3,
            "analytic_feasible": flops_ok, "measured_pool": pool,
            "note": "pool = pairs with BOTH analytic FLOPs and measured time within [0.98,1.02]"}
    json.dump(cost, open(OUT / "cost_pool.json", "w"), indent=1)
    print(f"[{time.time()-t0:.0f}s] cost pool: {len(pool)} pairs of {len(pairs)} enumerated; "
          f"ref {ref_t*1e3:.1f} ms", flush=True)
    print("calibration done", time.time() - t0)


if __name__ == "__main__":
    main()
