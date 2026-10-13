#!/usr/bin/env python3
"""GPU smoke: 50 optimizer steps on (a) AdamW S8 path with apply_step, (b) Muon reference path.
Measures throughput, VRAM, determinism of first step across paths; no eval, no endpoint."""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import s30m_core as C  # noqa: E402
from dv_core import apply_step, schedule_eta, param_fingerprint  # noqa: E402
from pilot_shim import loss_sum_count  # noqa: E402

STEPS = 50


def smoke(family: str):
    device = "cuda:0"
    stream = C.TokenStream()
    model = C.build_model(3421, device)
    fp0 = param_fingerprint(model)
    tgt = "model.layers.1.mlp.down_proj.weight"
    if family == "M":
        opts = C.build_muon(model, 0.01)
    else:
        opts = [C.build_adamw(model, 0.001)]
    bf16 = torch.cuda.is_bf16_supported()
    cursor, tokens = 0, 0
    first_loss = None
    t0 = time.time()
    opt_s = 0.0
    for k in range(1, STEPS + 1):
        eta = schedule_eta(0.001 if family != "M" else 0.01, k, 2048, 128, 0.1)
        for g in [g for opt in opts for g in opt.param_groups]:
            g["lr"] = eta
        model.train()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        lt, vt = 0.0, 0
        for _ in range(C.ACCUM):
            ids, cursor = stream.micro_batch(cursor, device)
            lengths = torch.full((C.MICRO,), C.SEQ + 1, device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(ids[:, :-1], use_cache=False).logits
            ls, c = loss_sum_count(logits, ids, lengths)
            ls.backward()
            vt += c
            lt += float(ls.detach())
        if first_loss is None:
            first_loss = lt / vt
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(vt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
        ts = time.perf_counter()
        if family == "M":
            for opt in opts:
                opt.step()
            torch.cuda.synchronize()
        else:
            apply_step(model, opts[0], family, 8.0, 1.0, "smoke", eta, tgt)
        opt_s += time.perf_counter() - ts
        tokens += vt
    wall = time.time() - t0
    vram = torch.cuda.max_memory_allocated() / 2**30
    out = {"family": family, "steps": STEPS, "wall_s": wall, "tok_per_s": tokens / wall,
           "step_s": wall / STEPS, "opt_step_s_total": opt_s,
           "projected_run_min_2048": wall / STEPS * 2048 / 60,
           "vram_GiB": vram, "first_step_nll": first_loss, "init_fp": fp0[:12],
           "final_nll_sample": lt / vt}
    print(json.dumps(out), flush=True)
    del model, opts
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


if __name__ == "__main__":
    a = smoke("S")
    b = smoke("M")
    assert a["first_step_nll"] == b["first_step_nll"], "CRN first-step mismatch"
    per_run = (a["projected_run_min_2048"] + b["projected_run_min_2048"]) / 2
    print(json.dumps({"crn_first_step_match": True,
                      "projected_13runs_hours_adamw": a["projected_run_min_2048"] * 10 / 60,
                      "projected_13runs_hours_muon": b["projected_run_min_2048"] * 3 / 60,
                      "projected_total_hours": (a["projected_run_min_2048"] * 10 +
                                                b["projected_run_min_2048"] * 3) / 60}))
