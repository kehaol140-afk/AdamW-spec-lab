"""N->L tail branches (final bounded contrast): preflight checks + two 256-step
continuations with recipe L (eff_lr = 0.5*base_lr, eff_wd = base_wd -> decay product
= 0.5x base). Sources are the OLD-R2 N mid-states (cursor 768/1792). Uses the FIXED
load chain (immutable base binding); per-step assertions on base/eff LR/WD/decay.
Outputs to outputs/r2f_nl_tail_v1/ (nothing old overwritten)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(str(ROOT) + "/.."))

from scripts.run_r2f import (load_full, immutable_base_wds, make_stream, sha256_file,
                             R2DIR, CORR)
from src.r2f_recipe import RECIPES, effective
from src.state_restore import eval_V, build_q_sets
from src.recipe_registry import lr_at, group_of_param
from vendored_fr.rng import capture_rng_state

OUT = ROOT / "outputs" / "r2f_nl_tail_v1"
OUT.mkdir(parents=True, exist_ok=True)


def preflight(src_path, start_step, recipe="L", q=None):
    """One-step reference check on the real source: eff LR/WD/decay-product correct,
    grads/moments untouched by config write-back, no CPU-step aliasing with payload,
    data-delta measured from L's OWN WD base."""
    bw = immutable_base_wds()
    model, opts, payload = load_full(src_path)
    # identity
    blob = torch.load(src_path, map_location="cpu", weights_only=False)
    assert blob["cursor"]["step"] == start_step, (blob["cursor"], start_step)
    # aliasing with payload
    for opt, os_ in zip(opts, blob["optimizer_states"]):
        for ka, kb in zip(opt.state.keys(), os_["state"].keys()):
            for kk in set(opt.state[ka].keys()) | set(os_["state"][kb].keys()):
                va, vb = opt.state[ka].get(kk), os_["state"][kb].get(kk)
                if torch.is_tensor(va) and torch.is_tensor(vb):
                    assert va.data_ptr() != vb.data_ptr(), "alias with payload"
    base_lr = lr_at(start_step + 1)
    cfg = RECIPES[recipe]
    expected_dp = cfg["d"] * base_lr * bw["muon"]
    eff = []
    for opt, key in ((opts[0], "muon"), (opts[1], "aux")):
        elr, ewd = effective(base_lr, bw["muon"] if key == "muon" else bw["aux"], recipe)
        for group in opt.param_groups:
            group["lr"] = elr
            group["weight_decay"] = ewd
        assert abs(elr * ewd - expected_dp) / expected_dp < 1e-12
        eff.append((elr, ewd))
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    theta_pre = {n: p.detach().clone() for n, p in model.named_parameters()}
    stream = make_stream(start_step, 1)
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    for mb in stream.micro_batches(0):
        b = mb.to("cuda")
        with autocast:
            lg = model(b).logits[..., :-1, :].contiguous()
            t = b[..., 1:].contiguous()
            loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), t.reshape(-1), reduction="mean")
        (loss / 4).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
    opts[0].step()
    opts[1].step()
    # data delta from L's OWN wd base: theta_next=(1-eff_lr*eff_wd)theta - eff_lr*v
    num = 0.0
    for n, p in model.named_parameters():
        if group_of_param(n) is None:
            continue
        d = (1 - eff[0][0] * eff[0][1]) * theta_pre[n].double() - p.detach().double()
        num += float(d.pow(2).sum())
    return {"source_sha": sha256_file(src_path), "cursor": start_step,
            "base_lr": base_lr, "eff_lr": eff[0][0], "eff_wd": eff[0][1],
            "decay_product": eff[0][0] * eff[0][1],
            "decay_product_ratio_vs_base": (eff[0][0] * eff[0][1]) / (base_lr * bw["muon"]),
            "muon_data_delta_norm": num ** 0.5}


def run_nl(run_id, src_path, start_step, q, logf):
    marker = OUT / f"{run_id}.complete.json"
    if marker.exists():
        return {"run_id": run_id, "status": "already_complete"}
    t0 = time.time()
    model, opts, payload = load_full(src_path)
    bw = immutable_base_wds()
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    stream = make_stream(start_step, 256)
    logl = open(OUT / f"{run_id}_train.jsonl", "w")
    cfg = RECIPES["L"]
    for k in range(1, 257):
        gstep = start_step + k
        base_lr = lr_at(gstep)
        eff = []
        for opt, key in ((opts[0], "muon"), (opts[1], "aux")):
            elr, ewd = effective(base_lr, bw["muon"] if key == "muon" else bw["aux"], "L")
            for group in opt.param_groups:
                group["lr"] = elr
                group["weight_decay"] = ewd
            dp = elr * ewd
            exp_dp = cfg["d"] * base_lr * bw["muon"]
            assert abs(dp - exp_dp) / exp_dp < 1e-12, f"{run_id} step{k} decay {dp}!={exp_dp}"
            assert ewd == 0.02, f"{run_id} step{k} wd {ewd} unexpected for L"
            eff.append((elr, ewd, dp))
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        micro = []
        for mb in stream.micro_batches(k - 1):
            b = mb.to("cuda")
            with autocast:
                lg = model(b).logits[..., :-1, :].contiguous()
                t = b[..., 1:].contiguous()
                loss = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), t.reshape(-1), reduction="mean")
            (loss / 4).backward()
            micro.append(float(loss.detach()))
        preclip = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0)
        opts[0].step()
        opts[1].step()
        logl.write(json.dumps({
            "run_id": run_id, "source_sha": sha256_file(src_path), "global_step": gstep,
            "config_sha": "r2f_nl_tail_v1", "recipe": "L",
            "base_lr": base_lr, "base_wd": bw["muon"],
            "eff_lr_muon": eff[0][0], "eff_wd_muon": eff[0][1],
            "decay_product_muon": eff[0][2],
            "decay_product_ratio": eff[0][2] / (base_lr * bw["muon"]),
            "train_loss": float(np.mean(micro)), "preclip_norm": float(preclip),
            "elapsed_s": time.time() - t0}) + "\n")
        if k in (1, 16, 64, 128, 256):
            res = eval_V(model, q, "select")
            assert model.training
            logl.write(json.dumps({"run_id": run_id, "monitor_rel_step": k, **res}) + "\n")
    logl.close()
    torch.save({"model_state": model.state_dict(),
                "optimizer_states": [o.state_dict() for o in opts],
                "rng": capture_rng_state(),
                "cursor": {"step": start_step + 256, "run_id": run_id},
                "meta": {"recipe": "L", "start_step": start_step}},
               OUT / f"{run_id}_step256_full.pt.tmp")
    b2 = torch.load(OUT / f"{run_id}_step256_full.pt.tmp", map_location="cpu", weights_only=False)
    assert b2["cursor"]["step"] == start_step + 256
    (OUT / f"{run_id}_step256_full.pt.tmp").rename(OUT / f"{run_id}_step256_full.pt")
    json.dump({"run_id": run_id, "recipe": "L", "n_steps": 256,
               "source_sha": sha256_file(src_path),
               "final_sha": sha256_file(OUT / f"{run_id}_step256_full.pt"),
               "wall_s": time.time() - t0},
              open(marker, "w"), indent=1)
    logf.write(f"[{time.time():.0f}] {run_id} COMPLETE\n")
    return {"run_id": run_id, "status": "complete"}


def main():
    t0 = time.time()
    logf = open(OUT / "runner_log.txt", "a")
    pre = {}
    for step in (512, 1536):
        src = R2DIR / f"s{step}_N_step256_full.pt"
        pre[str(step)] = preflight(src, step + 256)
        print(f"preflight s{step}: {pre[str(step)]}", flush=True)
    json.dump(pre, open(OUT / "preflight.json", "w"), indent=1)
    q = build_q_sets()
    for step in (512, 1536):
        r = run_nl(f"s{step}_N_to_L", R2DIR / f"s{step}_N_step256_full.pt", step + 256, q, logf)
        print(f"[{time.time()-t0:.0f}s] {r}", flush=True)
    logf.close()
    print(f"N->L runner done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
