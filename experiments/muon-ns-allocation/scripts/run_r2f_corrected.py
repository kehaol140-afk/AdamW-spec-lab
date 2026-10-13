"""R2F corrected runner: the four AH-source crossover branches with the FIXED base
binding (immutable manifest WD), per-step per-group assertions, and every-16-step
norm diagnostics. New run_ids (*_corrected) and a new output dir; nothing old is
overwritten. Sources are the ORIGINAL old-R2 AH full end-states (768/1792)."""
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
                              CORR, R2DIR)
from src.r2f_recipe import effective
from src.state_restore import eval_V, build_q_sets
from src.recipe_registry import lr_at, group_of_param
from vendored_fr.rng import capture_rng_state

CONFIG_SHA = "r2f_corrected_v1"


def run_branch_corrected(run_id, src_path, start_step, recipe, q, logf):
    complete_marker = CORR / f"{run_id}.complete.json"
    if complete_marker.exists():
        return {"run_id": run_id, "status": "already_complete"}
    t0 = time.time()
    model, opts, payload = load_full(src_path)
    bw = immutable_base_wds()
    base_wds = (bw["muon"], bw["aux"])
    src_sha = sha256_file(src_path)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    stream = make_stream(start_step, 256)
    logl = open(CORR / f"{run_id}_train.jsonl", "w")
    r_m = {"N": 1.0, "AH": 0.5}[recipe]
    snapshots = {}
    n_steps = 256
    theta_before_diag = None
    for k in range(1, n_steps + 1):
        gstep = start_step + k
        base_lr = lr_at(gstep)
        eff_pairs = []
        for opt, key in ((opts[0], "muon"), (opts[1], "aux")):
            elr, ewd = effective(base_lr, base_wds[0] if key == "muon" else base_wds[1], recipe)
            for group in opt.param_groups:
                group["lr"] = elr
                group["weight_decay"] = ewd
            dp = elr * ewd
            # HARD assertion: expected decay product = base_lr*base_wd for N/AH here
            expected_dp = base_lr * base_wds[0]
            assert abs(dp - expected_dp) / expected_dp < 1e-12, \
                f"{run_id} step{k} {key}: decay product {dp} != base {expected_dp}"
            assert ewd in (0.02, 0.04), f"{run_id} step{k} {key}: WD {ewd} hits old bug path"
            eff_pairs.append((elr, ewd, dp))
        # diagnostics snapshot (param norms) every 16 steps BEFORE the step
        if k % 16 == 1 or k == 1:
            theta_before_diag = {n_: p.detach().clone() for n_, p in model.named_parameters()}
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        micro = []
        for mb in stream.micro_batches(k - 1):
            batch = mb.to("cuda")
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                t = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       t.reshape(-1), reduction="mean")
            (loss / 4).backward()
            micro.append(float(loss.detach()))
        preclip = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0)
        opts[0].step()
        opts[1].step()
        rec = {"run_id": run_id, "source_checkpoint_sha": src_sha, "global_step": gstep,
               "config_sha": CONFIG_SHA, "recipe": recipe,
               "base_lr": base_lr, "base_wd_muon": base_wds[0], "base_wd_aux": base_wds[1],
               "eff_lr_muon": eff_pairs[0][0], "eff_wd_muon": eff_pairs[0][1],
               "decay_product_muon": eff_pairs[0][2],
               "eff_lr_aux": eff_pairs[1][0], "eff_wd_aux": eff_pairs[1][1],
               "decay_product_aux": eff_pairs[1][2],
               "train_loss": float(np.mean(micro)), "preclip_norm": float(preclip),
               "elapsed_s": time.time() - t0}
        # every-16-step norm diagnostics (data update norm / param norm per group)
        if k % 16 == 0 and theta_before_diag is not None:
            dn = {"muon": 0.0, "aux": 0.0}
            pn = {"muon": 0.0, "aux": 0.0}
            for n_, p in model.named_parameters():
                g = group_of_param(n_)
                key = "muon" if g is not None else "aux"
                elr, ewd = (eff_pairs[0] if key == "muon" else eff_pairs[1])[:2]
                # data update: eff_lr*v = (1-eff_lr*eff_wd)*theta_before - theta_after
                d = (1 - elr * ewd) * theta_before_diag[n_].double() - p.detach().double()
                dn[key] += float(d.pow(2).sum())
                pn[key] += float(p.detach().double().pow(2).sum())
            for key in dn:
                rec[f"param_norm_{key}"] = pn[key] ** 0.5
                rec[f"data_update_norm_{key}"] = dn[key] ** 0.5
                rec[f"data_over_param_{key}"] = (dn[key] ** 0.5) / max(pn[key] ** 0.5, 1e-30)
        logl.write(json.dumps(rec) + "\n")
        if k in (1, 16, 64):
            snapshots[k] = {n_: p.detach().cpu().clone() for n_, p in model.named_parameters()}
        if k in (1, 16, 64, 128, 256):
            res = eval_V(model, q, "select")
            assert model.training
            logl.write(json.dumps({"run_id": run_id, "monitor_rel_step": k, **res}) + "\n")
    logl.close()
    for k, snap in snapshots.items():
        torch.save({"model_state": snap, "meta": {"run_id": run_id, "rel_step": k}},
                   CORR / f"{run_id}_step{k}_model.pt")
    torch.save({"model_state": model.state_dict(),
                "optimizer_states": [o.state_dict() for o in opts],
                "rng": capture_rng_state(),
                "cursor": {"step": start_step + n_steps, "run_id": run_id},
                "meta": {"recipe": recipe, "start_step": start_step,
                         "base_wd_binding": "immutable_manifest"}},
               CORR / f"{run_id}_step256_full.pt.tmp")
    blob = torch.load(CORR / f"{run_id}_step256_full.pt.tmp", map_location="cpu", weights_only=False)
    assert blob["cursor"]["step"] == start_step + n_steps
    (CORR / f"{run_id}_step256_full.pt.tmp").rename(CORR / f"{run_id}_step256_full.pt")
    json.dump({"run_id": run_id, "recipe": recipe, "start_step": start_step,
               "n_steps": n_steps, "source_sha": src_sha,
               "final_sha": sha256_file(CORR / f"{run_id}_step256_full.pt"),
               "final_loss": float(np.mean(micro)), "wall_s": time.time() - t0},
              open(complete_marker, "w"), indent=1)
    logf.write(f"[{time.time():.0f}] {run_id} COMPLETE loss={np.mean(micro):.4f}\n")
    logf.flush()
    return {"run_id": run_id, "status": "complete"}


def main():
    t0 = time.time()
    logf = open(CORR / "runner_log.txt", "a")
    q = build_q_sets()
    plan = [
        ("s512_AH_to_N_corrected", R2DIR / "s512_AH_step256_full.pt", 768, "N"),
        ("s512_AH_to_AH_corrected", R2DIR / "s512_AH_step256_full.pt", 768, "AH"),
        ("s1536_AH_to_N_corrected", R2DIR / "s1536_AH_step256_full.pt", 1792, "N"),
        ("s1536_AH_to_AH_corrected", R2DIR / "s1536_AH_step256_full.pt", 1792, "AH"),
    ]
    for run_id, src, start, recipe in plan:
        r = run_branch_corrected(run_id, src, start, recipe, q, logf)
        print(f"[{time.time()-t0:.0f}s] {r}", flush=True)
    logf.close()
    print(f"corrected runner done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
