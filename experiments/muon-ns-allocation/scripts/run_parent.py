"""Parent trajectory: seed 3101, recipe of m1_muon_lr2.0_wd0.02_p0.5_s2030 (peak lr 0.01,
wd 0.02, p=0.5 data stream seed 2030, warmup32+cosine->0.1, 2048 updates), saving full
states at 512/1536/2048. AMO geometry observation SKIPPED (COMPARATOR_UNRESOLVED: paper
unreachable) — recorded, parent RNG untouched by any observation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition.model import build_model
from muon_transition.optimizers import build_muon_arm
from muon_transition.mixture import MixtureStream, build_step_manifest
from muon_transition.checkpoint import save_training_state
from src.recipe_registry import lr_at, PARENT_SEED, P1_STATES

OUT = ROOT / "outputs" / "p0_p1_v1_1" / "parent"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / "parent_log.txt", "w")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    torch.manual_seed(PARENT_SEED)
    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    manifest = build_step_manifest(a_train, b_train, 0.5, 2048, seed=2030)
    stream = MixtureStream(manifest, a_train, b_train)
    model = build_model(seed=PARENT_SEED).to("cuda")
    model.train()
    muon_opt, aux_opt = build_muon_arm(model, lr=0.01, weight_decay=0.02)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    targets_total = 0
    slots_total = 0
    losses = []
    for step in range(1, 2049):
        lr = lr_at(step)
        for opt in (muon_opt, aux_opt):
            for group in opt.param_groups:
                group["lr"] = lr
        for opt in (muon_opt, aux_opt):
            opt.zero_grad(set_to_none=True)
        micro = []
        for mb in stream.micro_batches(step - 1):
            batch = mb.to("cuda")
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                t = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       t.reshape(-1), reduction="mean")
            (loss / 4).backward()
            micro.append(float(loss.detach()))
            targets_total += int(t.numel())
            slots_total += int(batch.numel())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
        muon_opt.step()
        aux_opt.step()
        losses.append(float(np.mean(micro)))
        if step in (1, 32, 256, 512, 1024, 1536, 2048):
            log(f"[{time.time()-t0:.0f}s] step {step}: loss={losses[-1]:.4f} lr={lr:.6f}")
        if step in (512, 1536, 2048):
            info = save_training_state(
                OUT / f"parent_state_step{step}.pt", model, [muon_opt, aux_opt],
                cursor={"step": step, "run": "parent_seed3101"},
                meta={"recipe": "m1_muon_lr2.0_wd0.02_p0.5 template", "seed": PARENT_SEED,
                      "lr_used": lr})
            log(f"saved state at step {step}: {info['payload_sha256'][:16]}...")
    summary = {"parent_seed": PARENT_SEED, "final_loss": losses[-1],
               "targets_total": targets_total, "slots_total": slots_total,
               "targets_per_step_check": targets_total / 2048,
               "wall_seconds": time.time() - t0,
               "loss_first20": float(np.mean(losses[:20])),
               "loss_last20": float(np.mean(losses[-20:]))}
    json.dump(summary, open(OUT / "parent_summary.json", "w"), indent=1)
    log(json.dumps(summary, indent=1))
    # counters must match protocol expectations
    assert targets_total == 2048 * 32640, f"target count mismatch: {targets_total}"
    assert slots_total == 2048 * 32768
    log("counter checks passed (32640 targets / 32768 slots per update)")


if __name__ == "__main__":
    main()
