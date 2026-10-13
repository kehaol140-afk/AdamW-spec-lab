"""R2 three-arm 256-step real continuation (2 restore points x {N, MH, AH}).

Arms scale DATA updates only: effective_lr = r*base_lr, effective_wd = base_wd/r
(invariant decay product), recomputed from the original clock every step. Native
production path (opt.step) is used throughout; grads unscaled; global clip; monitor
on Q_select at relative steps 64/128/256; save step1 snapshot + step256 full state.
Includes per-point first-step verification and (budget permitting) one native repeat.
"""
from __future__ import annotations

import csv
import hashlib
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
from vendored_fr.rng import restore_rng_state
from src.recipe_registry import lr_at, group_of_param, TOTAL
from src.state_restore import eval_V, build_q_sets
from src.r2_data_scale import ARMS, effective_lr_wd

OUT = ROOT / "outputs" / "r2_amplitude_continuation_v1"
OUT.mkdir(parents=True, exist_ok=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_full(payload, step):
    """Restore model + both optimizers + RNG; NO lr write-back here (each step sets
    effective lr/wd from the base clock)."""
    model = build_model().to("cuda")
    model.load_state_dict(payload["model_state"])
    model.train()
    muon_opt, aux_opt = build_muon_arm(model, lr=1e-9, weight_decay=0.0)
    muon_opt.load_state_dict(payload["optimizer_states"][0])
    aux_opt.load_state_dict(payload["optimizer_states"][1])
    for opt in (muon_opt, aux_opt):
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.clone()
    restore_rng_state(payload["rng"])
    return model, [muon_opt, aux_opt]


def run_arm(payload, step, arm, stream, q, run_id, logf, monitor=True):
    """One 256-step real continuation of `arm` from restore point `step`."""
    t0 = time.time()
    model, opts = load_full(payload, step)
    base_wd = payload["optimizer_states"][0]["param_groups"][0]["weight_decay"]
    aux_base_wd = payload["optimizer_states"][1]["param_groups"][0]["weight_decay"]
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    r_m, r_a = ARMS[arm]["muon"], ARMS[arm]["aux"]
    metrics, monitors = [], []
    theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}  # after load
    first_step_info = None

    def train_mode_guard():
        assert model.training, "model must be in train mode during continuation"

    for k in range(1, 257):
        gstep = step + k
        base_lr = lr_at(gstep)
        for opt, r, wd in ((opts[0], r_m, base_wd), (opts[1], r_a, aux_base_wd)):
            eff_lr, eff_wd = effective_lr_wd(base_lr, wd, r)
            for group in opt.param_groups:
                group["lr"] = eff_lr
                group["weight_decay"] = eff_wd
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
        metrics.append({"run_id": run_id, "rel_step": k, "global_step": gstep,
                        "base_lr": base_lr, "eff_lr_muon": r_m * base_lr,
                        "eff_lr_aux": r_a * base_lr, "base_wd": base_wd,
                        "eff_wd_muon": base_wd / r_m, "eff_wd_aux": aux_base_wd / r_a,
                        "preclip_norm": float(preclip), "train_loss": float(np.mean(micro))})
        if k == 1:
            first_step_info = {n: p.detach().clone() for n, p in model.named_parameters()}
        if monitor and k in (64, 128, 256):
            res = eval_V(model, q, "select")
            train_mode_guard()
            monitors.append({"run_id": run_id, "rel_step": k, **res})
    # save step1 snapshot (model state incl. buffers) and step256 full state
    torch.save({"model_state": {n: p.detach().cpu() for n, p in first_step_info.items()},
                "meta": {"run_id": run_id, "step": 1}},
               OUT / f"{run_id}_step1_model.pt")
    torch.save({"model_state": model.state_dict(),
                "optimizer_states": [o.state_dict() for o in opts],
                "rng": __import__("vendored_fr.rng", fromlist=["capture_rng_state"]).capture_rng_state(),
                "cursor": {"step": step + 256, "run_id": run_id},
                "meta": {"arm": arm, "restore_step": step}},
               OUT / f"{run_id}_step256_full.pt")
    return metrics, monitors, first_step_info, theta_star, model


def first_step_verification(payload, step, stream, q, arms_first, theta_star):
    """Protocol 5.2: N must match an independent native reference bit-level; MH/AH
    data deltas must scale by r per group (FP64); moments identical on same input."""
    ref_model, ref_opts = load_full(payload, step)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    base_lr = lr_at(step + 1)
    for opt in ref_opts:
        for group in opt.param_groups:
            group["lr"] = base_lr
    for p in ref_model.parameters():
        p.grad = None
    for mb in stream.micro_batches(0):
        batch = mb.to("cuda")
        with autocast:
            logits = ref_model(batch).logits[..., :-1, :].contiguous()
            t = batch[..., 1:].contiguous()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   t.reshape(-1), reduction="mean")
        (loss / 4).backward()
    torch.nn.utils.clip_grad_norm_([p for p in ref_model.parameters() if p.grad is not None], 1.0)
    ref_opts[0].step()
    ref_opts[1].step()
    ref_params = {n: p.detach().clone() for n, p in ref_model.named_parameters()}
    out = {}
    n_max = 0.0
    for n, p in arms_first["N"].items():
        n_max = max(n_max, float((p - ref_params[n]).abs().max()))
    out["N_vs_independent_reference_maxdiff"] = n_max
    # common WD base and per-group data-delta ratios
    base_wd = payload["optimizer_states"][0]["param_groups"][0]["weight_decay"]
    aux_wd = payload["optimizer_states"][1]["param_groups"][0]["weight_decay"]
    for arm in ("MH", "AH"):
        ratios = {}
        for grp_name, is_muon in (("muon", True), ("aux", False)):
            wd = base_wd if is_muon else aux_wd
            r = ARMS[arm]["muon" if is_muon else "aux"]
            num = den = 0.0
            for n in ref_params:
                g = group_of_param(n)
                if (g is not None) != is_muon:
                    continue
                base = theta_star[n] * (1 - base_lr * wd)
                d_arm = (arms_first[arm][n].double() - base.double())
                d_nat = (arms_first["N"][n].double() - base.double())
                num += float(d_arm.pow(2).sum())
                den += float(d_nat.pow(2).sum())
            ratios[grp_name] = (num ** 0.5) / (den ** 0.5) if den > 0 else None
        out[f"{arm}_data_delta_ratio"] = ratios
    return out, ref_params


def main():
    t0 = time.time()
    logf = open(OUT / "run_log.txt", "a")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    q = build_q_sets()
    from muon_transition.m1_data_cache import get_domain_windows
    a_train, b_train = get_domain_windows()
    all_metrics, all_monitors, run_index = [], [], []
    # resume support: a run whose step256 full state exists on disk is skipped
    import os
    done_runs = {f.name.rsplit("_step256_full.pt", 1)[0]
                 for f in OUT.glob("*_step256_full.pt")}
    if done_runs:
        # reload prior run_index to preserve records
        if (OUT / "run_index.json").exists():
            prior = json.loads((OUT / "run_index.json").read_text())
            run_index = [r for r in prior if r["run_id"] in done_runs]
            log(f"resuming; completed runs on disk: {sorted(done_runs)}")
    for step in (512, 1536):
        p = ROOT / "outputs" / "p0_p1_v1_1" / "parent" / f"parent_state_step{step}.pt"
        payload = torch.load(p, map_location="cpu", weights_only=False)
        m = build_step_manifest(a_train, b_train, 0.5, step + 256, seed=2030)
        stream = MixtureStream({"steps": m["steps"][step:step + 256]}, a_train, b_train)
        arms_first = {}
        theta_star = None
        for arm in ("N", "MH", "AH"):
            run_id = f"s{step}_{arm}"
            if run_id in done_runs:
                # reload step1 snapshot for verification reuse; rebuild theta_star once
                snap = torch.load(OUT / f"{run_id}_step1_model.pt", map_location="cpu", weights_only=False)
                arms_first[arm] = {k: v.cuda() for k, v in snap["model_state"].items()}
                if theta_star is None:
                    theta_star = {k: v.cuda() for k, v in payload["model_state"].items()}
                continue
            metrics, monitors, first_info, theta_star, model = run_arm(
                payload, step, arm, stream, q, run_id, logf)
            arms_first[arm] = first_info
            all_metrics.extend(metrics)
            all_monitors.extend(monitors)
            run_index.append({"run_id": run_id, "restore_step": step, "arm": arm,
                              "steps": 256, "status": "complete",
                              "final_loss": metrics[-1]["train_loss"]})
            log(f"[{time.time()-t0:.0f}s] {run_id} done: final loss={metrics[-1]['train_loss']:.4f} "
                f"monitor V256={monitors[-1]['V']:.6f}")
            del model, first_info
            torch.cuda.empty_cache()
        if len(arms_first) == 3:
            ver, ref_params = first_step_verification(payload, step, stream, q, arms_first, theta_star)
            json.dump(ver, open(OUT / f"preflight_s{step}.json", "w"), indent=1)
            log(f"s{step} first-step verification: {json.dumps(ver)}")
            del ref_params
        del arms_first, theta_star
        torch.cuda.empty_cache()
    # native repeat (budget permitting; skip if already on disk)
    if "s512_N_repeat" not in done_runs:
        p = ROOT / "outputs" / "p0_p1_v1_1" / "parent" / "parent_state_step512.pt"
        payload = torch.load(p, map_location="cpu", weights_only=False)
        m = build_step_manifest(a_train, b_train, 0.5, 512 + 256, seed=2030)
        stream = MixtureStream({"steps": m["steps"][512:512 + 256]}, a_train, b_train)
        metrics, monitors, _, _, model = run_arm(payload, 512, "N", stream, q, "s512_N_repeat", logf)
        del model
        torch.cuda.empty_cache()
        rep = torch.load(OUT / "s512_N_step256_full.pt", map_location="cpu", weights_only=False)
        rep2 = torch.load(OUT / "s512_N_repeat_step256_full.pt", map_location="cpu", weights_only=False)
        same = all(torch.equal(rep["model_state"][k], rep2["model_state"][k]) for k in rep["model_state"])
        run_index.append({"run_id": "s512_N_repeat", "restore_step": 512, "arm": "N",
                          "steps": 256, "status": "complete",
                          "bitwise_equal_to_first_N": same})
        all_metrics.extend(metrics)
        all_monitors.extend(monitors)
        log(f"[{time.time()-t0:.0f}s] native repeat done; bitwise equal to first N: {same}")
        del rep, rep2

    with open(OUT / "training_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        w.writeheader(); w.writerows(all_metrics)
    with open(OUT / "monitor.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_monitors[0].keys()))
        w.writeheader(); w.writerows(all_monitors)
    json.dump(run_index, open(OUT / "run_index.json", "w"), indent=1)
    json.dump({"gpu_seconds": time.time() - t0}, open(OUT / "run_meta.json", "w"), indent=1)
    log(f"R2 continuation done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
