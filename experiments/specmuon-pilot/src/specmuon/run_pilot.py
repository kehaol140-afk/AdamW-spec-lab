"""SpecMuon P1 pilot runner (frozen design: configs/pilot_frozen_v1.json).

Stages:
  --stage smoke   gate 1: 2-step branch with the exact_polar arm (path check only)
  --stage replay  gate 2: native step {513,1537} replay vs R2 step1 snapshots
                         (bit-level: param maxdiff must be 0.0)
  --stage bench   gate 3: GPU cost measurement on real 56-matrix batches
  --stage pilot   frozen P1: 7 arms x 2 restore points x 256 steps

Semantics mirror run_parent.py / run_r2_amplitude.py exactly (read-only imports):
same data manifest continuation, same LR clock (lr_at), same wd, same loop ordering,
same autocast/clip. Native arms use torch.optim.Muon itself (SpecMuon(native_ns5) is
bit-equivalent by test t01; the upstream optimizer is the more faithful carrier).
Variant arms replace only the Muon orthogonalization via SpecMuon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition import config as mte_config
from muon_transition.model import build_model, unique_named_parameters
from muon_transition.mixture import MixtureStream, build_step_manifest
from muon_transition.checkpoint import parameter_fingerprint
from muon_transition.eval_nll import evaluate_q
from muon_transition.routing import route_parameters

from specmuon.spec_muon import SpecMuon

PARENT_DIR = LAB / "muon_spectral_allocation" / "outputs" / "p0_p1_v1_1" / "parent"
R2_DIR = LAB / "muon_spectral_allocation" / "outputs" / "r2_amplitude_continuation_v1"
M0_POOLS = mte_config.M0_OUTPUTS / "data_audit" / "eval_pools.pt"
OUT = ROOT / "results" / "p1"
BASE_WD = 0.02  # immutable parent wd (muon == aux), verified vs payload at load

# Frozen production LR clock (identical to muon_spectral_allocation
# src/recipe_registry.py::lr_at; asserted against R0_GATE anchors below).
_WARMUP, _COS_MIN, _TOTAL, _PEAK_LR = 32, 0.1, 2048, 0.01


def lr_at(step: int) -> float:
    import math

    if step <= _WARMUP:
        return _PEAK_LR * step / _WARMUP
    span = _TOTAL - _WARMUP
    w = (step - _WARMUP) / span
    return _PEAK_LR * (_COS_MIN + 0.45 * (1.0 + math.cos(math.pi * w)))


assert abs(lr_at(513) - 0.008793959719687522) < 1e-15, "LR clock drifted from R0_GATE anchor"
assert abs(lr_at(1537) - 0.002352918131184848) < 1e-15, "LR clock drifted from R0_GATE anchor"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_shas() -> dict:
    files = ["src/specmuon/run_pilot.py", "src/specmuon/spec_muon.py",
             "src/specmuon/operators.py", "src/specmuon/native_ns.py",
             "src/specmuon/io_utils.py"]
    return {f: file_sha(ROOT / f) for f in files}


# ---------------------------------------------------------------- restore / arms

def load_payload(step: int) -> dict:
    p = PARENT_DIR / f"parent_state_step{step}.pt"
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert payload.get("format") == "mte_state_v1"
    assert payload["cursor"]["step"] == step
    wd_muon = payload["optimizer_states"][0]["param_groups"][0]["weight_decay"]
    wd_aux = payload["optimizer_states"][1]["param_groups"][0]["weight_decay"]
    assert abs(wd_muon - BASE_WD) < 1e-15 and abs(wd_aux - BASE_WD) < 1e-15
    return payload


def build_arm(model, arm_cfg, device):
    """Return (muon_opt, aux_opt) per arm. Native arms: torch.optim.Muon.
    Variant arms: SpecMuon with the frozen spectral operator."""
    muon_params, aux_params, _ = route_parameters(model)
    m = mte_config.MUON_FREEZE
    name = arm_cfg["optimizer"]
    if name == "torch_muon":
        from torch.optim import Muon

        muon_opt = Muon(
            muon_params, lr=1e-9, weight_decay=0.0, momentum=m["momentum"],
            nesterov=m["nesterov"], ns_steps=m["ns_steps"],
            ns_coefficients=tuple(m["ns_coefficients"]), eps=m["eps"],
            adjust_lr_fn=m["adjust_lr_fn"],
        )
    elif name == "SpecMuon":
        muon_opt = SpecMuon(
            muon_params, lr=1e-9, weight_decay=0.0, momentum=m["momentum"],
            nesterov=m["nesterov"], ns_steps=m["ns_steps"],
            ns_coefficients=tuple(m["ns_coefficients"]), eps=m["eps"],
            adjust_lr_fn=m["adjust_lr_fn"],
            spectral_op=arm_cfg["spectral_op"],
            lam=arm_cfg.get("lam", 0.25), power=arm_cfg.get("power", 0.5),
        )
    else:
        raise ValueError(name)
    aux_opt = torch.optim.AdamW(aux_params, lr=1e-9,
                                betas=mte_config.ADAMW_BETAS,
                                eps=mte_config.ADAMW_EPS, weight_decay=0.0)
    return muon_opt, aux_opt


def parse_arm(arm: dict) -> dict:
    """Map the frozen config's optimizer spec to a build_arm spec.

    "SpecMuon(native_ns5)" is realized by torch.optim.Muon itself: test t01 proves
    SpecMuon(native_ns5) is bit-identical, and the upstream optimizer is the more
    faithful native carrier (it is exactly what produced the parent/R2 runs).
    """
    spec = arm["optimizer"]
    out = {"optimizer": None}
    if spec == "SpecMuon(native_ns5)":
        out["optimizer"] = "torch_muon"
    elif spec.startswith("SpecMuon(") and spec.endswith(")"):
        out["optimizer"] = "SpecMuon"
        out["spectral_op"] = spec[len("SpecMuon("):-1]
    else:
        raise ValueError(f"unknown frozen arm spec {spec!r}")
    return out


def normalize_arm(arm_cfg: dict) -> dict:
    spec = arm_cfg.get("optimizer")
    if isinstance(spec, str) and spec.startswith("SpecMuon("):
        merged = dict(parse_arm(arm_cfg))
        merged.update({k: v for k, v in arm_cfg.items() if k != "optimizer"})
        return merged
    return arm_cfg


def load_full(payload, arm_cfg, device):
    arm_cfg = normalize_arm(arm_cfg)
    model = build_model().to(device)
    model.load_state_dict(payload["model_state"])
    model.train()
    muon_opt, aux_opt = build_arm(model, arm_cfg, device)
    muon_opt.load_state_dict(payload["optimizer_states"][0])
    aux_opt.load_state_dict(payload["optimizer_states"][1])
    # R2 hardening: clone state tensors (AdamW step scalars alias the payload)
    for opt in (muon_opt, aux_opt):
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.clone()
    # re-assert the arm's spectral fields (load_state_dict may drop non-saved keys)
    if isinstance(muon_opt, SpecMuon):
        for g in muon_opt.param_groups:
            g["spectral_op"] = arm_cfg["spectral_op"]
            g["lam"] = arm_cfg.get("lam", 0.25)
            g["power"] = arm_cfg.get("power", 0.5)
    from vendored_fr.rng import restore_rng_state

    restore_rng_state(payload["rng"])
    return model, [muon_opt, aux_opt]


# ---------------------------------------------------------------- data / eval

_DATA = {}


def get_stream(step: int, n_cont: int):
    if "windows" not in _DATA:
        from muon_transition.m1_data_cache import get_domain_windows

        _DATA["windows"] = get_domain_windows()
    a_train, b_train = _DATA["windows"]
    m = build_step_manifest(a_train, b_train, 0.5, step + n_cont, seed=2030)
    return MixtureStream({"steps": m["steps"][step:step + n_cont]}, a_train, b_train)


def get_dev_pools():
    if "pools" not in _DATA:
        pools = torch.load(M0_POOLS, weights_only=True)
        _DATA["pools"] = {"A": pools["A_dev"], "B": pools["B_dev"]}
    return _DATA["pools"]


def init_eval(step: int, payload, arm_cfg, device):
    """Eval the restored (pre-branch) model once per restore point; cached."""
    cache = OUT / f"init_eval_s{step}.json"
    if cache.exists():
        return json.load(open(cache))
    model, opts = load_full(payload, arm_cfg, device)
    model.eval()
    q = evaluate_q(model, get_dev_pools(), device=device)
    json.dump(q, open(cache, "w"), indent=1)
    return q


# ---------------------------------------------------------------- branch core

def run_branch(cfg, arm, restore_step, device, n_steps=256, log_every=32,
               ckpt_every=64, keep_ckpt=False):
    arm_cfg = next(a for a in cfg["arms"] if a["name"] == arm)
    run_id = f"s{restore_step}_{arm}"
    bdir = OUT / "branches" / run_id
    bdir.mkdir(parents=True, exist_ok=True)
    payload = _DATA.setdefault(f"payload{restore_step}", load_payload(restore_step))

    stream = get_stream(restore_step, n_steps)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    lr_mult = arm_cfg.get("lr_mult", 1.0)

    meta = {
        "run_id": run_id, "arm": arm, "restore_step": restore_step,
        "n_steps": n_steps, "device": device, "lr_mult": lr_mult,
        "parent_ckpt_sha256": file_sha(PARENT_DIR / f"parent_state_step{restore_step}.pt"),
        "code_shas": code_shas(),
        "config_sha256": file_sha(ROOT / "configs" / "pilot_frozen_v1.json"),
        "torch_version": torch.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    metrics_path = bdir / "train_metrics.jsonl"
    ckpt_path = bdir / "ckpt_last.pt"

    start_k = 0
    model = opts = None
    norm_cfg = normalize_arm(arm_cfg)
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert st["run_id"] == run_id
        model, opts = load_full(payload, arm_cfg, device)
        model.load_state_dict(st["model_state"])
        for opt, os_ in zip(opts, st["optimizer_states"]):
            opt.load_state_dict(os_)
            for s in opt.state.values():
                for k, v in list(s.items()):
                    if torch.is_tensor(v):
                        s[k] = v.clone()
        if isinstance(opts[0], SpecMuon):
            for g in opts[0].param_groups:
                g["spectral_op"] = norm_cfg["spectral_op"]
                g["lam"] = norm_cfg.get("lam", 0.25)
                g["power"] = norm_cfg.get("power", 0.5)
        from vendored_fr.rng import restore_rng_state

        restore_rng_state(st["rng"])
        start_k = int(st["cursor"]["rel_step"])
        print(f"[{run_id}] resumed from rel_step {start_k}", flush=True)
    if model is None:
        model, opts = load_full(payload, arm_cfg, device)

    if start_k == 0:
        meta["param_fingerprint_restore"] = parameter_fingerprint(model)
        json.dump(meta, open(bdir / "branch_meta.json", "w"), indent=1)
    else:
        meta = json.load(open(bdir / "branch_meta.json"))

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    t_branch = time.perf_counter()
    fin = {"nonfinite_loss": False, "nonfinite_grad": False, "resume_fail": False}

    with open(metrics_path, "a") as mf:
        for k in range(start_k + 1, n_steps + 1):
            gstep = restore_step + k
            lr = lr_at(gstep) * lr_mult
            for opt in opts:
                for group in opt.param_groups:
                    group["lr"] = lr
                    group["weight_decay"] = BASE_WD
                opt.zero_grad(set_to_none=True)
            micro = []
            for mb in stream.micro_batches(k - 1):
                batch = mb.to(device)
                with autocast:
                    logits = model(batch).logits[..., :-1, :].contiguous()
                    t = batch[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        logits.float().reshape(-1, logits.shape[-1]),
                        t.reshape(-1), reduction="mean")
                (loss / mte_config.GRAD_ACCUM_STEPS).backward()
                micro.append(float(loss.detach()))
            preclip = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.grad is not None],
                mte_config.GRAD_CLIP_NORM)
            if not torch.isfinite(preclip):
                fin["nonfinite_grad"] = True
            t_opt0 = time.perf_counter()
            opts[0].step()
            t_muon = time.perf_counter() - t_opt0
            t_aux0 = time.perf_counter()
            opts[1].step()
            t_aux = time.perf_counter() - t_aux0
            train_loss = float(np.mean(micro))
            if not np.isfinite(train_loss):
                fin["nonfinite_loss"] = True
            st_extra = {}
            if isinstance(opts[0], SpecMuon) and getattr(opts[0], "last_stats", None):
                st_extra = {kk: opts[0].last_stats[kk] for kk in
                            ("matrices", "native_norm_sum", "max_rel_norm_err",
                             "t_native_probe_s", "t_svd_s")}
            rec = {"run_id": run_id, "rel_step": k, "global_step": gstep,
                   "lr": lr, "train_loss": train_loss,
                   "preclip_norm": float(preclip),
                   "t_muon_opt_s": t_muon, "t_aux_opt_s": t_aux,
                   **({"op": st_extra} if st_extra else {})}
            mf.write(json.dumps(rec) + "\n")
            if k % log_every == 0 or k == 1 or k == n_steps:
                mf.flush()
                print(f"[{run_id}] step {k}/{n_steps} loss={train_loss:.4f} "
                      f"lr={lr:.6f}", flush=True)
            if k % ckpt_every == 0 and k < n_steps:
                from vendored_fr.rng import capture_rng_state

                tmp = bdir / "ckpt_last.tmp"
                torch.save({"run_id": run_id, "model_state": model.state_dict(),
                            "optimizer_states": [o.state_dict() for o in opts],
                            "rng": capture_rng_state(),
                            "cursor": {"rel_step": k}}, tmp)
                tmp.rename(ckpt_path)

    wall = time.perf_counter() - t_branch
    model.eval()
    q = evaluate_q(model, get_dev_pools(), device=device)
    model.train()
    peak_mem = (torch.cuda.max_memory_allocated(device) / 2**20
                if device.startswith("cuda") else 0.0)
    result = {
        **meta, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": wall, "gpu_peak_mib": peak_mem,
        "param_fingerprint_final": parameter_fingerprint(model),
        "final_eval": q, **fin,
    }
    json.dump(result, open(bdir / "result.json", "w"), indent=1)
    if not keep_ckpt and ckpt_path.exists():
        ckpt_path.unlink()
    return result


# ---------------------------------------------------------------- gates

def stage_replay(cfg, device):
    """Native step {513,1537} must reproduce R2's step1 snapshots bit-exactly."""
    out = {}
    for step in (512, 1536):
        payload = load_payload(step)
        native_cfg = {"optimizer": "torch_muon"}
        model, opts = load_full(payload, native_cfg, device)
        stream = get_stream(step, 1)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        k = 1
        lr = lr_at(step + k)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = lr
                group["weight_decay"] = BASE_WD
            opt.zero_grad(set_to_none=True)
        micro = []
        for mb in stream.micro_batches(k - 1):
            batch = mb.to(device)
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
        ours = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
        ref = torch.load(R2_DIR / f"s{step}_N_step1_model.pt",
                         map_location="cpu", weights_only=False)["model_state"]
        keys_ok = sorted(ours) == sorted(ref)
        maxdiff = 0.0
        if keys_ok:
            for n in ours:
                maxdiff = max(maxdiff, float((ours[n].float() - ref[n].float()).abs().max()))
        out[f"s{step}"] = {
            "keys_match": keys_ok, "param_maxdiff_vs_r2_step1": maxdiff,
            "first_step_loss": float(np.mean(micro)),
            "first_step_preclip_norm": float(preclip), "lr_used": lr,
            "bit_exact": keys_ok and maxdiff == 0.0,
        }
        print(json.dumps(out[f"s{step}"], indent=1), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "replay_gate.json", "w"), indent=1)
    ok = all(v["bit_exact"] for v in out.values())
    print(f"REPLAY GATE: {'PASS' if ok else 'FAIL'}")
    return ok


def stage_smoke(cfg, device):
    """2-step exact_polar branch: exercises SVD path, norm logging, eval path."""
    payload = load_payload(512)
    model, opts = load_full(payload, {"optimizer": "SpecMuon", "spectral_op": "exact_polar"}, device)
    stream = get_stream(512, 2)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    losses, maxerr = [], 0.0
    for k in (1, 2):
        lr = lr_at(512 + k)
        for opt in opts:
            for group in opt.param_groups:
                group["lr"] = lr
                group["weight_decay"] = BASE_WD
            opt.zero_grad(set_to_none=True)
        micro = []
        for mb in stream.micro_batches(k - 1):
            batch = mb.to(device)
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                t = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       t.reshape(-1), reduction="mean")
            (loss / 4).backward()
            micro.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0)
        opts[0].step()
        opts[1].step()
        losses.append(float(np.mean(micro)))
        s = opts[0].last_stats
        maxerr = max(maxerr, s["max_rel_norm_err"])
        assert s["matrices"] == 56, f"expected 56 matrices, got {s['matrices']}"
        assert np.isfinite(losses[-1]), "non-finite loss in smoke"
        assert s["max_rel_norm_err"] <= 1e-5, f"norm err {s['max_rel_norm_err']:.2e}"
        print(f"[smoke] step {k}: loss={losses[-1]:.4f} matrices={s['matrices']} "
              f"norm_err={s['max_rel_norm_err']:.2e} t_svd={s['t_svd_s']*1e3:.0f}ms", flush=True)
    pools = get_dev_pools()
    model.eval()
    q = evaluate_q(model, {"A": pools["A"][:64], "B": pools["B"][:64]}, device=device)
    assert np.isfinite(q["v_q"])
    print(f"[smoke] eval path OK (subset v_q={q['v_q']:.4f})")
    print("SMOKE GATE: PASS")


def stage_bench(cfg, device):
    """GPU cost measurement on real 56-matrix batches (Phase 4)."""
    from specmuon.native_ns import vendored_zeropower_via_newtonschulz
    from specmuon.operators import svd_basis, native_update
    from specmuon.operators import exact_polar, head_spec_muon, bulk_spec_muon, muon_power

    payload = load_payload(512)
    opt_state = payload["optimizer_states"][0]
    names = [e["name"] for e in json.load(open(
        mte_config.M0_OUTPUTS / "routing_report.json"))["routing_table"]
        if e["route"] == "muon"]
    idxs = opt_state["param_groups"][0]["params"]
    mats = [opt_state["state"][i]["momentum_buffer"].to(device).float()
            for i in idxs]
    assert len(mats) == 56

    def time_batch(fn, warmup=2, reps=3):
        for _ in range(warmup):
            fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            ts.append(time.perf_counter() - t0)
        return {"mean_s": float(np.mean(ts)), "min_s": float(np.min(ts))}

    res = {}
    res["native_ns5_56"] = time_batch(
        lambda: [vendored_zeropower_via_newtonschulz(M) for M in mats])

    def run_op(opname, fn_of_M):
        def f():
            ns = []
            for M in mats:
                Y5 = vendored_zeropower_via_newtonschulz(M)
                T = float(Y5.float().norm())
                ns.append(fn_of_M(M, T))
            return ns
        return f

    res["exact_polar_56"] = time_batch(run_op("exact_polar", lambda M, T: exact_polar(M, target_norm=T)["Y"]))
    res["head_56"] = time_batch(run_op("head", lambda M, T: head_spec_muon(M, lam=0.25, target_norm=T)["Y"]))
    res["bulk_56"] = time_batch(run_op("bulk", lambda M, T: bulk_spec_muon(M, lam=0.25, target_norm=T)["Y"]))
    res["muon_p_56"] = time_batch(run_op("muon_p", lambda M, T: muon_power(M, p=0.5, target_norm=T)["Y"]))

    # informational: fp32-SVD variant and batched fp64 SVD (not the frozen path)
    def fp32_variant(M, T):
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        P = (U.to(torch.float64) @ Vh.to(torch.float64))
        n = P.norm()
        return P * (T / n)

    res["info_fp32_svd_polar_56"] = time_batch(run_op("fp32", fp32_variant))

    def batched_svd():
        by_shape = {}
        for M in mats:
            by_shape.setdefault(tuple(M.shape), []).append(M)
        out = []
        for shape, group in by_shape.items():
            stack = torch.stack(group)
            U, S, Vh = torch.linalg.svd(stack.to(torch.float64), full_matrices=False)
            out.append((U @ Vh).to(torch.float32))
        return out

    res["info_batched_fp64_svd_56"] = time_batch(batched_svd)

    # end-to-end optimizer-only step cost: native vs variant on the real model
    def opt_step_cost(arm_cfg, reps=3):
        model, opts = load_full(payload, arm_cfg, device)
        muon_params = opts[0].param_groups[0]["params"]
        # warmup 1: random grads create optimizer state (buffers) if absent
        gcpu = torch.Generator().manual_seed(0)
        for p_ in muon_params:
            p_.grad = torch.randn(p_.shape, generator=gcpu).to(device) * 0.01
        for group in opts[0].param_groups:
            group["lr"] = 0.005
            group["weight_decay"] = BASE_WD
        opts[0].step()
        # from now on use the real momentum buffers as deterministic grad stand-ins
        def set_grads():
            for p_ in muon_params:
                p_.grad = opts[0].state[p_]["momentum_buffer"].clone()
        set_grads()
        opts[0].step()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        ts = []
        for _ in range(reps):
            set_grads()
            t0 = time.perf_counter()
            opts[0].step()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            ts.append(time.perf_counter() - t0)
        return {"mean_s": float(np.mean(ts)), "min_s": float(np.min(ts))}

    res["opt_step_native_torch_muon"] = opt_step_cost({"optimizer": "torch_muon"})
    for arm in ("exact_polar", "head", "bulk", "muon_p"):
        res[f"opt_step_specmuon_{arm}"] = opt_step_cost(
            {"optimizer": "SpecMuon", "spectral_op": arm})

    # peak memory around a variant step
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
        model, opts = load_full(payload, {"optimizer": "SpecMuon", "spectral_op": "exact_polar"}, device)
        for n_, p_ in zip(names, opts[0].param_groups[0]["params"]):
            p_.grad = opts[0].state[p_]["momentum_buffer"].clone()
        opts[0].step()
        torch.cuda.synchronize(device)
        res["peak_mem_variant_step_mib"] = torch.cuda.max_memory_allocated(device) / 2**20

    json.dump(res, open(OUT / "bench_gpu.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pilot_frozen_v1.json")
    ap.add_argument("--stage", required=True,
                    choices=["smoke", "replay", "bench", "pilot"])
    ap.add_argument("--restore-point", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--threads", type=int, default=1,
                    help="torch intra-op threads; SVD parallelism comes from the "
                         "16-worker CPU pool inside SpecMuon (deterministic)")
    args = ap.parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = json.load(open(cfg_path))
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "smoke":
        stage_smoke(cfg, args.device)
    elif args.stage == "replay":
        ok = stage_replay(cfg, args.device)
        sys.exit(0 if ok else 2)
    elif args.stage == "bench":
        stage_bench(cfg, args.device)
    elif args.stage == "pilot":
        points = ([args.restore_point] if args.restore_point
                  else [rp["step"] for rp in cfg["parent"]["restore_points"]])
        arms = [a["name"] for a in cfg["arms"]]
        results = []
        for step in points:
            payload = load_payload(step)
            q0 = init_eval(step, payload, cfg["arms"][0], args.device)
            print(f"[pilot] init eval s{step}: v_q={q0['v_q']:.6f}", flush=True)
            del payload
            for arm in arms:
                arm_cfg = next(a for a in cfg["arms"] if a["name"] == arm)
                r = run_branch(cfg, arm, step, args.device)
                results.append(r)
                print(f"[pilot] DONE {r['run_id']}: v_q={r['final_eval']['v_q']:.6f} "
                      f"wall={r['wall_seconds']:.0f}s", flush=True)
        with open(OUT / "P1_INDEX.json", "w") as f:
            json.dump([{k: r[k] for k in ("run_id", "arm", "restore_step",
                                          "wall_seconds", "gpu_peak_mib")}
                       for r in results], f, indent=1)


if __name__ == "__main__":
    main()
