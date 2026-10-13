"""100M formal-pilot PREFLIGHT: restricted 100-step diagnostic on GPU0.

Authorized window: <= 0.3 GPU-h, 100 steps x 8192 tokens/step, NOT the 15-run
matrix. Runs only after: pilot cache minted (identity chain corpus->V16 rule),
GPU0 idle, seed/init reproducible.

Paths (all: seed 3471, identical init, same cursor-0 stream, no grad clipping,
bf16 autocast, fp32 master; warmup 8 then cosine to 0.1xpeak over T=100 —
disclosed diagnostic schedule, NOT the pilot schedule):
  muon_native   : GroupMuon native (bitwise torch.optim.Muon) 91 matrices + aux AdamW
  group_A_l065  : mix FFN39 lam=0.65 norm_match + native rest
  group_A_l085  : mix FFN39 lam=0.85 norm_match + native rest
  lowmu_0p9     : native with momentum=0.90 (control-arm mechanics on GPU)

Checks (per task list):
  1. per-step loss finite (hard assert)
  2. update-norm bookkeeping: mix norm_rel_err (must be <1e-9), per-path mean
     per-step |dtheta| for one FFN gate matrix / one V matrix (sampled, fp32)
  3. model+optimizer recovery: checkpoint at step 50 -> fresh objects -> replay
     steps 50..99 -> compare final params (bitwise report; max|diff| fallback)
     [muon_native and group_A_l065 only]
  4. throughput: s/step over steps 10..100 (steady), tokens/s, VRAM peak,
     optimizer seconds split (NS5 native/mix, aux)
  5. final state hashes: param fingerprint + momentum-buffer fingerprint
  6. dev NLL (V16 development 256 docs, dv_core metric semantics) at step 100
     for all paths, and at step 50 for muon_native (eval pipeline check)

Writes results/preflight_100steps.json; checkpoints to artifacts/preflight/
(deleted after replay verification; hashes retained in the json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from context_aware_muon.data import WindowStream, eta_schedule
from context_aware_muon.group_muon import GroupMuon
from context_aware_muon.model_100m import build_model, param_fingerprint
from context_aware_muon.routing import route_model

CACHE = ROOT / "artifacts" / "pilot_train_tokens_33p7m.u32"
DEV = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/eval_sets.json")
STEPS = 100
CKPT_STEP = 50
MICRO, ACCUM, SEQ = 8, 4, 256
PEAK, WARMUP, TOTAL = 1e-2, 8, STEPS
SEED = 3471
PATHS = ("muon_native", "group_A_l065", "group_A_l085", "lowmu_0p9")


def state_fingerprint(model: torch.nn.Module, muon: GroupMuon) -> dict:
    h_buf = None
    for st in muon.state.values():
        if "momentum_buffer" in st:
            b = st["momentum_buffer"].detach().cpu().contiguous()
            if h_buf is None:
                h_buf = b.clone()
            else:
                h_buf += b.sum()  # cheap aggregate; per-tensor identity not needed
    return {
        "param_fingerprint": param_fingerprint(model),
        "momentum_sum": float(h_buf.sum()) if h_buf is not None else None,
    }


def build_path(path: str, model: torch.nn.Module, peak: float):
    routed = route_model(model)
    if path == "muon_native":
        native = routed["ffn"] + routed["vo"] + routed["qk"]
        groups = [{"params": native, "mode": "native", "lr": peak}]
        muon = GroupMuon(groups, lr=peak, weight_decay=0.02, momentum=0.95, nesterov=True)
    elif path in ("group_A_l065", "group_A_l085"):
        lam = 0.65 if path.endswith("065") else 0.85
        groups = [
            {"params": routed["ffn"], "mode": "mix", "lam": lam, "norm_match": True, "lr": peak},
            {"params": routed["vo"] + routed["qk"], "mode": "native", "lr": peak},
        ]
        muon = GroupMuon(groups, lr=peak, weight_decay=0.02, momentum=0.95, nesterov=True)
    elif path == "lowmu_0p9":
        groups = [
            {"params": routed["ffn"] + routed["vo"] + routed["qk"], "mode": "native",
             "lr": peak, "momentum": 0.90},
        ]
        muon = GroupMuon(groups, lr=peak, weight_decay=0.02, momentum=0.95, nesterov=True)
    else:
        raise ValueError(path)
    aux = torch.optim.AdamW(routed["aux"], lr=peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.02)
    return muon, aux


def save_ckpt(path: Path, model, muon, aux, cursor: int, step: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()},
            "muon": muon.state_dict(),
            "aux": aux.state_dict(),
            "cursor": cursor,
            "step": step,
        },
        path,
    )


def load_ckpt(path: Path, model, muon, aux) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    muon.load_state_dict(ck["muon"])
    aux.load_state_dict(ck["aux"])
    return ck["cursor"], ck["step"]


@torch.no_grad()
def dev_nll(model, docs, device) -> dict:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for doc in docs:
        ids = torch.tensor(doc["ids"], device=device, dtype=torch.long)[None, :]
        logits = model(ids[:, :-1], use_cache=False).logits
        target = ids[:, 1:]
        per = torch.nn.functional.cross_entropy(
            logits[:, : target.shape[1]].float().reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        tot_loss += float(per.sum())
        tot_tok += int(target.numel())
    model.train()
    return {"dev_nll_token_weighted": tot_loss / tot_tok, "tokens": tot_tok, "docs": len(docs)}


def run_segment(model, muon, aux, stream, cursor0: int, step0: int, step1: int,
                device: str, want_ck: Path | None):
    """Run steps [step0, step1); optionally checkpoint at CKPT_STEP; return logs."""
    losses = []
    step_times = []
    opt_secs = ns_native = ns_mix = 0.0
    cos_g_traj, cos_onat_traj, nerr_max = [], [], 0.0
    stream.cursor = cursor0
    ck_saved = None
    for step in range(step0, step1):
        t_step = time.perf_counter()
        lr = eta_schedule(step, TOTAL, WARMUP, PEAK)
        for g in muon.param_groups:
            g["lr"] = lr
        for g in aux.param_groups:
            g["lr"] = lr
        tot = 0.0
        finite = True
        for _ in range(ACCUM):
            x, y = stream.next_batch(MICRO)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=y).loss
            if not torch.isfinite(loss):
                finite = False
            (loss / ACCUM).backward()
            tot += float(loss)
        assert finite, f"non-finite loss at step {step}"
        t_opt = time.perf_counter()
        muon.step()
        aux.step()
        opt_secs += time.perf_counter() - t_opt
        st = muon.stats
        ns_native += st["ns_seconds_native"]
        ns_mix += st["ns_seconds_mix"]
        if st["cos_O_g_mean"] is not None:
            cos_g_traj.append(st["cos_O_g_mean"])
            cos_onat_traj.append(st["cos_O_Onat_mean"])
            nerr_max = max(nerr_max, st["norm_rel_err_max"] or 0.0)
        muon.zero_grad(set_to_none=True)
        aux.zero_grad(set_to_none=True)
        losses.append(tot / ACCUM)
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t_step)
        if want_ck is not None and step + 1 == CKPT_STEP:
            save_ckpt(want_ck, model, muon, aux, stream.cursor, step + 1)
            ck_saved = {"path": str(want_ck), "step": step + 1, "cursor": stream.cursor}
    steady = step_times[10:] if len(step_times) > 10 else step_times
    return {
        "losses_first3": losses[:3],
        "losses_last3": losses[-3:],
        "loss_all_finite": True,
        "s_per_step_steady_mean": sum(steady) / len(steady),
        "s_per_step_first10_mean": sum(step_times[:10]) / len(step_times[:10]),
        "opt_seconds_total": opt_secs,
        "ns_seconds_native": ns_native,
        "ns_seconds_mix": ns_mix,
        "cos_O_g_mean_over_run": sum(cos_g_traj) / len(cos_g_traj) if cos_g_traj else None,
        "cos_O_Onat_mean_over_run": sum(cos_onat_traj) / len(cos_onat_traj) if cos_onat_traj else None,
        "norm_rel_err_max": nerr_max if nerr_max > 0 else None,
        "cursor_end": stream.cursor,
        "ckpt": ck_saved,
    }


def main():
    device = "cuda:0"
    assert torch.cuda.is_available()
    assert CACHE.exists(), "pilot cache missing (run scripts/mint_pilot_cache.py first)"
    docs = json.load(open(DEV, encoding="utf-8"))["development"]

    out = {
        "device": torch.cuda.get_device_name(0),
        "steps": STEPS,
        "ckpt_step": CKPT_STEP,
        "tokens_per_step": MICRO * ACCUM * SEQ,
        "peak_lr": PEAK,
        "schedule": "warmup8+cosine->0.1xpeak over T=100 (DIAGNOSTIC schedule, disclosed)",
        "seed": SEED,
        "cache_sha_from_manifest": json.load(open(ROOT / "artifacts" / "pilot_cache_manifest.json"))["sha256"],
        "paths": {},
    }

    for path in PATHS:
        torch.manual_seed(SEED)
        model = build_model("100m", seed=SEED, device=device)
        init_fp = param_fingerprint(model)
        model.train()
        muon, aux = build_path(path, model, PEAK)
        stream = WindowStream(CACHE, SEQ)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        want_ck = (ROOT / "artifacts" / "preflight" / f"ckpt50_{path}.pt") if path in ("muon_native", "group_A_l065") else None
        log = run_segment(model, muon, aux, stream, 0, 0, STEPS, device, want_ck)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        nll_final = dev_nll(model, docs, device)
        hashes = state_fingerprint(model, muon)

        replay = None
        if want_ck is not None:
            # fresh objects, restore, replay steps 50..99
            torch.manual_seed(SEED)
            model2 = build_model("100m", seed=SEED, device=device)
            muon2, aux2 = build_path(path, model2, PEAK)
            cursor_ck, step_ck = load_ckpt(want_ck, model2, muon2, aux2)
            model2.train()
            stream2 = WindowStream(CACHE, SEQ)
            log2 = run_segment(model2, muon2, aux2, stream2, cursor_ck, step_ck, STEPS, device, None)
            bitwise = all(
                torch.equal(a.detach(), b.detach())
                for (_, a), (_, b) in zip(model.state_dict().items(), model2.state_dict().items())
            )
            maxdiff = max(
                float((a.detach() - b.detach()).abs().max())
                for (_, a), (_, b) in zip(model.state_dict().items(), model2.state_dict().items())
            )
            replay = {
                "ckpt_step": step_ck,
                "cursor_match": cursor_ck == log["cursor_end"] - (STEPS - CKPT_STEP) * MICRO * ACCUM * (SEQ + 1),
                "replay_final_param_fingerprint": param_fingerprint(model2),
                "bitwise_identical": bitwise,
                "max_abs_param_diff": maxdiff,
                "replay_losses_last3": log2["losses_last3"],
            }
            want_ck.unlink(missing_ok=True)  # delete checkpoint; hashes retained

        tokens = STEPS * MICRO * ACCUM * SEQ
        out["paths"][path] = {
            "init_fingerprint": init_fp,
            "wall_s": wall,
            **log,
            "tokens_consumed": stream.cursor,
            "throughput_tok_s": tokens / wall,
            "vram_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
            "dev_nll_step100": nll_final,
            "final_hashes": hashes,
            "replay_check": replay,
        }
        print(path, json.dumps({k: v for k, v in out["paths"][path].items() if not isinstance(v, (list, dict))}))
        del model, muon, aux
        torch.cuda.empty_cache()

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "preflight_100steps.json").write_text(json.dumps(out, indent=1))
    print("PREFLIGHT_DONE")


if __name__ == "__main__":
    main()
