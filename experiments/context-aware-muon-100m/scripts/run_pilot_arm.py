"""Frozen pilot arm runner (CONTEXT_AWARE_MUON_100M_PILOT_V1).

One invocation = one preregistered arm of configs/pilot_arms_v1.json.
Common substrate per arm: seed 3471, MiniMind 100M, pilot cache (V16 rule,
sha f92715b4...), micro 8 x accum 4 x seq 256, bf16 autocast, fp32 master,
warmup 128 + cosine -> 0.1x peak over T=4096, NO gradient clipping,
diag_every=64, dev NLL (V16 development 256 docs, dv metric) at H in
{1024, 2048, 4096}, endpoint weights-only artifact + hashes.

Authorized execution only (see 100M_PILOT_EXECUTION_LOCK.md / 100M_GPU_GATE.md).
"""
from __future__ import annotations

import argparse
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
ARMS = ROOT / "configs" / "pilot_arms_v1.json"
STEPS, WARMUP = 4096, 128
MICRO, ACCUM, SEQ = 8, 4, 256
SEED = 3471
EVAL_HORIZONS = (1024, 2048, 4096)
WD, MOMENTUM = 0.02, 0.95


def resolve_lr(spec, results_dir: Path):
    if isinstance(spec, (int, float)):
        return float(spec)
    sel = json.load(open(results_dir / "selection.json"))
    if spec == "@B_muon_best":
        return sel["B_muon_best_lr"]
    if spec == "@C_A_best_lam":
        return sel["C_A_best_lam"]
    raise ValueError(spec)


def build_optimizers(arm: dict, model, peak: float):
    routed = route_model(model)
    kind = arm["kind"]
    if kind == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=peak, betas=(0.9, 0.95),
                                eps=1e-8, weight_decay=WD)
        return opt, None, None
    adjust = arm.get("adjust_lr_fn")  # None -> "original"
    if kind == "muon":
        native = routed["ffn"] + routed["vo"] + routed["qk"]
        groups = [{"params": native, "mode": "native", "lr": peak}]
        if "momentum" in arm:
            groups[0]["momentum"] = arm["momentum"]
    elif kind in ("group_A", "group_A_nat", "group_B"):
        lam = arm["lam"]
        norm_match = kind != "group_A_nat"
        mix_set = routed["ffn"] if kind != "group_B" else routed["ffn"] + routed["vo"]
        native_set = routed["vo"] + routed["qk"] if kind != "group_B" else routed["qk"]
        groups = [
            {"params": mix_set, "mode": "mix", "lam": lam, "norm_match": norm_match, "lr": peak},
            {"params": native_set, "mode": "native", "lr": peak},
        ]
    else:
        raise ValueError(kind)
    muon = GroupMuon(groups, lr=peak, weight_decay=WD, momentum=MOMENTUM,
                     nesterov=True, adjust_lr_fn=adjust, diag_every=64)
    aux = torch.optim.AdamW(routed["aux"], lr=peak, betas=(0.9, 0.95), eps=1e-8,
                            weight_decay=WD)
    return None, muon, aux


@torch.no_grad()
def dev_nll(model, docs, device) -> float:
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for doc in docs:
        ids = torch.tensor(doc["ids"], device=device, dtype=torch.long)[None, :]
        logits = model(ids[:, :-1], use_cache=False).logits
        target = ids[:, 1:]
        per = torch.nn.functional.cross_entropy(
            logits[:, : target.shape[1]].float().reshape(-1, logits.shape[-1]),
            target.reshape(-1), reduction="none").view_as(target)
        tot_loss += float(per.sum())
        tot_tok += int(target.numel())
    model.train()
    return tot_loss / tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm_id in configs/pilot_arms_v1.json")
    args = ap.parse_args()

    arms = json.load(open(ARMS))
    arm = arms["arms"][args.arm]
    out_dir = ROOT / "results" / "pilot" / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = out_dir / "run.lock.json"
    if lock.exists():
        raise SystemExit(f"arm {args.arm} already executed (lock present)")

    docs = json.load(open(DEV, encoding="utf-8"))["development"]
    sel_dir = ROOT / "results" / "pilot"
    peak = resolve_lr(arm["lr"], sel_dir)
    if arm.get("lam") is not None:
        arm = {**arm, "lam": resolve_lr(arm["lam"], sel_dir)}
    device = "cuda:0"
    assert torch.cuda.is_available() and CACHE.exists()

    torch.manual_seed(SEED)
    model = build_model("100m", seed=SEED, device=device)
    init_fp = param_fingerprint(model)
    model.train()
    opt, muon, aux = build_optimizers(arm, model, peak)
    stream = WindowStream(CACHE, SEQ)

    metrics = (out_dir / "metrics.jsonl").open("w")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    evals = {}
    for step in range(STEPS):
        lr = eta_schedule(step, STEPS, WARMUP, peak)
        targets = [opt] if opt is not None else [muon, aux]
        for o in targets:
            for g in o.param_groups:
                g["lr"] = lr
        tot, finite = 0.0, True
        for _ in range(ACCUM):
            x, y = stream.next_batch(MICRO)
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=y).loss
            if not torch.isfinite(loss):
                finite = False
            (loss / ACCUM).backward()
            tot += float(loss)
        assert finite, f"non-finite loss step {step}"
        ts = time.perf_counter()
        if opt is not None:
            opt.step()
        else:
            muon.step()
            aux.step()
        opt_s = time.perf_counter() - ts
        st = muon.stats if muon is not None else {}
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            muon.zero_grad(set_to_none=True)
            aux.zero_grad(set_to_none=True)
        metrics.write(json.dumps({
            "step": step, "lr": lr, "loss": tot / ACCUM,
            "opt_s": opt_s,
            "ns_native_s": st.get("ns_seconds_native"),
            "ns_mix_s": st.get("ns_seconds_mix"),
            "cos_O_g": st.get("cos_O_g_mean"),
            "cos_O_Onat": st.get("cos_O_Onat_mean"),
            "norm_rel_err_max": st.get("norm_rel_err_max"),
        }) + "\n")
        metrics.flush()
        if (step + 1) in EVAL_HORIZONS:
            evals[step + 1] = dev_nll(model, docs, device)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    # endpoint artifact: weights only (fp32) + hashes
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(sd, out_dir / "endpoint_weights.pt")
    run = {
        "arm_id": args.arm, "arm": arm, "peak_lr": peak, "seed": SEED,
        "init_fingerprint": init_fp,
        "final_fingerprint": param_fingerprint(model),
        "wall_s": wall,
        "throughput_tok_s": STEPS * MICRO * ACCUM * SEQ / wall,
        "vram_peak_gib": torch.cuda.max_memory_reserved() / 2**30,
        "tokens_consumed": stream.cursor,
        "dev_nll_by_horizon": evals,
        "steps": STEPS,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=1))
    lock.write_text(json.dumps({"arm_id": args.arm, "ts": time.time(), "status": "DONE"}))
    print(json.dumps({k: run[k] for k in ("arm_id", "peak_lr", "wall_s", "dev_nll_by_horizon")}))


if __name__ == "__main__":
    main()
