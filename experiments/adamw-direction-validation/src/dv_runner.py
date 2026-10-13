#!/usr/bin/env python3
"""adamw_direction_validation runner (layer 1 = dev-grade 2048-step matched matrix;
layer 2 = gated 3-seed 4096-step confirmation; layer 2 refuses to run until its
preregistered gate is satisfied and the new holdout manifest is UNSEALED).

Read-only dependencies on V16 project: pilot_core (model/build/checkpoint/batch/loss),
campaign_v16/cache/train_tokens.u32 (shared data stream), campaign_v16/eval_sets.json
(dev/screen reuse for layer 1). No V16 file is ever written."""
from __future__ import annotations
import json, math, random, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
V16 = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")
sys.path.insert(0, str(V16))
sys.path.insert(0, str(HERE))

from pilot_core import model_and_optimizer, training_batch, loss_sum_count  # noqa: E402
from src.dv_core import apply_step, schedule_eta, param_fingerprint, eval_docs, token_weighted  # noqa: E402

CFGF = HERE / "configs" / "frozen_v1.json"
RAW = HERE / "raw"
TARGET = "model.layers.1.mlp.down_proj.weight"
ANCHOR_MATS = ["model.layers.1.mlp.down_proj.weight",
               "model.layers.2.mlp.down_proj.weight",
               "model.layers.1.self_attn.o_proj.weight"]


def eval_sets():
    return json.load(open(V16 / "campaign_v16" / "eval_sets.json", encoding="utf-8"))


def anchor_scalars(named, A_map, target_A, meta):
    out = {}
    for m in ANCHOR_MATS:
        p = named[m]
        g = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
        md = meta["_m_plus"].get(m)
        Am = A_map.get(m)
        d = {}
        if g.numel() > 0:
            gd = g.double(); Ad = Am.double() if Am is not None else None
            d["g_norm"] = float(torch.linalg.vector_norm(gd))
            if Ad is not None:
                d["A_norm"] = float(torch.linalg.vector_norm(Ad))
                d["cos_g_A"] = float(torch.dot(gd.reshape(-1), Ad.reshape(-1)) /
                                     (d["g_norm"] * d["A_norm"] + 1e-30))
            if md is not None:
                mdd = md.double().to(gd.device)
                d["m_norm"] = float(torch.linalg.vector_norm(mdd))
                d["cos_g_m"] = float(torch.dot(gd.reshape(-1), mdd.reshape(-1)) /
                                     (d["g_norm"] * d["m_norm"] + 1e-30))
            if Am is not None:
                U, S, Vh = torch.linalg.svd(Amd := Am.double(), full_matrices=False)
                d["sigma1"] = float(S[0])
                d["head_energy_fraction"] = float((S[0] ** 2) / (S ** 2).sum().item())
        out[m] = d
    return out


def train_branch(task, cfg, device="cuda:0"):
    label = task["id"]
    mem = np.memmap(V16 / "campaign_v16" / "cache" / "train_tokens.u32", dtype=np.uint32, mode="r")
    torch.manual_seed(task["seed"]); torch.cuda.manual_seed_all(task["seed"])
    np.random.seed(task["seed"]); random.seed(task["seed"])
    binding = json.load(open(V16 / "bindings.local.json", encoding="utf-8"))
    pcfg = json.load(open(V16 / "config" / "local_v1_lowmem.json", encoding="utf-8"))
    model, opt, aliases, _ = model_and_optimizer(pcfg, binding, peak_lr=task["lr"], device=device)
    seq, mb, accum = cfg["runtime"]["sequence_length"], cfg["runtime"]["micro_batch_size"], cfg["runtime"]["accumulation_steps"]
    T = task["T"]
    dev_docs = eval_sets()[cfg["layer1"]["eval_split"]]
    step_log = RAW / "steps" / f"{label}.jsonl"
    step_log.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if step_log.exists():
        done = {json.loads(l)["step"] for l in step_log.open(encoding="utf-8")}
    bf16 = torch.cuda.is_bf16_supported()
    cursor, total_tokens, branch_step = 0, 0, 0
    origin = {"state_id": f"scratch_seed{task['seed']}", "loaded_checkpoint_sha": None, "global_step": 0}
    anchors_at = {max(1, int(T * f)) for f in (0.1, 0.25, 0.5, 0.75)}
    t0 = time.time()
    while branch_step < T:
        branch_step += 1
        eta = schedule_eta(task["lr"], branch_step, T, task["warmup"], cfg["schedule"]["final_lr_ratio"])
        for g in opt.param_groups:
            g["lr"] = eta
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_total, valid_total = 0.0, 0
        for _ in range(accum):
            ids, cursor = training_batch(mem, cursor, mb, seq, device)
            lengths = torch.full((mb,), ids.shape[1], device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(ids[:, :-1], use_cache=False).logits
                loss_sum, count = loss_sum_count(logits, ids, lengths)
            loss_sum.backward()
            valid_total += count
            loss_total += float(loss_sum.detach())
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(valid_total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), pcfg["baseline"]["global_grad_clip_norm"])
        cap = {}
        meta = apply_step(model, opt, task["family"], task["a"], task["b"], label, eta, TARGET,
                          anchor_recorder=(lambda n, A, tA, m: cap.update(anchor_scalars(n, A, tA, m)))
                          if branch_step in anchors_at else None,
                          anchor_names=tuple(ANCHOR_MATS) if branch_step in anchors_at else ())
        opt.zero_grad(set_to_none=True)
        total_tokens += valid_total
        row = {"branch": label, "step": branch_step, "eta": eta,
               "loss_sum": loss_total, "token_count": int(valid_total),
               "nll": loss_total / valid_total, "family_used": task["family"],
               "A_norm": meta["A_norm"], "D_norm": meta["D_norm"], "q_t": meta["q_t"],
               "opt_step_seconds": meta["opt_step_seconds"],
               "zero_norm_fallback": meta["zero_norm_fallback"]}
        if branch_step not in done:
            with step_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if cap:
            with (RAW / "anchors.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"branch": label, "step": branch_step, "q_t": meta["q_t"],
                                    "matrices": cap}, ensure_ascii=False) + "\n")
        if branch_step in task.get("eval_horizons", []):
            ident = {**origin, "global_step": branch_step, "param_fingerprint": param_fingerprint(model)}
            eval_docs(model, dev_docs, cfg["layer1"]["eval_split"], label, label, branch_step, ident, RAW, device)
            model.train()
    wall = time.time() - t0
    fp = param_fingerprint(model)
    torch.save({"model": model.state_dict(), "task": task, "final_step": T, "cursor": cursor,
                "param_fingerprint": fp}, RAW / "checkpoints" / f"{label}_endpoint.pt")
    return {"id": label, "wall_seconds": wall, "tokens": total_tokens, "param_fingerprint": fp}


def layer2_holdout_eval(cfg):
    """One-shot holdout_v2 opening: all layer-2 endpoints identity-locked first,
    evaluated once, manifest flipped to OPENED. Refuses if already opened."""
    man_path = HERE / "holdout_v2_manifest.json"
    man = json.load(open(man_path, encoding="utf-8"))
    if man.get("status") != "MINTED_SEALED":
        raise SystemExit("HOLDOUT_ALREADY_OPENED_OR_INVALID: " + str(man.get("status")))
    docs = [json.loads(l) for l in (HERE / "splits" / "holdout_v2.jsonl").open(encoding="utf-8")]
    import hashlib
    h = hashlib.sha256((HERE / "splits" / "holdout_v2.jsonl").read_bytes()).hexdigest()
    if h != man["sha256"]:
        raise SystemExit("HOLDOUT_V2_HASH_MISMATCH")
    binding = json.load(open(V16 / "bindings.local.json", encoding="utf-8"))
    pcfg = json.load(open(V16 / "config" / "local_v1_lowmem.json", encoding="utf-8"))
    model, _, _, _ = model_and_optimizer(pcfg, binding, 0.001, "cuda:0")
    rows_all = {}
    for arm in cfg["layer2"]["arms"]:
        for seed in cfg["layer2"]["seeds"]:
            cid = f"dv_{arm['name']}_s{seed}"
            ep = RAW / "checkpoints" / f"{cid}_endpoint.pt"
            if not ep.is_file():
                raise SystemExit(f"ENDPOINT_MISSING: {cid}")
            payload = torch.load(ep, map_location="cpu", weights_only=False)
            if payload["final_step"] != cfg["layer2"]["T"]:
                raise SystemExit(f"ENDPOINT_STEP_MISMATCH: {cid}")
            model.load_state_dict(payload["model"])
            ident = {"state_id": f"endpoint:{cid}", "loaded_checkpoint_sha": hashlib.sha256(ep.read_bytes()).hexdigest(),
                     "global_step": payload["final_step"], "param_fingerprint": payload["param_fingerprint"]}
            eval_docs(model, docs, "holdout_v2", "dvfinal", cid, cfg["layer2"]["T"], ident, RAW)
            model.train()
    man["status"] = "OPENED_ONE_SHOT"
    man["opened_at"] = time.time()
    json.dump(man, open(man_path, "w", encoding="utf-8"), indent=1)
    return "HOLDOUT_OPENED"


def main():
    cfg = json.load(open(CFGF, encoding="utf-8"))
    layer = sys.argv[1] if len(sys.argv) > 1 else "layer1"
    if layer == "layer1":
        tasks = cfg["layer1"]["arms"]
    elif layer == "layer2":
        gate = json.load(open(HERE / "layer1_gate.json", encoding="utf-8")) if (HERE / "layer1_gate.json").exists() else None
        if gate is None or not gate.get("pass"):
            raise SystemExit("LAYER2_BLOCKED: layer-1 gate not satisfied (preregistered).")
        hs = HERE / "holdout_v2_manifest.json"
        if not hs.exists() or json.load(open(hs)).get("status") != "MINTED_SEALED":
            raise SystemExit("LAYER2_BLOCKED: holdout_v2 not minted (corpus dependency).")
        tasks = [dict(arm, id=f"dv_{arm['name']}_s{s}", seed=s, T=cfg["layer2"]["T"],
                      warmup=cfg["layer2"]["warmup"])
                 for arm in cfg["layer2"]["arms"] for s in cfg["layer2"]["seeds"]]
    elif layer == "layer2_holdout":
        print(layer2_holdout_eval(cfg))
        return
    else:
        raise SystemExit("usage: dv_runner.py layer1|layer2|layer2_holdout")
    results = []
    for task in tasks:
        r = train_branch(task, cfg)
        results.append(r)
        print(json.dumps(r))
    json.dump(results, open(RAW / f"run_meta_{layer}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
