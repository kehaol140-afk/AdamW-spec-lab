#!/usr/bin/env python3
"""s30m_runner — staged T1 runner for the 30M direction-validation matrix.

Stages (adaptive, all selection on development H2048 token-weighted NLL only):
  1  LR pre-screen: S(1,1) @ L-index x LR {3e-4,1e-3,3e-3}      -> pick lr* (tie<1e-4 -> smaller)
  2  a screen:      S(a,1)  @ L-index, lr* x a {2,4,8}          -> pick a* (flat<1e-3 -> 4)
  3  direction:     F/P @ (lr*, a*, L-index)                    -> winner family for layer stage
  4  layer contrast S-a* + winner @ L-depth(layer 2)
  5  Muon reference: LR {5e-3,1e-2,2e-2} (wd 0.02, M-line wiring)

Resume: step logs are append-only with a done-set; anchors deduped on (branch, step).
All raw dirs pre-created. Endpoints saved with param fingerprints."""
from __future__ import annotations
import json, math, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from dv_core import apply_step, schedule_eta, param_fingerprint, eval_docs, token_weighted  # noqa: E402
import s30m_core as C  # noqa: E402

RAW = HERE / "raw"
CFGP = HERE / "configs" / "frozen_30m_t1.json"


def anchor_scalars(named, A_map, target_A, meta, target_block):
    from torch.linalg import vector_norm
    out = {}
    for m in C.anchor_names_for(target_block):
        p = named[m]
        g = p.grad.detach() if p.grad is not None else torch.zeros_like(p)
        md = meta["_m_plus"].get(m)
        Am = A_map.get(m)
        d = {}
        if g.numel() > 0:
            gd = g.double()
            Ad = Am.double() if Am is not None else None
            d["g_norm"] = float(vector_norm(gd))
            if Ad is not None:
                d["A_norm"] = float(vector_norm(Ad))
                d["cos_g_A"] = float(torch.dot(gd.reshape(-1), Ad.reshape(-1)) /
                                      (d["g_norm"] * d["A_norm"] + 1e-30))
            if md is not None:
                mdd = md.double().to(gd.device)
                d["m_norm"] = float(vector_norm(mdd))
                d["cos_g_m"] = float(torch.dot(gd.reshape(-1), mdd.reshape(-1)) /
                                      (d["g_norm"] * d["m_norm"] + 1e-30))
            if Ad is not None:
                U, S, Vh = torch.linalg.svd(Ad, full_matrices=False)
                d["sigma1"] = float(S[0])
                d["head_energy_fraction"] = float((S[0] ** 2) / (S ** 2).sum().item())
        out[m] = d
    return out


def run_branch(task, cfg, device="cuda:0"):
    label = task["id"]
    family = task["family"]
    stream = C.TokenStream()
    docs = C.dev_docs()
    step_log = RAW / "steps" / f"{label}.jsonl"
    done = set()
    if step_log.exists():
        done = {json.loads(l)["step"] for l in step_log.open(encoding="utf-8")}
    anchor_file = RAW / "anchors" / f"{label}.jsonl"
    anchor_done = set()
    if anchor_file.exists():
        anchor_done = {(json.loads(l)["step"]) for l in anchor_file.open(encoding="utf-8")}

    model = C.build_model(task["seed"], device)
    target_block = f"model.layers.{task['layer']}.mlp.down_proj.weight"
    fp0 = param_fingerprint(model)
    if family == "M":
        opts = C.build_muon(model, task["lr"])
    else:
        opts = [C.build_adamw(model, task["lr"])]
    bf16 = torch.cuda.is_bf16_supported() and device.startswith("cuda")
    T, warm = task["T"], task["warmup"]
    anchors_at = {max(1, int(T * f)) for f in (0.1, 0.25, 0.5, 0.75)}
    cursor, total_tokens, t_train, t_eval, t_opt = 0, 0, 0.0, 0.0, 0.0
    nonfinite = 0
    origin = {"state_id": f"scratch_seed{task['seed']}", "loaded_checkpoint_sha": None,
              "param_fingerprint_init": fp0}
    t0 = time.time()
    for branch_step in range(1, T + 1):
        eta = schedule_eta(task["lr"], branch_step, T, warm, cfg["schedule"]["final_lr_ratio"])
        for g in [g for opt in opts for g in opt.param_groups]:
            g["lr"] = eta
        model.train()
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        loss_total, valid_total = 0.0, 0
        for _ in range(C.ACCUM):
            ids, cursor = stream.micro_batch(cursor, device)
            lengths = torch.full((C.MICRO,), C.SEQ + 1, device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(ids[:, :-1], use_cache=False).logits
            from pilot_shim import loss_sum_count  # noqa: E402
            loss_sum, count = loss_sum_count(logits, ids, lengths)
            loss_sum.backward()
            valid_total += count
            loss_total += float(loss_sum.detach())
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(valid_total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), C.CLIP)
        row_extra = {}
        if family == "M":
            ts = time.perf_counter()
            for opt in opts:
                opt.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            opt_s = time.perf_counter() - ts
            t_opt += opt_s
            meta = {"A_norm": None, "D_norm": None, "q_t": None, "zero_norm_fallback": False,
                    "opt_step_seconds": opt_s}
        else:
            meta = apply_step(model, opts[0], family, task["a"], 1.0, label, eta, target_block,
                              anchor_recorder=(lambda n, A, tA, m: row_extra.update(
                                  anchor_scalars(n, A, tA, m, target_block)))
                              if branch_step in anchors_at and branch_step not in anchor_done else None,
                              anchor_names=tuple(C.anchor_names_for(target_block))
                              if branch_step in anchors_at and branch_step not in anchor_done else ())
            t_opt += meta["opt_step_seconds"]
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        total_tokens += valid_total
        if not math.isfinite(loss_total / valid_total):
            nonfinite += 1
        if branch_step not in done:
            with step_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"branch": label, "step": branch_step, "eta": eta,
                                    "loss_sum": loss_total, "token_count": int(valid_total),
                                    "nll": loss_total / valid_total, "family_used": family,
                                    "A_norm": meta["A_norm"], "D_norm": meta["D_norm"],
                                    "q_t": meta["q_t"], "opt_step_seconds": meta["opt_step_seconds"],
                                    "zero_norm_fallback": meta["zero_norm_fallback"],
                                    "wall_cum": time.time() - t0}) + "\n")
        if row_extra:
            with anchor_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"branch": label, "step": branch_step, "q_t": meta["q_t"],
                                    "matrices": row_extra}) + "\n")
        if branch_step in task.get("eval_horizons", []):
            ident = {**origin, "global_step": branch_step, "param_fingerprint": param_fingerprint(model)}
            te = time.perf_counter()
            eval_docs(model, docs, "development", label, label, branch_step, ident, RAW, device)
            t_eval += time.perf_counter() - te
            model.train()
    t_train = time.time() - t0 - t_eval
    torch.save({"model": model.state_dict(), "task": task, "final_step": T, "cursor": cursor,
                "param_fingerprint": param_fingerprint(model), "param_fingerprint_init": fp0},
               RAW / "checkpoints" / f"{label}_endpoint.pt")
    return {"id": label, "wall_seconds": t_train + t_eval, "train_seconds": t_train,
            "eval_seconds": t_eval, "opt_step_seconds_total": t_opt, "tokens": total_tokens,
            "param_fingerprint": param_fingerprint(model), "init_fingerprint": fp0,
            "nonfinite_steps": nonfinite}


def dev_nll(label, horizon):
    p = RAW / "eval" / f"{label}__H{horizon}__development.jsonl"
    return token_weighted([json.loads(l) for l in p.open(encoding="utf-8")])


def select_lr(cfg, state):
    lrs = cfg["stages"]["lr_grid"]
    rows = {lr: dev_nll(f"s30_S1_lr{lr:g}_L1", cfg["T"]) for lr in lrs}
    best = min(lrs, key=lambda lr: (rows[lr], lr))
    tie = [lr for lr in lrs if rows[best] - rows[lr] < 1e-4 and rows[lr] <= rows[best] + 1e-4]
    pick = min(tie) if tie else best
    state["lr_star"] = pick
    state["lr_screen"] = {str(lr): rows[lr] for lr in lrs}
    state["lr_boundary"] = pick == max(lrs) or pick == min(lrs)
    return pick


def select_a(cfg, state):
    lr = state["lr_star"]
    As = cfg["stages"]["a_grid"]
    rows = {a: dev_nll(f"s30_S{a}_lr{lr:g}_L1", cfg["T"]) for a in As}
    best = min(As, key=lambda a: rows[a])
    flat = abs(rows[4] - rows[best]) < 1e-3 if 4 in rows else False
    pick = 4 if flat else best
    state["a_star"] = pick
    state["a_screen"] = {str(a): rows[a] for a in As}
    state["a_boundary"] = best == max(As)
    state["a_flat_override"] = flat
    return pick


def select_family(cfg, state):
    lr, a = state["lr_star"], state["a_star"]
    f = dev_nll(f"s30_F{a}_lr{lr:g}_L1", cfg["T"])
    p = dev_nll(f"s30_P{a}_lr{lr:g}_L1", cfg["T"])
    pick = "F" if f <= p else "P"
    state["family_screen"] = {"F": f, "P": p}
    state["winner_family"] = pick
    return pick


def main():
    cfg = json.load(open(CFGP, encoding="utf-8"))
    for d in ("steps", "eval", "checkpoints", "anchors"):
        (RAW / d).mkdir(parents=True, exist_ok=True)
    state_path = RAW / "stage_state.json"
    state = json.load(open(state_path)) if state_path.exists() else {}
    seed, T, warm = cfg["seed"], cfg["T"], cfg["warmup"]
    H = cfg["eval_horizons"]
    L1, L2 = cfg["stages"]["layer_index"], cfg["stages"]["layer_depth"]
    results = []
    def launch(task):
        ep = RAW / "checkpoints" / f"{task['id']}_endpoint.pt"
        if ep.exists():
            print("skip(done)", task["id"])
            return
        r = run_branch(task, cfg)
        results.append(r)
        print(json.dumps(r), flush=True)

    for lr in cfg["stages"]["lr_grid"]:
        launch({"id": f"s30_S1_lr{lr:g}_L1", "family": "S", "a": 1, "lr": lr, "layer": L1,
                "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    if "lr_star" not in state:
        select_lr(cfg, state); state_path.write_text(json.dumps(state, indent=1))
    lr = state["lr_star"]
    for a in cfg["stages"]["a_grid"]:
        launch({"id": f"s30_S{a}_lr{lr:g}_L1", "family": "S", "a": a, "lr": lr, "layer": L1,
                "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    if "a_star" not in state:
        select_a(cfg, state); state_path.write_text(json.dumps(state, indent=1))
    a = state["a_star"]
    launch({"id": f"s30_F{a}_lr{lr:g}_L1", "family": "F", "a": a, "lr": lr, "layer": L1,
            "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    launch({"id": f"s30_P{a}_lr{lr:g}_L1", "family": "P", "a": a, "lr": lr, "layer": L1,
            "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    if "winner_family" not in state:
        select_family(cfg, state); state_path.write_text(json.dumps(state, indent=1))
    wf = state["winner_family"]
    launch({"id": f"s30_S{a}_lr{lr:g}_L{L2}", "family": "S", "a": a, "lr": lr, "layer": L2,
            "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    launch({"id": f"s30_{wf}{a}_lr{lr:g}_L{L2}", "family": wf, "a": a, "lr": lr, "layer": L2,
            "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    for lr_m in cfg["stages"]["muon_lr_grid"]:
        launch({"id": f"s30_M_lr{lr_m:g}", "family": "M", "a": None, "lr": lr_m, "layer": None,
                "T": T, "warmup": warm, "seed": seed, "eval_horizons": H})
    if results:
        json.dump(results, open(RAW / "run_meta_t1.json", "w"), indent=1)
    print("STAGE_STATE", json.dumps(state))


if __name__ == "__main__":
    main()
