"""dv_core — vendored from adamw_spectral_pilot/campaign_v16/core.py (read-only import forbidden:
core.py's apply_step writes the V16 budget ledger via begin_optimizer/bump).

Vendoring deltas (the ONLY deltas; math byte-identical where marked):
  D1 apply_step: `begin_optimizer(phase)` and the two `bump(...)` calls removed
     (ledger writes); the opt.step() duration is captured into meta["opt_step_seconds"].
  D2 apply_step: `torch.cuda.synchronize()` guarded by `torch.cuda.is_available()`
     (CPU unit tests / CPU-only execution).
  D3 eval_docs: HERE-relative output path replaced by explicit `out_dir` argument;
     resource()/count_forward()/bump() removed; eval device parameterized.
  D4 apply_step: PCFG["baseline"] lookups (betas (0.9,0.95), eps 1e-08) inlined as
     literals equal to config/local_v1_lowmem.json (new project has no campaign cfg).
Everything else is a verbatim copy of frozen core.py (source_hash
4afaa78e3633c302023c35f926cc1fd866cf9642978213410a751bb79e17aa12).
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------- n_a (verbatim)
def n_a(x, a_norm):
    xn = float(torch.linalg.vector_norm(x.double()))
    if xn == 0.0 or a_norm == 0.0:
        return torch.zeros_like(x), True, xn
    return (x.double() * (a_norm / xn)).to(x.dtype), False, xn


# ---------------------------------------------------------------- schedule_eta (verbatim)
def schedule_eta(peak, k, T, warmup, ratio):
    if k <= warmup:
        return peak * k / warmup
    return peak * (ratio + (1 - ratio) * (1 + math.cos(math.pi * (k - warmup) / (T - warmup))) / 2)


# ---------------------------------------------------------------- apply_step (deltas D1/D2)
def apply_step(model, opt, family, a, b, phase, eta, target_name, q_override=None,
               anchor_recorder=None, anchor_names=()):
    """Native opt.step() + verified displacement replacement. Returns meta with q_t."""
    named = dict(model.named_parameters())
    beta1, beta2 = (0.9, 0.95)
    eps = 1e-08
    per_beta1 = {}
    for group in opt.param_groups:
        if len(group["params"]) == 1 and group["betas"][0] != beta1:
            per_beta1[id(group["params"][0])] = group["betas"][0]
    A_map, meta = {}, {"family": family, "_m_plus": {}}
    for name, param in named.items():
        if param.grad is None:
            A_map[name] = None
            continue
        b1 = per_beta1.get(id(param), beta1)
        st = opt.state[param]
        if "step" in st:
            step_plus = int(st["step"].item() if torch.is_tensor(st["step"]) else st["step"]) + 1
            m_prev, v_prev = st["exp_avg"], st["exp_avg_sq"]
        else:
            step_plus, m_prev, v_prev = 1, torch.zeros_like(param), torch.zeros_like(param)
        m_plus = m_prev * b1 + param.grad * (1 - b1)
        v_plus = v_prev * beta2 + param.grad.square() * (1 - beta2)
        A_map[name] = (m_plus / (1 - b1 ** step_plus)) / ((v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)
        if name == target_name:
            meta["moments_step_plus"] = step_plus
        if anchor_names and name in anchor_names:
            meta["_m_plus"][name] = m_plus.detach().cpu().clone()
    target_A = A_map[target_name]
    meta["A_norm"] = float(torch.linalg.vector_norm(target_A.double()))
    if family in ("F", "P"):
        g_clip = named[target_name].grad.detach()
        if family == "P":
            b1 = per_beta1.get(id(named[target_name]), beta1)
            st = opt.state[named[target_name]]
            v_plus = (st["exp_avg_sq"] * beta2 + g_clip.square() * (1 - beta2)) if "step" in st \
                else g_clip.square() * (1 - beta2)
            step_plus = meta["moments_step_plus"]
            x = g_clip / ((v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)
        else:
            x = g_clip
        D, zero_fb, xn = n_a(x, meta["A_norm"])
        meta.update(zero_norm_fallback=zero_fb, input_norm=xn)
    else:
        D = target_A
        meta.update(zero_norm_fallback=False, input_norm=meta["A_norm"])
    if q_override is not None:
        dn = float(torch.linalg.vector_norm(D.double()))
        # q_override is the FULL target displacement norm ||eta*a*D||; rescale D so the
        # final displacement equals it exactly (no repeated eta/a multiplication).
        desired_D = (q_override / (eta * a)) if (a != 0 and eta != 0) else 0.0
        if dn > 0 and desired_D > 0:
            D = (D.double() * (desired_D / dn)).to(D.dtype)
        elif desired_D == 0:
            D = torch.zeros_like(D)
        meta["q_override"] = q_override
    if not torch.isfinite(D).all() or not torch.isfinite(target_A).all():
        raise RuntimeError("NON_FINITE_DIRECTION")
    started = time.perf_counter()                                  # D1 (was begin_optimizer)
    opt.step()
    meta["opt_step_seconds"] = time.perf_counter() - started      # D1 (was bump)
    if torch.cuda.is_available():                                  # D2 guard
        torch.cuda.synchronize()
    with torch.no_grad():
        for name, param in named.items():
            A_p = A_map[name]
            if A_p is None:
                continue
            if name == target_name:
                if not (family == "S" and a == 1.0 and q_override is None):
                    param.add_(eta * (target_A - a * D))
            elif b != 1.0:
                param.add_(eta * (1.0 - b) * A_p)
    if torch.cuda.is_available():                                  # D2 guard
        torch.cuda.synchronize()
    meta["D_norm"] = float(torch.linalg.vector_norm(D.double()))
    meta["q_t"] = eta * a * meta["D_norm"]
    if anchor_recorder is not None:
        anchor_recorder(named, A_map, target_A, meta)
    return meta


# ---------------------------------------------------------------- param_fingerprint (verbatim)
def param_fingerprint(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- eval_docs (delta D3)
def eval_docs(model, docs, split, phase, label, horizon, identity, out_dir, device="cuda:0"):
    """identity: {'state_id','loaded_checkpoint_sha','global_step','param_fingerprint'}"""
    import sys
    sys.path.insert(0, "/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")
    from pilot_core import loss_sum_count
    if identity.get("loaded_checkpoint_sha") is None and not str(identity.get("state_id", "")).startswith("scratch_"):
        raise RuntimeError("EVAL_WITHOUT_STATE_ORIGIN")
    path = Path(out_dir) / "eval" / f"{label}__H{horizon}__{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    old = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    if [x["doc_id"] for x in old] != [x["doc_id"] for x in docs[: len(old)]]:
        raise RuntimeError("PARTIAL_EVAL_IDENTITY_MISMATCH")
    if len(old) == len(docs):
        return old
    model.eval()
    started = time.perf_counter()
    rows = []
    for doc in docs[len(old):]:
        ids = torch.tensor(doc["ids"], device=device, dtype=torch.long)[None, :]
        with torch.no_grad():
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device=device))
        if count != doc["token_count"]:
            raise RuntimeError("MASK_OR_SHIFT_CHANGED")
        row = {"branch": label, "horizon": horizon, "split": split, "doc_id": doc["doc_id"],
               "loss_sum": float(loss), "token_count": int(count), **identity}
        rows.append(row)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    meta = {"eval_seconds": time.perf_counter() - started, "n_docs": len(docs)}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return old + rows


# ---------------------------------------------------------------- stats (verbatim)
def token_weighted(rows):
    return sum(r["loss_sum"] for r in rows) / sum(r["token_count"] for r in rows)


def paired_bootstrap(rows_a, rows_b, seed=9071, reps=1000):
    if [r["doc_id"] for r in rows_a] != [r["doc_id"] for r in rows_b]:
        raise RuntimeError("PAIRING_MISMATCH")
    la = np.array([r["loss_sum"] for r in rows_a]); lb = np.array([r["loss_sum"] for r in rows_b])
    ta = np.array([r["token_count"] for r in rows_a]); tb = np.array([r["token_count"] for r in rows_b])
    point = la.sum() / ta.sum() - lb.sum() / tb.sum()
    rng = np.random.default_rng(seed)
    draws = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, len(rows_a), len(rows_a))
        draws[i] = la[idx].sum() / ta[idx].sum() - lb[idx].sum() / tb[idx].sum()
    return {"point": point,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "ci97p5": [float(np.quantile(draws, 0.0125)), float(np.quantile(draws, 0.9875))]}
