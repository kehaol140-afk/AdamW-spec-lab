"""Group probes: production gradient collection, per-group NS_K update vectors,
three-part amplitude probes, geometry residuals.

Candidate construction (per state/batch):
  1. restore anchor; collect production G (global clip); build H from checkpoint M*;
  2. theta_0_plus = REAL muon_opt.step() + aux_opt.step() (native full one-step, K=5);
  3. explicit u vectors: u_l(K) = -lr*shape*NS_K(H_l) for K in {3,5,7} (bf16-deterministic,
     matches the internals of step());
  4. candidate points = theta_0_plus + sum over changed groups of (u_g(K)-u_g(5)).
A test cross-checks (a) step() vs manual assembly and (b) superposition vs direct
construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from src.recipe_registry import GROUPS, NS_COEFFS, EPS_NORM, MOMENTUM, group_of_param, shape_factor

from torch.optim._muon import _zeropower_via_newtonschulz


def collect_G(model, stream, step_index, autocast):
    for p in model.parameters():
        p.grad = None
    micro_losses, n_targets = [], 0
    for mb in stream.micro_batches(step_index):
        batch = mb.to("cuda")
        with autocast:
            logits = model(batch).logits[..., :-1, :].contiguous()
            targets = batch[..., 1:].contiguous()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), reduction="mean")
        (loss / 4).backward()
        micro_losses.append(float(loss.detach()))
        n_targets += int(targets.numel())
    preclip = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
    return {"preclip": float(preclip), "targets": n_targets,
            "loss_mean": float(np.mean(micro_losses))}


def compute_H(model, momenta, beta=MOMENTUM):
    """name -> (H fp32 tensor, group). M = lerp(M*, G); H = lerp(G, M) (production order)."""
    out = {}
    for name, p in model.named_parameters():
        g = group_of_param(name)
        if g is None or p.grad is None:
            continue
        m_i = momenta[name].lerp(p.grad, 1.0 - beta)
        out[name] = (p.grad.lerp(m_i, beta).detach(), g)
    return out


def ns_cache(H):
    """name -> {K: NS output (float)} for K in {3,5,7}; one call per matrix per K."""
    cache = {}
    for name, (h, _) in H.items():
        cache[name] = {K: _zeropower_via_newtonschulz(h, NS_COEFFS, K, EPS_NORM).float()
                       for K in (3, 5, 7)}
    return cache


def u_of(H, ns_c, K_vector, lr):
    """name -> u_l = -lr*shape_factor*NS_{K_l}(H_l) (data update component, no WD)."""
    u = {}
    for name, (h, g) in H.items():
        k = K_vector.get(g, 5)
        u[name] = (-lr * shape_factor(h.shape)) * ns_c[name][k]
    return u


def nu_g(u, names):
    return float(torch.sqrt(sum(u[n].pow(2).sum() for n in names)))


def geometry_E(ns_c):
    """name -> E_l = ||YY^T-I||_F/sqrt(r) at K=5; and per-K group summaries separately."""
    out = {}
    for name, per in ns_c.items():
        out[name] = {}
        for K, y in per.items():
            yy = y if y.shape[0] <= y.shape[1] else y.T
            r = min(yy.shape)
            e = float((yy @ yy.T - torch.eye(r, device=yy.device)).norm() / np.sqrt(r))
            out[name][K] = e
    return out


def group_E2(E_by_name, H, group, K):
    vals = [E_by_name[n][K] for n, (h, g) in H.items() if g == group]
    return float(np.mean([v * v for v in vals]))


def apply_candidate(name_to_param, theta_plus, u_native, u_cand_groups):
    """theta_cand = theta_0_plus with the listed groups' u replaced by u_cand groups.
    Only Muon-route parameters appear in theta_plus; aux params are identical across
    candidates by construction (same aux step on the same grads)."""
    with torch.no_grad():
        for n, p in theta_plus.items():
            name_to_param[n].copy_(p)
        for g, (uK, names) in u_cand_groups.items():
            for n in names:
                name_to_param[n].add_(uK[n] - u_native[n])


def restore_theta(name_to_param, theta_plus):
    with torch.no_grad():
        for n, p in theta_plus.items():
            name_to_param[n].copy_(p)
