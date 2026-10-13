"""Cost meter: analytic NS FLOPs per group/K-vector + isolated wall-time benchmark."""
from __future__ import annotations

import time

import numpy as np
import torch

from src.recipe_registry import GROUPS
from src.recipe_registry import NS_COEFFS, EPS_NORM

from torch.optim._muon import _zeropower_via_newtonschulz


def ns_flops_per_step(shape):
    r, s = min(shape), max(shape)
    return 4 * r * r * s + 2 * r ** 3


def group_step_flops(shapes_by_group):
    return {g: float(sum(ns_flops_per_step(sh) for sh in shapes)) for g, shapes in shapes_by_group.items()}


def kv_flops_ratio(step_flops, K_vector):
    tot = sum(K_vector.get(g, 5) * f for g, f in step_flops.items())
    ref = sum(5 * f for g, f in step_flops.items())
    return tot / ref


def benchmark_K(H_items, K_vector, n_blocks=24, warmup=4):
    """Median wall time of running NS_K over all matrices (spectral module only).

    H_items: iterable of (tensor, group) pairs. Alternating ref/candidate blocks are
    interleaved by the caller.
    """
    mats = [(tg[0], K_vector.get(tg[1], 5)) for _, tg in H_items]
    torch.cuda.synchronize()
    for _ in range(warmup):
        for h, k in mats:
            _zeropower_via_newtonschulz(h, NS_COEFFS, k, EPS_NORM)
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_blocks):
        t0 = time.perf_counter()
        for h, k in mats:
            _zeropower_via_newtonschulz(h, NS_COEFFS, k, EPS_NORM)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
