"""SpecMuon: torch-Muon-compatible optimizer with a pluggable spectral operator.

Semantics are IDENTICAL to torch.optim.Muon (torch 2.11, wheel sha 38e95918...):
decoupled weight decay at the unadjusted lr, momentum buffer `buf.lerp_(g, 1-mu)`,
Nesterov blend `g.lerp(buf, mu)`, LR shape adjustment sqrt(max(1, A/B)), update added
at the adjusted lr. The ONLY change is the orthogonalization map applied to the
Nesterov input:

    native_ns5    -> bit-identical to torch.optim.Muon
    exact_polar   -> baseline B
    head / bulk   -> SpecMuon variants (frozen lambda)
    muon_p        -> spectral power comparator (frozen p)

All non-native variants are strictly norm-matched per matrix to the native NS5 output
Frobenius norm (computed exactly at each step), so every arm applies updates of equal
per-matrix magnitude. Non-2D params are rejected exactly like torch Muon; parameter
routing and the aux AdamW arm are the caller's responsibility and are NOT touched here.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import torch


def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Verbatim port of torch.optim._muon._adjust_lr."""
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


_SVD_POOL: ThreadPoolExecutor | None = None
_SVD_POOL_WORKERS = 0


def _svd_batch_cpu(matrices: list[torch.Tensor], workers: int):
    """Deterministic fixed-worker-count parallel fp64 SVDs on CPU (LAPACK).

    Each call is single-threaded (intra-op threads are 1 in the pilot runner);
    map preserves order, so results are bit-identical for fixed `workers`.
    This is the same numerics as operators.svd_basis (CPU fp64 torch.linalg.svd).
    """
    global _SVD_POOL, _SVD_POOL_WORKERS
    if _SVD_POOL is None or _SVD_POOL_WORKERS != workers:
        if _SVD_POOL is not None:
            _SVD_POOL.shutdown(wait=True)
        _SVD_POOL = ThreadPoolExecutor(max_workers=workers)
        _SVD_POOL_WORKERS = workers
    return list(_SVD_POOL.map(
        lambda M: torch.linalg.svd(M.double(), full_matrices=False), matrices))


class SpecMuon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
        spectral_op: str = "native_ns5",
        lam: float = 0.25,
        power: float = 0.5,
    ) -> None:
        if spectral_op not in ("native_ns5", "exact_polar", "head", "bulk", "muon_p"):
            raise ValueError(f"unknown spectral_op: {spectral_op!r}")
        if not 0.0 <= lr:
            raise ValueError(f"lr should be >= 0 but is: {lr}")
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_coefficients": tuple(ns_coefficients),
            "eps": eps,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
            "spectral_op": spectral_op,
            "lam": lam,
            "power": power,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        f"SpecMuon only supports 2D parameters, got size {tuple(p.size())}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        import time as _time

        t_step0 = _time.perf_counter()
        stats = {"matrices": 0, "native_norm_sum": 0.0, "max_rel_norm_err": 0.0,
                 "t_native_probe_s": 0.0, "t_svd_s": 0.0}
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            op = group["spectral_op"]

            # ---- pass 1: momentum, nesterov input, native NS5 + target norms
            jobs = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.ndim != 2:
                    raise ValueError("Param gradient must be a 2D matrix")
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]
                buf.lerp_(grad, 1 - momentum)
                update = grad.lerp(buf, momentum) if group["nesterov"] else buf
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], p.shape)
                if op == "native_ns5":
                    from .native_ns import vendored_zeropower_via_newtonschulz

                    Y = vendored_zeropower_via_newtonschulz(
                        update, group["ns_coefficients"], group["ns_steps"], group["eps"]
                    )
                    p.mul_(1 - lr * weight_decay)
                    p.add_(Y.to(p.dtype), alpha=-adjusted_lr)
                    stats["matrices"] += 1
                    stats["native_norm_sum"] += float(Y.float().norm())
                    continue
                t_n0 = _time.perf_counter()
                from .operators import native_update

                native = native_update(update, ns_steps=group["ns_steps"])
                stats["t_native_probe_s"] += _time.perf_counter() - t_n0
                jobs.append((p, update, adjusted_lr, lr, weight_decay,
                             float(native.norm()), native))
            if op == "native_ns5" or not jobs:
                stats["t_step_s"] = _time.perf_counter() - t_step0
                self.last_stats = stats
                continue

            # ---- pass 2: parallel CPU fp64 SVDs of the nesterov inputs
            t_s0 = _time.perf_counter()
            bases = _svd_batch_cpu([u.detach().cpu() for (_, u, *_r) in jobs],
                                   workers=group.get("svd_workers", 16))

            # ---- pass 3: weights + GPU(fp64) reconstruction + norm match + apply
            kw = {}
            if op in ("head", "bulk"):
                kw["lam"] = group["lam"]
            if op == "muon_p":
                kw["p"] = group["power"]
            for (p, update, adjusted_lr, lr, weight_decay, native_norm, native), (U, S, Vh) in zip(jobs, bases):
                dev = p.device
                U_d, S_d, Vh_d = U.to(dev), S.to(dev), Vh.to(dev)
                if op == "exact_polar":
                    Y = U_d @ Vh_d
                else:
                    t = S_d / S_d[0].clamp_min(1e-300)
                    if op == "head":
                        w = 1.0 + kw["lam"] * t
                    elif op == "bulk":
                        w = 1.0 + kw["lam"] * (1.0 - t)
                    elif op == "muon_p":
                        w = t.pow(kw["p"])
                    else:
                        raise ValueError(op)
                    Y = (U_d * w.unsqueeze(0)) @ Vh_d
                n = Y.norm()
                if torch.isfinite(n) and n != 0:
                    Y = Y * (native_norm / n)
                rel = abs(float(Y.double().norm()) - native_norm) / native_norm
                stats["max_rel_norm_err"] = max(stats["max_rel_norm_err"], rel)
                stats["matrices"] += 1
                stats["native_norm_sum"] += native_norm
                p.mul_(1 - lr * weight_decay)
                p.add_(Y.to(p.dtype), alpha=-adjusted_lr)
            stats["t_svd_s"] += _time.perf_counter() - t_s0
        stats["t_step_s"] = _time.perf_counter() - t_step0
        self.last_stats = stats
        return loss
