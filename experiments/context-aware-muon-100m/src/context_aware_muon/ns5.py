"""Vendored Newton-Schulz orthogonalization (NS5) — verbatim semantics of
torch.optim._muon._zeropower_via_newtonschulz (torch 2.11.0, wheel file sha256
38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52).

Provenance: /home/b205/miniforge3/envs/fr_minimind/lib/python3.11/site-packages/torch/optim/_muon.py
Upstream permalink (in-file): https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L16

Project rule: orthogonalization is NS5-only; no singular-value decomposition of
any kind appears anywhere in this codebase (enforced by a source-scan test).
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5
DEFAULT_COEFFICIENTS = (DEFAULT_A, DEFAULT_B, DEFAULT_C)


def ns5(
    grad: Tensor,
    ns_coefficients: tuple[float, float, float] = DEFAULT_COEFFICIENTS,
    ns_steps: int = DEFAULT_NS_STEPS,
    eps: float = EPS,
) -> Tensor:
    """Newton-Schulz quintic iteration, bit-compatible with torch.optim.Muon's
    helper on identical inputs (same op order, same bf16 cast, same transpose rule).

    Scale-invariance: the input is Frobenius-normalized (clamped at eps) before the
    iterations, hence ns5(c*X) == ns5(X) up to bf16 rounding for any c > 0. This is
    the property that anchors the lam-equivalence point lam0 = 1/(1+momentum).
    """
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100")
    if len(grad.shape) != 2:
        raise ValueError("Input tensor must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


def adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Verbatim port of torch.optim._muon._adjust_lr."""
    A, B = param_shape[:2]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        import math

        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        import math

        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio


def lam_anchor(momentum: float) -> float:
    """lam0 = 1 - momentum.

    Truth from code, NOT from the torch docstring: torch.optim.Muon computes the
    Nesterov input as `grad.lerp(buf, momentum)` = (1-mu)*g + mu*buf (the
    docstring's B~ = g + mu*B does not match the implementation; Keller Jordan's
    reference code has the same lerp semantics). The mixing family
    M = lam*g + (1-lam)*buf therefore reproduces the native direction EXACTLY at
    lam0 = 1 - mu. The optimizer routes this exact value through the identical
    lerp expression (bitwise anchor); see group_muon.py.

    Empirical caveat (measured, documented): NS5 amplifies ~1e-7 relative input
    perturbations into O(0.1) changes of the singular-value profile on
    ill-conditioned mixtures (chaotic regime of the overdriven quintic); the
    bitwise routing avoids this floor at the anchor, and cos diagnostics away
    from the anchor must be read against this floor.
    """
    return 1.0 - momentum
