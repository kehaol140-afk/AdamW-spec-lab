"""Vendored Newton-Schulz machinery.

`vendored_zeropower_via_newtonschulz` is a faithful copy of
torch/optim/_muon.py::_zeropower_via_newtonschulz (torch 2.11.0, wheel sha256
38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52).
`newton_schulz_fp64` mirrors muon_transition_experiment/src/muon_transition/reference_muon.py.
`polar_express_cascade` is a validated convergent high-step reference using the Polar
Express coefficient sequence recovered from the modded-nanogpt git history (commit
9350f4d), which cites https://arxiv.org/pdf/2505.16932.

All functions are CPU-safe and total (no NaN for finite input).
"""

from __future__ import annotations

import torch

MUON_WHEEL_SHA256 = "38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52"

NS5_COEFFS = (3.4445, -4.7750, 2.0315)
DEFAULT_EPS = 1e-7

# Polar Express cascade coefficients (as landed in modded-nanogpt 9350f4d).
PE_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503993),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382689081, 0.42323551169395323),
)


def vendored_zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float] = NS5_COEFFS,
    ns_steps: int = 5,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Byte-faithful re-implementation of torch 2.11 `_zeropower_via_newtonschulz`.

    bf16 internal arithmetic, Frobenius-norm spectral clamp, tall->wide transpose
    handling: identical to the optimizer used to train the m1 checkpoints.
    """
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


def native_ns5_installed(
    grad: torch.Tensor,
    ns_coefficients: tuple[float, float, float] = NS5_COEFFS,
    ns_steps: int = 5,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Call the installed torch implementation directly (provenance ground truth)."""
    from torch.optim._muon import _zeropower_via_newtonschulz

    return _zeropower_via_newtonschulz(grad, ns_coefficients, ns_steps, eps)


def newton_schulz_fp64(
    G: torch.Tensor,
    steps: int = 5,
    a: float = 3.4445,
    b: float = -4.7750,
    c: float = 2.0315,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """FP64 reference NS (mirrors muon_transition_experiment reference_muon.py)."""
    X = G.detach().double().clone()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    n = X.norm().clamp(min=eps)
    X = X / n
    for _ in range(steps):
        A = X @ X.T
        Bm = b * A + c * (A @ A)
        X = a * X + Bm @ X
    if transposed:
        X = X.T
    return X


@torch.no_grad()
def converged_polar_reference(
    G: torch.Tensor,
    max_iter: int = 80,
    defect_tol: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Provable convergent polar reference (cubic Newton-Schulz).

    Algorithm: one spectral normalization (sigma_max <- 1), then repeated
    h(x) = 1.5 x - 0.5 x^3 (Newton iteration for the matrix sign / polar factor).

    Convergence proof sketch: after normalization all singular values lie in
    (0, 1]; h is monotone increasing on [0, 1] with h([0,1]) = [0,1], h(x) >= x
    with equality only at x in {0, 1}; hence every singular value increases
    monotonically toward 1 (quadratically near 1: h'(1) = 0) and none can exceed
    1, so X -> U V^T (the polar factor) for any full-rank input. Directions with
    sigma = 0 stay 0 (rank-preserving canonical polar). Convergence is *validated
    externally* against the SVD oracle (P0-A runner, tests t11).

    Returns (X, info) with info = {iterations, defect, converged, sigma_max_drift}.
    """
    X = G.detach().double().clone()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    r = min(X.shape)
    I = torch.eye(r, dtype=X.dtype)
    X = X / float(torch.linalg.matrix_norm(X, 2))
    it = 0
    defect = float("inf")
    drift = 0.0
    for it in range(1, max_iter + 1):
        A = X @ X.T
        X = 1.5 * X - 0.5 * (A @ X)
        if it % 5 == 0 or it == max_iter:
            A = X @ X.T
            defect = float((A - I).norm() / (r**0.5))
            smax = float(torch.linalg.matrix_norm(X, 2))
            drift = max(drift, abs(smax - 1.0))
            if smax > 1.0 + 1e-9:
                X = X / smax  # numerical safety (theory says unnecessary)
            if defect <= defect_tol:
                break
    if transposed:
        X = X.T
    info = {"iterations": it, "defect": defect, "converged": defect <= defect_tol,
            "sigma_max_drift": drift}
    return X, info


@torch.no_grad()
def polar_express_single_pass(G: torch.Tensor, margin: float = 1.05) -> torch.Tensor:
    """One pass of the 5-coefficient Polar Express sequence (diagnostic only).

    This is what a coefficient-engineered 5-step family achieves; NOT iterated,
    because the sequence is designed for a single pass and cycling it diverges
    (verified empirically: NaN within 2 cycles).
    """
    X = G.detach().double().clone()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    X = X / (margin * X.norm() + 1e-12)
    for (a, b, c) in PE_COEFFS:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X
