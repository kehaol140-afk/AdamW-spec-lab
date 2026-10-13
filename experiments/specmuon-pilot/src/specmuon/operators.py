"""Spectral operators on a momentum matrix (frozen contract, see docs/P0_PROTOCOL.md).

Every operator:
  1. uses the SVD basis of the input matrix `M` (fp64 internals),
  2. outputs `Y = U diag(w) V^T` scaled so that `||Y||_F == target_norm`
     (default target = ||NS5(M)||_F, the native per-matrix update norm),
  3. reduces to ExactPolar at zero bias,
  4. is total: finite output for zero/repeated singular values, low rank,
     transposed shapes, bf16-rounded inputs.
"""

from __future__ import annotations

import torch

from .native_ns import vendored_zeropower_via_newtonschulz

SVD_DTYPE = torch.float64


def svd_basis(M: torch.Tensor):
    """Economy SVD in fp64. Returns (U, S, Vh) with S descending."""
    return torch.linalg.svd(M.detach().to(SVD_DTYPE), full_matrices=False)


def _scale_to(Y: torch.Tensor, target_norm: float | torch.Tensor) -> torch.Tensor:
    n = Y.norm()
    if not torch.isfinite(n) or n == 0:
        return Y
    return Y * (float(target_norm) / n)


def native_update(M: torch.Tensor, ns_steps: int = 5) -> torch.Tensor:
    """Native NS5 output as fp32 (bf16 internal), i.e. baseline A itself."""
    return vendored_zeropower_via_newtonschulz(M, ns_steps=ns_steps).float()


def exact_polar(M: torch.Tensor, target_norm: float | None = None) -> dict:
    """Baseline B: P = U V^T from SVD, scaled to target_norm.

    Returns also P_unscaled (the perfectly orthogonal factor; the norm-matched
    Y has uniform singular values T/sqrt(r) instead).
    """
    U, S, Vh = svd_basis(M)
    P = U @ Vh
    native_norm = None
    if target_norm is None:
        native_norm = float(native_update(M).norm())
        target_norm = native_norm
    Y = _scale_to(P, target_norm)
    return {"Y": Y, "P_unscaled": P, "U": U, "S": S, "Vh": Vh,
            "target_norm": float(target_norm), "native_norm": native_norm}


def _weighted(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor,
              w: torch.Tensor, target_norm: float) -> torch.Tensor:
    Y = (U * w.unsqueeze(0)) @ Vh
    return _scale_to(Y, target_norm)


def head_spec_muon(M: torch.Tensor, lam: float = 0.25,
                    target_norm: float | None = None,
                    basis: tuple | None = None) -> dict:
    """C: Head-SpecMuon. w_i = 1 + lam * (sigma_i / sigma_1), lam > 0 tilts head up."""
    if basis is None:
        basis = svd_basis(M)
    U, S, Vh = basis
    t = S / S[0].clamp_min(1e-300)
    w = 1.0 + lam * t
    if target_norm is None:
        target_norm = float(native_update(M).norm())
    return {"Y": _weighted(U, S, Vh, w, target_norm), "w": w, "S": S,
            "U": U, "Vh": Vh, "target_norm": float(target_norm)}


def bulk_spec_muon(M: torch.Tensor, lam: float = 0.25,
                    target_norm: float | None = None,
                    basis: tuple | None = None) -> dict:
    """D: Bulk-SpecMuon. w_i = 1 + lam * (1 - sigma_i / sigma_1), lam > 0 tilts bulk up."""
    if basis is None:
        basis = svd_basis(M)
    U, S, Vh = basis
    t = S / S[0].clamp_min(1e-300)
    w = 1.0 + lam * (1.0 - t)
    if target_norm is None:
        target_norm = float(native_update(M).norm())
    return {"Y": _weighted(U, S, Vh, w, target_norm), "w": w, "S": S,
            "U": U, "Vh": Vh, "target_norm": float(target_norm)}


def muon_power(M: torch.Tensor, p: float = 0.5,
               target_norm: float | None = None,
               basis: tuple | None = None) -> dict:
    """E: Muon^p spectral-power comparator. w_i = (sigma_i / sigma_1)^p, p >= 0.

    p = 0 -> ExactPolar. p > 0 tilts toward head. Zero singular values get weight 0
    (documented difference vs ExactPolar, which completes rank-deficient inputs).
    """
    if p < 0:
        raise ValueError("muon_power requires p >= 0 (negative p is unbounded at 0).")
    if basis is None:
        basis = svd_basis(M)
    U, S, Vh = basis
    t = S / S[0].clamp_min(1e-300)
    w = t.pow(p)
    if target_norm is None:
        target_norm = float(native_update(M).norm())
    return {"Y": _weighted(U, S, Vh, w, target_norm), "w": w, "S": S,
            "U": U, "Vh": Vh, "target_norm": float(target_norm)}


OPERATORS = {
    "native_ns5": native_update,
    "exact_polar": lambda M, **kw: exact_polar(M, **kw)["Y"],
    "head": lambda M, lam=0.25, **kw: head_spec_muon(M, lam=lam, **kw)["Y"],
    "bulk": lambda M, lam=0.25, **kw: bulk_spec_muon(M, lam=lam, **kw)["Y"],
    "muon_p": lambda M, p=0.5, **kw: muon_power(M, p=p, **kw)["Y"],
}


def apply_operator(name: str, M: torch.Tensor, **kwargs) -> torch.Tensor:
    if name not in OPERATORS:
        raise KeyError(f"unknown operator {name!r}; known: {sorted(OPERATORS)}")
    return OPERATORS[name](M, **kwargs)
