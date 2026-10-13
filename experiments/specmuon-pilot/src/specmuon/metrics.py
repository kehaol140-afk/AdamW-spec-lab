"""Spectral metrics shared by P0-A/P0-B and the operator tests."""

from __future__ import annotations

import torch


def energy_shares(s: torch.Tensor, ks=(1, 8)) -> dict[int, float]:
    """E_k = sum_{i<=k} s_i^2 / sum_i s_i^2. s descending singular values."""
    p = s.double() ** 2
    total = float(p.sum())
    if total <= 0:
        return {k: 0.0 for k in ks}
    return {k: float(p[:k].sum() / total) for k in ks}


def participation_ratio(s: torch.Tensor) -> float:
    """PR = (sum s^2)^2 / sum s^4. Equals r for flat spectra, 1 for rank one."""
    p = s.double() ** 2
    d = float(p.pow(2).sum())
    if d <= 0:
        return 0.0
    return float(p.sum() ** 2 / d)


def orthogonality_defect(Y: torch.Tensor) -> float:
    """D(Y) = ||G - I||_F / sqrt(r), G = Y^T Y (tall) / Y Y^T (wide), r = min(m,n).

    0 for a perfectly orthogonalized update; matches the E_l metric of the prior
    muon_spectral_allocation project.
    """
    r = min(Y.shape)
    if Y.shape[0] >= Y.shape[1]:
        G = Y.double().T @ Y.double()
    else:
        G = Y.double() @ Y.double().T
    I = torch.eye(r, dtype=torch.float64)
    return float((G - I).norm() / (r**0.5))


def normalized_gap(A: torch.Tensor, B: torch.Tensor) -> float:
    """||A/||A||_F - B/||B||_F||_F / sqrt(2) in [0, 1]. 0 <=> same direction."""
    a = A.double() / A.double().norm().clamp_min(1e-300)
    b = B.double() / B.double().norm().clamp_min(1e-300)
    return float((a - b).norm() / (2**0.5))


def top_subspace_alignment(A: torch.Tensor, B: torch.Tensor, k: int = 8,
                           side: str = "left") -> float:
    """||U_k(A)^T U_k(B)||_F / sqrt(k) in [0, 1]; 1 = identical top-k subspace."""
    Ua = torch.linalg.svd(A.double(), full_matrices=False).U[:, :k]
    Ub = torch.linalg.svd(B.double(), full_matrices=False).U[:, :k]
    if side == "right":
        Ua = torch.linalg.svd(A.double().T, full_matrices=False).U[:, :k]
        Ub = torch.linalg.svd(B.double().T, full_matrices=False).U[:, :k]
    return float((Ua.T @ Ub).norm() / (k**0.5))


def top1_cosine(A: torch.Tensor, B: torch.Tensor, side: str = "left") -> float:
    """|cos| between top singular vectors (sign-ambiguous => absolute value)."""
    Ua = torch.linalg.svd(A.double(), full_matrices=False).U[:, 0]
    Ub = torch.linalg.svd(B.double(), full_matrices=False).U[:, 0]
    if side == "right":
        Ua = torch.linalg.svd(A.double().T, full_matrices=False).U[:, 0]
        Ub = torch.linalg.svd(B.double().T, full_matrices=False).U[:, 0]
    return float((Ua @ Ub).abs())


def directional_energy_shares(Y: torch.Tensor, U: torch.Tensor, V: torch.Tensor,
                              ks=(1, 8)) -> dict[int, float]:
    """Energy share of the input-basis top-k singular directions inside Y.

    Uses D = diag(U^T Y V): for operators that share the input singular basis,
    D are exactly the output singular values ordered by INPUT sigma. This is the
    basis-stable way to ask 'did the head (per the input spectrum) get heavier?'.
    """
    D = torch.diagonal(U.double().T @ Y.double() @ V.double())
    p = D ** 2
    total = float(p.sum())
    if total <= 0:
        return {k: 0.0 for k in ks}
    return {k: float(p[:k].sum() / total) for k in ks}


def cosine_matrices(A: torch.Tensor, B: torch.Tensor) -> float:
    """Cosine between vectorized matrices (used for cross-state stability)."""
    a = A.double().flatten()
    b = B.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))
