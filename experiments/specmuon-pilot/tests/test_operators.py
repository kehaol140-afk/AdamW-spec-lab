"""Operator unit/regression tests (CPU-only, no pytest dependency).

Run:  PYTHONPATH=src python tests/test_operators.py
Every test prints PASS/FAIL; exit code 1 on any failure.
Tolerances are the frozen ones from docs/P0_PROTOCOL.md.
"""

from __future__ import annotations

import math
import sys

import torch

from specmuon.metrics import (
    directional_energy_shares,
    energy_shares,
    orthogonality_defect,
)
from specmuon.native_ns import (
    converged_polar_reference,
    native_ns5_installed,
    polar_express_single_pass,
    vendored_zeropower_via_newtonschulz,
)
from specmuon.operators import bulk_spec_muon, exact_polar, head_spec_muon, muon_power, native_update, svd_basis

TOL_NORM = 1e-6
TOL_ZERO_BIAS = 1e-12
G = torch.Generator().manual_seed(2030)

RESULTS = []


def check(name: str, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, repr(e)))
        print(f"FAIL {name}: {e!r}")


def assert_close(a, b, tol, msg=""):
    if isinstance(a, float) and isinstance(b, float):
        d = abs(a - b)
    else:
        d = float((a.double() - b.double()).abs().max())
    if not (d <= tol):
        raise AssertionError(f"{msg} |delta|={d:.3e} > tol={tol:.1e}")


def rand_mat(m, n, scale=1.0):
    return torch.randn(m, n, generator=G) * scale


# ---------------------------------------------------------------- T1 vendored == installed
def t01_vendored_ns5_bitexact():
    for shape in [(64, 48), (48, 64), (1664, 512), (512, 1664), (512, 512)]:
        M = rand_mat(*shape)
        a = vendored_zeropower_via_newtonschulz(M)
        b = native_ns5_installed(M)
        assert torch.equal(a, b), f"vendored != installed for {shape}"


# ---------------------------------------------------------------- T2 zero bias -> ExactPolar
def t02_zero_bias_recovers_exact_polar():
    M = rand_mat(96, 64)
    T = float(native_update(M).norm())
    ep = exact_polar(M, target_norm=T)["Y"]
    h0 = head_spec_muon(M, lam=0.0, target_norm=T)["Y"]
    b0 = bulk_spec_muon(M, lam=0.0, target_norm=T)["Y"]
    p0 = muon_power(M, p=0.0, target_norm=T)["Y"]
    assert_close(ep, h0, TOL_ZERO_BIAS, "Head(0) != ExactPolar")
    assert_close(ep, b0, TOL_ZERO_BIAS, "Bulk(0) != ExactPolar")
    assert_close(ep, p0, TOL_ZERO_BIAS, "Muon^0 != ExactPolar")


# ---------------------------------------------------------------- T3 native != ExactPolar
def t03_native_distinct_from_exact_polar():
    M = rand_mat(96, 64)
    T = float(native_update(M).norm())
    ep = exact_polar(M, target_norm=T)
    y5 = native_update(M)
    gap = (y5.double() / y5.double().norm() - ep["Y"].double() / ep["Y"].double().norm()).norm()
    if not (gap > 1e-3):
        raise AssertionError(f"NS5 and ExactPolar unexpectedly identical (gap={gap:.2e})")
    # the unscaled polar factor is orthogonal to fp64 roundoff (the scaled arm is
    # uniform-valued T/sqrt(r) by design, so its defect is a pure scale signature)
    if not (orthogonality_defect(ep["P_unscaled"]) < 1e-10):
        raise AssertionError(f"polar factor not orthogonal: {orthogonality_defect(ep['P_unscaled']):.2e}")
    # native NS5 has a measurable defect
    if not (orthogonality_defect(y5) > 1e-3):
        raise AssertionError("NS5 defect unexpectedly tiny; baselines would be conflatable")


def _planted_spectrum_matrix(m=128, n=96, decay=1.0):
    """Random-orthogonal-basis matrix with a power-law spectrum sigma_i ~ (i+1)^-decay."""
    U = torch.linalg.qr(rand_mat(m, n).double())[0]  # m x n, orthonormal columns
    V = torch.linalg.qr(rand_mat(n, n).double())[0]
    S = torch.tensor([1.0 / ((i + 1) ** decay) for i in range(min(m, n))]).double()
    return (U * S.unsqueeze(0)) @ V.T


def _two_block_matrix(m=128, n=96, k=8, hi=1.0, lo=0.05):
    """Strongly separated two-block spectrum: top-k sigma = hi, rest = lo."""
    U = torch.linalg.qr(rand_mat(m, n).double())[0]
    V = torch.linalg.qr(rand_mat(n, n).double())[0]
    r = min(m, n)
    S = torch.full((r,), float(lo)).double()
    S[:k] = float(hi)
    return (U * S.unsqueeze(0)) @ V.T


# ---------------------------------------------------------------- T4 head raises head share
def _directional_profile(Y, U, V):
    """Output singular values ordered by INPUT sigma (direction-stable)."""
    return torch.diagonal(U.double().T @ Y.double() @ V.double())


def _head_rest_ratio(Y, U, V, k):
    """Mean per-direction energy of input-top-k directions vs the rest."""
    d2 = _directional_profile(Y, U, V) ** 2
    return float(d2[:k].mean() / d2[k:].mean())


def t04_head_raises_head_energy():
    k = 8
    # (a) two-block spectrum: the linear tilt must deliver its designed effect:
    #     head:rest per-direction energy ratio >= 1.4x at lambda=0.25
    #     ((1+0.25*1)^2 / (1+0.25*0.05)^2 = 1.46 in closed form)
    M = _two_block_matrix()
    T = float(native_update(M).norm())
    U, S, Vh = svd_basis(M)
    V = Vh.T
    ep = exact_polar(M, target_norm=T)["Y"]
    h = head_spec_muon(M, lam=0.25, target_norm=T)["Y"]
    r_ep = _head_rest_ratio(ep, U, V, k)
    r_h = _head_rest_ratio(h, U, V, k)
    if not (r_ep <= 1.0 + 1e-9):
        raise AssertionError(f"polar head:rest ratio not 1 ({r_ep})")
    if not (r_h / r_ep >= 1.4):
        raise AssertionError(f"Head ratio {r_h:.3f} vs polar {r_ep:.3f} (<1.4x)")
    if not (directional_energy_shares(h, U, V, ks=(k,))[k]
            > directional_energy_shares(ep, U, V, ks=(k,))[k] + 0.02):
        raise AssertionError("Head did not raise input-top-8 share on two-block spectrum")
    # (b) strictly increasing and monotone in lambda on any spectrum
    M2 = rand_mat(128, 96)
    T2 = float(native_update(M2).norm())
    U2, _, Vh2 = svd_basis(M2)
    V2 = Vh2.T
    e_prev = directional_energy_shares(
        exact_polar(M2, target_norm=T2)["Y"], U2, V2, ks=(k,))[k]
    for lam in (0.05, 0.25, 0.5, 1.0):
        hh = head_spec_muon(M2, lam=lam, target_norm=T2)["Y"]
        e_l = directional_energy_shares(hh, U2, V2, ks=(k,))[k]
        if not (e_l > e_prev + 1e-6):
            raise AssertionError("head directional share not strictly increasing in lambda")
        e_prev = e_l


# ---------------------------------------------------------------- T5 bulk raises bulk share
def t05_bulk_raises_bulk_energy():
    k = 8
    M = _two_block_matrix()
    T = float(native_update(M).norm())
    U, S, Vh = svd_basis(M)
    V = Vh.T
    ep = exact_polar(M, target_norm=T)["Y"]
    b = bulk_spec_muon(M, lam=0.25, target_norm=T)["Y"]
    r = min(M.shape)
    r_ep = _head_rest_ratio(ep, U, V, k)
    r_b = _head_rest_ratio(b, U, V, k)
    if not (r_b / r_ep <= 1 / 1.4):
        raise AssertionError(f"Bulk ratio {r_b:.3f} vs polar {r_ep:.3f} (not <= 1/1.4)")
    if not (directional_energy_shares(b, U, V, ks=(k,))[k]
            < directional_energy_shares(ep, U, V, ks=(k,))[k] - 0.02):
        raise AssertionError("Bulk did not lower input-top-8 share on two-block spectrum")
    tail = r - k
    e_ep_rest = 1.0 - directional_energy_shares(ep, U, V, ks=(k,))[k]
    e_b_rest = 1.0 - directional_energy_shares(b, U, V, ks=(k,))[k]
    if not (e_b_rest > e_ep_rest + 0.02):
        raise AssertionError("Bulk did not raise non-head share")
    # monotone decreasing in lambda on a random matrix
    M2 = rand_mat(128, 96)
    T2 = float(native_update(M2).norm())
    U2, _, Vh2 = svd_basis(M2)
    V2 = Vh2.T
    e_prev = directional_energy_shares(
        exact_polar(M2, target_norm=T2)["Y"], U2, V2, ks=(k,))[k]
    for lam in (0.05, 0.25, 0.5, 1.0):
        bb = bulk_spec_muon(M2, lam=lam, target_norm=T2)["Y"]
        e_l = directional_energy_shares(bb, U2, V2, ks=(k,))[k]
        if not (e_l < e_prev - 1e-6):
            raise AssertionError("bulk directional share not strictly decreasing in lambda")
        e_prev = e_l


# ---------------------------------------------------------------- T6 norm matching
def t06_norm_match_all_ops():
    torch.manual_seed(7)
    for shape in [(64, 48), (1664, 512), (512, 1664)]:
        M = rand_mat(*shape) * 0.01
        T = float(native_update(M).norm())
        outs = {
            "exact_polar": exact_polar(M, target_norm=T)["Y"],
            "head": head_spec_muon(M, lam=0.25, target_norm=T)["Y"],
            "bulk": bulk_spec_muon(M, lam=0.25, target_norm=T)["Y"],
            "muon_p": muon_power(M, p=0.5, target_norm=T)["Y"],
        }
        for op, Y in outs.items():
            rel = abs(float(Y.double().norm()) - T) / T
            if not (rel <= TOL_NORM):
                raise AssertionError(f"{op} {shape}: norm rel err {rel:.2e} > {TOL_NORM:.0e}")


# ---------------------------------------------------------------- T7 edge cases
def t07_zero_and_repeated_singular_values():
    # exact low rank (zeros in spectrum)
    U, _ = torch.linalg.qr(rand_mat(64, 64))
    V, _ = torch.linalg.qr(rand_mat(48, 48))
    S = torch.zeros(64, 48)
    S[0, 0], S[1, 1] = 3.0, 1.0
    M = (U[:, :64].double() @ S.double() @ V[:, :48].double().T).float()
    T = float(native_update(M).norm())
    for op, fn in [
        ("exact_polar", lambda: exact_polar(M, target_norm=T)["Y"]),
        ("head", lambda: head_spec_muon(M, lam=0.25, target_norm=T)["Y"]),
        ("bulk", lambda: bulk_spec_muon(M, lam=0.25, target_norm=T)["Y"]),
        ("muon_p", lambda: muon_power(M, p=0.5, target_norm=T)["Y"]),
    ]:
        Y = fn()
        if not torch.isfinite(Y).all():
            raise AssertionError(f"{op}: non-finite output on low-rank input")
        rel = abs(float(Y.double().norm()) - T) / T
        if not (rel <= TOL_NORM):
            raise AssertionError(f"{op}: low-rank norm err {rel:.2e}")

    # repeated singular values: invariance to basis rotation within degenerate block
    U2 = torch.linalg.qr(rand_mat(48, 48).double())[0]
    V2 = torch.linalg.qr(rand_mat(48, 48).double())[0]
    S2 = torch.diag(torch.tensor([2.0, 2.0, 2.0, 1.0, 0.5] + [0.25] * 43)).double()
    M2 = U2 @ S2 @ V2.T
    R = torch.eye(48, dtype=torch.float64)
    R[0, 0] = math.cos(0.7); R[0, 1] = -math.sin(0.7)
    R[1, 0] = math.sin(0.7); R[1, 1] = math.cos(0.7)
    # (U R) S (V R)^T == U R S R^T V^T == U S V^T when R commutes with S
    # (R only mixes the exactly-degenerate sigma=2 block and is orthogonal)
    Mb = (U2 @ R) @ S2 @ (V2 @ R).T
    assert_close(M2, Mb, 1e-12, "degenerate construction sanity")
    T2 = float(native_update(M2).norm())
    for fn in (lambda X: exact_polar(X, target_norm=T2)["Y"],
               lambda X: head_spec_muon(X, lam=0.25, target_norm=T2)["Y"],
               lambda X: bulk_spec_muon(X, lam=0.25, target_norm=T2)["Y"],
               lambda X: muon_power(X, p=0.5, target_norm=T2)["Y"]):
        assert_close(fn(M2), fn(Mb), 1e-9, "degenerate-subspace rotation changed output")


def t08_transpose_consistency():
    for shape in [(1664, 512), (512, 1664)]:
        M = rand_mat(*shape) * 0.1
        T = float(native_update(M).norm())
        Tt = float(native_update(M.T).norm())
        for fn in (lambda X: exact_polar(X, target_norm=T)["Y"],
                   lambda X: head_spec_muon(X, lam=0.25, target_norm=T)["Y"],
                   lambda X: bulk_spec_muon(X, lam=0.25, target_norm=T)["Y"],
                   lambda X: muon_power(X, p=0.5, target_norm=T)["Y"]):
            Y, Yt = fn(M), fn(M.T)
            # same input norm -> transpose operator is exact transpose (fp64 SVD basis)
            assert_close(Y, Yt.T, 1e-9, "op(M^T) != op(M)^T")
    # native NS5 transpose handling sanity: no shape error, finite output
    Y = vendored_zeropower_via_newtonschulz(rand_mat(512, 1664))
    assert torch.isfinite(Y).all() and Y.shape == (512, 1664)


def t09_bf16_rounded_inputs():
    for shape in [(256, 512), (512, 256), (832, 256)]:
        M = (rand_mat(*shape) * 0.05).bfloat16().float()  # bf16-quantized values
        T = float(native_update(M).norm())
        for op, fn in [
            ("exact_polar", lambda: exact_polar(M, target_norm=T)["Y"]),
            ("head", lambda: head_spec_muon(M, lam=0.25, target_norm=T)["Y"]),
            ("bulk", lambda: bulk_spec_muon(M, lam=0.25, target_norm=T)["Y"]),
            ("muon_p", lambda: muon_power(M, p=0.5, target_norm=T)["Y"]),
        ]:
            Y = fn()
            if not torch.isfinite(Y).all():
                raise AssertionError(f"{op}: non-finite on bf16-rounded {shape}")
            rel = abs(float(Y.double().norm()) - T) / T
            if not (rel <= TOL_NORM):
                raise AssertionError(f"{op}: bf16-input norm err {rel:.2e}")


def t10_tiny_matrix():
    M = rand_mat(1, 1)
    T = float(native_update(M).norm())
    Y = exact_polar(M, target_norm=T)["Y"]
    assert torch.isfinite(Y).all()
    rel = abs(float(Y.norm()) - T) / T
    if not (rel <= TOL_NORM):
        raise AssertionError(f"1x1 norm err {rel:.2e}")


# ---------------------------------------------------------------- T11 convergent reference
def t11_converged_reference_reaches_oracle():
    for shape in [(96, 64), (256, 512), (512, 1664)]:
        M = rand_mat(*shape) * 3.0
        Ycv, info = converged_polar_reference(M)
        U, S, Vh = svd_basis(M)
        P = U @ Vh
        gap = float((Ycv / Ycv.norm() - P / P.norm()).norm() / (2 ** 0.5))
        if not (gap <= 1e-3 and info["converged"]):
            raise AssertionError(f"cubic-Newton reference {shape}: gap={gap:.2e}, info={info}")
        # PE single pass: finite, near-polar but documented NOT to be iterated
        Ype = polar_express_single_pass(M)
        if not torch.isfinite(Ype).all():
            raise AssertionError("PE single pass produced non-finite values")
        # same-quintic K=20 does NOT converge to polar (why a validated
        # convergent iteration is required for high-step references)
        Y20 = vendored_zeropower_via_newtonschulz(M, ns_steps=20).float()
        if torch.isfinite(Y20).all():
            gap20 = float((Y20 / Y20.norm() - P / P.norm()).norm() / (2 ** 0.5))
            if not (gap20 > 1e-3):
                raise AssertionError(f"same-quintic K=20 unexpectedly at polar (gap={gap20:.2e})")


# ---------------------------------------------------------------- T12 weights positive/bounded
def t12_weight_functions_positive():
    M = rand_mat(96, 64)
    _, S, _ = svd_basis(M)
    t = S / S[0]
    for lam in (0.25,):
        wh = 1 + lam * t
        wb = 1 + lam * (1 - t)
        if not (wh.min() >= 1.0 and wb.min() >= 1.0 and wh.max() <= 1 + lam + 1e-12
                and wb.max() <= 1 + lam + 1e-12):
            raise AssertionError("weight functions out of declared range")
    if not (muon_power(M, p=0.5)["w"].min() >= 0.0):
        raise AssertionError("muon_p weights negative")


def main() -> int:
    tests = [
        ("t01_vendored_ns5_bitexact", t01_vendored_ns5_bitexact),
        ("t02_zero_bias_recovers_exact_polar", t02_zero_bias_recovers_exact_polar),
        ("t03_native_distinct_from_exact_polar", t03_native_distinct_from_exact_polar),
        ("t04_head_raises_head_energy", t04_head_raises_head_energy),
        ("t05_bulk_raises_bulk_energy", t05_bulk_raises_bulk_energy),
        ("t06_norm_match_all_ops", t06_norm_match_all_ops),
        ("t07_zero_and_repeated_singular_values", t07_zero_and_repeated_singular_values),
        ("t08_transpose_consistency", t08_transpose_consistency),
        ("t09_bf16_rounded_inputs", t09_bf16_rounded_inputs),
        ("t10_tiny_matrix", t10_tiny_matrix),
        ("t11_converged_reference_reaches_oracle", t11_converged_reference_reaches_oracle),
        ("t12_weight_functions_positive", t12_weight_functions_positive),
    ]
    for name, fn in tests:
        check(name, fn)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
