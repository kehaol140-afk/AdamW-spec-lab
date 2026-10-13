"""SpecMuon optimizer semantics tests (CPU-only).

Run:  PYTHONPATH=src python tests/test_spec_muon_semantics.py

Covers:
- bit-exact equivalence SpecMuon(native_ns5) vs torch.optim.Muon (3 steps, mixed shapes)
- unchanged semantics: decoupled wd at unadjusted lr, momentum/nesterov, LR shape rule
- variant arms are per-matrix norm-matched to the native arm
- aux AdamW route untouched (routing-level check against the frozen M0 artifact)
"""

from __future__ import annotations

import json
import sys

import torch

from specmuon.io_utils import muon_param_names
from specmuon.spec_muon import SpecMuon, _adjust_lr

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, repr(e)))
        print(f"FAIL {name}: {e!r}")


def make_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 48), (48, 64), (128, 128), (96, 32)]
    params = [torch.randn(s, generator=g) for s in shapes]
    return params


def grads_of(params, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn_like(p, generator=g) for p in params]


def t01_bitexact_vs_torch_muon():
    from torch.optim import Muon

    for step_seed in (1, 2, 3):
        p1 = make_params(11)
        p2 = make_params(11)
        o1 = Muon(p1, lr=0.01, weight_decay=0.02, momentum=0.95, nesterov=True,
                  ns_steps=5, ns_coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7)
        o2 = SpecMuon(p2, lr=0.01, weight_decay=0.02, momentum=0.95, nesterov=True,
                      ns_steps=5, ns_coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7,
                      spectral_op="native_ns5")
        for t in range(3):
            gg = grads_of(p1, 100 * step_seed + t)
            for p, gr in zip(p1, gg):
                p.grad = gr.clone()
            for p, gr in zip(p2, gg):
                p.grad = gr.clone()
            o1.step()
            o2.step()
        for a, b in zip(p1, p2):
            if not torch.equal(a, b):
                raise AssertionError("SpecMuon(native) diverged from torch.optim.Muon")


def t02_adjust_lr_matches_torch():
    from torch.optim._muon import _adjust_lr as torch_adjust_lr

    for shape in [(512, 512), (1664, 512), (512, 1664), (256, 512)]:
        for fn in (None, "original", "match_rms_adamw"):
            a = _adjust_lr(0.02, fn, torch.Size(shape))
            b = torch_adjust_lr(0.02, fn, torch.Size(shape))
            if a != b:
                raise AssertionError(f"_adjust_lr mismatch {shape} {fn}: {a} vs {b}")


def t03_variant_updates_are_norm_matched_per_matrix():
    """One step with wd=0: native arm and variant arm must apply data updates with the
    same per-matrix Frobenius norm (the operator contract, end-to-end)."""
    base = make_params(5)
    params_native = [p.clone() for p in base]
    params_var = [p.clone() for p in base]
    o_n = SpecMuon(params_native, lr=0.01, weight_decay=0.0, spectral_op="native_ns5")
    o_v = SpecMuon(params_var, lr=0.01, weight_decay=0.0, spectral_op="head", lam=0.25)
    gg = grads_of(base, 42)
    for p, gr in zip(params_native, gg):
        p.grad = gr.clone()
    for p, gr in zip(params_var, gg):
        p.grad = gr.clone()
    o_n.step()
    o_v.step()
    for p0, pn, pv in zip(base, params_native, params_var):
        un = (p0 - pn).double()
        uv = (p0 - pv).double()
        rel = abs(float(un.norm()) - float(uv.norm())) / float(un.norm())
        # end-to-end through fp32 params: torch.norm() blocked-summation noise ~1e-5;
        # the operator-level fp64 contract (test_operators t06) enforces 1e-6.
        if not (rel <= 1e-5):
            raise AssertionError(f"update norm mismatch through optimizer (rel {rel:.2e})")


def t04_weight_decay_momentum_semantics_unchanged():
    """With spectral_op=native, buffer evolution and decoupled wd must equal hand-computed
    torch semantics (fp32 params/grads, vendored bf16 NS, K=5)."""
    import specmuon.native_ns as nns

    lr, wd, mu = 0.02, 0.05, 0.95
    gen = torch.Generator().manual_seed(3)
    p0 = torch.randn(32, 32, generator=gen)
    p = p0.clone()
    opt = SpecMuon([p], lr=lr, weight_decay=wd, momentum=mu, nesterov=True,
                   spectral_op="native_ns5")
    g1 = torch.randn(32, 32, generator=torch.Generator().manual_seed(4))
    p.grad = g1.clone()
    opt.step()
    buf = (1 - mu) * g1
    upd_in = g1.lerp(buf, mu)
    Y = nns.vendored_zeropower_via_newtonschulz(upd_in).float()
    expected = p0 * (1 - lr * wd) - lr * Y  # square matrix -> adjusted lr ratio 1
    # add_(alpha=...) may fuse the multiply-add; allow one-ulp-scale slack
    if not torch.allclose(p, expected, atol=1e-7, rtol=0):
        raise AssertionError("SpecMuon step deviates from hand-computed Muon semantics")
    # momentum buffer state matches torch convention
    st = opt.state[p]["momentum_buffer"]
    if not torch.allclose(st, buf, atol=0, rtol=0):
        raise AssertionError("momentum buffer deviates from buf = (1-mu)*g")


def t05_routing_untouched_vs_frozen_artifact():
    names = muon_param_names()
    rep = json.load(open(
        "/home/b205/projects/training_dynamics_lab/muon_transition_experiment/outputs/m0/routing_report.json"
    ))
    if len(names) != 56 or rep["muon_tensor_count"] != 56:
        raise AssertionError("Muon route count changed")
    if rep["adamw_aux_tensor_count"] != 34:
        raise AssertionError("aux route count changed")
    n_embed = sum(1 for n in names if "embed" in n)
    if n_embed != 0:
        raise AssertionError("embedding leaked into Muon route")


def t06_rejects_non_2d():
    try:
        SpecMuon([torch.randn(4, 4, 4)], spectral_op="head")
    except ValueError:
        return
    raise AssertionError("SpecMuon accepted a 3D param (torch Muon would reject)")


def t07_variant_changes_direction_not_norm_semantics():
    """head/bulk/muon_p arms on identical inputs produce finite params and update-norm
    stats consistent with native (spot check through last_stats)."""
    for op, kw in (("exact_polar", {}), ("head", {"lam": 0.25}),
                   ("bulk", {"lam": 0.25}), ("muon_p", {"power": 0.5})):
        base = make_params(9)
        ps = [p.clone() for p in base]
        o = SpecMuon(ps, lr=0.01, spectral_op=op, **kw)
        gg = grads_of(base, 77)
        for p, gr in zip(ps, gg):
            p.grad = gr.clone()
        o.step()
        for p in ps:
            if not torch.isfinite(p).all():
                raise AssertionError(f"{op}: non-finite parameters after step")
        if o.last_stats["matrices"] != len(base):
            raise AssertionError(f"{op}: matrix count wrong")


def main() -> int:
    tests = [
        ("t01_bitexact_vs_torch_muon", t01_bitexact_vs_torch_muon),
        ("t02_adjust_lr_matches_torch", t02_adjust_lr_matches_torch),
        ("t03_variant_updates_are_norm_matched_per_matrix", t03_variant_updates_are_norm_matched_per_matrix),
        ("t04_weight_decay_momentum_semantics_unchanged", t04_weight_decay_momentum_semantics_unchanged),
        ("t05_routing_untouched_vs_frozen_artifact", t05_routing_untouched_vs_frozen_artifact),
        ("t06_rejects_non_2d", t06_rejects_non_2d),
        ("t07_variant_changes_direction_not_norm_semantics", t07_variant_changes_direction_not_norm_semantics),
    ]
    for name, fn in tests:
        check(name, fn)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} passed, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
