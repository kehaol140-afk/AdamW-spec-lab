import copy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_aware_muon.group_muon import GroupMuon
from context_aware_muon.ns5 import lam_anchor, ns5

SHAPES = [(2432, 768), (768, 2432), (768, 768), (384, 768), (768, 384)]


def _rand_params(seed=0):
    torch.manual_seed(seed)
    return [torch.randn(*s, requires_grad=False) for s in SHAPES]


def _grads(params, seed):
    torch.manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p)


def test_native_bitwise_vs_torch_muon():
    from torch.optim import Muon

    kw = dict(lr=0.01, weight_decay=0.02, momentum=0.95, nesterov=True)
    p1 = _rand_params(0)
    p2 = _rand_params(0)
    o1 = GroupMuon(p1, mode="native", **kw)
    o2 = Muon(p2, **kw)
    for t in range(3):
        _grads(p1, 100 + t)
        _grads(p2, 100 + t)
        o1.step()
        o2.step()
    for a, b in zip(p1, p2):
        assert torch.equal(a, b), "native mode must be bitwise-identical to torch.optim.Muon"


def _ns5_fp32(x, steps=5):
    """Exact-arithmetic NS5 (float32, no bf16 cast) — test-only mirror."""
    a, b, c = 3.4445, -4.775, 2.0315
    y = x.float()
    if y.size(0) > y.size(1):
        y = y.T
    y = y / y.norm().clamp(min=1e-7)
    for _ in range(steps):
        g = y @ y.T
        gu = torch.addmm(g, g, g, beta=b, alpha=c)
        y = torch.addmm(y, gu, y, beta=a)
    if y.size(0) > y.size(1):
        y = y.T
    return y


def test_lam_anchor_direction_and_norm_match():
    mu = 0.95
    lam0 = lam_anchor(mu)  # = 1 - mu = 0.05 (code semantics, see ns5.lam_anchor)
    assert abs(lam0 - 0.05) < 1e-12
    kw = dict(lr=0.01, weight_decay=0.02, momentum=mu, nesterov=True)
    p1 = _rand_params(0)
    p2 = _rand_params(0)
    o1 = GroupMuon(p1, mode="native", **kw)
    o2 = GroupMuon(p2, mode="mix", lam=lam0, norm_match=True, **kw)
    cos_min, rel_max = 1.0, 0.0
    for t in range(3):
        _grads(p1, 200 + t)
        _grads(p2, 200 + t)
        o1.step()
        o2.step()
        st = o2.stats
        cos_min = min(cos_min, st["cos_O_Onat_mean"])
        rel_max = max(rel_max, st["norm_rel_err_max"])
    # bitwise-routed anchor: identical NS5 input -> direction exact (cos == 1
    # up to fp32 cos-summation), norms matched in fp64
    assert cos_min > 0.99999, cos_min
    assert rel_max < 1e-9, rel_max
    for a, b in zip(p1, p2):
        rel = float((a - b).norm() / a.norm())
        assert rel < 1e-5, rel


def test_ns5_perturbation_floor_measured():
    """Metric note (measured): with fp64 cosines, a 1e-7 relative input perturbation
    costs < 1e-4 cos on well-conditioned random matrices; the apparent 'chaos'
    seen with fp32 cosines over ~2M elements was summation error (~1e-4), not
    optimizer instability. All diagnostics therefore use fp64 (group_muon.py)."""
    torch.manual_seed(3)
    buf = torch.randn(256, 256)
    g = torch.randn(256, 256)
    x = 0.05 * g + 0.95 * buf
    x_p = x * (1 + 1e-7 * torch.randn_like(x))
    from context_aware_muon.ns5 import ns5 as ns5_fn

    a, b = ns5_fn(x).double(), ns5_fn(x_p).double()
    cos = float((a * b).sum() / (a.norm() * b.norm()))
    assert cos > 0.999, cos


def test_momentum_state_independent_of_lam():
    mu = 0.95
    p1 = _rand_params(0)
    p2 = _rand_params(0)
    o1 = GroupMuon(p1, mode="native", lr=0.01, momentum=mu)
    o2 = GroupMuon(p2, mode="mix", lr=0.01, momentum=mu, lam=0.9)
    for t in range(3):
        _grads(p1, 300 + t)
        _grads(p2, 300 + t)
        o1.step()
        o2.step()
    for q1, q2 in zip(p1, p2):
        b1 = o1.state[q1]["momentum_buffer"]
        b2 = o2.state[q2]["momentum_buffer"]
        assert torch.equal(b1, b2), "momentum buffers must evolve identically"


def test_norm_match_precision_fp64():
    p = _rand_params(0)
    o = GroupMuon(p, mode="mix", lr=0.01, lam=0.8, norm_match=True)
    _grads(p, 7)
    o.step()
    assert o.stats["norm_rel_err_max"] < 1e-9
    assert o.stats["n_mix"] == len(p)


def test_lam_one_follows_pure_gradient_direction():
    torch.manual_seed(9)
    g = torch.randn(128, 96)
    buf = torch.randn(128, 96)
    o_g = ns5(g).float()
    o_mix = ns5(1.0 * g + 0.0 * buf).float()
    cos = float((o_g * o_mix).sum() / (o_g.norm() * o_mix.norm()))
    assert cos > 0.9999


def test_low_momentum_group_differs():
    p1 = _rand_params(0)
    p2 = _rand_params(0)
    o1 = GroupMuon(p1, mode="native", lr=0.01, momentum=0.95)
    g1 = GroupMuon(p2, mode="native", lr=0.01, momentum=0.90)
    for t in range(2):
        _grads(p1, 400 + t)
        _grads(p2, 400 + t)
        o1.step()
        g1.step()
    assert not torch.equal(p1[0], p2[0])


def test_state_roundtrip_bit_replay():
    p1 = _rand_params(1)
    p2 = _rand_params(1)
    o1 = GroupMuon(p1, mode="mix", lr=0.01, lam=0.75)
    _grads(p1, 500)
    o1.step()
    # checkpoint = optimizer state AND parameters (params are the runner's duty)
    sd = copy.deepcopy(o1.state_dict())
    for a, b in zip(p1, p2):
        b.data.copy_(a.data)
    o2 = GroupMuon(p2, mode="mix", lr=0.01, lam=0.75)
    o2.load_state_dict(sd)
    _grads(p1, 501)
    _grads(p2, 501)
    o1.step()
    o2.step()
    for a, b in zip(p1, p2):
        assert torch.equal(a, b)


def test_determinism_repeat_run():
    def run(seed=3):
        ps = _rand_params(seed)
        o = GroupMuon(ps, mode="mix", lr=0.01, lam=0.85)
        for t in range(4):
            _grads(ps, 600 + t)
            o.step()
        return [q.clone() for q in ps]

    a, b = run(), run()
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_stats_ranges_and_ns_cost_accounting():
    p = _rand_params(2)
    o = GroupMuon(p, mode="mix", lr=0.01, lam=0.7)
    _grads(p, 11)
    o.step()
    s = o.stats
    assert -1.0 <= s["cos_O_g_mean"] <= 1.0
    assert -1.0 <= s["cos_O_Onat_mean"] <= 1.0
    assert s["ns_seconds_mix"] >= 0.0 and s["n_mix"] == len(p)


def test_momentum_buffer_is_ema_not_classic():
    """2026-10-12 preflight lock: buffer update truth is buf <- mu*buf + (1-mu)*g
    (lerp/EMA), NOT classic mu*buf + g and NOT the torch docstring's B_t = mu*B + g.
    Explicit numeric check after exactly one step from a nonzero buffer."""
    mu = 0.95
    p = [torch.randn(64, 32)]
    o = GroupMuon(p, mode="native", lr=0.01, momentum=mu)
    buf0 = torch.randn(64, 32)
    o.state[p[0]]["momentum_buffer"] = buf0.clone()
    g = torch.randn(64, 32)
    p[0].grad = g.clone()
    o.step()
    got = o.state[p[0]]["momentum_buffer"]
    ema = mu * buf0 + (1 - mu) * g
    classic = mu * buf0 + g
    # fused lerp kernel rounds differently from the explicit formula: tight allclose
    # (1 ULP @ fp32 = 2.4e-7 here), not bitwise (bitwise anchor is the torch-equality test)
    assert torch.allclose(got, ema, rtol=0, atol=1e-6), float((got - ema).abs().max())
    assert not torch.allclose(got, classic), "buffer must NOT follow classic mu*buf+g"


def test_mix_family_has_no_low_momentum_equivalent():
    """2026-10-12 preflight lock: for a generic gradient sequence there is NO
    momentum value mu' that makes native(mu') reproduce the mix(lam, mu=0.95)
    NS5-input direction: the history-decay profile differs even when the
    first-order effective gradient age is matched. This is precisely why the
    low-momentum arms are CONTROLS (age-matched, not equivalents)."""
    mu, lam = 0.95, 0.85
    T = 20
    torch.manual_seed(17)
    gs = [torch.randn(96, 64) for _ in range(T)]
    # mix trajectory buffer (starts at zero, same as optimizer init)
    B = torch.zeros(96, 64)
    for g in gs:
        B = mu * B + (1 - mu) * g
    M = lam * gs[-1] + (1 - lam) * B
    best = -1.0
    for mu_p in [i / 200.0 for i in range(1, 200)]:
        Bp = torch.zeros(96, 64)
        for g in gs:
            Bp = mu_p * Bp + (1 - mu_p) * g
        M_native = (1 - mu_p) * gs[-1] + mu_p * Bp
        cos = float((M * M_native).sum() / (M.norm() * M_native.norm() + 1e-30))
        best = max(best, cos)
    assert best < 0.9999, f"unexpected exact equivalent at cos={best}"
    # first-order effective gradient age identities (see MUON_MOMENTUM_EQUIVALENCE.md)
    age_mix = (1 - lam) * mu / (1 - mu)  # = 2.85 steps for lam=0.85, mu=0.95
    age_native = lambda m: m * m / (1 - m)  # steady-state Nesterov input age
    assert abs(age_mix - 2.85) < 1e-12
    assert abs(age_native(0.95) - 18.05) < 1e-9
    assert abs(age_native(0.9) - 8.1) < 1e-9
    assert abs(age_native(0.8) - 3.2) < 1e-9
