"""CPU unit tests for the 30M port. Run: python tests/test_s30m_cpu.py"""
import json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import s30m_core as C  # noqa: E402
from dv_core import apply_step, schedule_eta, n_a  # noqa: E402
from pilot_shim import loss_sum_count  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append(("PASS", name))
        print("PASS", name)
    except Exception as e:
        RESULTS.append(("FAIL", f"{name}: {e!r}"))
        print("FAIL", name, repr(e))


def tiny_model():
    """A small stand-in model with the SAME structural roles (2D matrices + 1D norms + tied head)
    so CPU tests run in seconds; tests math portability, not 30M size."""
    torch.manual_seed(0)
    m = torch.nn.ModuleDict({
        "layers": torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "mlp": torch.nn.ModuleDict({"down_proj": torch.nn.Linear(8, 12, bias=False)}),
                "self_attn": torch.nn.ModuleDict({"o_proj": torch.nn.Linear(8, 8, bias=False)}),
                "norm": torch.nn.LayerNorm(8),
            }) for _ in range(2)]),
    })
    for p in m.parameters():
        p.data.normal_()
    return m


def named(m):
    return {k: p for k, p in m.named_parameters()}


def test_model_build():
    model = C.build_model(3421)
    n = sum(p.numel() for p in model.parameters())
    assert n == 30_025_216, n
    t = model.model.layers[1].mlp.down_proj.weight
    assert tuple(t.shape) == (512, 1664)
    m2 = C.build_model(3421)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), m2.parameters()))


def test_adamw_groups():
    model = C.build_model(3421)
    opt = C.build_adamw(model, 1e-3)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert all(p.ndim >= 2 for p in opt.param_groups[0]["params"])
    assert opt.param_groups[1]["weight_decay"] == 0.0
    assert all(p.ndim < 2 for p in opt.param_groups[1]["params"])
    assert tuple(opt.param_groups[0]["betas"]) == (0.9, 0.95) and opt.param_groups[0]["eps"] == 1e-8


def test_S11_bit_native():
    m1, m2 = tiny_model(), tiny_model()
    o1 = torch.optim.AdamW([p for p in m1.parameters() if p.ndim >= 2], lr=1e-3, betas=(0.9, 0.95))
    o2 = torch.optim.AdamW([p for p in m2.parameters() if p.ndim >= 2], lr=1e-3, betas=(0.9, 0.95))
    for k in range(1, 4):
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            if p1.ndim >= 2:
                g = torch.randn_like(p1)
                p1.grad = g.clone(); p2.grad = g.clone()
        o1.step()
        apply_step(m2, o2, "S", 1.0, 1.0, "t", 1e-3, "layers.0.mlp.down_proj.weight")
        o1.zero_grad(); o2.zero_grad()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.equal(p1, p2), (n1, n2)


def test_Sa_displacement():
    m = tiny_model()
    eta, a = 1e-3, 8.0
    opt = torch.optim.AdamW([p for p in m.parameters() if p.ndim >= 2], lr=eta,
                            betas=(0.9, 0.95), weight_decay=0.0)
    tgt = "layers.0.mlp.down_proj.weight"
    for p in m.parameters():
        if p.ndim >= 2:
            p.grad = torch.randn_like(p)
    pre = {n: p.detach().clone() for n, p in m.named_parameters()}
    meta = apply_step(m, opt, "S", a, 1.0, "t", eta, tgt)
    st = opt.state[named(m)[tgt]]
    step = int(st["step"])
    A = (st["exp_avg"] / (1 - 0.9 ** step)) / ((st["exp_avg_sq"] / (1 - 0.95 ** step)).sqrt() + 1e-8)
    got = named(m)[tgt]
    expect = pre[tgt] - eta * a * A  # native -eta*A then correction eta*(A - a*A)
    assert torch.allclose(got, expect, atol=1e-7), (got - expect).abs().max()
    for n, p in m.named_parameters():
        if n != tgt and p.ndim >= 2:  # others: native AdamW update -eta*A_p (wd=0)
            stp = opt.state[p]
            Ap = (stp["exp_avg"] / (1 - 0.9 ** int(stp["step"]))) / \
                 ((stp["exp_avg_sq"] / (1 - 0.95 ** int(stp["step"]))).sqrt() + 1e-8)
            assert torch.allclose(p, pre[n] - eta * Ap, atol=1e-7), n
        elif n != tgt:
            assert torch.equal(p, pre[n]), n
    assert abs(meta["q_t"] - eta * a * float(A.double().norm())) < 1e-9


def test_F_P_norm_match():
    m = tiny_model()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.ndim >= 2], lr=0.0, betas=(0.9, 0.95))
    tgt = "layers.0.mlp.down_proj.weight"
    g = torch.randn_like(named(m)[tgt])
    for p in m.parameters():
        if p.ndim >= 2:
            p.grad = torch.randn_like(p)
    named(m)[tgt].grad = g.clone()
    meta = apply_step(m, opt, "F", 8.0, 1.0, "t", 1e-3, tgt)
    st = opt.state[named(m)[tgt]]
    step = int(st["step"])
    A = (st["exp_avg"] / (1 - 0.9 ** step)) / ((st["exp_avg_sq"] / (1 - 0.95 ** step)).sqrt() + 1e-8)
    assert abs(meta["D_norm"] - meta["A_norm"]) <= 2e-6 * meta["A_norm"]
    assert not meta["zero_norm_fallback"]
    metaP = apply_step(m, opt, "P", 8.0, 1.0, "t", 1e-3, tgt)
    assert abs(metaP["D_norm"] - metaP["A_norm"]) <= 2e-6 * metaP["A_norm"]


def test_muon_routing():
    model = C.build_model(3421)
    opts = C.build_muon(model, 1e-2)
    muon, aux = opts
    assert len(muon.param_groups[0]["params"]) == 56
    assert len(aux.param_groups[0]["params"]) == 34
    assert muon.param_groups[0]["weight_decay"] == 0.02
    x = torch.randint(0, 6400, (2, 16))
    loss = model(x).logits.float().pow(2).mean()
    loss.backward()
    muon.step(); aux.step()
    bufs = sum(1 for s in muon.state.values() if "momentum_buffer" in s)
    assert bufs == 56, bufs


def test_window_identity_with_v16():
    """First micro-batch windows from the 30M cache must equal V16 training_batch on the V16 cache."""
    import sys as _s
    _s.path.insert(0, "/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")
    from pilot_core import training_batch
    s30 = C.TokenStream()
    ids30, _ = s30.micro_batch(0, "cpu")
    v16mem = np.memmap("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/"
                       "campaign_v16/cache/train_tokens.u32", dtype=np.uint32, mode="r")
    idsV, _ = training_batch(v16mem, 0, C.MICRO, C.SEQ, "cpu")
    assert ids30.shape == idsV.shape == (8, 257)
    assert torch.equal(ids30, idsV)


def test_schedule():
    assert schedule_eta(1e-3, 1, 2048, 128, 0.1) == 1e-3 / 128
    assert abs(schedule_eta(1e-3, 128, 2048, 128, 0.1) - 1e-3) < 1e-12
    assert abs(schedule_eta(1e-3, 2048, 2048, 128, 0.1) - 1e-4) < 1e-12


def test_loss_sum_count_full_windows():
    ids = torch.randint(0, 6400, (4, 257))
    logits = torch.randn(4, 256, 6400)
    lengths = torch.full((4,), 257, dtype=torch.long)
    s, c = loss_sum_count(logits, ids, lengths)
    assert c == 4 * 256 and torch.isfinite(s)


for n, f in sorted(locals().items()):
    if n.startswith("test_"):
        check(n, f)
fails = [r for r in RESULTS if r[0] == "FAIL"]
print(f"{len(RESULTS) - len(fails)}/{len(RESULTS)} CPU TESTS PASS")
sys.exit(1 if fails else 0)
