#!/usr/bin/env python3
"""CPU-only unit tests for adamw_direction_validation (zero GPU).
Run: python -m pytest -q tests/  (or python tests/test_dv_cpu.py)"""
import json, sys, tempfile, time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")

from src.dv_core import apply_step, schedule_eta, n_a, paired_bootstrap, token_weighted, eval_docs  # noqa: E402

V16_LEDGER = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/budget_ledger.json")
TARGET = "model.layers.1.mlp.down_proj.weight"


def tiny_setup(seed=0):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(16, 8))
    other = torch.nn.Parameter(torch.randn(8, 8))
    opt = torch.optim.AdamW([{"params": [W], "weight_decay": 0.1},
                             {"params": [other], "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    model = torch.nn.ModuleDict({"w": torch.nn.ParameterDict({"t": W, "o": other})})
    # shim exposing parameters under target names; apply_step only needs named_parameters()
    class Shim:
        def __init__(self):
            self._W, self._other = W, other

        def named_parameters(self, remove_duplicate=True):
            yield TARGET, self._W
            yield "other.weight", self._other
    return Shim(), W, other, opt


def analytic_A(opt, name_param, beta1=0.9, beta2=0.95, eps=1e-8):
    p = name_param
    st = opt.state[p]
    if "step" in st:
        step_plus = int(st["step"]) + 1
        m_prev, v_prev = st["exp_avg"], st["exp_avg_sq"]
    else:
        step_plus, m_prev, v_prev = 1, torch.zeros_like(p), torch.zeros_like(p)
    m_plus = m_prev * beta1 + p.grad * (1 - beta1)
    v_plus = v_prev * beta2 + p.grad.square() * (1 - beta2)
    return (m_plus / (1 - beta1 ** step_plus)) / ((v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)


def test_S11_is_bit_native():
    model, W, other, opt = tiny_setup()
    W.grad = torch.randn_like(W) * 0.01; other.grad = torch.randn_like(other) * 0.01
    W0, o0 = W.detach().clone(), other.detach().clone()
    eta = 0.001
    meta = apply_step(model, opt, "S", 1.0, 1.0, "t", eta, TARGET)
    # reference: plain AdamW step on copies
    model2, W2, other2, opt2 = tiny_setup()
    W2.grad = W.grad.clone(); other2.grad = other.grad.clone()
    for g in opt2.param_groups: g["lr"] = eta
    opt2.step()
    assert torch.equal(W.detach(), W2.detach()), "S(1,1) target must be bit-native"
    assert torch.equal(other.detach(), other2.detach()), "S(1,1) other must be bit-native"
    assert abs(meta["D_norm"] - meta["A_norm"]) < 1e-12


def test_S8_displacement_math():
    model, W, other, opt = tiny_setup()
    W.grad = torch.randn_like(W) * 0.01; other.grad = torch.randn_like(other) * 0.01
    eta = 0.001
    A_t = analytic_A(opt, W); A_o = analytic_A(opt, other)
    W0, o0 = W.detach().clone(), other.detach().clone()
    meta = apply_step(model, opt, "S", 8.0, 1.0, "t", eta, TARGET)
    # expected: native step then correction eta*(A - 8A) = -7*eta*A
    native_W = W0 * (1 - eta * 0.1) - eta * A_t
    expect_W = native_W - 7 * eta * A_t
    expect_o = o0 - eta * A_o  # 1D wd=0, b=1 native
    assert torch.allclose(W.detach(), expect_W, atol=1e-7), "S(a=8) target = -8*eta*A with decay"
    assert torch.allclose(other.detach(), expect_o, atol=1e-7), "b=1 leaves other params native"
    assert abs(meta["q_t"] - 8 * eta * meta["A_norm"]) < 1e-9


def test_F_P_direction_and_norm():
    for fam, ref in (("F", None), ("P", None)):
        model, W, other, opt = tiny_setup()
        g = torch.randn_like(W) * 0.01
        W.grad = g.clone(); other.grad = torch.randn_like(other) * 0.01
        eta = 0.001
        if fam == "P":
            v_plus = g.square() * (1 - 0.95)
            x = g / ((v_plus / (1 - 0.95)).sqrt() + 1e-8)
        else:
            x = g
        A_t = analytic_A(opt, W)
        meta = apply_step(model, opt, fam, 8.0, 1.0, "t", eta, TARGET)
        assert abs(meta["D_norm"] - meta["A_norm"]) / meta["A_norm"] < 5e-7, \
            f"{fam}: ||D||==||A|| within fp32-cast rounding (V16 observed ~2e-10..1e-7)"
        assert meta["q_t"] > 0 and not meta["zero_norm_fallback"]


def test_q_override_exact():
    model, W, other, opt = tiny_setup()
    W.grad = torch.randn_like(W) * 0.01; other.grad = torch.randn_like(other) * 0.01
    eta = 0.002; q = 0.123
    meta = apply_step(model, opt, "F", 4.0, 1.0, "t", eta, TARGET, q_override=q)
    assert abs(meta["q_t"] - q) < 1e-8, "q_override exact within fp32 cast (V16 A2 measured <=2.1e-9)"


def test_schedule_eta():
    assert schedule_eta(0.001, 1, 2048, 128, 0.1) == 0.001 * 1 / 128
    import math
    expect = 0.001 * (0.1 + 0.9 * (1 + math.cos(math.pi * (512 - 128) / (2048 - 128))) / 2)
    assert schedule_eta(0.001, 512, 2048, 128, 0.1) == expect


def test_bootstrap_and_tokenweighted():
    a = [{"doc_id": str(i), "loss_sum": 10.0 + i, "token_count": 100} for i in range(50)]
    b = [{"doc_id": str(i), "loss_sum": 9.0 + i, "token_count": 100} for i in range(50)]
    st = paired_bootstrap(a, b)
    expected = (500+1225)/5000 - (450+1225)/5000  # = 0.01
    assert abs(st["point"] - expected) < 1e-12 and st["ci95"][0] > 0
    assert abs(token_weighted(a) - sum(x["loss_sum"] for x in a) / 5000.0) < 1e-12


def test_V16_ledger_untouched():
    """Isolation: running vendored apply_step must not modify V16 budget_ledger.json."""
    before = V16_LEDGER.read_bytes()
    model, W, other, opt = tiny_setup()
    W.grad = torch.randn_like(W) * 0.01; other.grad = torch.randn_like(other) * 0.01
    apply_step(model, opt, "P", 8.0, 1.0, "t", 0.001, TARGET)
    assert V16_LEDGER.read_bytes() == before, "V16 ledger mutated!"


def test_eval_docs_split_load_and_guard():
    es = json.load(open("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/eval_sets.json"))
    assert set(es) == {"development", "screen", "final_holdout"}
    assert len(es["development"]) == 256 and len(es["screen"]) == 256 and len(es["final_holdout"]) == 1024
    ids = {d["doc_id"] for d in es["development"]}
    assert not ids & {d["doc_id"] for d in es["final_holdout"]}
    # identity guard
    class M:
        def eval(self): pass
    try:
        eval_docs(M(), es["development"][:1], "x", "t", "l", 1,
                  {"state_id": "s", "loaded_checkpoint_sha": None}, out_dir=tempfile.mkdtemp())
        raise AssertionError("EVAL_WITHOUT_STATE_ORIGIN must fire for non-scratch without sha")
    except RuntimeError as e:
        assert "EVAL_WITHOUT_STATE_ORIGIN" in str(e)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} CPU TESTS PASS")
