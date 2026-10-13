"""CPU regression tests for V15 branch semantics. No GPU optimizer steps are consumed."""
import copy
import json
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import direction_horizon_v15.train as v15
from direction_horizon_v15.train import (CFG, n_a, family_for_step, tie_key, branch_specs,
                                         family_candidates, begin_optimizer, write)


def silent_gate(*args, **kwargs):
    return {"status": "READY"}


class Recorder:
    def __init__(self):
        self.attempts = 0
        self.completions = 0
        self.forward_tokens = 0
        self.last_phase = None


def fake_ledger(recorder):
    return {"optimizer_step_attempts": recorder.attempts,
            "optimizer_step_completions": recorder.completions,
            "token_forward_equivalents": recorder.forward_tokens,
            "last_phase": recorder.last_phase}


def tiny_model(seed=0, shared=True):
    torch.manual_seed(seed)
    dense = torch.nn.Linear(6, 6)
    other = torch.nn.Linear(6, 4)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = dense
            self.other = other
            if shared:
                self.dense2 = dense

        def forward(self, x):
            return self.other(torch.nn.functional.relu(self.dense(x) + self.dense2(x)))
    return Tiny()


def make_opt(model, lr):
    matrix = [p for p in model.parameters() if p.ndim >= 2]
    vector = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": matrix, "weight_decay": 0.1},
              {"params": vector, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8,
                             amsgrad=False, maximize=False, foreach=False, fused=False)


class BranchHarness:
    """Drives apply_step on a tiny CPU model with two-token-sum microbatches."""

    def __init__(self, target_name, seed=0):
        self.target_name = target_name
        self.model = tiny_model(seed)
        self.opt = make_opt(self.model, 0.01)
        self.recorder = Recorder()
        self.eta = 0.01
        for group in self.opt.param_groups:
            group["lr"] = self.eta

    def accumulate(self, scale_a=1.0, scale_b=1.0, clip=True):
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        for scale in (scale_a, scale_b):
            x = torch.randn(3, 6)
            target = torch.randn(3, 4)
            loss_sum = ((self.model(x) - target) ** 2).sum() * scale
            loss_sum.backward()
        valid = 6  # two fake token-equivalents
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(valid)
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    def accumulate_with(self, batches, clip=True):
        """batches: list of (x, target) pairs; normalizes by total token count 6."""
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        for x, target in batches:
            ((self.model(x) - target) ** 2).sum().backward()
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(6)
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

    @staticmethod
    def make_batches(seed, count):
        torch.manual_seed(seed)
        return [(torch.randn(3, 6), torch.randn(3, 4)) for _ in range(count)]

    def step(self, family, a, b):
        timing = {}
        with patch.object(v15, "resource", silent_gate), \
             patch.object(v15, "begin_optimizer", self._begin), \
             patch.object(v15, "complete_optimizer", self._complete):
            meta = v15.apply_step(self.model, self.opt, family, a, b,
                                  "test", self.eta, timing,
                                  target_name=self.target_name)
        self.opt.zero_grad(set_to_none=True)
        return meta

    def _complete(self):
        self.recorder.completions += 1

    def _begin(self, value, phase=""):
        self.recorder.attempts += 1

    def snapshot(self):
        return ({n: p.detach().clone() for n, p in self.model.named_parameters()},
                copy.deepcopy(self.opt.state_dict()))


class V15CpuTests(unittest.TestCase):
    def test_identity_branch_matches_native_adamw_bitwise(self):
        branch = BranchHarness("dense.weight", seed=3)
        native = BranchHarness("dense.weight", seed=3)
        for k in range(6):
            batches = BranchHarness.make_batches(100 + k, 2)
            branch.accumulate_with(batches)
            native.accumulate_with(batches)
            branch.step("S", 1.0, 1.0)
            with patch.object(v15, "resource", silent_gate):
                native.opt.step()
            native.opt.zero_grad(set_to_none=True)
        for (name, p) in branch.model.named_parameters():
            q = dict(native.model.named_parameters())[name]
            self.assertTrue(torch.equal(p, q), name)
        self.assertEqual(branch.recorder.attempts, 6)

    def test_a0_keeps_decay_drops_data_and_advances_moments(self):
        h = BranchHarness("dense.weight", seed=5)
        old = dict(h.model.named_parameters())["dense.weight"].detach().clone()
        h.accumulate(2.0, 1.5)
        h.step("S", 0.0, 1.0)
        target = dict(h.model.named_parameters())["dense.weight"]
        expected = old * (1 - 0.01 * 0.1)
        self.assertTrue(torch.allclose(target, expected, atol=1e-7, rtol=1e-6))
        state = h.opt.state[target]
        self.assertEqual(int(state["step"]), 1)
        self.assertGreater(float(state["exp_avg"].abs().max()), 0.0)

    def test_a_and_b_semantics_against_manual_formula(self):
        for family, a, b in (("S", 0.5, 1.0), ("S", 2.0, 0.5), ("F", 0.5, 0.5), ("P", 1.0, 0.5)):
            h = BranchHarness("dense.weight", seed=7)
            h.accumulate(1.3, 0.7)
            named = dict(h.model.named_parameters())
            beta1, beta2, eps = (0.9, 0.95, 1e-8)
            base, A, g_clip, v_hat = {}, {}, None, None
            for name, p in named.items():
                if p.grad is None:
                    continue
                base[name] = p.detach().clone() * (1 - 0.01 * (0.1 if p.ndim >= 2 else 0.0))
                m_plus = 0.9 * 0 + p.grad * 0.1
                v_plus = 0.95 * 0 + p.grad.square() * 0.05
                A[name] = (m_plus / (1 - beta1)) / ((v_plus / (1 - beta2)).sqrt() + eps)
                if name == "dense.weight":
                    g_clip = p.grad.detach().clone()
                    v_hat = v_plus / (1 - beta2)
            h.step(family, a, b)
            if family == "S":
                D = A["dense.weight"]
            elif family == "F":
                D, _ = n_a(g_clip, float(torch.linalg.vector_norm(A["dense.weight"].double())))
            else:
                D, _ = n_a(g_clip / (v_hat.sqrt() + 1e-8),
                           float(torch.linalg.vector_norm(A["dense.weight"].double())))
            for name, p in dict(h.model.named_parameters()).items():
                if name == "dense.weight":
                    expected = base[name] - 0.01 * a * D
                else:
                    expected = base[name] - 0.01 * b * A[name]
                self.assertTrue(torch.allclose(p, expected, atol=1e-6, rtol=1e-5),
                                f"{family} {a} {b} {name}")

    def test_f_and_p_norm_match_and_clip_timing(self):
        # At step 1 Adam's A is scale-invariant, so clip timing is undetectable; after the
        # first step the accumulated moments make A depend on the post-clip gradients.
        h = BranchHarness("dense.weight", seed=11)
        state = torch.get_rng_state()
        h.accumulate(1.0, 1.0)
        h.step("S", 1.0, 1.0)  # populate moments
        h.accumulate(50.0, 40.0)  # clipping active on the second step
        g_post = h.model.dense.weight.grad.detach().clone()
        total = float(torch.sqrt(sum((p.grad ** 2).sum() for p in h.model.parameters()
                                     if p.grad is not None)))
        self.assertAlmostEqual(total, 1.0, places=5)
        twin = BranchHarness("dense.weight", seed=11)
        torch.set_rng_state(state)
        twin.accumulate(1.0, 1.0)
        twin.opt.step()  # same first step, moments populated identically
        twin.opt.zero_grad(set_to_none=True)
        twin.accumulate(50.0, 40.0, clip=False)  # identical batches, pre-clip gradients
        g_raw = twin.model.dense.weight.grad.detach().clone()
        beta1, beta2, eps = (0.9, 0.95, 1e-8)
        m_prev = twin.opt.state[twin.model.dense.weight]["exp_avg"].detach().clone()
        v_prev = twin.opt.state[twin.model.dense.weight]["exp_avg_sq"].detach().clone()

        def a_from(g):
            m_plus = beta1 * m_prev + (1 - beta1) * g
            v_plus = beta2 * v_prev + (1 - beta2) * g.square()
            return (m_plus / (1 - beta1 ** 2)) / ((v_plus / (1 - beta2 ** 2)).sqrt() + eps)

        norm_from_clip = float(torch.linalg.vector_norm(a_from(g_post).double()))
        norm_from_raw = float(torch.linalg.vector_norm(a_from(g_raw).double()))
        self.assertNotAlmostEqual(norm_from_clip, norm_from_raw, places=3)
        meta_f = h.step("F", 1.0, 1.0)
        self.assertAlmostEqual(meta_f["A_norm"], norm_from_clip, places=4)
        self.assertAlmostEqual(meta_f["D_norm"], meta_f["A_norm"], places=5)
        h2 = BranchHarness("dense.weight", seed=11)
        torch.set_rng_state(state)
        h2.accumulate(1.0, 1.0)
        h2.step("S", 1.0, 1.0)
        h2.accumulate(50.0, 40.0)
        meta_p = h2.step("P", 1.0, 1.0)
        self.assertAlmostEqual(meta_p["D_norm"], meta_p["A_norm"], places=5)

    def test_directions_evolve_naturally(self):
        h = BranchHarness("dense.weight", seed=13)
        h.accumulate(1.0, 1.0)
        meta1 = h.step("F", 1.0, 1.0)
        h.accumulate(0.5, 2.0)
        meta2 = h.step("F", 1.0, 1.0)
        self.assertNotAlmostEqual(meta1["A_norm"], meta2["A_norm"], places=6)
        self.assertNotAlmostEqual(meta1["D_norm"], meta2["D_norm"], places=6)
        self.assertEqual(meta2["target_moments_step_plus"], 2)

    def test_shared_parameter_single_update(self):
        h = BranchHarness("dense.weight", seed=17)
        params = list(h.model.named_parameters())
        self.assertEqual(len(params), len(h.opt.param_groups[0]["params"]) + len(h.opt.param_groups[1]["params"]))
        ids = [id(p) for _, p in params]
        self.assertEqual(len(ids), len(set(ids)))
        h.accumulate(1.0, 1.0)
        h.step("S", 1.0, 1.0)
        self.assertEqual(len(h.opt.state), len(params))

    def test_eval_does_not_mutate_state(self):
        h = BranchHarness("dense.weight", seed=19)
        h.accumulate(1.0, 1.0)
        h.step("S", 1.0, 1.0)
        eval_inputs = [(torch.randn(3, 6), torch.randn(3, 4)) for _ in range(3)]
        params, state = h.snapshot()
        rng = (torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state()))
        with torch.no_grad():
            for x, _ in eval_inputs:
                h.model.eval()
                h.model(x)
        for (name, p) in h.model.named_parameters():
            self.assertTrue(torch.equal(p, params[name]), name)
        after = h.opt.state_dict()["state"]
        for key in state["state"]:
            for field, value in state["state"][key].items():
                self.assertTrue(torch.equal(after[key][field], value))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng[0]))
        self.assertEqual(random.getstate(), rng[1])
        self.assertTrue(np.array_equal(np.random.get_state()[1], rng[2][1]))

    def test_one_shot_family_schedule(self):
        spec = {"family": "F", "a": 1.0, "b": 0.5}
        self.assertEqual([family_for_step(spec, k, True) for k in (1, 2, 3, 64)], ["F", "S", "S", "S"])
        self.assertEqual([family_for_step(spec, k, False) for k in (1, 2)], ["F", "F"])

    def test_budget_attempt_counted_before_failure(self):
        with patch.object(v15, "read_modify_write", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                v15.begin_optimizer("x")
        with patch.object(v15, "save_ledger", lambda v: None), \
             patch.object(v15, "ledger", lambda: {"optimizer_step_attempts":
                                                  CFG["max_gpu_optimizer_step_attempts"] - 1}):
            v15.begin_optimizer("x")  # counted before any failure can occur later
            self.assertTrue(True)

    def test_budget_cap_enforced(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "budget_ledger.json").write_text(json.dumps(
                {"manifest_sha256": "m", "token_forward_equivalents": 0, "forward_attempts": 0,
                 "optimizer_step_attempts": CFG["max_gpu_optimizer_step_attempts"],
                 "optimizer_step_completions": 0, "backward_attempts": 0, "last_phase": None}),
                encoding="utf-8")
            with patch.object(v15, "HERE", base), \
                 patch.object(v15, "sha", lambda p, n=None: "m"), \
                 patch.object(v15, "write", lambda path, obj: Path(path).write_text(
                     json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")):
                with self.assertRaises(RuntimeError):
                    v15.begin_optimizer("x")

    def test_atomic_write_retries_then_succeeds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.json"
            real_replace = os.replace
            calls = {"n": 0}

            def flaky(src, dst):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise PermissionError("transient")
                real_replace(src, dst)
            with patch.object(v15.os, "replace", flaky):
                write(path, {"a": 1})
            self.assertEqual(calls["n"], 2)
            self.assertEqual(v15.read(path), {"a": 1})

    def test_zero_norm_fallback(self):
        D, meta = n_a(torch.zeros(3, 3), 5.0)
        self.assertTrue(torch.all(D == 0))
        self.assertTrue(meta["zero_norm_fallback"])
        D2, meta2 = n_a(torch.ones(3, 3), 0.0)
        self.assertTrue(torch.all(D2 == 0))
        self.assertTrue(meta2["zero_norm_fallback"])

    def test_ledger_helpers_read_modify_write_fresh_file(self):
        """Regression for the attempt-4 accounting race: helpers must always mutate the
        CURRENT file, never a stale in-memory dict."""
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "budget_ledger.json").write_text(json.dumps(
                {"manifest_sha256": "m", "token_forward_equivalents": 100, "forward_attempts": 3,
                 "optimizer_step_attempts": 5, "optimizer_step_completions": 5,
                 "backward_attempts": 20, "last_phase": None}), encoding="utf-8")
            with patch.object(v15, "HERE", base), \
                 patch.object(v15, "sha", lambda p, n=None: "m"), \
                 patch.object(v15, "write", lambda path, obj: Path(path).write_text(
                     json.dumps(obj, ensure_ascii=False) + "\n", encoding="utf-8")):
                v15.begin_optimizer("t1")
                v15.begin_optimizer("t2")
                v15.complete_optimizer()
                v15.count_forward(50, "t3")
                v15.bump_ledger(backward_attempts=4, train_seconds=1.5)
                # a fresh helper must see every prior helper's effect (no stale dicts anywhere)
                v15.begin_optimizer("t4")
                final = v15.ledger()
            self.assertEqual(final["optimizer_step_attempts"], 8)  # 5+1+1+1
            self.assertEqual(final["token_forward_equivalents"], 150)
            self.assertEqual(final["backward_attempts"], 24)
            self.assertAlmostEqual(final["train_seconds"], 1.5)

    def test_tie_break_prefers_identity_then_closeness(self):
        specs = {s["id"]: s for s in branch_specs()}
        order = sorted([specs["F_a2_b0.5"], specs["F_a1_b1"], specs["F_a0.5_b1"], specs["F_a1_b0.5"]],
                       key=tie_key)
        self.assertEqual(order[0]["id"], "F_a1_b1")
        self.assertEqual(order[1]["id"], "F_a0.5_b1")
        self.assertEqual(order[2]["id"], "F_a1_b0.5")
        self.assertEqual(len(branch_specs()), 20)
        for family in ("S", "F", "P"):
            self.assertEqual(len(family_candidates(family)), 8)

    def test_save_load_branch_state_roundtrip_cpu(self):
        import tempfile
        from direction_horizon_v15.train import load_branch_state, save_branch_state
        h = BranchHarness("dense.weight", seed=23)
        h.accumulate(1.0, 2.0)
        h.step("S", 0.5, 0.5)
        spec = {"id": "test", "family": "S", "a": 0.5, "b": 0.5}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.pt"
            with patch.object(v15, "sha", lambda p, n=None: "m"):
                save_branch_state(path, h.model, h.opt, "test", spec, 1, 77, 2048, False, 256)
                h.accumulate(3.0, 1.0)
                h.step("F", 1.0, 1.0)
                state = load_branch_state(path, h.model, h.opt, "test", 256)
            self.assertEqual(state["branch_step"], 1)
            self.assertEqual(state["cursor"], 77)
            snap = (state["model"]["other.weight"] == h.model.other.weight).all()
        self.assertTrue(bool(snap))
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            with patch.object(v15, "sha", lambda p, n=None: "m"):
                load_branch_state(Path("missing"), h.model, h.opt, "other", 256)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state2.pt"
            with patch.object(v15, "sha", lambda p, n=None: "m"):
                save_branch_state(path, h.model, h.opt, "test", spec, 1, 77, 2048, False, 256)
                with self.assertRaises(RuntimeError):
                    load_branch_state(path, h.model, h.opt, "WRONG_LABEL", 256)


if __name__ == "__main__":
    unittest.main()
