import copy
import math
import os
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

os.environ["USE_TF"] = "0"

from pilot_core import (ROOT, adamw_next, atomic_json, isolated_probe, load_checkpoint,
                        loss_sum_count, model_and_optimizer, read_json, save_checkpoint, schedule_lr, sha_file,
                        spectral_candidate, split_head)


class AdamWContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_one_and_256_steps_fp64_against_torch(self):
        torch.manual_seed(7)
        shapes = [(3, 3), (5, 3), (7,)]
        measured = {}
        for shape in shapes:
            with self.subTest(shape=shape):
                p = torch.nn.Parameter(torch.randn(shape, dtype=torch.float64))
                opt = torch.optim.AdamW([p], lr=0.003, betas=(0.9, 0.95), eps=1e-8,
                                        weight_decay=0.1, amsgrad=False, maximize=False,
                                        foreach=False, fused=False)
                max_error = 0.0
                for step in range(256):
                    grad = torch.randn_like(p) * (1 + step / 100)
                    old = p.detach().clone()
                    expected, direction, _ = adamw_next(old, grad, opt.state[p], lr=0.003,
                                                         wd=0.1, beta1=0.9, beta2=0.95,
                                                         eps=1e-8)
                    p.grad = grad
                    opt.step()
                    max_error = max(max_error, float((p.detach() - expected).abs().max()))
                    torch.testing.assert_close(p.detach(), expected, rtol=1e-10, atol=1e-12)
                    self.assertIsNotNone(direction)
                self.assertLess(max_error, 1e-12)
                measured[str(shape)] = max_error
        atomic_json(ROOT / "artifacts" / "numerics" / "adamw_parity.json",
                    {"status": "PASS", "dtype": "float64", "steps_per_shape": 256,
                     "shapes": measured, "rtol": 1e-10, "atol": 1e-12,
                     "torch_version": torch.__version__})

    def test_none_gradient_vs_zero_gradient_and_fixed_decay(self):
        for grad in (None, torch.zeros(2, 2, dtype=torch.float64)):
            p = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
            opt = torch.optim.AdamW([p], lr=0.01, weight_decay=0.1, foreach=False, fused=False)
            expected, direction, _ = adamw_next(p.detach(), grad, opt.state[p], lr=0.01,
                                                 wd=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
            p.grad = grad
            opt.step()
            torch.testing.assert_close(p.detach(), expected, rtol=0, atol=1e-12)
            if grad is None:
                self.assertIsNone(direction)
                self.assertEqual(len(opt.state[p]), 0)
            else:
                self.assertIsNotNone(direction)
                self.assertLess(float((p - 0.999).abs().max()), 1e-12)

    def test_svd_identity_equivalence_and_decay(self):
        torch.manual_seed(5)
        a = torch.randn(7, 5, dtype=torch.float64)
        old = torch.randn_like(a)
        head, rest, _ = split_head(a)
        torch.testing.assert_close(head + rest, a, rtol=1e-12, atol=1e-12)
        normal = old * (1 - 0.003 * 0.1) - 0.003 * a
        identity = spectral_candidate(old, head, rest, 0.003, 0.1, 1, 1)
        torch.testing.assert_close(identity, normal, rtol=1e-12, atol=1e-12)
        zero_scale = spectral_candidate(old, head, rest, 0.003, 0.1, 2, 0)
        torch.testing.assert_close(zero_scale, old * (1 - 0.003 * 0.1))
        torch.testing.assert_close(0.5 * head + rest, 0.5 * (head + 2 * rest))
        self.assertGreater(float((old * (1 - 0.003 * 0.1) - old * (1 - 0.006 * 0.1)).norm()), 0)

    def test_output_orthogonality_but_loss_cross_term(self):
        torch.manual_seed(11)
        a = torch.randn(6, 4, dtype=torch.float64)
        head, rest, _ = split_head(a)
        x = torch.randn(4, 9, dtype=torch.float64)
        cross = float(((head @ x) * (rest @ x)).sum())
        self.assertLess(abs(cross), 1e-11)
        joint = ((1.7 * head + rest) @ x).square().sum()
        separate = 1.7 ** 2 * (head @ x).square().sum() + (rest @ x).square().sum()
        torch.testing.assert_close(joint, separate, rtol=1e-12, atol=1e-12)
        # An arbitrary non-isotropic Hessian can couple orthogonal SVD components.
        h = head.reshape(-1)
        r = rest.reshape(-1)
        v = h + r
        coupling = float((h @ v) * (v @ r))
        self.assertGreater(abs(coupling), 1e-4)

    def test_paired_diagonal_residual_not_just_off_diagonal(self):
        m = torch.tensor([[2., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
        a = torch.tensor([[0., 0., 1.], [0., 0., 0.]], dtype=torch.float64)
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        paired = torch.einsum("ki,kl,il->i", u, a, vh)
        self.assertEqual(float(paired.square().sum()), 0.0)
        self.assertEqual(float(a.square().sum()), 1.0)

    def test_token_weighting_shift_and_real_eos_value(self):
        torch.manual_seed(17)
        ids = torch.tensor([[1, 2, 2, 0], [1, 3, 0, 0]])
        lengths = torch.tensor([3, 2])
        logits = torch.randn(2, 4, 5, dtype=torch.float64)
        total, count = loss_sum_count(logits, ids, lengths)
        first = torch.nn.functional.cross_entropy(logits[0, :2], ids[0, 1:3], reduction="sum")
        second = torch.nn.functional.cross_entropy(logits[1, :1], ids[1, 1:2], reduction="sum")
        self.assertEqual(count, 3)
        torch.testing.assert_close(total, (first + second).float(), rtol=1e-6, atol=1e-6)
        self.assertNotAlmostEqual(float(total / count), float((first / 2 + second) / 2), places=5)

    def test_lookahead_window_257_ids_256_logits(self):
        ids = torch.tensor([[1, 2, 3, 4], [1, 3, 2, 4]])
        lengths = torch.tensor([4, 4])
        logits = torch.randn(2, 3, 5)
        loss, count = loss_sum_count(logits, ids, lengths)
        expected = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 5), ids[:, 1:].reshape(-1), reduction="sum")
        self.assertEqual(count, 6)
        torch.testing.assert_close(loss, expected)

    def test_eval_input_slice_matches_full_forward_prefix(self):
        torch.manual_seed(29)
        ids = torch.tensor([[1, 3, 4, 2, 5, 2, 6]])
        length = ids.shape[1]
        embedding = torch.nn.Embedding(8, 4)
        head = torch.nn.Linear(4, 8)
        with torch.no_grad():
            logits_full = head(embedding(ids))
            logits_sliced = head(embedding(ids[:, :-1]))
        torch.testing.assert_close(logits_sliced, logits_full[:, :-1], rtol=0, atol=0)
        loss, count = loss_sum_count(logits_sliced, ids, torch.tensor([length]))
        expected = torch.nn.functional.cross_entropy(
            logits_sliced.reshape(-1, 8), ids[:, 1:].reshape(-1), reduction="sum")
        self.assertEqual(count, length - 1)
        torch.testing.assert_close(loss, expected)

    def test_probe_restores_buffers_moments_rng_grads_mode_cursor(self):
        torch.manual_seed(3)
        random.seed(3)
        np.random.seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        model.register_buffer("nonpersistent_marker", torch.tensor([2.]), persistent=False)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False, fused=False)
        model.train()
        model(torch.randn(4, 3)).sum().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        state = {"cursor": 9, "step": 1}
        before_model = copy.deepcopy(model.state_dict())
        before_opt = copy.deepcopy(opt.state_dict())
        before_torch = torch.get_rng_state().clone()
        before_np = copy.deepcopy(np.random.get_state())
        before_py = random.getstate()
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with isolated_probe(model, opt, state):
                model.eval()
                model(torch.randn(4, 3)).sum().backward(); opt.step()
                model.nonpersistent_marker.add_(4)
                state["cursor"] = 100
                random.random(); np.random.random()
                raise RuntimeError("intentional")
        self.assertTrue(model.training)
        self.assertEqual(state, {"cursor": 9, "step": 1})
        for k, v in before_model.items():
            torch.testing.assert_close(model.state_dict()[k], v, rtol=0, atol=0)
        for k in before_opt["state"]:
            for name, val in before_opt["state"][k].items():
                other = opt.state_dict()["state"][k][name]
                if torch.is_tensor(val):
                    torch.testing.assert_close(other, val, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(random.getstate(), before_py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], before_np[1]))
        torch.testing.assert_close(model.nonpersistent_marker, torch.tensor([2.]), rtol=0, atol=0)

    def test_checkpoint_continuous_vs_resume(self):
        (ROOT / "work").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "work") as temp:
            torch.manual_seed(23); random.seed(23); np.random.seed(23)
            model = torch.nn.Linear(4, 3)
            opt = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False, fused=False)
            aliases = {name: [name] for name, _ in model.named_parameters()}
            data = torch.randn(8, 2, 4)
            target = torch.randn(8, 2, 3)
            def step(k):
                opt.zero_grad(set_to_none=True)
                torch.nn.functional.mse_loss(model(data[k]), target[k]).backward()
                opt.step(); opt.zero_grad(set_to_none=True)
            for k in range(4): step(k)
            path = Path(temp) / "state.pt"
            save_checkpoint(path, model, opt, step=4, cursor=4, total_tokens=24,
                            config_hash="config", data_hash="data", aliases=aliases)
            digest = sha_file(path)
            expected_rng = (float(torch.rand(())), random.random(), float(np.random.random()))
            for k in range(4, 8): step(k)
            continuous = copy.deepcopy(model.state_dict())
            continuous_opt = copy.deepcopy(opt.state_dict())
            payload = load_checkpoint(path, model, opt, config_hash="config", data_hash="data", aliases=aliases)
            self.assertEqual((payload["successful_optimizer_step"], payload["data_cursor"],
                              payload["total_effective_tokens"]), (4, 4, 24))
            restored_rng = (float(torch.rand(())), random.random(), float(np.random.random()))
            self.assertEqual(restored_rng, expected_rng)
            for k in range(4, 8): step(k)
            for name, value in continuous.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            for key in continuous_opt["state"]:
                for field, value in continuous_opt["state"][key].items():
                    other = opt.state_dict()["state"][key][field]
                    if torch.is_tensor(value):
                        torch.testing.assert_close(other, value, rtol=0, atol=0)
            self.assertEqual(sha_file(path), digest)
            atomic_json(ROOT / "artifacts" / "numerics" / "resume_replay_cpu.json",
                        {"status": "PASS", "steps": "4+restore+4 vs continuous 8",
                         "max_parameter_abs_error": 0, "max_optimizer_state_abs_error": 0,
                         "checkpoint_file_unchanged": True})

    def test_shared_parameter_updated_once(self):
        tied = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.first = tied; self.second = tied
        model = Tied()
        self.assertEqual(len(list(model.parameters())), 1)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1,
                                foreach=False, fused=False)
        tied.grad = torch.zeros_like(tied)
        opt.step()
        torch.testing.assert_close(tied, torch.full_like(tied, 0.99), rtol=0, atol=1e-12)

    def test_local_minimind_fp32_actual_target_update(self):
        binding_file = ROOT / "bindings.local.json"
        if not binding_file.exists():
            self.skipTest("preflight binding unavailable")
        binding = read_json(binding_file)
        cfg = read_json(ROOT / "config" / "local_v1.json")
        torch.manual_seed(31)
        model, opt, aliases, count = model_and_optimizer(cfg, binding, 0.001, "cpu")
        self.assertLessEqual(count, cfg["model"]["max_parameters"])
        ids = torch.randint(0, 6400, (1, 9), dtype=torch.long)
        lengths = torch.tensor([9])
        logits = model(ids, use_cache=False).logits
        loss_sum, token_count = loss_sum_count(logits, ids, lengths)
        (loss_sum / token_count).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["baseline"]["global_grad_clip_norm"])
        target = model.model.layers[1].mlp.down_proj.weight
        old = target.detach().clone()
        grad = target.grad.detach().clone()
        expected, direction, _ = adamw_next(old, grad, opt.state[target], lr=0.001,
                                             wd=0.1, beta1=0.9, beta2=0.95, eps=1e-8)
        opt.step()
        max_error = float((target.detach() - expected).abs().max())
        torch.testing.assert_close(target.detach(), expected, rtol=1e-5, atol=1e-7)
        self.assertGreater(float(direction.norm()), 0)
        opt.zero_grad(set_to_none=True)
        state = {"cursor": 0}
        baseline = target.detach().clone()
        outputs = []
        for _ in range(3):
            with isolated_probe(model, opt, state):
                outputs.append(float(model(ids, use_cache=False).logits.sum().detach()))
                with torch.no_grad():
                    target.add_(1)
                state["cursor"] = 999
            torch.testing.assert_close(target.detach(), baseline, rtol=0, atol=0)
            self.assertEqual(state["cursor"], 0)
        self.assertEqual(outputs, [outputs[0]] * 3)
        atomic_json(ROOT / "artifacts" / "numerics" / "identity_probe_cpu.json",
                    {"status": "PASS", "data": "synthetic token IDs for CPU numerical test only",
                     "model_parameters": count, "actual_fp32_adamw_max_abs_error": max_error,
                     "repeat_logit_sum": outputs, "repeat_spread": max(outputs) - min(outputs),
                     "model_optimizer_rng_cursor_restored": True})


if __name__ == "__main__":
    unittest.main()
