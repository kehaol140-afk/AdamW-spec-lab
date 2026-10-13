"""Task-2 tests: one-step equivalence, superposition, tri-probe norms, geometry, FLOPs."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.optim import Muon
from torch.optim._muon import _zeropower_via_newtonschulz

from src.recipe_registry import NS_COEFFS, EPS_NORM, shape_factor
from src.cost_meter import ns_flops_per_step, group_step_flops, kv_flops_ratio


def manual_muon_step(param, grad, buf, lr, wd, K=5):
    buf2 = buf.lerp(grad, 0.05)                    # beta=0.95
    update = grad.lerp(buf2, 0.95)                 # nesterov
    y = _zeropower_via_newtonschulz(update, NS_COEFFS, K, EPS_NORM)
    newp = param * (1 - lr * wd) - lr * shape_factor(param.shape) * y.float()
    return newp, buf2


class TestOneStepEquivalence(unittest.TestCase):
    def test_k5_manual_equals_production_step(self):
        torch.manual_seed(0)
        p = torch.randn(64, 32)
        opt = Muon([p], lr=0.01, weight_decay=0.02, momentum=0.95, nesterov=True,
                   ns_steps=5, ns_coefficients=NS_COEFFS, eps=EPS_NORM, adjust_lr_fn=None)
        g = torch.randn(64, 32)
        p.grad = g
        # seed the buffer like the anchor (non-zero initial momentum)
        buf0 = torch.randn(64, 32) * 0.1
        opt.state[p] = {"momentum_buffer": buf0.clone()}
        before = p.detach().clone()
        opt.step()
        manual, _ = manual_muon_step(before, g, buf0.clone(), 0.01, 0.02, K=5)
        # one fp32 rounding path difference between manual expression and mul_/add_ order
        self.assertLess(float((p - manual).abs().max()), 5e-8)

    def test_superposition_equals_direct(self):
        torch.manual_seed(1)
        p = torch.randn(64, 64)
        g = torch.randn(64, 64)
        lr, wd = 0.01, 0.02
        u5 = -lr * shape_factor(p.shape) * _zeropower_via_newtonschulz(
            g, NS_COEFFS, 5, EPS_NORM).float()
        u3 = -lr * shape_factor(p.shape) * _zeropower_via_newtonschulz(
            g, NS_COEFFS, 3, EPS_NORM).float()
        theta0 = p * (1 - lr * wd) + u5
        superp = theta0 + (u3 - u5)
        direct = p * (1 - lr * wd) + u3
        # documented superposition rounding (addendum 6.1): extra add/subtract path
        self.assertLess(float((superp - direct).abs().max()), 1e-6)


class TestTriProbe(unittest.TestCase):
    def test_norm_identities(self):
        torch.manual_seed(2)
        h = torch.randn(32, 16)
        lr = 0.01
        sf = shape_factor(h.shape)
        u5 = -lr * sf * _zeropower_via_newtonschulz(h, NS_COEFFS, 5, EPS_NORM).float()
        u3 = -lr * sf * _zeropower_via_newtonschulz(h, NS_COEFFS, 3, EPS_NORM).float()
        n5, n3 = float(u5.norm()), float(u3.norm())
        gamma = n3 / n5
        live = u5 * gamma
        direct = u3 / gamma
        self.assertAlmostEqual(float(live.norm()), n3, places=6)
        self.assertAlmostEqual(float(direct.norm()), n5, places=6)


class TestGeometry(unittest.TestCase):
    def test_identity_has_zero_E(self):
        y = torch.eye(64, 128)[:, :64].contiguous()  # orthonormal-ish columns
        r = min(y.shape)
        e = float((y @ y.T - torch.eye(r)).norm() / np.sqrt(r))
        self.assertLess(e, 1e-5)

    def test_ns_outputs_have_small_E(self):
        torch.manual_seed(3)
        h = torch.randn(64, 64)
        y = _zeropower_via_newtonschulz(h, NS_COEFFS, 5, EPS_NORM).float()
        r = min(y.shape)
        e = float((y @ y.T - torch.eye(r)).norm() / np.sqrt(r))
        self.assertLess(e, 0.5)  # quintic leaves sigma' in [0.5,1.5]


class TestCost(unittest.TestCase):
    def test_flops_formulas(self):
        self.assertEqual(ns_flops_per_step((512, 512)), 6 * 512 ** 3)
        self.assertEqual(ns_flops_per_step((256, 512)), 4 * 256 ** 2 * 512 + 2 * 256 ** 3)
        # orientation invariance
        self.assertEqual(ns_flops_per_step((512, 256)), ns_flops_per_step((256, 512)))

    def test_ratio_and_same_budget_pairs(self):
        shapes = {g: [(512, 512)] * 8 for g in ("q", "o")}
        shapes.update({g: [(256, 512)] * 8 for g in ("k", "v")})
        shapes.update({g: [(1664, 512)] * 8 for g in ("gate", "up", "down")})
        sf = group_step_flops(shapes)
        self.assertAlmostEqual(kv_flops_ratio(sf, {}), 1.0)
        # same-shape swap pair is exactly 1.0
        kv = {"q": 3, "o": 7}
        self.assertAlmostEqual(kv_flops_ratio(sf, kv), 1.0, places=9)
        # cross-shape pair far outside [0.98,1.02]
        kv_bad = {"k": 3, "gate": 7}
        r = kv_flops_ratio(sf, kv_bad)
        self.assertTrue(r < 0.98 or r > 1.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
