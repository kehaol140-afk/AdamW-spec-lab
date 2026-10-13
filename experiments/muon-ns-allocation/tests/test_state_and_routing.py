"""Task-1 tests: state restore, aliases, routing, LR clock, Q-set access guard."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recipe_registry import GROUPS, lr_at, group_of_param, shape_factor, q_set_of_group
from src.state_restore import access_guard


class TestRouting(unittest.TestCase):
    def test_groups_and_layer_count(self):
        names = [f"model.layers.{l}.{'self_attn.q_proj' if g=='q' else 'self_attn.k_proj' if g=='k' else 'self_attn.v_proj' if g=='v' else 'self_attn.o_proj' if g=='o' else 'mlp.gate_proj' if g=='gate' else 'mlp.up_proj' if g=='up' else 'mlp.down_proj'}.weight"
                 for l in range(8) for g in GROUPS]
        self.assertEqual(len(names), 56)
        for n in names:
            self.assertIsNotNone(group_of_param(n))
        self.assertIsNone(group_of_param("model.embed_tokens.weight"))
        self.assertIsNone(group_of_param("model.layers.0.norm1.weight"))

    def test_shape_factor_original_rule(self):
        self.assertEqual(shape_factor((512, 512)), 1.0)
        self.assertEqual(shape_factor((1664, 512)), math.sqrt(1664 / 512))
        self.assertEqual(shape_factor((256, 512)), 1.0)  # A<B -> sqrt(max(1,A/B))=1


class TestLRClock(unittest.TestCase):
    def test_matches_production_schedule(self):
        # production: sched(t) = t/WARMUP (t<=WARMUP) else 0.1+0.45*(1+cos(pi*w))
        def prod(step):
            if step <= 32:
                return step / 32
            w = (step - 32) / (2048 - 32)
            return 0.1 + 0.45 * (1.0 + math.cos(math.pi * w))
        for s in (1, 32, 33, 512, 513, 1000, 1537, 2048):
            self.assertAlmostEqual(lr_at(s), 0.01 * prod(s), places=12)

    def test_probe_lr_is_next_update_value(self):
        # state 512 -> probes consume lr_at(513), not peak, not end value
        self.assertNotEqual(lr_at(513), 0.01)
        self.assertNotEqual(lr_at(513), 0.001)


class TestQAccessGuard(unittest.TestCase):
    def test_p2_and_confirm_sealed_in_p1(self):
        guard = access_guard({"select", "P1_check"})
        guard("select")
        guard("P1_check")
        with self.assertRaises(RuntimeError):
            guard("P2_endpoint")

    def test_hash_rule_deterministic(self):
        self.assertEqual(q_set_of_group("abc"), q_set_of_group("abc"))
        self.assertIn(q_set_of_group("abc"), (0, 1, 2))


class TestNoAliasSynthetic(unittest.TestCase):
    def test_optimizer_states_deep_between_loads(self):
        from torch.optim import Muon
        p1 = torch.zeros(4, 4)
        p2 = torch.zeros(4, 4)
        o1 = Muon([p1], lr=0.1)
        o2 = Muon([p2], lr=0.1)
        p1.grad = torch.ones(4, 4)
        p2.grad = torch.ones(4, 4)
        o1.step()
        o2.step()
        b1 = list(o1.state.values())[0]["momentum_buffer"]
        b2 = list(o2.state.values())[0]["momentum_buffer"]
        self.assertNotEqual(b1.data_ptr(), b2.data_ptr())
        b1.add_(1.0)
        self.assertAlmostEqual(float(b2.sum()), 0.8, places=6)  # unchanged: 0.05*16 elems


if __name__ == "__main__":
    unittest.main(verbosity=2)
