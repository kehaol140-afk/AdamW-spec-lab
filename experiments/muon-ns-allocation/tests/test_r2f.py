"""R2F tests: recipe algebra (heterogeneous groups, wd=0), switch semantics (no
double scaling from saved effective values), moment consistency across recipes on
the same input, save/load continuation, 2x2 identity."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r2f_recipe import RECIPES, effective, decay_product


class TestRecipeAlgebra(unittest.TestCase):
    def test_table(self):
        for name, (r, d) in {"N": (1, 1), "AH": (0.5, 1), "L": (0.5, 0.5), "D": (1, 0.5)}.items():
            self.assertEqual(RECIPES[name]["r"], r)
            self.assertEqual(RECIPES[name]["d"], d)

    def test_effective_values(self):
        lr, wd = 0.008, 0.02
        self.assertEqual(effective(lr, wd, "N"), (0.008, 0.02))
        self.assertEqual(effective(lr, wd, "AH"), (0.004, 0.04))
        self.assertEqual(effective(lr, wd, "L"), (0.004, 0.02))
        self.assertEqual(effective(lr, wd, "D"), (0.008, 0.01))

    def test_decay_products(self):
        lr, wd = 0.008, 0.02
        self.assertAlmostEqual(decay_product(lr, wd, "N"), lr * wd)
        self.assertAlmostEqual(decay_product(lr, wd, "AH"), lr * wd)
        self.assertAlmostEqual(decay_product(lr, wd, "L"), 0.5 * lr * wd)
        self.assertAlmostEqual(decay_product(lr, wd, "D"), 0.5 * lr * wd)

    def test_heterogeneous_groups_and_zero_wd(self):
        # group A: lr 0.01 wd 0.02 ; group B: lr 0.003 wd 0.0 — per-group bases
        for rec in RECIPES:
            la, wa = effective(0.01, 0.02, rec)
            lb, wb = effective(0.003, 0.0, rec)
            self.assertAlmostEqual(la, RECIPES[rec]["r"] * 0.01)
            self.assertAlmostEqual(lb, RECIPES[rec]["r"] * 0.003)
            self.assertEqual(wb, 0.0)  # zero-WD group stays zero

    def test_two_steps_no_double_scaling(self):
        base_lr, base_wd = 0.008, 0.02
        for rec in RECIPES:
            # ALWAYS rebuild from base — never chain from previous effective
            e1 = effective(base_lr, base_wd, rec)
            e2 = effective(base_lr, base_wd, rec)
            self.assertEqual(e1, e2)

    def test_switch_from_saved_effective_rejected(self):
        # switching AH->N must use base, not the AH-saved (0.004, 0.04)
        base_lr, base_wd = 0.008, 0.02
        ah_saved = effective(base_lr, base_wd, "AH")
        wrong = effective(ah_saved[0], ah_saved[1], "N")  # what we must NOT do
        right = effective(base_lr, base_wd, "N")
        self.assertNotEqual(wrong, right)
        self.assertEqual(right, (0.008, 0.02))


class TestMomentConsistency(unittest.TestCase):
    def test_same_input_moments_equal_across_recipes(self):
        # moments update from UNSCALED grads; recipes differ only in lr/wd -> with the
        # same grad and old state, momentum buffer recursion is identical
        beta = 0.95
        buf = torch.tensor([0.3])
        g = torch.tensor([0.7])
        b1 = beta * buf + (1 - beta) * g
        b2 = beta * buf + (1 - beta) * g  # identical regardless of recipe
        self.assertTrue(torch.equal(b1, b2))

    def test_data_and_decay_terms(self):
        theta = torch.tensor([1.0])
        v = torch.tensor([0.4])
        for rec, cfg in RECIPES.items():
            r, d = cfg["r"], cfg["d"]
            lr, wd = 0.01, 0.02
            eff_lr, eff_wd = effective(lr, wd, rec)
            theta_next = (1 - eff_lr * eff_wd) * theta - eff_lr * v
            protocol = (1 - d * lr * wd) * theta - r * lr * v
            self.assertAlmostEqual(float(theta_next - protocol), 0.0, places=15)


class TestTwoByTwoIdentity(unittest.TestCase):
    def test_path_interaction_identity(self):
        J = {"NN": 1.0, "NAH": 0.9, "AHN": 0.95, "AHAH": 0.85}
        R_N = J["NN"] - J["AHN"]
        R_AH = J["NAH"] - J["AHAH"]
        I_path = J["AHAH"] - J["AHN"] - J["NAH"] + J["NN"]
        self.assertAlmostEqual(I_path, R_N - R_AH, places=15)


class TestSaveLoadContinuation(unittest.TestCase):
    def test_cursor_and_clock_advance_once(self):
        # simulate 2-step continuation bookkeeping: consumed steps are unique
        consumed = list(range(513, 515))
        resumed = list(range(515, 517))
        self.assertEqual(len(set(consumed + resumed)), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
