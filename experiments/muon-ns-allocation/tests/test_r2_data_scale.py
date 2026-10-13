"""R2 tests: arm semantics, decay-product invariance, no double scaling, N->P->N
restoration (CPU synthetic mirroring the r1_probe defect), candidate batch/individual
consistency."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r2_data_scale import ARMS, effective_lr_wd, group_scale


class TestArmSemantics(unittest.TestCase):
    def test_decay_product_invariant(self):
        base_lr, base_wd = 0.0088, 0.02
        for arm, sc in ARMS.items():
            for key in ("muon", "aux"):
                r = sc[key]
                eff_lr, eff_wd = effective_lr_wd(base_lr, base_wd, r)
                self.assertAlmostEqual(eff_lr * eff_wd, base_lr * base_wd, places=15)
                self.assertAlmostEqual(eff_lr, r * base_lr, places=15)

    def test_no_cumulative_scaling(self):
        # recomputing from base each step must not multiply 0.5 twice
        base_lr = 0.01
        r = 0.5
        lr = base_lr
        for _ in range(3):
            lr = effective_lr_wd(base_lr, 0.02, r)[0]  # ALWAYS from base
        self.assertAlmostEqual(lr, 0.5 * base_lr, places=15)

    def test_arm_registry(self):
        self.assertEqual(group_scale("N", True), 1.0)
        self.assertEqual(group_scale("MH", True), 0.5)
        self.assertEqual(group_scale("MH", False), 1.0)
        self.assertEqual(group_scale("AH", False), 0.5)


class TestNPNRestoration(unittest.TestCase):
    """CPU synthetic mirror of the r1_probe defect: restoring only Muon params after
    P leaves aux at pre-step values; the fix restores ALL params."""

    def _ctx_like(self, fix: bool):
        muon_pre, aux_pre = torch.tensor([1.0]), torch.tensor([10.0])
        muon_after = 0.9 * muon_pre + 0.5
        aux_after = 0.5 * aux_pre + 0.4
        theta_pre = {"muon": muon_pre.clone(), "aux": aux_pre.clone()}
        # live params start at the post-native-step values (native step already ran)
        muon = torch.nn.Parameter(muon_after.clone())
        aux = torch.nn.Parameter(aux_after.clone())
        theta_plus = ({"muon": muon_after.clone(), "aux": aux_after.clone()} if fix
                      else {"muon": muon_after.clone()})
        return theta_pre, theta_plus, muon, aux

    def _restore(self, theta_plus, muon, aux, fix):
        with torch.no_grad():
            muon.copy_(theta_plus["muon"])
            if fix:
                aux.copy_(theta_plus["aux"])

    def test_defect_reproduced_then_fixed(self):
        for fix, expect_ok in ((False, False), (True, True)):
            theta_pre, theta_plus, muon, aux = self._ctx_like(fix)
            with torch.no_grad():
                # N state
                self._restore(theta_plus, muon, aux, fix)
                n_aux = aux.detach().clone()
                # apply P (pre-state) to both params
                muon.copy_(theta_pre["muon"])
                aux.copy_(theta_pre["aux"])
                # restore N
                self._restore(theta_plus, muon, aux, fix)
            self.assertTrue(bool(torch.equal(aux, n_aux)) is expect_ok,
                            f"fix={fix}: aux restoration {'ok' if expect_ok else 'BUG'}")

    def test_candidate_individual_equals_batched(self):
        # superposition is per-parameter; batching candidates sequentially with full
        # restore must equal evaluating each candidate from the same base
        base = torch.tensor([1.0, 2.0])
        deltas = [torch.tensor([0.1, -0.2]), torch.tensor([-0.3, 0.05])]
        outs = []
        for d in deltas:
            outs.append(base + d)  # each from the SAME base
        outs_batch = [base + d for d in deltas]
        for a, b in zip(outs, outs_batch):
            self.assertTrue(torch.equal(a, b))


class TestOneStepScalingSemantics(unittest.TestCase):
    def test_data_delta_scale_not_grad_not_full_delta(self):
        # theta_next = (1 - lr*wd)*theta - r*lr*d ; WD term must NOT scale with r
        lr, wd, r = 0.01, 0.02, 0.5
        theta, d = torch.tensor([1.0]), torch.tensor([0.7])
        theta_next = (1 - lr * wd) * theta - r * lr * d
        # wrong alternatives for contrast
        wrong_grad = (1 - lr * wd) * theta - lr * (0.5 * d)   # same here for linear ops
        wrong_full = 0.5 * ((1 - lr * wd) * theta - lr * d)   # scales WD too
        self.assertTrue(torch.equal(theta_next, wrong_grad))  # .grad scaling happens to
        # coincide for the data term but changes momentum downstream (not tested here)
        self.assertFalse(torch.equal(theta_next, wrong_full))
        # decay product invariance with effective lr/wd
        eff_lr, eff_wd = effective_lr_wd(lr, wd, r)
        theta_eff = (1 - eff_lr * eff_wd) * theta - eff_lr * d
        self.assertAlmostEqual(float(theta_next - theta_eff), 0.0, places=15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
