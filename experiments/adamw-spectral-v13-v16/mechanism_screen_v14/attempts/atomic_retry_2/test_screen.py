import math
import unittest

import torch

from mechanism_screen_v14 import screen


class ScreenUnitTests(unittest.TestCase):
    def test_candidate_grid_is_fixed(self):
        rows = screen.candidates()
        self.assertEqual(len(rows), 32)
        self.assertEqual({x["family"] for x in rows},
                         {"scalar", "spectral", "fresh_gradient", "fresh_preconditioned"})
        self.assertEqual(sum(x["family"] == "scalar" for x in rows), 8)
        self.assertEqual(sum(x["family"] == "spectral" for x in rows), 12)

    def test_zero_norm_fallback_and_match(self):
        a = torch.tensor([[3., 4.]])
        z, meta = screen.normalize_to_a(torch.zeros_like(a), a)
        self.assertTrue(meta["zero_norm_fallback"])
        self.assertTrue(torch.equal(z, torch.zeros_like(a)))
        x, meta = screen.normalize_to_a(torch.tensor([[1., 2.]]), a)
        self.assertFalse(meta["zero_norm_fallback"])
        self.assertAlmostEqual(torch.linalg.vector_norm(x).item(), 5., places=6)

    def test_tie_prefers_simple_original_controls(self):
        rows = screen.candidates()
        scalar = [c for c in rows if c["family"] == "scalar"]
        self.assertEqual(min(scalar, key=screen.tie_key)["id"], "scalar_g1_s1")

    def test_token_weighting_differs_from_document_mean(self):
        rows = [{"loss_sum": 1., "token_count": 1}, {"loss_sum": 90., "token_count": 10}]
        self.assertAlmostEqual(screen.aggregate(rows), 91/11)
        self.assertNotAlmostEqual(screen.aggregate(rows), (1+9)/2)

    def test_frozen_state_formula_and_spectral_identity(self):
        # No optimizer step, no model creation: verify the exact intervention algebra.
        a = torch.tensor([[2., 0.], [0., 1.]])
        u,s,v = torch.linalg.svd(a.double(), full_matrices=False)
        head = (s[0]*torch.outer(u[:,0],v[0])).float()
        self.assertTrue(torch.allclose(head+(a-head),a))
        b = torch.ones_like(a)
        eta = .003
        ordinary = b-eta*a
        spectral_rho_one = b-eta*(head+(a-head))
        self.assertTrue(torch.equal(ordinary,spectral_rho_one))
        self.assertTrue(torch.equal(b-eta*0*a,b))


if __name__ == "__main__":
    unittest.main()
