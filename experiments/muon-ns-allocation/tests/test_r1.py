"""R1 tests: tie keys, extension union, norm directions, Z/P semantics, selection
guards (calibration-only). Synthetic; no GPU."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r1_selection import tie_key, select_family, boundary_touch, compute_extension


class TestTieKeys(unittest.TestCase):
    def test_native_closest_wins_among_ties(self):
        # strictly best (gap > tie tol) -> only the best is near-best
        cands = [
            {"id": "far", "a": 0.5, "b": 1.25, "structure": "NS5", "V_mean": 1.000005},
            {"id": "native", "a": 1.0, "b": 1.0, "structure": "NS5", "V_mean": 1.000003},
            {"id": "mid", "a": 0.75, "b": 1.0, "structure": "NS5", "V_mean": 1.000000},
        ]
        sel, near = select_family(cands, lambda r: True, None)
        self.assertEqual(sel["id"], "mid")
        self.assertEqual(len(near), 1)
        # exact tie (within 1e-6) -> closest-to-native rule picks native
        for c in cands:
            c["V_mean"] = 2.0
        sel, near = select_family(cands, lambda r: True, None)
        self.assertEqual(sel["id"], "native")
        self.assertEqual(len(near), 3)

    def test_b_tie_break_before_a(self):
        c1 = {"id": "x", "a": 0.5, "b": 1.0, "structure": "NS5", "V_mean": 1.0}
        c2 = {"id": "y", "a": 1.0, "b": 0.75, "structure": "NS5", "V_mean": 1.0}
        self.assertLess(tie_key(c2), tie_key(c1))  # |log b| smaller wins


class TestExtension(unittest.TestCase):
    def _rows(self, state, best_a, best_b, struct="NS5"):
        rows = []
        for a in (0.5, 0.75, 1.0):
            for b in (0.75, 1.0, 1.25):
                for s in ("NS5", "NS3_MATCHED"):
                    if (a, b, s) == (best_a, best_b, struct):
                        v = 2.0
                    elif s == "NS3_MATCHED":
                        # give NS3M an interior optimum (0.75, 1.0) so its family
                        # does not spuriously touch boundaries in synthetic rows
                        v = 4.0 + 0.1 * (abs(a - 0.75) + abs(b - 1.0))
                    else:
                        v = 3.0 + 0.1 * (abs(a - best_a) + abs(b - best_b))
                    rows.append({"id": f"{a}|{b}|{s}", "a": a, "b": b,
                                 "structure": s, "V_mean": v})
        return rows

    def test_low_a_boundary_triggers_025(self):
        cal = {"512": self._rows("512", 0.5, 1.0)}
        A_f, B_f, trig, union = compute_extension(cal, [0.5, 0.75, 1.0], [0.75, 1.0, 1.25])
        self.assertIn(0.25, A_f)
        self.assertNotIn(1.5, A_f)
        self.assertNotIn(0.5, B_f)
        self.assertNotIn(1.5, B_f)

    def test_high_b_boundary_only_for_A_S_families(self):
        cal = {"512": self._rows("512", 1.0, 1.25)}
        A_f, B_f, trig, union = compute_extension(cal, [0.5, 0.75, 1.0], [0.75, 1.0, 1.25])
        self.assertIn(1.5, B_f)

    def test_g_family_b_fixed_no_b_trigger(self):
        rows = []
        for a in (0.5, 0.75, 1.0):
            rows.append({"id": f"{a}|1.0|NS5", "a": a, "b": 1.0, "structure": "NS5",
                         "V_mean": 2.0 if a == 0.5 else 3.0})
            rows.append({"id": f"{a}|1.0|NS3M", "a": a, "b": 1.0, "structure": "NS3_MATCHED",
                         "V_mean": 5.0})
        cal = {"512": rows}
        A_f, B_f, trig, union = compute_extension(cal, [0.5, 0.75, 1.0], [0.75, 1.0, 1.25])
        self.assertIn(0.25, A_f)
        self.assertEqual(B_f, [0.75, 1.0, 1.25])  # no b trigger

    def test_union_across_states(self):
        cal = {"512": self._rows("512", 0.5, 1.0), "1536": self._rows("1536", 1.0, 1.25)}
        A_f, B_f, trig, union = compute_extension(cal, [0.5, 0.75, 1.0], [0.75, 1.0, 1.25])
        self.assertIn(0.25, A_f)
        self.assertIn(1.5, B_f)


class TestZPSemantics(unittest.TestCase):
    def test_z_equals_a0_point_semantically(self):
        # Z (zero Muon data) == candidate with a=0 (any b/structure) by construction:
        # delta = -u5 for all Muon matrices. Verified at engine level by identical
        # parameter tensors; here we assert the algebra on a tiny array.
        import numpy as np
        u5 = np.array([0.3, -0.2])
        a = 0.0
        b = 1.25
        delta_candidate = a * b * u5 - u5  # O-group form with a=0
        delta_Z = -u5
        self.assertLess(np.abs(delta_candidate - delta_Z).max(), 1e-15)


if __name__ == "__main__":
    unittest.main(verbosity=2)
