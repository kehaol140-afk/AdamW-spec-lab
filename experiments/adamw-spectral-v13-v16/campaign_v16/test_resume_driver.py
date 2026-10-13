"""CPU-only control-plane regression tests; never import campaign_v16.core."""
import unittest
import sys

from campaign_v16 import tools_resume_driver as driver


class ResumeDriverTests(unittest.TestCase):
    def test_controller_import_is_light(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_original_wall_deadline_blocks_worker(self):
        ledger = {"wall_started_at": 1000, "pause_seconds": 0,
                  "optimizer_step_attempts": 4, "token_forward_equivalents": 100}
        self.assertEqual(driver.budget_status(ledger, {"stop": None}, 1000 + 72 * 3600 + 1),
                         "BLOCKED_WALL")

    def test_cumulative_pause_cannot_be_reset_by_episode(self):
        ledger = {"wall_started_at": 1000, "pause_seconds": 7200,
                  "optimizer_step_attempts": 4, "token_forward_equivalents": 100}
        self.assertEqual(driver.budget_status(ledger, {"stop": "STOP_RESOURCE"}, 2000),
                         "BLOCKED_PAUSE_BUDGET")
        self.assertEqual(ledger["pause_seconds"], 7200)

    def test_ram_needs_three_consecutive_ten_second_samples(self):
        readings = iter([2.7, 2.65, 2.61])
        slept = []
        ok = driver.ram_ok(sample=lambda: next(readings) * 2**30,
                           sleep=lambda s: slept.append(s))
        self.assertTrue(ok)
        self.assertEqual(slept, [10, 10])
        readings = iter([2.7, 2.59, 2.8])
        self.assertFalse(driver.ram_ok(sample=lambda: next(readings) * 2**30,
                                       sleep=lambda _: None))


if __name__ == "__main__":
    unittest.main()
