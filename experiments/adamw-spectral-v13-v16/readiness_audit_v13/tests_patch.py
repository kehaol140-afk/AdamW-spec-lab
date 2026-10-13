"""Regression tests for the V1.3 child-only atomic-write and resume patch."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from . import common
from .engine import accepted_source_hashes


class PatchRegressionTests(unittest.TestCase):
    def test_atomic_write_retries_transient_windows_sharing_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            path.write_text('{"value":0}', encoding="utf-8")
            original = os.replace
            calls = []
            def flaky(source, target):
                calls.append((source, target))
                if len(calls) <= 2:
                    raise PermissionError(5, "simulated sharing violation")
                return original(source, target)
            with (mock.patch.object(common.os, "replace", side_effect=flaky),
                  mock.patch.object(common.time, "sleep", return_value=None)):
                common.write_json(path, {"value": 7})
            self.assertEqual(len(calls), 3)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 7})

    def test_cumulative_ledger_and_explicit_previous_source(self):
        root = Path(__file__).resolve().parent
        run = common.read_json(root / "run_manifest.json")
        ledger = common.read_json(root / "budget_ledger.json")
        old = common.read_json(root / "attempts" / "attempt1_before_atomic_retry" / "run_manifest.json")
        self.assertEqual(run["attempt_id"], 2)
        self.assertIn(old["source_hash"], run["accepted_previous_source_hashes"])
        self.assertIn(old["source_hash"], accepted_source_hashes())
        self.assertGreaterEqual(ledger["token_forward_equivalents"],
                                common.read_json(root / "attempts" / "attempt1_before_atomic_retry" /
                                                 "budget_ledger.json")["token_forward_equivalents"])
        self.assertGreaterEqual(ledger["gpu_optimizer_step_attempts"], 3)


if __name__ == "__main__":
    unittest.main()
