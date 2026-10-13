import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from . import common
from .engine import _apply_state, effect_from_rows, state_identity


class ReadinessCpuTests(unittest.TestCase):
    def test_digest_order_and_changes(self):
        a = torch.tensor([[1., 2.]])
        b = torch.tensor([3.])
        first = common.tensor_digest({"z": b, "a": a})
        self.assertEqual(first, common.tensor_digest({"a": a.clone(), "z": b.clone()}))
        self.assertNotEqual(first, common.tensor_digest({"a": a+0.001, "z": b}))
        self.assertNotEqual(first, common.tensor_digest({"a": a.double(), "z": b}))
        self.assertNotEqual(first, common.tensor_digest({"a": a.reshape(2), "z": b}))
        self.assertNotEqual(common.tensor_digest({"a": None}),
                            common.tensor_digest({"a": torch.zeros(1)}))
        self.assertNotEqual(common.tensor_digest({"a": None}), common.tensor_digest({}))

    def test_optimizer_logical_names_aliases_and_groups(self):
        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Linear(2, 2, bias=False)
                self.b = self.a
                self.unused = torch.nn.Parameter(torch.ones(1))
        model = Tied()
        opt = torch.optim.AdamW([{"params": [model.a.weight], "weight_decay": 0.1},
                                 {"params": [model.unused], "weight_decay": 0}], lr=0.01)
        snap = common.optimizer_logical_snapshot(model, opt)
        self.assertEqual(snap["aliases"]["a.weight"], ["a.weight", "b.weight"])
        self.assertEqual(len(snap["groups"]), 2)
        self.assertNotEqual(snap["groups"][0]["options"]["weight_decay"],
                            snap["groups"][1]["options"]["weight_decay"])

    def test_compare_tensors_stages(self):
        result = common.compare_tensors(torch.tensor([1., 2.]), torch.tensor([1., 2.1]))
        self.assertEqual(result["different_elements"], 1)
        self.assertEqual(result["max_difference_flat_index"], 1)
        self.assertGreater(result["max_abs"], 0)
        self.assertEqual(common.compare_tensors(None, torch.zeros(1))["same_presence"], False)

    def test_state_isolation_exception_restores_submodule_modes_and_nonpersistent_buffers(self):
        random.seed(3); np.random.seed(3); torch.manual_seed(3)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        model.register_buffer("extra", torch.tensor([4.]), persistent=False)
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.train(); model[1].eval()
        model(torch.randn(4, 2)).sum().backward(); opt.step()
        pre = copy.deepcopy(model.state_dict())
        cursor = {"i": 5}
        rng = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with common.isolated_state(model, opt, cursor):
                model.eval(); model[1].train(); cursor["i"] = 90
                with torch.no_grad():
                    model.extra.add_(1)
                    model[0].weight.add_(1)
                torch.rand(1)
                raise RuntimeError("probe")
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        self.assertEqual(cursor, {"i": 5})
        self.assertEqual(float(model.extra), 4.)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for key, value in pre.items():
            torch.testing.assert_close(model.state_dict()[key], value)

    def test_four_state_tied_and_none_semantics(self):
        model = torch.nn.Linear(2, 2, bias=True)
        frozen = {"base": {"weight": torch.ones_like(model.weight),
                            "bias": torch.tensor([2., 2.])},
                  "delta": {"weight": torch.full_like(model.weight, -0.5),
                            "bias": torch.zeros_like(model.bias)},
                  "updated": {"weight": torch.full_like(model.weight, 0.5),
                              "bias": torch.tensor([2., 2.])},
                  "A": {"weight": torch.ones_like(model.weight)},
                  "eta": 0.5, "formula_tolerance": 1e-6}
        self.assertLessEqual(state_identity(frozen, "weight")["F11_actual_max_abs"], 1e-6)
        _apply_state(model, frozen, "weight", "F00")
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
        _apply_state(model, frozen, "weight", "F10")
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.5))
        torch.testing.assert_close(model.bias, torch.tensor([2., 2.]))
        _apply_state(model, frozen, "weight", "F01")
        torch.testing.assert_close(model.weight, torch.ones_like(model.weight))
        _apply_state(model, frozen, "weight", "F11")
        torch.testing.assert_close(model.weight, frozen["updated"]["weight"])

    def test_one_shift_mask_and_token_weight(self):
        from pilot_core import loss_sum_count
        ids = torch.tensor([[1, 4, 5, 2]])
        logits = torch.randn(1, 3, 8)
        loss, count = loss_sum_count(logits, ids, torch.tensor([4]))
        self.assertEqual(count, 3)
        self.assertEqual(logits.shape[1], ids.shape[1] - 1)
        rows = {state: [{"loss_sum": 1.0, "token_count": 1},
                        {"loss_sum": 10.0, "token_count": 10}]
                for state in ("F00", "F10", "F01", "F11")}
        result = effect_from_rows(rows)
        self.assertAlmostEqual(result["nll"]["F00"], 1.0)

    def test_paired_label_effect_uses_same_docs_and_weights(self):
        def make(a, b):
            return {"F00": [{"loss_sum": a, "token_count": 1}, {"loss_sum": b, "token_count": 10}],
                    "F10": [{"loss_sum": a-1, "token_count": 1}, {"loss_sum": b-2, "token_count": 10}],
                    "F01": [{"loss_sum": a, "token_count": 1}, {"loss_sum": b, "token_count": 10}],
                    "F11": [{"loss_sum": a-1, "token_count": 1}, {"loss_sum": b-2, "token_count": 10}]}
        legacy, natural = effect_from_rows(make(5, 50)), effect_from_rows(make(5, 50.5))
        self.assertAlmostEqual(natural["I_before"] - legacy["I_before"], 0)
        self.assertAlmostEqual(legacy["I_before"], 3/11)

    def test_budget_counts_attempted_before_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            where = Path(tmp)
            (where / "config.json").write_text(json.dumps({"max_token_forward_equivalents": 5,
                 "max_gpu_optimizer_step_attempts": 1}), encoding="utf-8")
            with mock.patch.object(common, "HERE", where):
                ledger = common.Ledger("same-run")
                ledger.forward(4); ledger.backward(); ledger.optimizer_attempt()
                with self.assertRaises(common.BudgetStop):
                    ledger.forward(2)
                with self.assertRaises(common.BudgetStop):
                    ledger.optimizer_attempt()
                resumed = common.Ledger("same-run")
                self.assertEqual(resumed.data["token_forward_equivalents"], 4)
                self.assertEqual(resumed.data["gpu_optimizer_step_attempts"], 1)
                self.assertEqual(resumed.data["gpu_optimizer_steps_completed"], 0)
                with self.assertRaisesRegex(RuntimeError, "MANIFEST_CHANGED"):
                    common.Ledger("different-run")


if __name__ == "__main__":
    unittest.main()
