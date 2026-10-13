import copy
import random
import unittest

import numpy as np
import torch

from replay_audit_v12.core import (natural_prefix, legacy_window, packed_windows,
                                   reanalyze_parent, training_windows)
from replay_audit_v12.replay import four_states, interactions, isolated_state


class V12CpuTests(unittest.TestCase):
    def test_long_document_has_no_artificial_eos(self):
        content = list(range(10, 310))
        legacy = legacy_window(content, 1, 2, 256)
        natural = natural_prefix(content, 1, 2, 256)
        self.assertEqual(legacy[-1], 2)
        self.assertEqual(natural[-1], content[255])
        self.assertEqual(len(natural), 257)

    def test_short_document_keeps_real_eos(self):
        self.assertEqual(natural_prefix([7, 8], 1, 2, 256), [1, 7, 8, 2])

    def test_one_shift_and_count(self):
        from pilot_core import loss_sum_count
        ids = torch.tensor([[1, 4, 5, 2]])
        logits = torch.randn(1, 3, 8)
        loss, count = loss_sum_count(logits, ids, torch.tensor([4]))
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 8),
                                                      ids[:, 1:].reshape(-1), reduction="sum")
        self.assertEqual(count, 3)
        torch.testing.assert_close(loss, expected)

    def test_four_states_and_interaction(self):
        base = {"target": torch.tensor([2.0]), "other": torch.tensor([3.0]),
                "unused": torch.tensor([4.0])}
        delta = {"target": torch.tensor([0.5]), "other": torch.tensor([-0.25]),
                 "unused": torch.tensor([0.0])}
        states = four_states(base, delta, "target")
        self.assertEqual([float(states[key]["target"]) for key in ("F00", "F10", "F01", "F11")],
                         [2.0, 2.5, 2.0, 2.5])
        self.assertEqual([float(states[key]["other"]) for key in ("F00", "F10", "F01", "F11")],
                         [3.0, 3.0, 2.75, 2.75])
        self.assertTrue(all(float(state["unused"]) == 4.0 for state in states.values()))
        result = interactions({"F00": 10.0, "F10": 8.0, "F01": 9.0, "F11": 6.5})
        self.assertEqual(result, {"I_before": 2.0, "I_after": 2.5, "interaction": -0.5})

    def test_four_states_match_real_cpu_adamw_and_none_gradient(self):
        target = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
        other = torch.nn.Parameter(torch.tensor([-1.0, 4.0]))
        unused = torch.nn.Parameter(torch.tensor([5.0]))
        optimizer = torch.optim.AdamW([target, other, unused], lr=0.01,
                                      weight_decay=0.1, foreach=False, fused=False)
        old = {"target": target.detach().clone(), "other": other.detach().clone(),
               "unused": unused.detach().clone()}
        target.grad = torch.tensor([0.2, -0.3])
        other.grad = torch.tensor([0.4, 0.1])
        unused.grad = None
        optimizer.step()
        base = {"target": old["target"] * (1 - 0.01 * 0.1),
                "other": old["other"] * (1 - 0.01 * 0.1), "unused": old["unused"]}
        updated = {"target": target.detach().clone(), "other": other.detach().clone(),
                   "unused": unused.detach().clone()}
        delta = {name: updated[name] - base[name] for name in base}
        states = four_states(base, delta, "target")
        for name in updated:
            torch.testing.assert_close(states["F11"][name], updated[name], rtol=0, atol=1e-7)
        torch.testing.assert_close(states["F01"]["target"], base["target"])
        torch.testing.assert_close(states["F01"]["other"], updated["other"])
        torch.testing.assert_close(states["F00"]["unused"], old["unused"])

    def test_isolated_state_restores_everything(self):
        random.seed(71); np.random.seed(71); torch.manual_seed(71)
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3))
        model.register_buffer("nonpersistent", torch.tensor([2.0]), persistent=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        model.train()
        model(torch.randn(4, 3)).sum().backward(); optimizer.step()
        cursor = {"position": 9}
        parameters = [p.detach().clone() for p in model.parameters()]
        gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        before_rng = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with isolated_state(model, optimizer, cursor):
                model.eval(); cursor["position"] = 999
                with torch.no_grad():
                    for p in model.parameters(): p.add_(1)
                    model.nonpersistent.add_(1)
                random.random(); np.random.rand(); torch.rand(1)
                raise RuntimeError("probe")
        self.assertTrue(model.training)
        self.assertEqual(cursor, {"position": 9})
        self.assertEqual(float(model.nonpersistent), 2.0)
        for p, old, grad in zip(model.parameters(), parameters, gradients):
            torch.testing.assert_close(p, old)
            if grad is not None: torch.testing.assert_close(p.grad, grad)
        for key, fields in optimizer_state["state"].items():
            for field, value in fields.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][key][field], value)
        self.assertEqual(random.getstate(), before_rng[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before_rng[1][1]))
        torch.testing.assert_close(torch.get_rng_state(), before_rng[2])

    def test_token_weighted_is_not_document_mean(self):
        documents = [(10.0, 1), (10.0, 10)]
        weighted = sum(loss for loss, _ in documents) / sum(count for _, count in documents)
        document_mean = sum(loss / count for loss, count in documents) / len(documents)
        self.assertNotEqual(weighted, document_mean)
        self.assertAlmostEqual(weighted, 20 / 11)

    def test_windows_do_not_overlap(self):
        windows = training_windows(1052996, 256, 8, 4)
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(a["end_token_exclusive"] <= b["start_token"]
                            for a, b in zip(windows, windows[1:])))

    def test_packed_sources_clustered(self):
        docs = [{"doc_id": "a", "content_ids": list(range(20))},
                {"doc_id": "b", "content_ids": list(range(20, 40))}]
        rows = packed_windows(docs, 1, 2, 16)
        self.assertTrue(rows)
        self.assertTrue(all(row["descriptive_only"] for row in rows))
        self.assertTrue(all(row["cluster_id"] for row in rows))

    def test_parent_gate_recalculation(self):
        result = reanalyze_parent()
        self.assertEqual(result["parent_next_action"], "STOP")
        self.assertEqual(result["main_confirm"]["distinct_documents"], 64)
        self.assertTrue(all(row["distinct_documents"] == 16 for row in result["cell_best_scalar"]))


if __name__ == "__main__":
    unittest.main()
