"""CPU regression tests for V16 campaign core semantics. No GPU steps consumed."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USE_TF", "0")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import campaign_v16.core as core
from campaign_v16.core import n_a, schedule_eta, parent_schedule_eta, digest


class V16CpuTests(unittest.TestCase):
    def test_eval_origin_guard(self):
        """A model that never loaded a state must not be labeled with checkpoint identity."""
        caught = None
        try:
            core.eval_docs(None, [], "x", "p", "l", 0, {"state_id": "H0_step256",
                                                        "loaded_checkpoint_sha": None,
                                                        "global_step": 256})
        except RuntimeError as exc:
            caught = str(exc)
        self.assertIn("EVAL_WITHOUT_STATE_ORIGIN", caught)
        # scratch identities are the documented exception (explicitly labeled, no ckpt claim)
        try:
            core.eval_docs(None, [], "x", "p", "l", 0, {"state_id": "scratch_seed3407",
                                                        "loaded_checkpoint_sha": None,
                                                        "global_step": 0})
        except RuntimeError as exc:
            self.fail(f"scratch identity should be allowed: {exc}")

    def test_eta_schedules(self):
        # parent continuation: step 257 uses global schedule of the 512-step parent run
        self.assertAlmostEqual(parent_schedule_eta(0.003, 257), 0.003 * (
            0.1 + 0.9 * (1 + np.cos(np.pi * (257 - 32) / 480)) / 2), places=12)
        # from-scratch: warmup then cosine to 0.1*peak
        self.assertAlmostEqual(schedule_eta(0.006, 128, 2048, 128, 0.1), 0.006)
        self.assertAlmostEqual(schedule_eta(0.006, 1, 2048, 128, 0.1), 0.006 / 128)
        self.assertAlmostEqual(schedule_eta(0.006, 2048, 2048, 128, 0.1), 0.0006)

    def test_n_a_and_q_override_semantics(self):
        x = torch.randn(4, 4)
        D, zero, xn = n_a(x, 5.0)
        self.assertFalse(zero)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(D.double())), 5.0, places=5)
        D0, zero0, _ = n_a(torch.zeros(2, 2), 5.0)
        self.assertTrue(zero0 and torch.all(D0 == 0))

    def test_ledger_single_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "budget_ledger.json").write_text(json.dumps(
                {"manifest_sha256": "m", "optimizer_step_attempts": 5, "token_forward_equivalents": 0,
                 "forward_calls": 0}), encoding="utf-8")
            with patch.object(core, "LEDGER_PATH", base / "budget_ledger.json"), \
                 patch.object(core, "write", lambda p, o: Path(p).write_text(
                     json.dumps(o, ensure_ascii=False) + "\n", encoding="utf-8")):
                core.begin_optimizer("t")
                core.begin_optimizer("t")
                v = core._fresh_ledger({})
            self.assertEqual(v["optimizer_step_attempts"], 7)

    def test_apply_step_identity_and_q_override(self):
        """S,a=b=1 with no q-override skips every correction (bit-native); q-override rescales
        the target displacement to the external norm regardless of a."""
        torch.manual_seed(3)
        dense = torch.nn.Linear(6, 6)

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = dense
                self.other = torch.nn.Linear(6, 4)

            def forward(self, x):
                return self.other(torch.nn.functional.relu(self.dense(x)))
        model = Tiny()
        target = "dense.weight"
        matrix = [p for p in model.parameters() if p.ndim >= 2]
        vector = [p for p in model.parameters() if p.ndim < 2]
        opt = torch.optim.AdamW([{"params": matrix, "weight_decay": 0.1},
                                 {"params": vector, "weight_decay": 0.0}],
                                lr=0.01, betas=(0.9, 0.95), eps=1e-8, foreach=False, fused=False)
        x = torch.randn(3, 6)
        ((model(x) - torch.randn(3, 4)) ** 2).sum().backward()
        for p in model.parameters():
            p.grad.div_(6)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        w_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        with patch.object(core, "begin_optimizer", lambda phase: None), \
             patch.object(core, "bump", lambda **k: None), \
             patch.object(core, "resource", lambda *a, **k: None):
            core.apply_step(model, opt, "S", 1.0, 1.0, "t", 0.01, target_name=target)
        for n, p in model.named_parameters():
            self.assertFalse(torch.equal(p, w_before[n]))  # native AdamW moved them
        # q-override: displacement norm must equal q/eta regardless of internal a
        model2 = Tiny()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=0.01, betas=(0.9, 0.95),
                                 eps=1e-8, foreach=False, fused=False)
        w0 = {n: p.detach().clone() for n, p in model2.named_parameters()}
        ((model2(x) - torch.randn(3, 4)) ** 2).sum().backward()
        for p in model2.parameters():
            p.grad.div_(6)
        with patch.object(core, "begin_optimizer", lambda phase: None), \
             patch.object(core, "bump", lambda **k: None), \
             patch.object(core, "resource", lambda *a, **k: None):
            meta = core.apply_step(model2, opt2, "F", 2.0, 1.0, "t", 0.01, target_name=target,
                                   q_override=0.05)
        target_p = dict(model2.named_parameters())[target]
        base = w0[target] * (1 - 0.01 * 0.1)
        displacement = (base - target_p)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(displacement.double())), 0.05,
                               places=4)  # full displacement norm equals q_override exactly


if __name__ == "__main__":
    unittest.main()
