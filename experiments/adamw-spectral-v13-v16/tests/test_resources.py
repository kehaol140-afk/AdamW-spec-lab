import copy
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from pilot_core import resource_decision, require_resource, resource_sample


class ResourcePolicyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {"protocol_id": "ADAMW_A_SPECTRAL_LOCAL_V1_1_LOWMEM_20261002",
                    "runtime": {"system_ram_free_reserve_gib": 1.0,
                                "system_ram_start_min_gib": 1.5,
                                "system_ram_warning_gib": 1.5,
                                "process_rss_soft_limit_gib": 4}}

    def test_preinit_admits_1_9_gib(self):
        self.assertEqual(resource_decision(self.cfg, int(1.9 * 2 ** 30), True)[0], "READY")

    def test_runtime_warns_but_allows_1_25_gib(self):
        self.assertEqual(resource_decision(self.cfg, int(1.25 * 2 ** 30), False)[0], "WARNING")

    def test_runtime_blocks_0_9_gib(self):
        self.assertEqual(resource_decision(self.cfg, int(0.9 * 2 ** 30), False)[0], "RESOURCE_BLOCKED")

    def test_original_config_retains_2_gib_gate(self):
        old = {"runtime": {"system_ram_free_reserve_gib": 2}}
        self.assertEqual(resource_decision(old, int(1.9 * 2 ** 30), True)[0], "RESOURCE_BLOCKED")

    def test_missing_readout_blocks(self):
        self.assertEqual(resource_decision(self.cfg, None, False)[0], "RESOURCE_BLOCKED")

    def test_oom_propagates(self):
        with patch("pilot_core.resource_sample", side_effect=MemoryError("OOM")):
            with self.assertRaises(MemoryError):
                require_resource(self.cfg, "test", admission=True)

    def test_gate_does_not_change_rng(self):
        random.seed(19); np.random.seed(19); torch.manual_seed(19)
        before = (random.getstate(), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone())
        resource_decision(self.cfg, int(1.25 * 2 ** 30), False)
        self.assertEqual(random.getstate(), before[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], before[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), before[2]))

    def test_sampling_before_init_does_not_query_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("pilot_core.ROOT", Path(directory)), \
                 patch("pilot_core.source_hash", return_value="test-source"), \
                 patch("pilot_core.ram_available", return_value=int(1.9 * 2 ** 30)), \
                 patch("pilot_core.torch.cuda.is_initialized", return_value=False), \
                 patch("pilot_core.torch.cuda.mem_get_info", side_effect=AssertionError("CUDA queried")):
                row = resource_sample(self.cfg, "before_model", admission=True)
                self.assertEqual(row["status"], "READY")
                self.assertIsNone(row["gpu_free_bytes"])
                self.assertIsNone(row["gpu_peak_reserved_bytes"])

    def test_warning_gate_preserves_training_objects(self):
        param = torch.nn.Parameter(torch.ones(2))
        param.grad = torch.tensor([0.5, -0.5])
        optimizer = torch.optim.AdamW([param], lr=0.01)
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)
        cursor = 2048
        value = param.detach().clone()
        grad = param.grad.clone()
        state = copy.deepcopy(optimizer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            with patch("pilot_core.ROOT", Path(directory)), \
                 patch("pilot_core.source_hash", return_value="test-source"), \
                 patch("pilot_core.ram_available", return_value=int(1.25 * 2 ** 30)), \
                 patch("pilot_core.torch.cuda.is_initialized", return_value=False):
                row = require_resource(self.cfg, "during_training")
        self.assertEqual(row["status"], "WARNING")
        self.assertTrue(torch.equal(param, value))
        self.assertTrue(torch.equal(param.grad, grad))
        self.assertEqual(cursor, 2048)
        for key, fields in state["state"].items():
            for name, old in fields.items():
                self.assertTrue(torch.equal(old, optimizer.state_dict()["state"][key][name]))


if __name__ == "__main__":
    unittest.main()
