"""Real-chain tests for the CONFIRMED_WRONG_DECAY fix: source-checkpoint load ->
immutable base binding -> recipe write-back -> REAL optimizer.step. Uses the actual
old-R2 AH end-states and one real production batch (GPU, one step each)."""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(str(ROOT) + "/.."))

from scripts.run_r2f import load_full, immutable_base_wds, make_stream, R2DIR
from src.r2f_recipe import effective
from src.recipe_registry import lr_at, group_of_param


def state_hashes(model, opts):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    for opt in opts:
        for k in sorted(opt.state.keys(), key=str):
            for kk in sorted(opt.state[k].keys()):
                v = opt.state[k][kk]
                if torch.is_tensor(v):
                    h.update(v.detach().cpu().contiguous().numpy().tobytes())
                else:
                    h.update(str(v).encode())
    return h.hexdigest()


def one_real_step(model, opts, base_wds, recipe, stream, gstep):
    base_lr = lr_at(gstep)
    eff = []
    for opt, key in ((opts[0], "muon"), (opts[1], "aux")):
        elr, ewd = effective(base_lr, base_wds[0] if key == "muon" else base_wds[1], recipe)
        for group in opt.param_groups:
            group["lr"] = elr
            group["weight_decay"] = ewd
        eff.append((elr, ewd))
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    for opt in opts:
        opt.zero_grad(set_to_none=True)
    for mb in stream.micro_batches(0):
        batch = mb.to("cuda")
        with autocast:
            logits = model(batch).logits[..., :-1, :].contiguous()
            t = batch[..., 1:].contiguous()
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   t.reshape(-1), reduction="mean")
        (loss / 4).backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
    opts[0].step()
    opts[1].step()
    return base_lr, eff


class TestRealChainFix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = R2DIR / "s512_AH_step256_full.pt"
        cls.payload_raw = torch.load(cls.src, map_location="cpu", weights_only=False)
        cls.payload_sha = hashlib.sha256(
            cls.payload_raw["optimizer_states"][1]["param_groups"][0]["weight_decay"].__str__().encode()).hexdigest()
        cls.base = immutable_base_wds()
        cls.stream = make_stream(768, 1)

    def test_base_binding_immutable(self):
        # AH source saves 0.04; the fixed binding must still return 0.02 for both groups
        self.assertEqual(self.base, {"muon": 0.02, "aux": 0.02})
        saved = self.payload_raw["optimizer_states"][0]["param_groups"][0]["weight_decay"]
        self.assertEqual(saved, 0.04)  # historical record, NOT our base
        self.assertNotEqual(saved, self.base["muon"])

    def test_recipe_writeback_does_not_touch_state(self):
        model, opts, payload = load_full(self.src)
        h0 = state_hashes(model, opts)
        base_lr = lr_at(769)
        for opt, key in ((opts[0], "muon"), (opts[1], "aux")):
            elr, ewd = effective(base_lr, self.base["muon"] if key == "muon" else self.base["aux"], "N")
            for group in opt.param_groups:
                group["lr"] = elr
                group["weight_decay"] = ewd
        h1 = state_hashes(model, opts)  # param_groups lr/wd are config, not tensors
        self.assertEqual(h0, h1, "recipe write-back must not change weights/moments/m/v/step")
        # payload untouched
        self.assertEqual(
            self.payload_raw["optimizer_states"][0]["param_groups"][0]["weight_decay"], 0.04)

    def test_all_four_configs_eff_values(self):
        base_lr = lr_at(769)
        for recipe, (lr_m, wd_m) in {"N": (1.0, 1.0), "AH": (0.5, 2.0)}.items():
            for key in ("muon", "aux"):
                elr, ewd = effective(base_lr, self.base[key], recipe)
                self.assertAlmostEqual(elr, lr_m * base_lr, places=15)
                self.assertAlmostEqual(ewd, wd_m * 0.02, places=15)
                self.assertAlmostEqual(elr * ewd, base_lr * 0.02, places=15,
                                       msg=f"{recipe}/{key} decay product must equal base")

    def test_real_step_moments_invariant_and_no_double_scaling(self):
        # same source, same first batch: AH->N and AH->AH must leave momentum identical
        # after the FIRST step (moments follow UNSCALED grads before any divergence);
        # a second step must not multiply WD again (diverged grads after step 1 are
        # normal path feedback and are NOT used for the moment comparison)
        res = {}
        for recipe in ("N", "AH"):
            model, opts, _ = load_full(self.src)
            blr, eff = one_real_step(model, opts, (self.base["muon"], self.base["aux"]),
                                     recipe, self.stream, 769)
            mom = [st["momentum_buffer"].clone() for st in opts[0].state.values()]
            self.assertAlmostEqual(eff[0][1], 0.02 if recipe == "N" else 0.04, places=15)
            # second step from the SAME base (no chaining from previous effective)
            blr2, eff2 = one_real_step(model, opts, (self.base["muon"], self.base["aux"]),
                                       recipe, make_stream(768, 1), 770)
            self.assertAlmostEqual(eff2[0][1], 0.02 if recipe == "N" else 0.04, places=15,
                                   msg="second step must not re-multiply WD")
            res[recipe] = mom
        for buf_a, buf_b in zip(res["N"], res["AH"]):
            self.assertTrue(torch.equal(buf_a, buf_b),
                            "first-step moments must be identical under the same grads")

    def test_data_delta_from_correct_wd_base(self):
        # one real step for both recipes; data delta measured from EACH recipe's own
        # correct WD base: theta_next = (1 - eff_lr*eff_wd) theta - eff_lr*v
        outs = {}
        for recipe in ("N", "AH"):
            model, opts, _ = load_full(self.src)
            theta_pre = {n: p.detach().clone() for n, p in model.named_parameters()}
            blr, eff = one_real_step(model, opts, (self.base["muon"], self.base["aux"]),
                                     recipe, self.stream, 769)
            elr_m, ewd_m = eff[0]
            num = 0.0
            for n, p in model.named_parameters():
                if group_of_param(n) is None:
                    continue
                d = (1 - elr_m * ewd_m) * theta_pre[n].double() - p.detach().double()
                num += float(d.pow(2).sum())
            outs[recipe] = num ** 0.5
        # AH data delta should be ~0.5x N's (same direction, halved lr)
        ratio = outs["AH"] / outs["N"]
        self.assertLess(abs(ratio - 0.5), 1e-3, f"expected ~0.5, got {ratio}")

    def test_cpu_step_not_shared_across_branches(self):
        m1, o1, _ = load_full(self.src)
        m2, o2, _ = load_full(self.src)
        for oa, ob in zip(o1, o2):
            for ka, kb in zip(oa.state.keys(), ob.state.keys()):
                va, vb = oa.state[ka].get("step"), ob.state[kb].get("step")
                if torch.is_tensor(va) and torch.is_tensor(vb):
                    self.assertNotEqual(va.data_ptr(), vb.data_ptr())


if __name__ == "__main__":
    unittest.main(verbosity=2)
