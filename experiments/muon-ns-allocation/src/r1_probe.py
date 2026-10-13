"""R1 probe engine: unified-clock candidates (a, b, structure) on the O group.

Per (state, batch): restore -> write lr(s+1) to ALL groups -> production G -> H ->
u5 (all Muon) + u3 (O group) -> c = nu5/nu3 (FP64, whole O group) -> native theta_0_plus
via real opt.step() -> candidates by superposition. Norm checks recorded per pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LAB / "muon_transition_experiment"))
sys.path.insert(0, str(LAB / "muon_transition_experiment" / "src"))
sys.path.insert(0, str(LAB / "minimind_base"))

from muon_transition.model import build_model
from muon_transition.optimizers import build_muon_arm
from muon_transition.mixture import MixtureStream, build_step_manifest
from muon_transition.routing import route_parameters
from vendored_fr.rng import restore_rng_state
from src.recipe_registry import lr_at, group_of_param, shape_factor, NS_COEFFS, EPS_NORM
from src.state_restore import eval_V
from torch.optim._muon import _zeropower_via_newtonschulz

O_NAMES = [f"model.layers.{l}.self_attn.o_proj.weight" for l in range(8)]


def restore_unified(payload, step):
    model = build_model().to("cuda")
    model.load_state_dict(payload["model_state"])
    model.train()
    muon_opt, aux_opt = build_muon_arm(model, lr=1e-9, weight_decay=0.0)
    muon_opt.load_state_dict(payload["optimizer_states"][0])
    aux_opt.load_state_dict(payload["optimizer_states"][1])
    for opt in (muon_opt, aux_opt):
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.clone()
    restore_rng_state(payload["rng"])
    lr = lr_at(step + 1)
    for opt in (muon_opt, aux_opt):
        for group in opt.param_groups:
            group["lr"] = lr
    return model, [muon_opt, aux_opt], lr


class BatchContext:
    """Caches everything shared by all candidates of one (state, batch)."""

    def __init__(self, payload, step, batch_slice):
        from muon_transition.m1_data_cache import get_domain_windows
        a_train, b_train = get_domain_windows()
        m = build_step_manifest(a_train, b_train, 0.5, batch_slice[1], seed=2030)
        steps = m["steps"][batch_slice[0]:batch_slice[1]]
        model, opts, lr = restore_unified(payload, step)
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        stream = MixtureStream({"steps": steps}, a_train, b_train)
        self.theta_pre = {n: p.detach().clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.grad = None
        micro_losses, self.n_targets = [], 0
        for mb in stream.micro_batches(0):
            batch = mb.to("cuda")
            with autocast:
                logits = model(batch).logits[..., :-1, :].contiguous()
                t = batch[..., 1:].contiguous()
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                       t.reshape(-1), reduction="mean")
            (loss / 4).backward()
            micro_losses.append(float(loss.detach()))
            self.n_targets += int(t.numel())
        self.preclip = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1.0))
        _, _, table = route_parameters(model)
        muon_names = [t["name"] for t in table if t["route"] == "muon"]
        state = payload["optimizer_states"][0]["state"]
        momenta = {nm: state[idx]["momentum_buffer"].to("cuda") for idx, nm in enumerate(muon_names)}
        self.H, self.muon_names_all = {}, []
        for name, p in model.named_parameters():
            g = group_of_param(name)
            if g is None or p.grad is None:
                continue
            m_i = momenta[name].lerp(p.grad, 0.05)
            self.H[name] = (p.grad.lerp(m_i, 0.95).detach(), g)
            self.muon_names_all.append(name)
        # unified-clock native one-step
        opts[0].step()
        opts[1].step()
        # R1-REPAIR (R2 protocol 3.1): snapshot ALL unique parameters after the native
        # one-step (Muon AND aux), so restore_theta() returns the model to the exact
        # post-step state. The original round saved Muon-only, leaving aux params at
        # their pre-step values after a P/Z evaluation -> extension calibration reused
        # a wrong aux baseline (see docs/R1_REPAIR_FOR_R2.md).
        self.theta_plus = {n: p.detach().clone() for n, p in model.named_parameters()}
        self.lr = lr
        self.wd = payload["optimizer_states"][0]["param_groups"][0]["weight_decay"]
        # u vectors (cached on CPU after build)
        self.u5 = {}
        for name, (h, g) in self.H.items():
            y = _zeropower_via_newtonschulz(h, NS_COEFFS, 5, EPS_NORM).float()
            self.u5[name] = ((-lr * shape_factor(h.shape)) * y).cpu()
        self.u3 = {}
        for name in O_NAMES:
            h, _ = self.H[name]
            y = _zeropower_via_newtonschulz(h, NS_COEFFS, 3, EPS_NORM).float()
            self.u3[name] = ((-lr * shape_factor(h.shape)) * y).cpu()
        nu5 = nu3 = 0.0
        for name in O_NAMES:
            nu5 += float(self.u5[name].double().pow(2).sum())
            nu3 += float(self.u3[name].double().pow(2).sum())
        self.nu5, self.nu3 = nu5 ** 0.5, nu3 ** 0.5
        self.c = (nu5 / nu3) ** 0.5 if nu3 > 0 else None
        self.loss_mean = float(np.mean(micro_losses))
        # realized WD-only base for norm checks: theta_plus - u5
        self._n2p = dict(model.named_parameters())
        self._model = model

    def apply_candidate(self, a, b, structure):
        """Superpose candidate params onto the live model. Full-parameter base restore
        (aux included); Muon-coordinate delta only."""
        with torch.no_grad():
            for n, p in self.theta_plus.items():
                self._n2p[n].copy_(p)
            for name in self.muon_names_all:
                if name in O_NAMES:
                    if structure == "NS5":
                        d = (a * b - 1.0) * self.u5[name]
                    else:
                        d = (a * b * self.c * self.u3[name].cuda() - self.u5[name].cuda()).cpu()
                    self._n2p[name].add_(d.cuda())
                else:
                    self._n2p[name].add_(((a - 1.0) * self.u5[name]).cuda())

    def apply_zero_data(self):
        with torch.no_grad():
            for n, p in self.theta_plus.items():
                self._n2p[n].copy_(p)
            for name in self.muon_names_all:
                self._n2p[name].sub_(self.u5[name].cuda())

    def apply_pre_state(self):
        with torch.no_grad():
            for n, p in self.theta_pre.items():
                self._n2p[n].copy_(p)

    def restore_theta(self):
        with torch.no_grad():
            for n, p in self.theta_plus.items():
                self._n2p[n].copy_(p)

    def realized_O_norm(self, a, b, structure):
        """FP64 realized O-group data-delta norm (theta_cand - theta_after_common_WD)."""
        tot = 0.0
        for name in O_NAMES:
            if structure == "NS5":
                d = (a * b) * self.u5[name]
            else:
                d = (a * b * self.c) * self.u3[name]
            tot += float(d.double().pow(2).sum())
        return tot ** 0.5


def evaluate_batch(ctx, q, side, points, repeat_native=False):
    """points: list of dicts {id, a, b, structure} or {id, kind: native|Z|P}."""
    rows = []
    norm_checks = []
    for pt in points:
        kind = pt.get("kind")
        if kind == "native" or (pt.get("a") == 1.0 and pt.get("b") == 1.0
                                and pt.get("structure") == "NS5"):
            ctx.restore_theta()
            res = eval_V(ctx._model, q, side)
            rows.append({"point_id": pt["id"], **res})
            if repeat_native:
                ctx.restore_theta()
                rows.append({"point_id": pt["id"] + "::repeat", **eval_V(ctx._model, q, side)})
            continue
        if kind == "Z":
            ctx.apply_zero_data()
            res = eval_V(ctx._model, q, side)
            ctx.restore_theta()
            rows.append({"point_id": "Z_NO_MUON_DATA", **res})
            continue
        if kind == "P":
            ctx.apply_pre_state()
            res = eval_V(ctx._model, q, side)
            ctx.restore_theta()
            rows.append({"point_id": "P_PRE_STATE", **res})
            continue
        a, b, s = pt["a"], pt["b"], pt["structure"]
        ctx.apply_candidate(a, b, s)
        res = eval_V(ctx._model, q, side)
        ctx.restore_theta()
        rows.append({"point_id": pt["id"], "a": a, "b": b, "structure": s, **res})
        if s == "NS3_MATCHED":
            n5 = ctx.realized_O_norm(a, b, "NS5")
            n3 = ctx.realized_O_norm(a, b, "NS3_MATCHED")
            norm_checks.append({"point_id": pt["id"], "a": a, "b": b,
                                "realized_rel_diff": abs(n5 - n3) / max(n5, 1e-30)})
    return rows, norm_checks
