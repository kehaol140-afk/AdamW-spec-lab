"""GroupMuon: native-Muon-compatible multi-matrix optimizer with per-group
direction mixing (current gradient x historical momentum) before NS5.

Design contract (frozen semantics; see 100M_METHOD_DESIGN.md):
  1. Momentum state evolution is IDENTICAL to torch.optim.Muon for every matrix
     (buf <- mu*buf + (1-mu)*g  [EMA semantics: buf.lerp_(grad, 1-momentum)];
     NOT the classic buf <- mu*buf + g, and NOT the torch docstring's
     B_t = mu*B_{t-1} + g_t — code truth, amended 2026-10-12, see
     MUON_MOMENTUM_EQUIVALENCE.md), independent of the mixing lam.
 2. native mode matrices follow torch.optim.Muon bitwise (same op order; vendored
    ns5 + adjust_lr; bf16 add) — regression-tested against torch.optim.Muon.
  3. mix mode matrices use NS5(lam*g + (1-lam)*buf) as the direction input.
     lam0 = 1 - mu makes the mix IDENTICAL to the native Nesterov input
     (1-mu)*g + mu*buf (bitwise-routed anchor; see ns5.lam_anchor — amended
     2026-10-12 from the stale "1/(1+mu)" derivation which relied on the
     torch docstring formula rather than the lerp code).
 4. norm_match=True (default): per-matrix update is rescaled in fp64 so its
    Frobenius norm EQUALS the native-mode displacement norm of the SAME step
    (native reference NS5 computed on B~ = g + mu*buf). This isolates the pure
    direction effect; cost = a second NS5 per mixed matrix (measured, reported).
 5. Decoupled weight decay uses the RAW lr (not shape-adjusted), exactly as
    torch.optim.Muon does.
 6. No SVD anywhere; no dynamic spectral exponent; no gating network.

Aux params (embedding/norms) are NOT handled here: the runner attaches a plain
torch.optim.AdamW (same lr, same schedule) per the M-line aux_lr_rule.
"""

from __future__ import annotations

import time

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .ns5 import adjust_lr, lam_anchor, ns5


class GroupMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        weight_decay: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
        eps: float = 1e-7,
        adjust_lr_fn: str | None = None,
        lam: float = 0.75,
        norm_match: bool = True,
        mode: str = "native",
        diag_every: int = 1,
    ) -> None:
        # NOTE: unlike torch.optim.Muon we accept ONLY 2D params as well; groups carry
        # per-group 'mode' ('native'|'mix') and optional 'momentum' override (low-mu arm).
        if mode not in ("native", "mix"):
            raise ValueError(f"mode must be 'native' or 'mix', got {mode}")
        if diag_every < 1:
            raise ValueError("diag_every must be >= 1")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
            eps=eps,
            adjust_lr_fn=adjust_lr_fn,
            mode=mode,
            lam=lam,
            norm_match=norm_match,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"GroupMuon supports 2D params only, got {tuple(p.shape)}")
        self.stats: dict = {}
        self._step_count = 0          # optimizer-level step counter (diag anchoring)
        self.diag_every = diag_every  # 1 = fp64 cosine diagnostics every step (test default);
        # pilot freeze uses 64: cos(O,g)/cos(O,O_nat) sampled every 64 steps + norms always

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        agg = {
            "ns_seconds_native": 0.0,
            "ns_seconds_mix": 0.0,
            "n_native": 0,
            "n_mix": 0,
            "cos_O_g": [],
            "cos_O_Onat": [],
            "norm_rel_err": [],
        }

        self._step_count += 1
        diag_step = (self._step_count % self.diag_every == 0) or self._step_count == 1

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            coef = tuple(group["ns_coefficients"])
            eps = group["eps"]
            mode = group.get("mode", "native")
            lam = group.get("lam", 0.75)
            norm_match = group.get("norm_match", True)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise RuntimeError("param gradient must be 2D")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )
                buf = state["momentum_buffer"]

                # ---- momentum state evolution: identical to torch Muon, always
                buf.lerp_(grad, 1 - momentum)

                if mode == "native":
                    t0 = time.perf_counter()
                    update = grad.lerp(buf, momentum) if nesterov else buf
                    update = ns5(update, coef, ns_steps, eps)
                    agg["ns_seconds_native"] += time.perf_counter() - t0
                    agg["n_native"] += 1
                    adjusted_lr = adjust_lr(lr, group["adjust_lr_fn"], p.shape)
                    p.mul_(1 - lr * weight_decay)
                    p.add_(update, alpha=-adjusted_lr)
                elif mode == "mix":
                    # native reference direction (needed for norm matching + diagnostics)
                    t0 = time.perf_counter()
                    btilde = grad.lerp(buf, momentum) if nesterov else buf.clone()
                    o_nat = ns5(btilde, coef, ns_steps, eps)
                    t_nat = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    # bitwise anchor: at lam == 1-momentum the native Nesterov input
                    # IS the mix; use the identical lerp expression (see ns5.lam_anchor)
                    if abs(lam - (1.0 - momentum)) < 1e-12:
                        mix = grad.lerp(buf, momentum) if nesterov else buf
                    else:
                        mix = lam * grad + (1.0 - lam) * buf
                    o_mix = ns5(mix, coef, ns_steps, eps)
                    t_mix = time.perf_counter() - t0
                    agg["ns_seconds_mix"] += t_nat + t_mix
                    agg["n_mix"] += 1

                    o_mix_d = o_mix.double()
                    o_nat_d = o_nat.double()
                    # NOTE: cosines in fp64 — fp32 summation over ~2M elements has
                    # ~1e-4 error and would mask/forge direction diagnostics.
                    # Sampled at diag anchors only (diag_every; F-1 cost rule);
                    # norm matching below is unconditional (it IS the update rule).
                    if diag_step:
                        gf = grad.double()
                        agg["cos_O_g"].append(
                            float(torch.sum(o_mix_d * gf) / (o_mix_d.norm() * gf.norm() + 1e-30))
                        )
                        agg["cos_O_Onat"].append(
                            float(torch.sum(o_mix_d * o_nat_d) / (o_mix_d.norm() * o_nat_d.norm() + 1e-30))
                        )

                    adjusted_lr = adjust_lr(lr, group["adjust_lr_fn"], p.shape)
                    p.mul_(1 - lr * weight_decay)
                    if norm_match:
                        n_nat = o_nat_d.norm()
                        n_mix = o_mix_d.norm()
                        scale = float(n_nat / (n_mix + 1e-30))
                        agg["norm_rel_err"].append(
                            float((o_mix_d.norm() * scale - n_nat).abs() / (n_nat + 1e-30))
                        )
                        delta = o_mix_d * (-adjusted_lr * scale)
                        p.add_(delta.to(p.dtype))
                    else:
                        p.add_(o_mix, alpha=-adjusted_lr)
                else:
                    raise RuntimeError(f"unknown mode {mode}")

        self.stats = {
            "ns_seconds_native": agg["ns_seconds_native"],
            "ns_seconds_mix": agg["ns_seconds_mix"],
            "n_native": agg["n_native"],
            "n_mix": agg["n_mix"],
            "cos_O_g_mean": _mean(agg["cos_O_g"]),
            "cos_O_Onat_mean": _mean(agg["cos_O_Onat"]),
            "norm_rel_err_max": _max(agg["norm_rel_err"]),
        }
        return loss


def _mean(xs):
    return float(sum(xs) / len(xs)) if xs else None


def _max(xs):
    return float(max(xs)) if xs else None
