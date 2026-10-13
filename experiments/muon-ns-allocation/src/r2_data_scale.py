"""R2 data-update scaling semantics: effective_lr = r * base_lr,
effective_weight_decay = base_wd / r (decay product preserved), recomputed from the
base clock every step; plus arm registry and first-step verification helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = {"N": {"muon": 1.0, "aux": 1.0},
        "MH": {"muon": 0.5, "aux": 1.0},
        "AH": {"muon": 0.5, "aux": 0.5}}


def group_scale(arm: str, is_muon: bool) -> float:
    return ARMS[arm]["muon" if is_muon else "aux"]


def effective_lr_wd(base_lr: float, base_wd: float, r: float):
    """Data-update scaling with invariant decay product base_lr*base_wd."""
    return r * base_lr, base_wd / r


def set_group_lr_wd(opt, base_lr: float, base_wd: float, r: float, is_muon: bool):
    eff_lr, eff_wd = effective_lr_wd(base_lr, base_wd, r)
    for group in opt.param_groups:
        group["lr"] = eff_lr
        group["weight_decay"] = eff_wd
    return eff_lr, eff_wd
