"""R2F recipes: (r, d) double multipliers rebuilt from the immutable base each step.

eff_lr = r * base_lr ; eff_wd = (d / r) * base_wd  (decay product = d * base_lr * base_wd).
N=(1,1) AH=(0.5,1) L=(0.5,0.5) D=(1,0.5)."""
from __future__ import annotations

RECIPES = {
    "N":  {"r": 1.0, "d": 1.0},
    "AH": {"r": 0.5, "d": 1.0},
    "L":  {"r": 0.5, "d": 0.5},
    "D":  {"r": 1.0, "d": 0.5},
}


def effective(base_lr: float, base_wd: float, recipe: str):
    r = RECIPES[recipe]["r"]
    d = RECIPES[recipe]["d"]
    return r * base_lr, (d / r) * base_wd


def decay_product(base_lr, base_wd, recipe):
    """Actual decay product = eff_lr * eff_wd = d * base_lr * base_wd."""
    eff_lr, eff_wd = effective(base_lr, base_wd, recipe)
    return eff_lr * eff_wd
