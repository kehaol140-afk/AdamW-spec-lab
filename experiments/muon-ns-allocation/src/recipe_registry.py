"""Recipe registry: groups, layers, LR clock, coefficients, target split rule."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT.parent

GROUPS = ["q", "k", "v", "o", "gate", "up", "down"]
GROUP_PATTERNS = {"q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
                  "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight",
                  "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
                  "down": "mlp.down_proj.weight"}
K_GRID = [3, 5, 7]
NS_COEFFS = (3.4445, -4.775, 2.0315)
EPS_NORM = 1e-7
MOMENTUM = 0.95
WARMUP, COS_MIN, TOTAL = 32, 0.1, 2048
PEAK_LR = 0.01
WD = 0.02
P_MIX = 0.5
MANIFEST_SEED = 2030
PARENT_SEED = 3101
P1_STATES = [512, 1536]
Q_VERSION = "P0P1_Q_v1"
Q_WINDOWS = {"select": 128, "P1_check": 128, "P2_endpoint": 256}
TEMPLATE_RUN = "m1_muon_lr2.0_wd0.02_p0.5_s2030"
TEMPLATE_DIR = LAB / "muon_transition_experiment" / "outputs" / "m1" / TEMPLATE_RUN

# group -> list of param names (built from model once; verified against routing)


def group_of_param(name: str):
    tail = name.split("model.layers.")[-1]
    for g, pat in GROUP_PATTERNS.items():
        if tail.endswith(pat):
            return g
    return None


def lr_at(step: int) -> float:
    """Production clock: the LR update `step` consumes (1-based), warmup then cosine."""
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    span = TOTAL - WARMUP
    w = (step - WARMUP) / span
    return PEAK_LR * (COS_MIN + 0.45 * (1.0 + math.cos(math.pi * w)))


def shape_factor(shape) -> float:
    a, b = int(shape[0]), int(shape[1])
    return math.sqrt(max(1.0, a / b))


def q_set_of_group(group_id: str) -> int:
    d = hashlib.sha256((Q_VERSION + "|" + str(group_id)).encode()).digest()[0]
    return d % 3  # 0 select, 1 P1_check, 2 P2_endpoint


def load_cfg():
    return json.loads((ROOT / "configs" / "p0_p1_v1_1.json").read_text())
