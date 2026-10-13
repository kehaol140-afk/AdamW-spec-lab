import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_aware_muon.model_100m import (
    EXPECTED_PARAMS,
    build_model,
    param_fingerprint,
)


def test_100m_param_count_and_tie():
    m = build_model("100m", seed=1)
    n = sum(p.numel() for p in m.parameters())
    assert n == EXPECTED_PARAMS["100m"], n
    assert m.model.embed_tokens.weight is m.lm_head.weight
    # target block shapes preserved
    d = m.model.layers[0].mlp.down_proj.weight
    assert tuple(d.shape) == (768, 2432), tuple(d.shape)


def test_30m_param_count():
    m = build_model("30m", seed=1)
    assert sum(p.numel() for p in m.parameters()) == EXPECTED_PARAMS["30m"]


def test_init_determinism():
    a = param_fingerprint(build_model("30m", seed=42))
    b = param_fingerprint(build_model("30m", seed=42))
    c = param_fingerprint(build_model("30m", seed=43))
    assert a == b and a != c
