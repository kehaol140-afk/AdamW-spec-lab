import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from context_aware_muon.model_100m import build_model
from context_aware_muon.routing import (
    EXPECTED_COUNTS_100M,
    EXPECTED_COUNTS_30M,
    muon_tensor_count,
    route_model,
    verify_routing,
)


def test_routing_counts_and_disjoint_100m():
    m = build_model("100m", seed=1)
    r = route_model(m)
    assert verify_routing(r, "100m") == EXPECTED_COUNTS_100M
    ids = [id(p) for ps in r.values() for p in ps]
    assert len(ids) == len(set(ids)) == 39 + 26 + 26 + 54


def test_routing_counts_30m():
    m = build_model("30m", seed=1)
    assert verify_routing(route_model(m), "30m") == EXPECTED_COUNTS_30M


def test_group_sizes():
    assert muon_tensor_count("native") == 0
    assert muon_tensor_count("A") == 39
    assert muon_tensor_count("B") == 65


def test_muon_route_all_2d_and_shapes_preserved():
    m = build_model("100m", seed=1)
    r = route_model(m)
    for role in ("ffn", "vo", "qk"):
        for p in r[role]:
            assert p.ndim == 2
    shapes = {tuple(p.shape) for p in r["ffn"]}
    assert (2432, 768) in shapes and (768, 2432) in shapes  # gate/up vs down, no reshape
