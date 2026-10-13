"""Parameter routing for Group-Muon-A / Group-Muon-B on the dense MiniMind 100M.

Roles (13 layers, dense FFN):
  ffn   : layers.{0..12}.mlp.{gate_proj,up_proj,down_proj}.weight   -> 39 matrices (Group-A mix set)
  vo    : layers.{0..12}.self_attn.{v_proj,o_proj}.weight          -> 26 matrices (Group-B extension)
  qk    : layers.{0..12}.self_attn.{q_proj,k_proj}.weight           -> 26 matrices (always native Muon)
  aux   : embed_tokens (tied with lm_head, single object), model.norm, per-layer
          q_norm/k_norm/input_layernorm/post_attention_layernorm   -> 54 tensors (AdamW)

Group-Muon-A: mix(FFN) + native(QK,VO)
Group-Muon-B: mix(FFN,VO) + native(QK)
"""

from __future__ import annotations

from collections import OrderedDict

from torch import nn


def route_module(module_name: str) -> str:
    if module_name == "model.embed_tokens.weight":
        return "aux"
    if module_name == "model.norm.weight":
        return "aux"
    if ".mlp.gate_proj.weight" in module_name or ".mlp.up_proj.weight" in module_name or ".mlp.down_proj.weight" in module_name:
        return "ffn"
    if ".self_attn.v_proj.weight" in module_name or ".self_attn.o_proj.weight" in module_name:
        return "vo"
    if ".self_attn.q_proj.weight" in module_name or ".self_attn.k_proj.weight" in module_name:
        return "qk"
    return "aux"  # q_norm/k_norm/layernorms and anything scalar


def route_model(model: nn.Module) -> "OrderedDict[str, list[nn.Parameter]]":
    out: OrderedDict[str, list[nn.Parameter]] = OrderedDict(
        (k, []) for k in ("ffn", "vo", "qk", "aux")
    )
    seen = set()
    for name, p in model.named_parameters():
        role = route_module(name)
        out[role].append(p)
        seen.add(id(p))
    # tied head must be registered once
    n_total = len(list(model.parameters()))
    assert len(seen) == n_total, "duplicate parameter objects (tie broken?)"
    for role in ("ffn", "vo", "qk"):
        for p in out[role]:
            assert p.ndim == 2, f"non-2D tensor in muon route {role}: {tuple(p.shape)}"
    for p in out["aux"]:
        assert id(p) not in [id(q) for q in out["ffn"] + out["vo"] + out["qk"]]
    return out


EXPECTED_COUNTS_100M = {"ffn": 39, "vo": 26, "qk": 26, "aux": 54}
EXPECTED_COUNTS_30M = {"ffn": 24, "vo": 16, "qk": 16, "aux": 34}


def verify_routing(routed, scale: str = "100m") -> dict:
    exp = EXPECTED_COUNTS_100M if scale == "100m" else EXPECTED_COUNTS_30M
    got = {k: len(v) for k, v in routed.items()}
    if got != exp:
        raise RuntimeError(f"routing mismatch ({scale}): {got} != {exp}")
    return got


def muon_tensor_count(group_mode: str) -> int:
    """Matrices touched by the direction mix, by arm:
    'native' -> 0; 'A' -> 39 (FFN); 'B' -> 65 (FFN+V/O)."""
    return {"native": 0, "A": 39, "B": 65}[group_mode]
