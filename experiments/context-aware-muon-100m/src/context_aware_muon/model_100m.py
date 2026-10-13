"""100M MiniMind model builder + deterministic initialization.

Structure (G1b-verified on this machine, 2026-09-16):
  hidden 768 / 13 layers / 12 Q heads - 6 KV heads (head_dim 64, GQA 2:1)
  FFN 2432 (frozen default formula ceil(768*pi/64)*64), vocab 6400, tied head.
  Unique parameters: 100,784,000.

Codebase (read-only vendored by path): minimind_base/model/model_minimind.py
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch

MINIMIND_MODEL_PATH = Path(
    "/home/b205/projects/training_dynamics_lab/minimind_base/model/model_minimind.py"
)

CONFIG_100M = dict(
    hidden_size=768,
    num_hidden_layers=13,
    num_attention_heads=12,
    num_key_value_heads=6,
    intermediate_size=2432,
    vocab_size=6400,
    dropout=0.0,
    flash_attn=True,
    use_moe=False,
)

CONFIG_30M = dict(
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    num_key_value_heads=4,
    intermediate_size=1664,
    vocab_size=6400,
    dropout=0.0,
    flash_attn=True,
    use_moe=False,
)

EXPECTED_PARAMS = {"100m": 100_784_000, "30m": 30_025_216}

_module_cache: dict = {}


def _load_minimind_module():
    if "m" not in _module_cache:
        spec = importlib.util.spec_from_file_location(
            "cam_minimind_model", str(MINIMIND_MODEL_PATH)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module_cache["m"] = mod
    return _module_cache["m"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(scale: str = "100m", seed: int = 3471, device: str = "cpu") -> torch.nn.Module:
    """Deterministic construction: same seed -> bit-identical initial parameters
    (verified by fingerprint in tests). Tied embedding kept as a single object."""
    cfg = {"100m": CONFIG_100M, "30m": CONFIG_30M}[scale]
    mod = _load_minimind_module()
    with torch.no_grad():
        torch.manual_seed(seed)
        model = mod.MiniMindForCausalLM(mod.MiniMindConfig(**cfg))
    model.to(device)
    n = sum(p.numel() for p in model.parameters())
    if n != EXPECTED_PARAMS[scale]:
        raise RuntimeError(f"param count mismatch for {scale}: {n} != {EXPECTED_PARAMS[scale]}")
    assert model.model.embed_tokens.weight is model.lm_head.weight, "tied head broken"
    return model


def param_fingerprint(model: torch.nn.Module) -> str:
    """Stable hash over parameter names+bytes (init/endpoint identity checks)."""
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
