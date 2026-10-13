"""s30m_core — 30M adaptation layer over the frozen dv_core math kernel.

Model: MiniMind 30,025,216 (h512/8L/int1664/8Q-4KV/head64, vocab 6400 tied, seq 256)
built from minimind_base (M-line model card, routing_report.json verified).

Semantics (frozen, see 30M_FROZEN_PROTOCOL.md):
- AdamW arms (S/F/P): V16/dv conventions — betas (0.9,0.95) eps 1e-8, matrix(>=2D) wd 0.1 /
  1D wd 0, grads divided by total valid tokens, global clip 1.0, then dv_core.apply_step.
- Muon reference arm: M-line wiring verbatim — torch.optim.Muon (sha 38e95918) on the 7-pattern
  2D matrices (56 tensors) + AdamW aux on the remaining 34 tensors, same LR on both,
  wd 0.02 (M-line best-cell convention), momentum 0.95/Nesterov/NS5/(3.4445,-4.775,2.0315)/eps 1e-7.
- Data: artifacts/train_tokens_s30m.u32 (17M tokens; first 8.5M byte-identical to V16 cache);
  V16 window semantics (seq+1 windows, stride seq, no wrap), micro 8 x accum 4 = 8192 tok/step.
- Eval: V16 development 256 docs via dv_core.eval_docs (same tokenizer 71f32c68).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
LAB = HERE.parent
MINIMIND_BASE = LAB / "minimind_base"
V16 = LAB / "adamw_spectral_pilot"

MODEL_KW = dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                num_key_value_heads=4, vocab_size=6400, use_moe=False, dropout=0.0,
                flash_attn=True)
SEQ = 256
MICRO = 8
ACCUM = 4
TOKENS_PER_STEP = MICRO * SEQ * ACCUM  # 8192
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
MATRIX_WD = 0.1
CLIP = 1.0
MUON_FREEZE = dict(momentum=0.95, nesterov=True, ns_steps=5,
                   ns_coefficients=(3.4445, -4.7750, 2.0315), eps=1e-7, adjust_lr_fn=None)
MUON_WD = 0.02
MUON_ROUTE_PATTERNS = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                       "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


def build_model(seed: int, device: str = "cpu") -> torch.nn.Module:
    import sys
    if str(MINIMIND_BASE) not in sys.path:
        sys.path.insert(0, str(MINIMIND_BASE))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    torch.manual_seed(seed)
    model = MiniMindForCausalLM(MiniMindConfig(**MODEL_KW)).float()
    assert model.model.embed_tokens.weight is model.lm_head.weight
    n = sum(p.numel() for p in model.parameters())
    assert n == 30_025_216, n
    return model.to(device)


def build_adamw(model: torch.nn.Module, lr: float) -> torch.optim.AdamW:
    named = dict(model.named_parameters())
    matrix = [p for p in named.values() if p.ndim >= 2]
    one_d = [p for p in named.values() if p.ndim < 2]
    return torch.optim.AdamW(
        [{"params": matrix, "weight_decay": MATRIX_WD}, {"params": one_d, "weight_decay": 0.0}],
        lr=lr, betas=ADAMW_BETAS, eps=ADAMW_EPS, amsgrad=False, maximize=False,
        foreach=False, fused=False)


def build_muon(model: torch.nn.Module, lr: float):
    from torch.optim import Muon
    named = dict(model.named_parameters())
    muon_params = [p for name, p in named.items() if any(pat in name for pat in MUON_ROUTE_PATTERNS)]
    aux_params = [p for name, p in named.items() if not any(pat in name for pat in MUON_ROUTE_PATTERNS)]
    assert len(muon_params) == 56 and len(aux_params) == 34, (len(muon_params), len(aux_params))
    muon_opt = Muon(muon_params, lr=lr, weight_decay=MUON_WD, momentum=MUON_FREEZE["momentum"],
                    nesterov=MUON_FREEZE["nesterov"], ns_steps=MUON_FREEZE["ns_steps"],
                    ns_coefficients=tuple(MUON_FREEZE["ns_coefficients"]), eps=MUON_FREEZE["eps"],
                    adjust_lr_fn=MUON_FREEZE["adjust_lr_fn"])
    aux_opt = torch.optim.AdamW(aux_params, lr=lr, betas=ADAMW_BETAS, eps=ADAMW_EPS,
                                weight_decay=MUON_WD)
    return [muon_opt, aux_opt]


class TokenStream:
    """V16 training_batch semantics: consecutive (seq+1) windows, stride seq, no wrap."""

    def __init__(self, cache_path: Path | None = None):
        self.mem = np.memmap(cache_path or (HERE / "artifacts" / "train_tokens_s30m.u32"),
                             dtype=np.uint32, mode="r")

    def micro_batch(self, cursor: int, device: str):
        batch, seq = MICRO, SEQ
        needed = batch * seq + 1
        if cursor + needed > len(self.mem):
            raise RuntimeError("DATA_INSUFFICIENT: token cache exhausted, no wrapping")
        array = np.stack([np.asarray(self.mem[cursor + i * seq:cursor + i * seq + seq + 1],
                                     dtype=np.int64) for i in range(batch)])
        return torch.from_numpy(array).to(device), cursor + batch * seq


def dev_docs():
    return json.load(open(V16 / "campaign_v16" / "eval_sets.json", encoding="utf-8"))["development"]


def anchor_names_for(target_block: str) -> tuple:
    layer_idx = int(target_block.split(".")[2])
    other = 1 if layer_idx == 2 else 2
    return (target_block,
            f"model.layers.{other}.mlp.down_proj.weight",
            f"model.layers.{layer_idx}.self_attn.o_proj.weight")
