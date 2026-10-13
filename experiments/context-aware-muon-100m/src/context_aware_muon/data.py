"""Deterministic token cache (V16 packing family) + window stream + schedule.

Packing rule (smoke/pilot variant, corpus order):
  stream = concat over docs of [BOS] + enc(text) + [EOS];  uint32 flat cache.
Windows: consecutive (seq_len+1)-slices with stride seq_len -> each yields
  seq_len prediction targets (minimind forward shifts internally).
NOTE: the formal pilot must re-vendor the exact V16 doc-partition rule
(normalize -> sha256 partition -> train role) from adamw_spectral_pilot
prepare_data.py before freezing its data config; smoke uses corpus order
(deterministic, disclosed). See 100M_PILOT_PROTOCOL_DRAFT.md gap list.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

BOS, EOS = 1, 2
CORPUS = Path("/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl")
TOKENIZER = Path("/home/b205/projects/training_dynamics_lab/minimind_base/model/tokenizer.json")


def build_cache(out_path: Path, max_tokens: int, corpus: Path = CORPUS) -> dict:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    out = np.empty(max_tokens, dtype=np.uint32)
    filled = 0
    docs = 0
    with open(corpus, "r", encoding="utf-8") as f:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            ids = tok.encode(text, add_special_tokens=False).ids
            seq = [BOS] + ids + [EOS]
            take = min(len(seq), max_tokens - filled)
            if take <= 0:
                break
            out[filled : filled + take] = np.asarray(seq[:take], dtype=np.uint32)
            filled += take
            docs += 1
            if filled >= max_tokens:
                break
    out = out[:filled]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.tofile(out_path)
    h = hashlib.sha256(out.tobytes()).hexdigest()
    meta = {
        "tokens": int(filled),
        "docs": docs,
        "sha256": h,
        "corpus": str(corpus),
        "packing": "[BOS]+enc+[EOS] concatenated, corpus order (smoke variant)",
    }
    (Path(out_path).with_suffix(".meta.json")).write_text(json.dumps(meta, indent=1))
    return meta


class WindowStream:
    """Deterministic window stream over a uint32 cache; cursor-based, no RNG."""

    def __init__(self, cache_path: Path, seq_len: int = 256):
        self.tokens = np.fromfile(str(cache_path), dtype=np.uint32)
        self.seq_len = seq_len
        self.cursor = 0

    def next_batch(self, micro: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = self.seq_len + 1
        need = micro * n
        if self.cursor + need > len(self.tokens):
            self.cursor = 0  # (pilot never wraps; smoke safety)
        w = torch.from_numpy(self.tokens[self.cursor : self.cursor + need].astype(np.int64)).view(
            micro, n
        )
        self.cursor += need
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()


def eta_schedule(step: int, total: int, warmup: int, peak: float, final_ratio: float = 0.1) -> float:
    """dv/V16 semantics: linear warmup then cosine to final_ratio*peak."""
    import math

    if step < warmup:
        return peak * (step + 1) / warmup
    u = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (final_ratio + (1 - final_ratio) / 2 * (1 + math.cos(math.pi * u)))
