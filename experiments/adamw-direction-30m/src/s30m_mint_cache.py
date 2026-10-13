#!/usr/bin/env python3
"""Mint the 30M-project train token cache by replaying the V16 partition/packing rules
from the server corpus start, extending beyond V16's 8.5M-token cache.

Guarantees recorded:
- first 8,500,000 tokens byte-identical to campaign_v16/cache/train_tokens.u32 (stream identity)
- packing rule identical to V16 prepare(): per train doc, [BOS] + encode(text) + [EOS]
- roles per doc_roles partition (CUTS 70/76/82/94/97/100) so eval-roled docs stay out of train
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

import numpy as np

CORPUS = Path("/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl")
V16_CACHE = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/cache/train_tokens.u32")
TOKDIR = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/minimind_repo/model")
OUT = Path(__file__).resolve().parent.parent / "artifacts"
TARGET_TOKENS = 17_000_000
CUTS = (70, 76, 82, 94, 97, 100)
ROLES = ("train", "lr_select", "probe_select", "probe_confirm", "branch_eval_reserved", "final_holdout_reserved")

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKDIR), local_files_only=True)
    assert tok.bos_token_id == 1 and tok.eos_token_id == 2, (tok.bos_token_id, tok.eos_token_id)
    role_counts = {r: 0 for r in ROLES}
    seen = set()
    stream = np.zeros(TARGET_TOKENS, dtype=np.uint32)
    n = 0
    scanned = 0
    train_docs = 0
    with CORPUS.open("rb") as f:
        while n < TARGET_TOKENS:
            raw = f.readline(2_097_152 + 1)
            if not raw:
                raise SystemExit("CORPUS_EXHAUSTED before token target")
            if len(raw) > 2_097_152:
                continue
            scanned += len(raw)
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            if any(k in row for k in ("conversations", "messages", "instruction", "response")):
                continue
            text = " ".join(row["text"].split())
            if not text:
                continue
            doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
            bucket = int(doc_id[:16], 16) % 100
            role = next(r for cut, r in zip(CUTS, ROLES) if bucket < cut)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            role_counts[role] += 1
            if role != "train":
                continue
            ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
            take = min(len(ids), TARGET_TOKENS - n)
            stream[n:n + take] = np.asarray(ids[:take], dtype=np.uint32)
            n += take
            train_docs += 1
    cache_path = OUT / "train_tokens_s30m.u32"
    stream[:n].tofile(cache_path)
    sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    v16 = np.fromfile(V16_CACHE, dtype=np.uint32)
    prefix_match = bool(np.array_equal(stream[: len(v16)], v16))
    manifest = {
        "status": "MINTED", "tokens": int(n), "target": TARGET_TOKENS,
        "train_documents_consumed": train_docs, "scanned_bytes": scanned,
        "sha256": sha, "v16_prefix_match": prefix_match, "v16_tokens": int(len(v16)),
        "corpus": str(CORPUS), "tokenizer_dir": str(TOKDIR),
        "tokenizer_sha256": hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest(),
        "packing": "per train doc [BOS]+encode(text)+[EOS]; concatenation; V16 prepare() rules",
        "roles": role_counts,
    }
    (OUT / "cache_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    if not prefix_match:
        raise SystemExit("V16_PREFIX_MISMATCH")

if __name__ == "__main__":
    main()
