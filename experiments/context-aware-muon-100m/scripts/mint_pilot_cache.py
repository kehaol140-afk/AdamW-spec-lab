"""Mint the FROZEN-PILOT 33,685,504-token train cache for context_aware_muon.

V16 doc-partition rule re-vendored verbatim from adamw_spectral_pilot/prepare_data.py
(same source family as adamw_direction_scale30m/src/s30m_mint_cache.py, which this
script cross-checks by byte prefix):
  normalize " ".join(text.split()) -> doc_id=sha256 -> bucket=int(id[:16],16)%100
  cuts (70,76,82,94,97,100) -> roles; ONLY role 'train' is packed.
  packing: per train doc [BOS]+encode(text,add_special_tokens=False)+[EOS], concatenated.

Guarantees recorded in the manifest:
  - first 8,500,000 tokens byte-identical to campaign_v16/cache/train_tokens.u32
  - first 17,000,000 tokens byte-identical to s30m cache train_tokens_s30m.u32
  - size == 4096 steps x 8224 window tokens (micro8 x accum4 x (seq256+1)) exactly:
    pilot T=4096 consumes the cache with NO wrap.

Target size: 4096 * 8 * 4 * 257 = 33,685,504 window tokens.
(8192 prediction tokens/step as preregistered; window stream over-reads 1 column.)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path("/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl")
TOKDIR = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/minimind_repo/model")
V16_CACHE = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot/campaign_v16/cache/train_tokens.u32")
S30M_CACHE = Path("/home/b205/projects/training_dynamics_lab/adamw_direction_scale30m/artifacts/train_tokens_s30m.u32")
OUT = ROOT / "artifacts"

TARGET_TOKENS = 4096 * 8 * 4 * 257  # 33,685,504
CUTS = (70, 76, 82, 94, 97, 100)
ROLES = ("train", "lr_select", "probe_select", "probe_confirm", "branch_eval_reserved", "final_holdout_reserved")


def main() -> None:
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
    import time

    t0 = time.perf_counter()
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
            stream[n : n + take] = np.asarray(ids[:take], dtype=np.uint32)
            n += take
            train_docs += 1
    cache_path = OUT / "pilot_train_tokens_33p7m.u32"
    stream[:n].tofile(cache_path)
    sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()

    v16 = np.fromfile(V16_CACHE, dtype=np.uint32)
    v16_prefix = bool(np.array_equal(stream[: len(v16)], v16))
    s30m = np.fromfile(S30M_CACHE, dtype=np.uint32)
    s30m_prefix = bool(np.array_equal(stream[: len(s30m)], s30m))

    manifest = {
        "status": "MINTED" if (v16_prefix and s30m_prefix) else "PREFIX_MISMATCH",
        "tokens": int(n),
        "target": TARGET_TOKENS,
        "train_documents_consumed": train_docs,
        "scanned_bytes": scanned,
        "sha256": sha,
        "v16_prefix_match_8p5m": v16_prefix,
        "s30m_prefix_match_17m": s30m_prefix,
        "corpus": str(CORPUS),
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "tokenizer_dir": str(TOKDIR),
        "tokenizer_sha256": hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest(),
        "packing": "per train doc [BOS]+encode(text)+[EOS]; concatenated; V16 prepare() rules (cuts 70/76/82/94/97/100)",
        "roles": role_counts,
        "no_wrap_check": "size == 4096 steps x 8224 window tokens; pilot consumes exactly, no wrap",
        "mint_seconds": round(time.perf_counter() - t0, 1),
    }
    (OUT / "pilot_cache_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    if not (v16_prefix and s30m_prefix):
        raise SystemExit("PREFIX_MISMATCH")


if __name__ == "__main__":
    main()
