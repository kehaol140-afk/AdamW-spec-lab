#!/usr/bin/env python3
"""Corpus verification chain (CPU-only, read-only):
V1 reproduce partition ledger from server corpus prefix (byte-compare doc_roles.jsonl)
V2 full-corpus id map; locate the 886 legacy eval docs (cross-ledger GAP_2)
V3 tokenize the 1068 holdout_v2 candidates with the frozen tokenizer
V4 tokenizer-chain check: re-encode frozen dev docs, compare ids exactly
V5 content dedup: candidate token sequences vs all 1536 frozen eval docs
V6 train-cache spot check: candidate 32-token prefixes absent from 8.5M cache
"""
import hashlib, json, sys
from pathlib import Path

V16 = Path("/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot")
DV = Path("/home/b205/projects/training_dynamics_lab/adamw_direction_validation")
CORPUS = Path("/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl")
PREFIX_BYTES = 5429819
EXPECT_PREFIX_SHA = "2cb92156c151a54826974e57bbffaf0c508433f2e0fd35580d9394c8c5bdf2f7"
CUTS = (70, 76, 82, 94, 97, 100)
ROLES = ("train", "lr_select", "probe_select", "probe_confirm", "branch_eval_reserved", "final_holdout_reserved")
OUT = DV / "verification_out"
OUT.mkdir(exist_ok=True)

def role_for(text):
    normalized = " ".join(text.split())
    doc_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    bucket = int(doc_id[:16], 16) % 100
    return normalized, doc_id, next(r for cut, r in zip(CUTS, ROLES) if bucket < cut)

def scan(max_bytes=None, want_texts_for=None):
    """Stream corpus; return ordered (doc_id, role[, normalized text for ids in want_texts_for])."""
    seen = set()
    out = []
    texts = {} if want_texts_for is not None else None
    scanned = 0
    with CORPUS.open("rb") as f:
        while True:
            raw = f.readline(2097152 + 1)
            if not raw:
                break
            if max_bytes is not None and (len(raw) > 2097152 or scanned + len(raw) > max_bytes):
                break
            scanned += len(raw)
            try:
                row = json.loads(raw)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str) or any(
                        k in row for k in ("conversations", "messages", "instruction", "response")):
                    continue
            except Exception:
                continue
            text, doc_id, role = role_for(row["text"])
            if not text or doc_id in seen:
                continue
            seen.add(doc_id)
            out.append((doc_id, role))
            if texts is not None and doc_id in want_texts_for:
                texts[doc_id] = text
    return out, scanned, texts

# ---- V1: prefix scan reproduces the ledger ----
print("V1: scanning prefix (5,429,819 B)...")
docs, scanned, _ = scan(PREFIX_BYTES)
print("  docs:", len(docs), "| scanned:", scanned)
lines = "".join(json.dumps({"doc_id": d, "role": r}) + "\n" for d, r in docs)
ledger = (V16 / "artifacts" / "doc_roles.jsonl").read_text(encoding="utf-8")
v1 = (lines == ledger)
print("  ledger byte-identical:", v1)
if not v1:
    (OUT / "rescan_prefix.jsonl").write_text(lines, encoding="utf-8")
    sys.exit("V1 FAILED")

# ---- V2: full-corpus scan; locate legacy eval ids ----
print("V2: full-corpus scan (1.24 GB)...")
es = json.load(open(V16 / "campaign_v16" / "eval_sets.json", encoding="utf-8"))
eval_ids = {}
for split, ds in es.items():
    for d in ds:
        eval_ids[d["doc_id"]] = split
alldocs, total_scanned, _ = scan()
print("  full docs:", len(alldocs), "| bytes:", total_scanned)
ledger_ids = {d for d, _ in docs}
legacy_missing = [i for i in eval_ids if i not in ledger_ids]
located = sum(1 for i in legacy_missing if any(d == i for d, _ in alldocs))
full_id_set = {d for d, _ in alldocs}
loc = {i: (i in full_id_set) for i in legacy_missing}
print("  legacy eval ids not in ledger:", len(legacy_missing), "| located in full corpus:", sum(loc.values()))

# ---- V3/V4/V5 ----
print("V3-V5: tokenizing with frozen tokenizer...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(str(V16 / "minimind_repo" / "model"), local_files_only=True)
cand_ids = [l.strip() for l in (DV / "splits" / "holdout_v2_candidate_ids.txt").open() if l.strip()]
# V4 tokenizer-chain: re-encode first 16 dev docs by matching text from prefix
_, _, prefix_texts = scan(PREFIX_BYTES, want_texts_for={d["doc_id"] for d in es["development"][:16]})
ok4 = 0
for d in es["development"][:16]:
    t = prefix_texts.get(d["doc_id"])
    if t is None:
        continue
    ids = [1] + tok.encode(t, add_special_tokens=False) + [2]
    if ids == d["ids"]:
        ok4 += 1
print("  V4 tokenizer chain (dev docs re-encoded identically):", ok4, "/ 16")
# V5 dedup: encode all eval doc texts found anywhere in corpus; compare token seq with candidates
eval_seq = {}
for d_id in eval_ids:
    eval_seq[d_id] = None
need = set(eval_ids) - set(prefix_texts.keys())
_, _, full_texts_by_need = scan(None, want_texts_for=need)
for d_id in eval_ids:
    t = prefix_texts.get(d_id) or full_texts_by_need.get(d_id)
    if t is None:
        continue
    eval_seq[d_id] = tuple([1] + tok.encode(t, add_special_tokens=False) + [2])
encoded_eval = {s for s in eval_seq.values() if s is not None}
print("  eval docs encoded:", len(encoded_eval), "/", len(eval_ids))
_, _, cand_texts = scan(PREFIX_BYTES, want_texts_for=set(cand_ids))
bad_dup, missing = [], []
seqs = {}
for c in cand_ids:
    t = cand_texts.get(c)
    if t is None:
        missing.append(c); continue
    s = tuple([1] + tok.encode(t, add_special_tokens=False) + [2])
    seqs[c] = s
    if s in encoded_eval:
        bad_dup.append(c)
print("  V5 candidates missing from corpus:", len(missing), "| token-seq duplicates of frozen eval docs:", len(bad_dup))
# V6 cache spot check: first 32 tokens of each candidate absent from train cache
import numpy as np
mem = np.memmap(V16 / "campaign_v16" / "cache" / "train_tokens.u32", dtype=np.uint32, mode="r")
cache_bytes = mem.tobytes()
hits = 0
for c, s in seqs.items():
    probe = np.asarray(s[:32], dtype=np.uint32).tobytes()
    if probe in cache_bytes:
        hits += 1
print("  V6 candidate 32-token prefixes found in train cache:", hits, "/", len(seqs))

json.dump({
 "v1_prefix_ledger_reproduced": v1, "v1_docs": len(docs),
 "v2_full_docs": len(alldocs), "v2_legacy_missing_from_ledger": len(legacy_missing),
 "v2_legacy_located_in_full_corpus": sum(loc.values()),
 "v4_tokenizer_chain_matches": ok4, "v4_checked": 16,
 "v5_candidates": len(cand_ids), "v5_missing": missing, "v5_token_seq_dup_of_eval": bad_dup,
 "v6_cache_prefix_hits": hits, "v6_checked": len(seqs),
 "corpus": {"path": str(CORPUS), "bytes": total_scanned,
            "prefix_sha256": EXPECT_PREFIX_SHA, "full_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest()},
}, open(OUT / "corpus_verification.json", "w"), indent=1)
print("WROTE", OUT / "corpus_verification.json")
