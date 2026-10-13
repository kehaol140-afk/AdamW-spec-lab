"""Low-memory, offline JSONL preparation; no Torch/model is imported here."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
from tokenizers import Tokenizer
from resource_policy import resource_decision

ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("run.py", "pilot_core.py", "pilot_workflow.py", "prepare_data.py",
                "resource_policy.py", "tests/test_numerics.py", "tests/test_resources.py",
                "RUN_COMMANDS_RESOLVED.ps1", "RUN_LOWMEM_SMOKE.ps1")
ROLES = ("train", "lr_select", "probe_select", "probe_confirm",
         "branch_eval_reserved", "final_holdout_reserved")
CUTS = (70, 76, 82, 94, 97, 100)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_object(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_hash() -> str:
    return sha_object({name: sha_file(ROOT / name) for name in SOURCE_FILES})


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(arg: str):
    path = Path(arg)
    if not path.is_absolute():
        path = ROOT / path
    return read_json(path), sha_file(path)


def event(kind: str, **fields) -> None:
    path = ROOT / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.time(), "kind": kind, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def prepare_resource(cfg: dict, cfg_hash: str, phase: str) -> dict:
    available = psutil.virtual_memory().available
    status, reason = resource_decision(cfg, available, False)
    rss = psutil.Process().memory_info().rss
    limit = int(cfg["runtime"]["process_rss_soft_limit_gib"] * 2 ** 30)
    if rss > limit:
        status, reason = "RESOURCE_BLOCKED", f"RSS={rss}, limit={limit}"
    row = {"time": time.time(), "protocol_id": cfg["protocol_id"],
           "config_hash": cfg_hash, "source_hash": source_hash(), "pid": os.getpid(),
           "stage": "prepare", "phase": phase, "gpu_optimizer_updates": 0,
           "ram_available_bytes": available, "process_rss_bytes": rss,
           "ram_runtime_stop_bytes": int(cfg["runtime"]["system_ram_free_reserve_gib"] * 2 ** 30),
           "ram_runtime_warn_bytes": int(cfg["runtime"].get("system_ram_warning_gib", cfg["runtime"]["system_ram_free_reserve_gib"]) * 2 ** 30),
           "gpu_free_bytes": None, "gpu_allocated_bytes": None, "gpu_reserved_bytes": None,
           "gpu_peak_allocated_bytes": None, "gpu_peak_reserved_bytes": None,
           "status": status, "reason": reason}
    path = ROOT / "logs" / "resources.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
    if status != "READY":
        event("resource", stage="prepare", phase=phase, status=status,
              ram_available_bytes=available, reason=reason)
    return row


def role_for(text: str):
    normalized = " ".join(text.split())
    doc_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    bucket = int(doc_id[:16], 16) % 100
    return normalized, doc_id, next(role for cut, role in zip(CUTS, ROLES) if bucket < cut)


def prepare(cfg: dict, cfg_hash: str) -> dict:
    binding = read_json(ROOT / "bindings.local.json")
    if Path(binding["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("ENV_BLOCKED: current Python differs from binding")
    source_name = binding.get("source_data_path")
    if not source_name:
        result = {"status": "DATA_INSUFFICIENT", "reason": "source_data_path missing",
                  "config_hash": cfg_hash, "source_hash": source_hash()}
        atomic_json(ROOT / "artifacts" / "data_manifest.json", result)
        event("prepare", status=result["status"], reason=result["reason"])
        return result
    source = Path(source_name)
    if not source.is_file() or binding.get("source_format") != "jsonl_text":
        raise RuntimeError("DATA_INSUFFICIENT: bound source missing or unsupported schema")
    tok_path = Path(binding["tokenizer_path"]) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    if max(vocab.values()) >= vocab_size:
        raise RuntimeError("ENV_BLOCKED: tokenizer ID/vocabulary mismatch")
    config_tok = read_json(Path(binding["tokenizer_path"]) / "tokenizer_config.json")
    bos = config_tok.get("bos_token_id", 1)
    eos = config_tok.get("eos_token_id", 2)
    if not isinstance(bos, int): bos = tokenizer.token_to_id(str(bos))
    if not isinstance(eos, int): eos = tokenizer.token_to_id(str(eos))
    if bos is None or eos is None or not 0 <= bos < vocab_size or not 0 <= eos < vocab_size:
        raise RuntimeError("ENV_BLOCKED: tokenizer BOS/EOS unresolved")
    sample = prepare_resource(cfg, cfg_hash, "before_scan")
    available = sample["ram_available_bytes"]
    if sample["status"] == "RESOURCE_BLOCKED":
        result = {"status": "RESOURCE_BLOCKED", "reason": sample["reason"],
                  "ram_available_bytes": available, "source_path": str(source),
                  "config_hash": cfg_hash, "source_hash": source_hash()}
        atomic_json(ROOT / "artifacts" / "data_manifest.json", result)
        event("prepare", status=result["status"], ram_available_bytes=available)
        return result
    data_cfg = cfg["data"]
    seq = cfg["runtime"]["sequence_length"]
    needed = {"lr_select": data_cfg["lr_select_blocks"],
              "probe_select": data_cfg["probe_select_blocks"],
              "probe_confirm": data_cfg["probe_confirm_blocks"]}
    formal_needed = 8 * seq * cfg["baseline"]["successful_optimizer_steps"] + 2 * 8 * seq + 1
    smoke_needed = 8 * seq * cfg["numerics"]["real_data_smoke_steps"] + 1
    cap = data_cfg["max_train_target_tokens"] + 1
    train_path = ROOT / "artifacts" / "train_tokens.u32"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    cache = np.memmap(train_path, dtype=np.uint32, mode="w+", shape=(cap,))
    roles_path = ROOT / "artifacts" / "doc_roles.jsonl"
    roles_temp = roles_path.with_name(roles_path.name + ".tmp")
    eval_rows = {role: [] for role in ROLES if role != "train"}
    role_counts = {role: 0 for role in ROLES}
    seen = set()
    n_train = 0
    scanned = 0
    scan_hash = hashlib.sha256()
    invalid = 0
    complete = True
    resource_blocked = False
    t0 = time.perf_counter()
    with source.open("rb") as input_file, roles_temp.open("w", encoding="utf-8") as roles_out:
        while scanned < data_cfg["max_scan_bytes"]:
            raw = input_file.readline(min(data_cfg["max_jsonl_line_bytes"] + 1,
                                          data_cfg["max_scan_bytes"] - scanned + 1))
            if not raw:
                break
            if len(raw) > data_cfg["max_jsonl_line_bytes"] or scanned + len(raw) > data_cfg["max_scan_bytes"]:
                complete = False
                break
            scanned += len(raw)
            scan_hash.update(raw)
            try:
                row = json.loads(raw)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str) or any(
                        key in row for key in ("conversations", "messages", "instruction", "response")):
                    invalid += 1
                    continue
                text, doc_id, role = role_for(row["text"])
                if not text or doc_id in seen:
                    continue
                seen.add(doc_id)
                roles_out.write(json.dumps({"doc_id": doc_id, "role": role}) + "\n")
                role_counts[role] += 1
                if role == "train" and n_train < cap:
                    ids = [bos] + tokenizer.encode(text, add_special_tokens=False).ids + [eos]
                    take = min(len(ids), cap - n_train)
                    cache[n_train:n_train + take] = np.asarray(ids[:take], dtype=np.uint32)
                    n_train += take
                elif role in needed and len(eval_rows[role]) < needed[role]:
                    ids = [bos] + tokenizer.encode(text, add_special_tokens=False).ids[:seq - 1] + [eos]
                    eval_rows[role].append({"doc_id": doc_id, "ids": ids})
                if len(seen) % 500 == 0:
                    sample = prepare_resource(cfg, cfg_hash, "scan_500_documents")
                    if sample["status"] == "RESOURCE_BLOCKED":
                        resource_blocked = True
                        complete = False
                        break
                if n_train >= formal_needed and all(len(eval_rows[r]) >= n for r, n in needed.items()):
                    complete = False
                    break
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                invalid += 1
        else:
            complete = False
    os.replace(roles_temp, roles_path)
    cache.flush()
    del cache
    with train_path.open("r+b") as f:
        f.truncate(n_train * 4)
    eval_path = ROOT / "artifacts" / "eval_blocks.json"
    atomic_json(eval_path, eval_rows)
    smoke_ready = n_train >= smoke_needed
    formal_ready = n_train >= formal_needed and all(len(eval_rows[r]) >= n for r, n in needed.items())
    st = source.stat()
    result = {"status": "RESOURCE_BLOCKED" if resource_blocked else
              "READY" if formal_ready else "READY_FOR_SMOKE" if smoke_ready else "DATA_INSUFFICIENT",
              "smoke_ready": smoke_ready, "formal_ready": formal_ready,
              "source_path": str(source), "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns,
              "scanned_bytes": scanned, "scanned_sha256": scan_hash.hexdigest(),
              "full_source_sha256": scan_hash.hexdigest() if complete and scanned == st.st_size else None,
              "scan_complete": complete and scanned == st.st_size,
              "invalid_schema_rows": invalid, "role_document_counts": role_counts,
              "eval_block_counts": {key: len(value) for key, value in eval_rows.items()},
              "train_token_count": n_train, "train_cache_sha256": sha_file(train_path),
              "eval_cache_sha256": sha_file(eval_path),
              "doc_assignments_file": str(roles_path.relative_to(ROOT)),
              "doc_assignments_sha256": sha_file(roles_path),
              "tokenizer_hash": binding["tokenizer_fingerprint"],
              "tokenizer_vocab_size": vocab_size, "bos_token_id": bos, "eos_token_id": eos,
              "config_hash": cfg_hash, "source_hash": source_hash(),
              "partition": "normalized whitespace text SHA256; first 16 hex mod 100; cuts 70/76/82/94/97/100",
              "train_packing": "BOS + text + EOS concatenated; windows length seq+1 stride seq; shift once",
              "eval_packing": "one block per document, no cross-document packing; explicit length mask",
              "dedup_scope": "exact normalized text within consumed scan window only",
              "prepare_seconds": time.perf_counter() - t0,
              "ram_available_at_end_bytes": psutil.virtual_memory().available}
    result["data_hash"] = sha_object({k: v for k, v in result.items() if k not in
                                      ("status", "data_hash", "prepare_seconds", "ram_available_at_end_bytes")})
    prepare_resource(cfg, cfg_hash, "after_scan")
    atomic_json(ROOT / "artifacts" / "data_manifest.json", result)
    atomic_json(ROOT / "artifacts" / "protocol_resolved.json",
                {"protocol_id": cfg["protocol_id"], "config_hash": cfg_hash,
                 "bindings": binding, "model": cfg["model"], "data_hash": result["data_hash"]})
    event("prepare", status=result["status"], scanned_bytes=scanned,
          train_tokens=n_train, eval_blocks=result["eval_block_counts"],
          ram_available_bytes=result["ram_available_at_end_bytes"])
    return result
