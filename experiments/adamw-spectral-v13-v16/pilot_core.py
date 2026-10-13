"""Local, offline implementation of ADAMW_A_SPECTRAL_LOCAL_V1_20261001."""
from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from resource_policy import resource_decision

ROOT = Path(__file__).resolve().parent
ROLES = ("train", "lr_select", "probe_select", "probe_confirm",
         "branch_eval_reserved", "final_holdout_reserved")
ROLE_CUTS = (70, 76, 82, 94, 97, 100)
SOURCE_FILES = ("run.py", "pilot_core.py", "pilot_workflow.py", "prepare_data.py",
                "resource_policy.py", "tests/test_numerics.py", "tests/test_resources.py",
                "RUN_COMMANDS_RESOLVED.ps1", "RUN_LOWMEM_SMOKE.ps1")


def sha_file(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = limit
        while True:
            b = f.read(min(1024 * 1024, remaining) if remaining is not None else 1024 * 1024)
            if not b:
                break
            h.update(b)
            if remaining is not None:
                remaining -= len(b)
                if remaining <= 0:
                    break
    return h.hexdigest()


def sha_object(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def source_hash() -> str:
    return sha_object({name: sha_file(ROOT / name) for name in SOURCE_FILES})


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def event(kind: str, **fields) -> None:
    path = ROOT / "logs" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.time(), "kind": kind, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        f.flush()
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(row)
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            columns = next(csv.reader(f))
        if set(columns) != set(row):
            raise ValueError(f"CSV schema mismatch: {path}")
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def config_path(arg: str) -> Path:
    p = Path(arg)
    return p if p.is_absolute() else ROOT / p


def load_config(arg: str) -> tuple[dict, str]:
    p = config_path(arg)
    return read_json(p), sha_file(p)


def read_binding() -> dict:
    p = ROOT / "bindings.local.json"
    if not p.is_file():
        raise RuntimeError("ENV_BLOCKED: run preflight first")
    b = read_json(p)
    if Path(b["python_executable"]).resolve() != Path(sys.executable).resolve():
        raise RuntimeError("ENV_BLOCKED: current Python differs from bindings.local.json")
    return b


def fingerprint(config_hash: str, data_hash: str | None = None) -> dict:
    b = read_binding()
    return {"source_hash": source_hash(), "config_hash": config_hash,
            "binding_hash": sha_object(b), "data_hash": data_hash}


@contextlib.contextmanager
def run_lock(stage: str):
    path = ROOT / "artifacts" / "run.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"pid": os.getpid(), "stage": stage, "time": time.time()}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"project lock exists: {path}; inspect recorded PID, do not kill other processes")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f)
        yield
    finally:
        if path.exists() and read_json(path).get("pid") == os.getpid():
            path.unlink()


def repo_candidates(root: Path) -> list[Path]:
    direct = [root / "minimind", root / "minimind-master"]
    valid = [p for p in direct if (p / "model" / "model_minimind.py").is_file()]
    if valid:
        return valid
    excluded = {".git", "dataset", "data", "cache", "checkpoints", "outputs", "results",
                "envs", "venv", ".venv", "adamw_spectral_pilot_handoff", "adamw_spectral_pilot"}
    queue = [(root, 0)]
    while queue:
        base, depth = queue.pop(0)
        if depth >= 4:
            continue
        for p in base.iterdir():
            if not p.is_dir() or p.name.lower() in excluded:
                continue
            if (p / "model" / "model_minimind.py").is_file():
                valid.append(p)
            else:
                queue.append((p, depth + 1))
    return valid


def data_candidates(root: Path) -> list[Path]:
    def has_text_schema(path: Path) -> bool:
        try:
            with path.open("rb") as f:
                for _ in range(5):
                    raw = f.readline(2 * 1024 * 1024 + 1)
                    if not raw:
                        break
                    if len(raw) > 2 * 1024 * 1024:
                        return False
                    row = json.loads(raw)
                    if isinstance(row, dict) and isinstance(row.get("text"), str) and not any(
                            k in row for k in ("conversations", "messages", "instruction", "response")):
                        return True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return False

    out = [p for p in (ROOT / "data").glob("*.jsonl") if has_text_schema(p)] if (ROOT / "data").exists() else []
    excluded = {".git", "cache", "checkpoints", "outputs", "results", "envs", "venv", ".venv",
                "adamw_spectral_pilot_handoff", "adamw_spectral_pilot"}
    queue = [(root, 0)]
    while queue:
        base, depth = queue.pop(0)
        if depth > 4:
            continue
        for p in base.iterdir():
            if p.is_dir():
                if depth < 4 and p.name.lower() not in excluded:
                    queue.append((p, depth + 1))
            elif (p.suffix.lower() == ".jsonl" and
                  not any(tag in p.name.lower() for tag in ("sft", "dpo", "ppo", "reward", "chat")) and
                  has_text_schema(p)):
                out.append(p)
    return sorted(set(out))


def ram_available() -> int:
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 0


def resource_sample(cfg: dict, phase: str, *, stage: str = "smoke",
                    gpu_updates: int = 0, admission: bool = False,
                    vram_limit: int | None = None) -> dict:
    """Log a measured resource boundary without initializing CUDA for logging."""
    import psutil
    runtime = cfg["runtime"]
    available = ram_available()
    status, reason = resource_decision(cfg, available, admission)
    rss = psutil.Process().memory_info().rss
    if rss > runtime["process_rss_soft_limit_gib"] * 2 ** 30:
        status, reason = "RESOURCE_BLOCKED", f"RSS={rss}, limit={int(runtime['process_rss_soft_limit_gib'] * 2 ** 30)}"
    cuda = torch.cuda.is_initialized()
    reserved = torch.cuda.memory_reserved(0) if cuda else None
    if vram_limit is not None and reserved is not None and reserved > vram_limit:
        status, reason = "RESOURCE_BLOCKED", f"CUDA reserved={reserved}, limit={vram_limit}"
    config_name = "local_v1_lowmem.json" if cfg.get("protocol_id", "").endswith("LOWMEM_20261002") else "local_v1.json"
    config_path = ROOT / "config" / config_name
    row = {"time": time.time(), "protocol_id": cfg["protocol_id"],
           "pid": os.getpid(), "config_hash": sha_file(config_path) if config_path.is_file() else None,
           "source_hash": source_hash(),
           "stage": stage, "phase": phase, "gpu_optimizer_updates": gpu_updates,
           "ram_available_bytes": available,
           "ram_preinit_min_bytes": int(runtime.get("system_ram_start_min_gib", runtime["system_ram_free_reserve_gib"]) * 2 ** 30),
           "ram_runtime_warn_bytes": int(runtime.get("system_ram_warning_gib", runtime["system_ram_free_reserve_gib"]) * 2 ** 30),
           "ram_runtime_stop_bytes": int(runtime["system_ram_free_reserve_gib"] * 2 ** 30),
           "process_rss_bytes": rss,
           "gpu_free_bytes": torch.cuda.mem_get_info(0)[0] if cuda else None,
           "gpu_allocated_bytes": torch.cuda.memory_allocated(0) if cuda else None,
           "gpu_reserved_bytes": reserved,
           "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if cuda else None,
           "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if cuda else None,
           "process_vram_limit_bytes": vram_limit, "status": status, "reason": reason}
    path = ROOT / "logs" / "resources.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
    if status != "READY":
        event("resource", phase=phase, stage=stage, status=status,
              ram_available_bytes=available, gpu_optimizer_updates=gpu_updates, reason=reason)
    return row


def require_resource(cfg: dict, phase: str, *, stage: str = "smoke",
                     gpu_updates: int = 0, admission: bool = False,
                     vram_limit: int | None = None) -> dict:
    row = resource_sample(cfg, phase, stage=stage, gpu_updates=gpu_updates,
                          admission=admission, vram_limit=vram_limit)
    if row["status"] == "RESOURCE_BLOCKED":
        raise RuntimeError(f"RESOURCE_BLOCKED: {phase}: {row['reason']}")
    return row


def preflight(cfg: dict, cfg_hash: str, root: Path) -> dict:
    root = root.resolve()
    repos = repo_candidates(root)
    ranked = [p for p in (root / "minimind", root / "minimind-master") if p in repos]
    repo = ranked[0] if ranked else (repos[0] if len(repos) == 1 else None)
    files = data_candidates(root)
    status = "READY"
    reasons = []
    if repo is None:
        status = "AMBIGUOUS_REPO" if repos else "ENV_BLOCKED"
        reasons.append("MiniMind repo unresolved")
    if not files:
        status = "DATA_INSUFFICIENT" if status == "READY" else status
        reasons.append("No local pretraining text JSONL found in bounded search")
    if len(files) > 1:
        status = "DATA_INSUFFICIENT" if status == "READY" else status
        reasons.append("Multiple data candidates require an explicit source_data_path binding")
    tok_dir = repo / "model" if repo else None
    tok_files = [tok_dir / "tokenizer.json", tok_dir / "tokenizer_config.json"] if tok_dir else []
    if not tok_dir or not all(p.is_file() for p in tok_files):
        status = "ENV_BLOCKED"
        reasons.append("Local tokenizer files missing")
    source_files = [repo / "model" / "model_minimind.py", repo / "trainer" / "train_pretrain.py"] if repo else []
    b = {"status": status, "repo_root": str(repo) if repo else None,
         "repo_relative_to_search_root": str(repo.relative_to(root)) if repo else None,
         "python_executable": str(Path(sys.executable).resolve()),
         "tokenizer_path": str(tok_dir) if tok_dir else None,
         "tokenizer_relative_to_repo": "model" if tok_dir else None,
         "source_data_path": str(files[0]) if len(files) == 1 else None,
         "source_data_relative_to_search_root": str(files[0].relative_to(root)) if len(files) == 1 else None,
         "source_format": "jsonl_text" if len(files) == 1 else None,
         "source_fingerprint": sha_object({str(p.relative_to(repo)): sha_file(p) for p in source_files if p.is_file()}) if repo else None,
         "tokenizer_fingerprint": sha_object({p.name: sha_file(p) for p in tok_files}) if all(p.is_file() for p in tok_files) else None,
         "repo_candidates": list(map(str, repos)), "data_candidates": list(map(str, files)),
         "search_root": str(root)}
    atomic_json(ROOT / "bindings.local.json", b)
    import platform
    dep = {}
    for name in ("torch", "transformers", "tokenizers", "numpy", "psutil"):
        try:
            mod = __import__(name)
            dep[name] = getattr(mod, "__version__", "installed")
        except Exception as e:
            dep[name] = f"MISSING_OR_BROKEN: {e}"
            status = "ENV_BLOCKED"
    gpu = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpu.append({"index": i, "name": torch.cuda.get_device_name(i),
                        "free_bytes": free, "total_bytes": total})
    model_check = None
    if repo and tok_dir and all(p.is_file() for p in tok_files) and all(
            not str(v).startswith("MISSING_OR_BROKEN") for v in dep.values()):
        try:
            model, optimizer, aliases, count = model_and_optimizer(
                cfg, b, cfg["baseline"]["peak_lr_grid"][1], "cpu")
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), local_files_only=True)
            model_check = {"status": "PASS", "parameter_count": count,
                           "vocab_size": len(tokenizer), "max_token_id": max(tokenizer.get_vocab().values()),
                           "bos_token_id": tokenizer.bos_token_id,
                           "eos_token_id": tokenizer.eos_token_id,
                           "pad_token_id": tokenizer.pad_token_id,
                           "tied_embedding": model.model.embed_tokens.weight is model.lm_head.weight,
                           "primary_parameter": "model.layers.1.mlp.down_proj.weight",
                           "primary_shape": list(model.model.layers[1].mlp.down_proj.weight.shape),
                           "attention_shape": list(model.model.layers[1].self_attn.o_proj.weight.shape),
                           "parameter_aliases": aliases}
            del model, optimizer
        except Exception as e:
            model_check = {"status": "FAILED", "reason": str(e)}
            status = "MODEL_SIZE_UNRESOLVED" if "MODEL_SIZE_UNRESOLVED" in str(e) else "ENV_BLOCKED"
            reasons.append("MiniMind model/tokenizer API check failed")
    b["status"] = status
    atomic_json(ROOT / "bindings.local.json", b)
    report = {"status": status, "reasons": reasons, "bindings": b,
              "python_version": platform.python_version(), "dependencies": dep,
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
              "gpu": gpu, "model_check": model_check,
              "ram_available_bytes": ram_available(), "config_hash": cfg_hash,
              "protocol_hash": sha_file(ROOT / "PROTOCOL_LOCAL_V1.md"),
              "source_hash": source_hash(), "time": time.time()}
    resource = resource_sample(cfg, "preflight_end", stage="preflight")
    report["ram_available_bytes"] = resource["ram_available_bytes"]
    if resource["status"] == "RESOURCE_BLOCKED":
        report["resource_status"] = "RESOURCE_BLOCKED"
        report["reasons"].append(resource["reason"])
    if not gpu:
        report["resource_status"] = "RESOURCE_BLOCKED"
        report["reasons"].append("bound Python cannot access CUDA")
    atomic_json(ROOT / "artifacts" / "preflight.json", report)
    event("preflight", status=status, resource_status=report.get("resource_status"), reasons=reasons)
    return report


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def partition_text(text: str) -> tuple[str, str]:
    doc_id = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    bucket = int(doc_id[:16], 16) % 100
    role = next(role for cut, role in zip(ROLE_CUTS, ROLES) if bucket < cut)
    return doc_id, role


def prepare(cfg: dict, cfg_hash: str) -> dict:
    from transformers import AutoTokenizer
    b = read_binding()
    if not b.get("source_data_path"):
        manifest = {"status": "DATA_INSUFFICIENT", "reason": "source_data_path missing",
                    "config_hash": cfg_hash, "source_hash": source_hash()}
        atomic_json(ROOT / "artifacts" / "data_manifest.json", manifest)
        event("prepare", status="DATA_INSUFFICIENT", reason="source_data_path missing")
        return manifest
    source = Path(b["source_data_path"])
    if not source.is_file() or b.get("source_format") != "jsonl_text":
        raise RuntimeError("DATA_INSUFFICIENT: bound source absent or schema not confirmed")
    tok = AutoTokenizer.from_pretrained(b["tokenizer_path"], local_files_only=True)
    if max(tok.get_vocab().values()) >= len(tok):
        raise RuntimeError("ENV_BLOCKED: tokenizer max ID exceeds vocabulary")
    available = ram_available()
    if resource_decision(cfg, available, False)[0] == "RESOURCE_BLOCKED":
        manifest = {"status": "RESOURCE_BLOCKED", "reason": resource_decision(cfg, available, False)[1],
                    "ram_available_bytes": available, "source_path": str(source),
                    "config_hash": cfg_hash, "source_hash": source_hash()}
        atomic_json(ROOT / "artifacts" / "data_manifest.json", manifest)
        event("prepare", status="RESOURCE_BLOCKED", ram_available_bytes=available)
        return manifest
    limits = cfg["data"]
    max_bytes = limits["max_scan_bytes"]
    line_limit = limits["max_jsonl_line_bytes"]
    seq = cfg["runtime"]["sequence_length"]
    needed = {"lr_select": limits["lr_select_blocks"],
              "probe_select": limits["probe_select_blocks"],
              "probe_confirm": limits["probe_confirm_blocks"]}
    formal_needed = 8 * seq * cfg["baseline"]["successful_optimizer_steps"] + 2 * 8 * seq + 1
    train_cap = limits["max_train_target_tokens"] + 1
    train_path = ROOT / "artifacts" / "train_tokens.u32"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    token_mem = np.memmap(train_path, dtype=np.uint32, mode="w+", shape=(train_cap,))
    n_train = 0
    eval_rows = {role: [] for role in ROLES if role != "train"}
    role_counts = {role: 0 for role in ROLES}
    roles_path = ROOT / "artifacts" / "doc_roles.jsonl"
    roles_tmp = roles_path.with_name(roles_path.name + ".tmp")
    seen = set()
    scanned = 0
    scan_hash = hashlib.sha256()
    invalid_schema = 0
    complete = True
    with source.open("rb") as f, roles_tmp.open("w", encoding="utf-8") as roles_out:
        while scanned < max_bytes:
            raw = f.readline(min(line_limit + 1, max_bytes - scanned + 1))
            if not raw:
                break
            if len(raw) > line_limit or scanned + len(raw) > max_bytes:
                complete = False
                break
            scanned += len(raw)
            scan_hash.update(raw)
            try:
                row = json.loads(raw)
                if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                    invalid_schema += 1
                    continue
                if any(key in row for key in ("conversations", "messages", "instruction", "response")):
                    invalid_schema += 1
                    continue
                text = normalize_text(row["text"])
                if not text:
                    continue
                doc_id, role = partition_text(text)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                roles_out.write(json.dumps({"doc_id": doc_id, "role": role}) + "\n")
                role_counts[role] += 1
                if role == "train" and n_train < train_cap:
                    ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False) + [tok.eos_token_id]
                    take = min(len(ids), train_cap - n_train)
                    token_mem[n_train:n_train + take] = np.asarray(ids[:take], dtype=np.uint32)
                    n_train += take
                elif role in needed and len(eval_rows[role]) < needed[role]:
                    ids = [tok.bos_token_id] + tok.encode(text, add_special_tokens=False)[:seq - 1] + [tok.eos_token_id]
                    if len(ids) >= 2:
                        eval_rows[role].append({"doc_id": doc_id, "ids": ids})
                if n_train >= formal_needed and all(len(eval_rows[r]) >= n for r, n in needed.items()):
                    complete = False
                    break
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                invalid_schema += 1
        else:
            complete = False
    os.replace(roles_tmp, roles_path)
    token_mem.flush()
    del token_mem
    with train_path.open("r+b") as f:
        f.truncate(n_train * 4)
    atomic_json(ROOT / "artifacts" / "eval_blocks.json", eval_rows)
    smoke_ok = n_train >= 8 * seq * cfg["numerics"]["real_data_smoke_steps"] + 1
    formal_ok = n_train >= formal_needed and all(
        len(eval_rows[role]) >= count for role, count in needed.items())
    st = source.stat()
    manifest = {"status": "READY" if formal_ok else "READY_FOR_SMOKE" if smoke_ok else "DATA_INSUFFICIENT",
                "smoke_ready": smoke_ok, "formal_ready": formal_ok, "source_path": str(source),
                "source_size": st.st_size, "source_mtime_ns": st.st_mtime_ns,
                "scanned_bytes": scanned, "scanned_sha256": scan_hash.hexdigest(),
                "full_source_sha256": scan_hash.hexdigest() if complete and scanned == st.st_size else None,
                "scan_complete": complete and scanned == st.st_size,
                "invalid_schema_rows": invalid_schema, "role_document_counts": role_counts,
                "eval_block_counts": {k: len(v) for k, v in eval_rows.items()},
                "train_token_count": n_train, "train_cache_sha256": sha_file(train_path),
                "eval_cache_sha256": sha_file(ROOT / "artifacts" / "eval_blocks.json"),
                "doc_assignments_file": str(roles_path.relative_to(ROOT)),
                "doc_assignments_sha256": sha_file(roles_path),
                "tokenizer_hash": b["tokenizer_fingerprint"],
                "config_hash": cfg_hash, "source_hash": source_hash(),
                "partition": "normalized whitespace text SHA256; first 16 hex mod 100; cuts 70/76/82/94/97/100",
                "train_packing": "BOS + text + EOS concatenated; windows length seq+1 stride seq; shift once",
                "eval_packing": "one block per document, no cross-document packing; explicit length mask",
                "dedup_scope": "exact normalized text within consumed scan window only"}
    manifest["data_hash"] = sha_object({k: v for k, v in manifest.items() if k not in ("status", "data_hash")})
    atomic_json(ROOT / "artifacts" / "data_manifest.json", manifest)
    atomic_json(ROOT / "artifacts" / "protocol_resolved.json",
                {"protocol_id": cfg["protocol_id"], "config_hash": cfg_hash, "bindings": b,
                 "model": cfg["model"], "data_hash": manifest["data_hash"]})
    event("prepare", status=manifest["status"], train_tokens=n_train,
          eval_blocks=manifest["eval_block_counts"], scanned_bytes=scanned)
    return manifest


def adamw_direction(param: torch.Tensor, grad: torch.Tensor | None, state: dict,
                    *, beta1: float, beta2: float, eps: float) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Direction using the next moments and this parameter's own next step."""
    if grad is None:
        return None, None
    step = int(state.get("step", 0)) + 1
    m = state.get("exp_avg", torch.zeros_like(param)) * beta1 + grad * (1 - beta1)
    v = state.get("exp_avg_sq", torch.zeros_like(param)) * beta2 + grad.square() * (1 - beta2)
    a = (m / (1 - beta1 ** step)) / ((v / (1 - beta2 ** step)).sqrt() + eps)
    return a, m


def adamw_next(param: torch.Tensor, grad: torch.Tensor | None, state: dict,
               *, lr: float, wd: float, beta1: float, beta2: float, eps: float):
    a, m = adamw_direction(param, grad, state, beta1=beta1, beta2=beta2, eps=eps)
    return (param.clone() if a is None else param * (1 - lr * wd) - lr * a), a, m


def split_head(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if a.ndim != 2:
        raise ValueError("SVD target must be a matrix")
    u, s, vh = torch.linalg.svd(a.to(device="cpu", dtype=torch.float64), full_matrices=False)
    head = s[0] * torch.outer(u[:, 0], vh[0, :])
    return head, a.to(device="cpu", dtype=torch.float64) - head, s


def spectral_candidate(old: torch.Tensor, a_head: torch.Tensor, a_rest: torch.Tensor,
                       eta: float, wd: float, rho: float, scale: float) -> torch.Tensor:
    return old * (1 - eta * wd) - eta * scale * (rho * a_head + a_rest)


def schedule_lr(peak: float, k: int, cfg: dict) -> float:
    warm = cfg["baseline"]["warmup_steps"]
    total = cfg["baseline"]["successful_optimizer_steps"]
    if not 1 <= k <= total:
        raise ValueError("optimizer step out of protocol range")
    return peak * (k / warm if k <= warm else
                   0.1 + 0.9 * (1 + math.cos(math.pi * (k - warm) / (total - warm))) / 2)


def loss_sum_count(logits: torch.Tensor, ids: torch.Tensor, lengths: torch.Tensor):
    """One label shift; position mask uses lengths, never token values."""
    seq = ids.shape[1]
    target = ids[:, 1:]
    if logits.shape[1] not in (seq, seq - 1):
        raise ValueError(f"logits length={logits.shape[1]} incompatible with token window={seq}")
    mask = torch.arange(seq - 1, device=ids.device)[None, :] < (lengths[:, None] - 1)
    per = F.cross_entropy(logits[:, :target.shape[1]].float().reshape(-1, logits.shape[-1]),
                          target.reshape(-1), reduction="none").view_as(target)
    return (per * mask).sum(), int(mask.sum().item())


def save_checkpoint(path: Path, model, optimizer, *, step: int, cursor: int,
                    total_tokens: int, config_hash: str, data_hash: str, aliases: dict):
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("checkpoint only at accumulation boundary after zero_grad")
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "all_buffers": {name: value.detach().clone() for name, value in model.named_buffers()},
               "parameter_aliases": aliases, "successful_optimizer_step": step,
               "scheduler_position": step, "data_cursor": cursor, "total_effective_tokens": total_tokens,
               "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
               "grad_scaler": None, "config_hash": config_hash, "data_hash": data_hash,
               "source_hash": source_hash()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer, *, config_hash: str, data_hash: str, aliases: dict):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (("config_hash", config_hash), ("data_hash", data_hash),
                          ("source_hash", source_hash()), ("parameter_aliases", aliases)):
        if payload[key] != expected:
            raise RuntimeError(f"checkpoint {key} mismatch")
    model.load_state_dict(payload["model"])
    if set(payload["all_buffers"]) != set(dict(model.named_buffers())):
        raise RuntimeError("checkpoint buffer map mismatch")
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(payload["all_buffers"][name].to(value.device))
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["cpu_rng"])
    if payload["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    return payload


@contextlib.contextmanager
def isolated_probe(model, optimizer, state: dict):
    """Restore the full mutable training state even when a probe raises."""
    snapshot = {"model": copy.deepcopy(model.state_dict()),
                "all_buffers": {name: value.detach().clone() for name, value in model.named_buffers()},
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "grads": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                "mode": model.training, "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "state": copy.deepcopy(state)}
    try:
        yield
    finally:
        model.load_state_dict(snapshot["model"])
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(snapshot["all_buffers"][name].to(value.device))
        optimizer.load_state_dict(snapshot["optimizer"])
        for p, g in zip(model.parameters(), snapshot["grads"]):
            p.grad = g
        model.train(snapshot["mode"])
        torch.set_rng_state(snapshot["cpu_rng"])
        if snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        state.clear()
        state.update(snapshot["state"])


def model_and_optimizer(cfg: dict, binding: dict, peak_lr: float, device: str):
    from transformers import AutoTokenizer
    repo = Path(binding["repo_root"])
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    tok = AutoTokenizer.from_pretrained(binding["tokenizer_path"], local_files_only=True)
    if max(tok.get_vocab().values()) >= len(tok):
        raise RuntimeError("tokenizer ID/vocab mismatch")
    c = cfg["model"]
    mc = MiniMindConfig(hidden_size=c["hidden_size"], num_hidden_layers=c["num_hidden_layers"],
                        intermediate_size=c["intermediate_size"], num_attention_heads=c["num_attention_heads"],
                        num_key_value_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
                        dropout=c["dropout"], use_moe=False, use_cache=False, vocab_size=len(tok),
                        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                        max_position_embeddings=cfg["runtime"]["sequence_length"], flash_attn=True,
                        tie_word_embeddings=True)
    model = MiniMindForCausalLM(mc).float()
    if model.model.embed_tokens.weight is not model.lm_head.weight:
        raise RuntimeError("MiniMind embedding tying unresolved")
    count = sum(p.numel() for p in model.parameters())
    if count > c["max_parameters"]:
        raise RuntimeError(f"MODEL_SIZE_UNRESOLVED: measured parameters={count}")
    target = model.model.layers[1].mlp.down_proj.weight
    attn = model.model.layers[1].self_attn.o_proj.weight
    if tuple(target.shape) != (c["hidden_size"], c["intermediate_size"]) or tuple(attn.shape) != (c["hidden_size"], c["hidden_size"]):
        raise RuntimeError("target matrix shape unresolved")
    aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(p), []).append(name)
    alias_map = {names[0]: names for names in aliases.values()}
    groups = [[], []]
    for name, p in model.named_parameters():
        groups[0 if p.ndim >= 2 else 1].append(p)
    bc = cfg["baseline"]
    opt = torch.optim.AdamW([{"params": groups[0], "weight_decay": bc["matrix_weight_decay"]},
                             {"params": groups[1], "weight_decay": 0.0}],
                            lr=peak_lr, betas=tuple(bc["betas"]), eps=bc["eps"],
                            amsgrad=False, maximize=False, foreach=False, fused=False)
    return model.to(device), opt, alias_map, count


def training_batch(mem, cursor: int, batch: int, seq: int, device: str):
    needed = batch * seq + 1
    if cursor + needed > len(mem):
        raise RuntimeError("DATA_INSUFFICIENT: train token cache exhausted, no wrapping")
    array = np.stack([np.asarray(mem[cursor + i * seq:cursor + i * seq + seq + 1], dtype=np.int64)
                      for i in range(batch)])
    return torch.from_numpy(array).to(device), cursor + batch * seq


def train_steps(model, opt, cfg: dict, mem, *, peak_lr: float, step: int, cursor: int,
                total_tokens: int, stop: int, device: str, log_prefix: str,
                vram_limit: int | None = None, update_counter: list[int] | None = None):
    import psutil
    seq = cfg["runtime"]["sequence_length"]
    mb = cfg["runtime"]["micro_batch_size"]
    accum = cfg["runtime"]["accumulation_steps"]
    bc = cfg["baseline"]
    precision = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model.train()
    for k in range(step + 1, stop + 1):
        completed = update_counter[0] if update_counter is not None else k - 1
        require_resource(cfg, "before_step", stage=log_prefix, gpu_updates=completed,
                         vram_limit=vram_limit)
        eta = schedule_lr(peak_lr, k, cfg)
        for group in opt.param_groups:
            group["lr"] = eta
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        valid_total = 0
        loss_total = 0.0
        for _ in range(accum):
            ids, cursor = training_batch(mem, cursor, mb, seq, device)
            lengths = torch.full((mb,), seq + 1, device=device, dtype=torch.long)
            with torch.autocast(device_type="cuda", dtype=precision, enabled=precision == torch.bfloat16):
                logits = model(ids[:, :-1], use_cache=False).logits
                loss_sum, count = loss_sum_count(logits, ids, lengths)
            loss_sum.backward()
            valid_total += count
            loss_total += float(loss_sum.detach())
        if valid_total <= 0:
            raise RuntimeError("zero valid targets")
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(valid_total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), bc["global_grad_clip_norm"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_backward = time.perf_counter()
        require_resource(cfg, "before_optimizer", stage=log_prefix, gpu_updates=completed,
                         vram_limit=vram_limit)
        opt.step()
        if update_counter is not None:
            update_counter[0] += 1
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t_step = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        total_tokens += valid_total
        rss = psutil.Process().memory_info().rss
        row = {"stage": log_prefix, "step": k, "lr": eta, "eta_wd": eta * bc["matrix_weight_decay"],
               "loss_sum": loss_total, "token_count": valid_total, "nll": loss_total / valid_total,
               "cursor": cursor, "total_effective_tokens": total_tokens,
               "forward_backward_seconds": t_backward - t0, "optimizer_seconds": t_step - t_backward,
               "rss_bytes": rss,
               "allocated_vram_bytes": torch.cuda.memory_allocated() if device.startswith("cuda") else 0,
               "reserved_vram_bytes": torch.cuda.memory_reserved() if device.startswith("cuda") else 0}
        append_csv(ROOT / "logs" / "runtime.csv", row)
        event("train_step", **row)
        require_resource(cfg, "after_optimizer", stage=log_prefix,
                         gpu_updates=update_counter[0] if update_counter is not None else k,
                         vram_limit=vram_limit)
    return stop, cursor, total_tokens


def require_data(cfg: dict, config_hash: str | None = None, *, formal: bool = False):
    b = read_binding()
    manifest = read_json(ROOT / "artifacts" / "data_manifest.json")
    if not manifest.get("formal_ready" if formal else "smoke_ready", False):
        label = "RESOURCE_BLOCKED" if manifest.get("status") == "RESOURCE_BLOCKED" else "DATA_INSUFFICIENT"
        raise RuntimeError(f"{label}: {manifest}")
    if manifest["source_hash"] != source_hash():
        raise RuntimeError("data manifest source hash mismatch; rerun prepare")
    expected_data_hash = sha_object({k: v for k, v in manifest.items() if k not in
                                    ("status", "data_hash", "prepare_seconds", "ram_available_at_end_bytes")})
    if manifest.get("data_hash") != expected_data_hash:
        raise RuntimeError("data manifest self-hash mismatch")
    if config_hash is not None and manifest["config_hash"] != config_hash:
        raise RuntimeError("data manifest config hash mismatch; rerun prepare")
    if manifest["tokenizer_hash"] != b["tokenizer_fingerprint"]:
        raise RuntimeError("data manifest tokenizer fingerprint mismatch")
    source = Path(manifest["source_path"])
    st = source.stat()
    if st.st_size != manifest["source_size"] or st.st_mtime_ns != manifest["source_mtime_ns"]:
        raise RuntimeError("bound source file metadata changed")
    if sha_file(source, manifest["scanned_bytes"]) != manifest["scanned_sha256"]:
        raise RuntimeError("consumed source scan window hash mismatch")
    for name, hash_key in (("eval_blocks.json", "eval_cache_sha256"),
                           ("doc_roles.jsonl", "doc_assignments_sha256")):
        if sha_file(ROOT / "artifacts" / name) != manifest[hash_key]:
            raise RuntimeError(f"{name} hash mismatch")
    path = ROOT / "artifacts" / "train_tokens.u32"
    if sha_file(path) != manifest["train_cache_sha256"]:
        raise RuntimeError("train cache hash mismatch")
    return b, manifest, np.memmap(path, dtype=np.uint32, mode="r")


def resource_check(cfg: dict) -> dict:
    require_resource(cfg, "before_cuda_model_init", stage="admission", admission=True)
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_BLOCKED: bound Python has no CUDA")
    free, total = torch.cuda.mem_get_info(0)
    available_ram = require_resource(cfg, "after_cuda_query", stage="admission")["ram_available_bytes"]
    limit = min(cfg["runtime"]["vram_soft_limit_gib"] * 2 ** 30,
                free - cfg["runtime"]["vram_free_reserve_gib"] * 2 ** 30)
    if limit <= 0:
        raise RuntimeError("RESOURCE_BLOCKED: VRAM reserve unavailable")
    return {"free_vram": free, "total_vram": total, "process_vram_limit": limit,
            "available_ram": available_ram}


def eval_blocks(model, rows, device: str):
    model.eval()
    out = []
    with torch.no_grad():
        for row in rows:
            ids = torch.tensor(row["ids"], dtype=torch.long, device=device)[None, :]
            lengths = torch.tensor([ids.shape[1]], device=device)
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, lengths)
            out.append({"doc_id": row["doc_id"], "loss_sum": float(loss), "token_count": count})
    return out


def smoke(cfg: dict, cfg_hash: str):
    b, data, mem = require_data(cfg, cfg_hash)
    fp = fingerprint(cfg_hash, data["data_hash"])
    atomic_json(ROOT / "artifacts" / "smoke_run_manifest.json",
                {"status": "ADMISSION_PENDING", **fp, "gpu_optimizer_updates": 0})
    res = resource_check(cfg)
    with run_lock("smoke"):
        updates = [0]
        seed = cfg["baseline"]["discovery_seed"]
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(cfg["runtime"]["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        peak = cfg["baseline"]["peak_lr_grid"][1]
        model, opt, aliases, count = model_and_optimizer(cfg, b, peak, "cuda:0")
        require_resource(cfg, "after_model_init", gpu_updates=updates[0],
                         vram_limit=res["process_vram_limit"])
        atomic_json(ROOT / "artifacts" / "smoke_run_manifest.json", {"status": "RUNNING", **fp,
                    "parameter_count": count, "resource": res, "gpu_optimizer_updates": 0})
        result = {"status": "FAILED", **fp, "parameter_count": count, "resource": res,
                  "gpu_optimizer_updates": 0}
        try:
            # The replay uses 8 + 4 steps, within the 20-step smoke/replay cap.
            step, cursor, tokens = train_steps(model, opt, cfg, mem, peak_lr=peak, step=0, cursor=0,
                                               total_tokens=0, stop=4, device="cuda:0", log_prefix="smoke_a",
                                               vram_limit=res["process_vram_limit"], update_counter=updates)
            ckpt = ROOT / "artifacts" / "smoke_step4.pt"
            require_resource(cfg, "before_checkpoint_save", gpu_updates=updates[0],
                             vram_limit=res["process_vram_limit"])
            save_checkpoint(ckpt, model, opt, step=step, cursor=cursor, total_tokens=tokens,
                            config_hash=cfg_hash, data_hash=data["data_hash"], aliases=aliases)
            require_resource(cfg, "after_checkpoint_save", gpu_updates=updates[0],
                             vram_limit=res["process_vram_limit"])
            ckpt_sha = sha_file(ckpt)
            step, cursor, tokens = train_steps(model, opt, cfg, mem, peak_lr=peak, step=step, cursor=cursor,
                                               total_tokens=tokens, stop=8, device="cuda:0", log_prefix="smoke_continuous",
                                               vram_limit=res["process_vram_limit"], update_counter=updates)
            continuous = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            end_opt = copy.deepcopy(opt.state_dict())
            require_resource(cfg, "before_checkpoint_restore", gpu_updates=updates[0],
                             vram_limit=res["process_vram_limit"])
            ck = load_checkpoint(ckpt, model, opt, config_hash=cfg_hash, data_hash=data["data_hash"], aliases=aliases)
            require_resource(cfg, "after_checkpoint_restore", gpu_updates=updates[0],
                             vram_limit=res["process_vram_limit"])
            step, cursor, tokens = train_steps(model, opt, cfg, mem, peak_lr=peak,
                                               step=ck["successful_optimizer_step"], cursor=ck["data_cursor"],
                                               total_tokens=ck["total_effective_tokens"], stop=8,
                                               device="cuda:0", log_prefix="smoke_replay",
                                               vram_limit=res["process_vram_limit"], update_counter=updates)
            max_error = max(float((v.cpu() - continuous[k]).abs().max()) for k, v in model.state_dict().items())
            max_state_error = 0.0
            replay_opt = opt.state_dict()
            if replay_opt["param_groups"] != end_opt["param_groups"]:
                raise RuntimeError("P0_FAIL: replay optimizer parameter groups mismatch")
            for key, state_row in end_opt["state"].items():
                for field, value in state_row.items():
                    other = replay_opt["state"][key][field]
                    if torch.is_tensor(value):
                        max_state_error = max(max_state_error, float((value.cpu() - other.cpu()).abs().max()))
                    elif value != other:
                        raise RuntimeError(f"P0_FAIL: replay optimizer {field} mismatch")
            if max_error > cfg["numerics"]["fp32_atol"] or max_state_error > cfg["numerics"]["fp32_atol"] or sha_file(ckpt) != ckpt_sha:
                raise RuntimeError(f"P0_FAIL: replay parameter/state max errors={max_error}/{max_state_error}")
            result.update(status="PASS", replay_max_abs_error=max_error,
                          replay_optimizer_state_max_abs_error=max_state_error, checkpoint_sha256=ckpt_sha,
                          optimizer_step=step, effective_tokens=tokens, parameter_count=count,
                          gpu_optimizer_updates=updates[0])
            atomic_json(ROOT / "artifacts" / "numerics" / "resume_replay.json", result)
            atomic_json(ROOT / "artifacts" / "numerics" / "memory_smoke.json", res)
            event("smoke", status="PASS", replay_max_abs_error=max_error)
        except Exception as e:
            result["status"] = "BLOCKED" if str(e).startswith("RESOURCE_BLOCKED") else "FAILED"
            result["reason"] = str(e)
            result["gpu_optimizer_updates"] = updates[0]
            result["last_successful_checkpoint"] = str(ckpt) if 'ckpt_sha' in locals() else None
            result["uncheckpointed_gpu_updates"] = max(0, updates[0] - 4) if result["last_successful_checkpoint"] else updates[0]
            event("smoke", status="FAILED", reason=str(e))
            raise
        finally:
            atomic_json(ROOT / "artifacts" / "smoke_run_manifest.json", result)
        return result
