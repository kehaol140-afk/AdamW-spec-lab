"""Identity, state, accounting and resource primitives for the V1.3 child."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import struct
import time
from pathlib import Path

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V12 = ROOT / "replay_audit_v12"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1 * (attempt + 1))


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha_file(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        left = limit
        while left is None or left > 0:
            block = stream.read(min(left, 1024 * 1024) if left is not None else 1024 * 1024)
            if not block:
                break
            digest.update(block)
            if left is not None:
                left -= len(block)
    return digest.hexdigest()


def sha_object(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_hash():
    names = ("__init__.py", "common.py", "audit.py", "engine.py", "cli.py",
             "tests_v13.py", "run_parent_tests_isolated.py", "config.json", "RUN_VERIFIED.ps1")
    return sha_object({name: sha_file(HERE / name) for name in names})


def _field(digest, label, payload):
    label = label.encode("utf-8")
    digest.update(struct.pack("<Q", len(label)))
    digest.update(label)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def tensor_digest(values):
    """Stable logical-name digest; distinguish absent, None, dtype, shape and bytes."""
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if value is None:
            _field(digest, name, b"NONE")
            continue
        if not torch.is_tensor(value):
            _field(digest, name, b"SCALAR:" + repr(value).encode("utf-8"))
            continue
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps({"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                              sort_keys=True, separators=(",", ":")).encode("utf-8")
        # view(uint8) handles bfloat16 without changing its numerical representation.
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        _field(digest, name + "/metadata", metadata)
        _field(digest, name + "/bytes", raw)
    return digest.hexdigest()


def optimizer_logical_snapshot(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    missing = [id(p) for group in optimizer.param_groups for p in group["params"] if id(p) not in names]
    if missing:
        raise RuntimeError("OPTIMIZER_PARAMETER_MISSING_LOGICAL_NAME")
    groups = []
    seen = set()
    for group in optimizer.param_groups:
        members = [names[id(p)] for p in group["params"]]
        if any(name in seen for name in members):
            raise RuntimeError("OPTIMIZER_DUPLICATE_UNIQUE_PARAMETER")
        seen.update(members)
        groups.append({"members": members,
                       "options": {k: v for k, v in group.items() if k != "params"}})
    state = {}
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            name = names[id(parameter)]
            for field, value in optimizer.state.get(parameter, {}).items():
                state[name + "/" + str(field)] = value.detach().cpu().clone() if torch.is_tensor(value) else value
    aliases = {}
    for name, p in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(names[id(p)], []).append(name)
    return {"state": state, "groups": groups, "aliases": aliases,
            "state_digest": tensor_digest(state), "group_digest": sha_object(groups),
            "alias_digest": sha_object(aliases)}


def compare_tensors(left, right):
    if left is None or right is None:
        return {"same_presence": left is right, "left_none": left is None, "right_none": right is None}
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return {"same_scalar": left == right, "left": repr(left), "right": repr(right),
                "different_elements": int(left != right)}
    if left.shape != right.shape or left.dtype != right.dtype:
        return {"same_shape_dtype": False, "left_shape": list(left.shape),
                "right_shape": list(right.shape), "left_dtype": str(left.dtype),
                "right_dtype": str(right.dtype)}
    a, b = left.detach().cpu().double(), right.detach().cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        return {"finite": False, "left_nonfinite": int((~torch.isfinite(a)).sum()),
                "right_nonfinite": int((~torch.isfinite(b)).sum())}
    diff = (a - b).abs()
    flat = diff.reshape(-1)
    index = int(flat.argmax()) if flat.numel() else None
    na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    denominator = max(na, nb)
    cos = float((a * b).sum() / (na * nb)) if na and nb else (1.0 if na == nb == 0 else None)
    return {"same_shape_dtype": True, "finite": True,
            "max_abs": float(flat.max()) if index is not None else 0.0,
            "relative_l2": float(torch.linalg.vector_norm(a - b)) / denominator if denominator else 0.0,
            "relative_l2_denominator": denominator,
            "cosine": cos, "different_elements": int((diff != 0).sum()),
            "max_difference_flat_index": index,
            "left_at_max": float(a.reshape(-1)[index]) if index is not None else None,
            "right_at_max": float(b.reshape(-1)[index]) if index is not None else None}


@contextlib.contextmanager
def isolated_state(model, optimizer, cursor):
    snapshot = {"model": copy.deepcopy(model.state_dict()),
                "buffers": {name: b.detach().clone() for name, b in model.named_buffers()},
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "grads": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                "modes": {name: module.training for name, module in model.named_modules()},
                "cursor": copy.deepcopy(cursor), "python": random.getstate(),
                "numpy": np.random.get_state(), "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    try:
        yield
    finally:
        model.load_state_dict(snapshot["model"])
        with torch.no_grad():
            for name, buffer in model.named_buffers():
                buffer.copy_(snapshot["buffers"][name].to(buffer.device))
        optimizer.load_state_dict(snapshot["optimizer"])
        for parameter, grad in zip(model.parameters(), snapshot["grads"]):
            parameter.grad = grad
        for name, module in model.named_modules():
            module.training = snapshot["modes"][name]
        cursor.clear(); cursor.update(snapshot["cursor"])
        random.setstate(snapshot["python"])
        np.random.set_state(snapshot["numpy"])
        torch.set_rng_state(snapshot["cpu"])
        if snapshot["cuda"] is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda"])


class BudgetStop(RuntimeError):
    pass


class Ledger:
    def __init__(self, run_hash):
        self.path = HERE / "budget_ledger.json"
        self.config = read_json(HERE / "config.json")
        if self.path.exists():
            self.data = read_json(self.path)
            if self.data["run_manifest_sha256"] != run_hash:
                raise RuntimeError("BUDGET_RUN_MANIFEST_CHANGED")
        else:
            self.data = {"run_manifest_sha256": run_hash, "token_forward_equivalents": 0,
                         "forward_attempts": 0, "backward_attempts": 0,
                         "gpu_optimizer_step_attempts": 0, "gpu_optimizer_steps_completed": 0,
                         "elapsed_seconds": 0.0, "attempts": []}
            self.save()

    def save(self):
        write_json(self.path, self.data)

    def forward(self, token_count):
        if self.data["token_forward_equivalents"] + token_count > self.config["max_token_forward_equivalents"]:
            raise BudgetStop("TOKEN_FORWARD_BUDGET_EXHAUSTED")
        self.data["token_forward_equivalents"] += int(token_count)
        self.data["forward_attempts"] += 1
        self.save()

    def backward(self):
        self.data["backward_attempts"] += 1
        self.save()

    def optimizer_attempt(self):
        if self.data["gpu_optimizer_step_attempts"] >= self.config["max_gpu_optimizer_step_attempts"]:
            raise BudgetStop("GPU_OPTIMIZER_ATTEMPT_BUDGET_EXHAUSTED")
        self.data["gpu_optimizer_step_attempts"] += 1
        self.save()

    def optimizer_complete(self):
        self.data["gpu_optimizer_steps_completed"] += 1
        self.save()


def resource_gate(ledger, phase, *, admission=False, vram_limit=None):
    cfg = read_json(ROOT / "config" / "local_v1_lowmem.json")["runtime"]
    available = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    initialized = torch.cuda.is_initialized()
    reserved = torch.cuda.memory_reserved(0) if initialized else None
    minimum = cfg["system_ram_start_min_gib"] if admission else cfg["system_ram_free_reserve_gib"]
    reason = None
    if available < minimum * 2 ** 30:
        reason = "RAM_BELOW_REQUIRED"
    elif rss > cfg["process_rss_soft_limit_gib"] * 2 ** 30:
        reason = "RSS_ABOVE_LIMIT"
    elif vram_limit is not None and reserved is not None and reserved > vram_limit:
        reason = "VRAM_ABOVE_LIMIT"
    status = "BLOCKED" if reason else ("WARNING" if not admission and
             available < cfg["system_ram_warning_gib"] * 2 ** 30 else "READY")
    row = {"time": time.time(), "phase": phase, "pid": os.getpid(), "status": status,
           "reason": reason, "available_ram_bytes": available, "rss_bytes": rss,
           "cuda_reserved_bytes": reserved, "cuda_allocated_bytes": torch.cuda.memory_allocated(0) if initialized else None,
           "budget": {k: ledger.data[k] for k in ("token_forward_equivalents", "forward_attempts",
                       "backward_attempts", "gpu_optimizer_step_attempts", "gpu_optimizer_steps_completed")}}
    append_jsonl(HERE / "logs" / "resources.jsonl", row)
    if reason:
        raise RuntimeError("RESOURCE_BLOCKED:" + reason)
    return row


@contextlib.contextmanager
def run_lock():
    if (ROOT / "artifacts" / "run.lock").exists():
        raise RuntimeError("PARENT_RUN_LOCK_PRESENT")
    path = HERE / "run.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        if path.exists() and path.read_text().strip() == str(os.getpid()):
            path.unlink()
