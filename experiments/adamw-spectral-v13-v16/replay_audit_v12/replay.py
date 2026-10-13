"""Opt-in GPU counterfactual replay. Never changes the parent checkout."""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch

from .core import HERE, ROOT, read_json, sha_file, sha_object, verify_parent, write_json


class BudgetStop(RuntimeError):
    pass


def four_states(base: dict, data_delta: dict, target: str) -> dict:
    """Pure four-state construction; None-gradient parameters have zero data_delta."""
    if set(base) != set(data_delta) or target not in base:
        raise ValueError("four-state parameter map mismatch")
    return {
        "F00": {name: value.clone() for name, value in base.items()},
        "F10": {name: value + (data_delta[name] if name == target else 0) for name, value in base.items()},
        "F01": {name: value + (data_delta[name] if name != target else 0) for name, value in base.items()},
        "F11": {name: value + data_delta[name] for name, value in base.items()},
    }


def interactions(nll: dict) -> dict:
    if set(nll) != {"F00", "F10", "F01", "F11"}:
        raise ValueError("four NLL states required")
    return {"I_before": nll["F00"] - nll["F10"],
            "I_after": nll["F01"] - nll["F11"],
            "interaction": nll["F11"] - nll["F10"] - nll["F01"] + nll["F00"]}


@contextlib.contextmanager
def isolated_state(model, optimizer, cursor: dict):
    """Restore all mutable state even if a measurement raises."""
    snapshot = {"model": copy.deepcopy(model.state_dict()),
                "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "grads": [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
                "mode": model.training, "cursor": copy.deepcopy(cursor),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
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
        model.train(snapshot["mode"])
        cursor.clear(); cursor.update(snapshot["cursor"])
        random.setstate(snapshot["python_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        torch.set_rng_state(snapshot["cpu_rng"])
        if snapshot["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])


class Ledger:
    def __init__(self, eval_manifest_hash: str, child_source_hash: str):
        self.path = HERE / "budget_ledger.json"
        self.maximum = read_json(HERE / "config.json")["max_eval_token_forward_equivalents"]
        self.max_steps = read_json(HERE / "config.json")["max_gpu_optimizer_calls"]
        if self.path.exists():
            self.value = read_json(self.path)
            if (self.value["eval_manifest_hash"] != eval_manifest_hash or
                    self.value["child_source_hash"] != child_source_hash):
                raise RuntimeError("CHILD_BUDGET_FINGERPRINT_MISMATCH")
        else:
            self.value = {"eval_manifest_hash": eval_manifest_hash,
                          "child_source_hash": child_source_hash,
                          "token_forward_equivalents": 0, "forward_calls": 0,
                          "backward_calls": 0, "gpu_optimizer_calls": 0,
                          "gpu_optimizer_calls_attempted": 0,
                          "replay_started": False}
            self.save()

    def save(self):
        write_json(self.path, self.value)

    def forward(self, count: int):
        if self.value["token_forward_equivalents"] + count > self.maximum:
            raise BudgetStop(f"EVAL_BUDGET_STOP: {self.value['token_forward_equivalents']}+{count}>{self.maximum}")
        self.value["token_forward_equivalents"] += count
        self.value["forward_calls"] += 1
        self.save()

    def backward(self):
        self.value["backward_calls"] += 1
        self.save()

    def begin_optimizer(self):
        if self.value["gpu_optimizer_calls_attempted"] >= self.max_steps:
            raise BudgetStop("GPU_OPTIMIZER_BUDGET_STOP")
        self.value["gpu_optimizer_calls_attempted"] += 1
        self.save()

    def complete_optimizer(self):
        self.value["gpu_optimizer_calls"] += 1
        self.save()


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()


def resource_gate(cfg: dict, phase: str, ledger: Ledger, *, admission: bool = False,
                  vram_limit: int | None = None):
    runtime = cfg["runtime"]
    available = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    ram_limit = int(runtime["system_ram_start_min_gib" if admission else "system_ram_free_reserve_gib"] * 2 ** 30)
    warning = int(runtime["system_ram_warning_gib"] * 2 ** 30)
    initialized = torch.cuda.is_initialized()
    reserved = torch.cuda.memory_reserved(0) if initialized else None
    reason = None
    if available < ram_limit:
        reason = f"RAM {phase} available={available}, required={ram_limit}"
    elif rss > runtime["process_rss_soft_limit_gib"] * 2 ** 30:
        reason = f"RSS {phase}={rss} exceeds 4GiB limit"
    elif vram_limit is not None and reserved is not None and reserved > vram_limit:
        reason = f"VRAM {phase} reserved={reserved}, limit={vram_limit}"
    status = "RESOURCE_BLOCKED" if reason else "WARNING" if not admission and available < warning else "READY"
    row = {"time": time.time(), "phase": phase, "pid": os.getpid(), "status": status,
           "child_source_hash": ledger.value["child_source_hash"],
           "evaluation_manifest_sha256": ledger.value["eval_manifest_hash"],
           "reason": reason, "available_ram_bytes": available, "rss_bytes": rss,
           "gpu_optimizer_calls": ledger.value["gpu_optimizer_calls"],
           "gpu_optimizer_calls_attempted": ledger.value["gpu_optimizer_calls_attempted"],
           "token_forward_equivalents": ledger.value["token_forward_equivalents"],
           "cuda_allocated_bytes": torch.cuda.memory_allocated(0) if initialized else None,
           "cuda_reserved_bytes": reserved,
           "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if initialized else None,
           "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0) if initialized else None}
    append_jsonl(HERE / "logs" / "resources.jsonl", row)
    if reason:
        raise RuntimeError("RESOURCE_BLOCKED: " + reason)
    return row


@contextlib.contextmanager
def child_lock():
    if (ROOT / "artifacts" / "run.lock").exists():
        raise RuntimeError("PARENT_RUN_LOCK_PRESENT")
    path = HERE / "run.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump({"pid": os.getpid(), "time": time.time()}, stream)
        yield
    finally:
        if path.exists() and read_json(path).get("pid") == os.getpid():
            path.unlink()


def load_child_views() -> dict:
    groups = defaultdict(list)
    with (HERE / "eval_views.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not 2 <= len(row["ids"]) <= 257 or row["token_count"] != len(row["ids"]) - 1:
                raise RuntimeError("CHILD_EVAL_VIEW_INVALID")
            groups[row["view"]].append(row)
    return groups


def apply_state(model, state: dict):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))


def build_update(model, optimizer, checkpoint: Path, cfg: dict, binding: dict,
                 data_hash: str, aliases: dict, mem, window: dict, ledger: Ledger,
                 vram_limit: int) -> dict:
    from pilot_core import adamw_direction, load_checkpoint, loss_sum_count, schedule_lr, training_batch
    child_cfg = read_json(HERE / "config.json")
    ck = load_checkpoint(checkpoint, model, optimizer,
                         config_hash=sha_file(ROOT / "config" / "local_v1_lowmem.json"),
                         data_hash=data_hash, aliases=aliases)
    if ck["successful_optimizer_step"] != child_cfg["checkpoint_step"]:
        raise RuntimeError("PARENT_CHECKPOINT_STEP_MISMATCH")
    eta = schedule_lr(child_cfg["peak_lr"], child_cfg["checkpoint_step"] + 1, cfg)
    for group in optimizer.param_groups:
        group["lr"] = eta
    cursor = window["start_token"]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    valid_total = 0
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    for _ in range(cfg["runtime"]["accumulation_steps"]):
        ids, cursor = training_batch(mem, cursor, cfg["runtime"]["micro_batch_size"],
                                     cfg["runtime"]["sequence_length"], "cuda:0")
        lengths = torch.full((ids.shape[0],), ids.shape[1], device="cuda:0", dtype=torch.long)
        ledger.forward(ids.shape[0] * (ids.shape[1] - 1))
        with torch.autocast(device_type="cuda", dtype=precision, enabled=precision == torch.bfloat16):
            logits = model(ids[:, :-1], use_cache=False).logits
            loss_sum, count = loss_sum_count(logits, ids, lengths)
        loss_sum.backward(); ledger.backward()
        valid_total += count
    if cursor != window["end_token_exclusive"] - 1 or valid_total != window["target_tokens"]:
        raise RuntimeError("TRAIN_WINDOW_CURSOR_OR_TOKEN_MISMATCH")
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(valid_total)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["baseline"]["global_grad_clip_norm"])
    named = dict(model.named_parameters())
    decay = {id(parameter): float(group["weight_decay"])
             for group in optimizer.param_groups for parameter in group["params"]}
    old, base, directions, moments_plus, grads, none_names = {}, {}, {}, {}, {}, []
    gradient_digest = hashlib.sha256()
    for name, parameter in named.items():
        p = parameter.detach()
        old[name] = p.cpu().clone()
        if parameter.grad is None:
            none_names.append(name)
            base[name] = old[name].clone()
            directions[name] = torch.zeros_like(old[name])
            continue
        gradient_digest.update(name.encode("utf-8"))
        gradient_digest.update(parameter.grad.detach().cpu().contiguous().numpy().tobytes())
        wd = decay[id(parameter)]
        base[name] = (p * (1 - eta * wd)).cpu().clone()
        a, m_plus = adamw_direction(p, parameter.grad, optimizer.state[parameter],
                                    beta1=cfg["baseline"]["betas"][0],
                                    beta2=cfg["baseline"]["betas"][1], eps=cfg["baseline"]["eps"])
        directions[name] = a.cpu().clone()
        if name in child_cfg["targets"]:
            moments_plus[name] = m_plus.cpu().clone()
            grads[name] = parameter.grad.detach().cpu().clone()
    resource_gate(cfg, "before_isolated_optimizer", ledger, vram_limit=vram_limit)
    ledger.begin_optimizer()
    optimizer.step(); ledger.complete_optimizer()
    torch.cuda.synchronize()
    resource_gate(cfg, "after_isolated_optimizer", ledger, vram_limit=vram_limit)
    updated = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
    optimizer_digest = hashlib.sha256()
    state = optimizer.state_dict()
    optimizer_digest.update(json.dumps(state["param_groups"], sort_keys=True).encode("utf-8"))
    for key in sorted(state["state"]):
        for field, value in sorted(state["state"][key].items()):
            optimizer_digest.update(f"{key}/{field}".encode("utf-8"))
            optimizer_digest.update(value.detach().cpu().contiguous().numpy().tobytes()
                                    if torch.is_tensor(value) else str(value).encode("utf-8"))
    delta = {name: updated[name] - base[name] for name in named}
    max_error = max(float((updated[name] - (base[name] - eta * directions[name])).abs().max())
                    for name in named)
    limit = cfg["numerics"]["fp32_atol"] + cfg["numerics"]["fp32_rtol"] * max(
        float(value.abs().max()) for value in updated.values())
    if max_error > limit:
        raise RuntimeError(f"P0_FAIL: full real AdamW step parity={max_error} > {limit}")
    optimizer.zero_grad(set_to_none=True)
    return {"base": base, "delta": delta, "updated": updated,
            "directions": {name: directions[name] for name in child_cfg["targets"]},
            "moments_plus": moments_plus, "grads": grads, "none_gradient_names": none_names,
            "eta": eta, "valid_tokens": valid_total, "parity_max_abs": max_error,
            "gradient_sha256": gradient_digest.hexdigest(),
            "optimizer_state_sha256": optimizer_digest.hexdigest(),
            "generation_precision": "bf16_autocast" if precision == torch.bfloat16 else "fp32"}


def evaluate(model, rows: list[dict], ledger: Ledger, *, build_id: int, target: str,
             state_name: str, view: str, cfg: dict, vram_limit: int) -> tuple[list[dict], float]:
    from pilot_core import loss_sum_count
    model.eval()
    outputs = []
    with torch.no_grad():
        for row in rows:
            resource_gate(cfg, "before_eval_document", ledger, vram_limit=vram_limit)
            ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
            lengths = torch.tensor([ids.shape[1]], device="cuda:0")
            ledger.forward(row["token_count"])
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, lengths)
            if count != row["token_count"]:
                raise RuntimeError("EVAL_MASK_COUNT_MISMATCH")
            item = {"build_window_id": build_id, "target": target, "state": state_name,
                    "view": view, "doc_id": row["doc_id"], "source_doc_ids": row["source_doc_ids"],
                    "cluster_id": row["cluster_id"], "loss_sum": float(loss), "token_count": count}
            append_jsonl(HERE / "results" / "replay_losses.jsonl", item)
            outputs.append(item)
    return outputs, sum(r["loss_sum"] for r in outputs) / sum(r["token_count"] for r in outputs)


def gradient_signal(model, rows: list[dict], parameter, directions: dict,
                    ledger: Ledger, *, build_id: int, target: str, baseline: str,
                    view: str, cfg: dict, vram_limit: int) -> dict:
    from pilot_core import loss_sum_count
    model.eval()
    total = torch.zeros_like(parameter, dtype=torch.float64, device="cpu")
    tokens = 0
    for row in rows:
        resource_gate(cfg, "before_probe_gradient", ledger, vram_limit=vram_limit)
        ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
        lengths = torch.tensor([ids.shape[1]], device="cuda:0")
        ledger.forward(row["token_count"])
        logits = model(ids[:, :-1], use_cache=False).logits
        loss, count = loss_sum_count(logits, ids, lengths)
        grad = torch.autograd.grad(loss, parameter, retain_graph=False)[0]
        ledger.backward()
        total += grad.detach().cpu().double()
        tokens += count
    gradient = total / tokens
    result = {"build_window_id": build_id, "target": target, "baseline": baseline,
              "view": view, "token_count": tokens, "evaluation_precision": "fp32_no_tf32"}
    gnorm = float(torch.linalg.vector_norm(gradient))
    anorm = float(torch.linalg.vector_norm(directions["A"].double()))
    for label, value in directions.items():
        vec = value.double()
        norm = float(torch.linalg.vector_norm(vec))
        dot = float((gradient * vec).sum())
        result[label] = {"dot": dot, "cosine": dot / (gnorm * norm) if gnorm and norm else None,
                         "direction_norm": norm,
                         "A_norm_matched_dot": dot * anorm / norm if norm else None}
    append_jsonl(HERE / "results" / "gradient_signals.jsonl", result)
    return result


def finite_difference(model, rows: list[dict], parameter, direction: torch.Tensor,
                      ledger: Ledger, *, cfg: dict, build_id: int, target: str,
                      baseline: str, vram_limit: int) -> dict:
    """Small central difference on an independent four-document evaluation subset."""
    from pilot_core import loss_sum_count
    subset = rows[:4]
    unit = direction.double() / torch.linalg.vector_norm(direction.double())
    h = read_json(HERE / "config.json")["finite_difference_unit_step"]
    def measured_nll():
        loss_total, count_total = 0.0, 0
        model.eval()
        with torch.no_grad():
            for row in subset:
                ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
                ledger.forward(row["token_count"])
                logits = model(ids[:, :-1], use_cache=False).logits
                loss, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device="cuda:0"))
                loss_total += float(loss); count_total += count
        return loss_total / count_total
    original = parameter.detach().clone()
    try:
        with torch.no_grad(): parameter.copy_(original + h * unit.to(parameter.device, parameter.dtype))
        plus = measured_nll()
        with torch.no_grad(): parameter.copy_(original - h * unit.to(parameter.device, parameter.dtype))
        minus = measured_nll()
        with torch.no_grad(): parameter.copy_(original)
        grad = torch.zeros_like(parameter, dtype=torch.float64, device="cpu")
        tokens = 0
        for row in subset:
            ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
            ledger.forward(row["token_count"])
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device="cuda:0"))
            grad += torch.autograd.grad(loss, parameter)[0].detach().cpu().double()
            ledger.backward(); tokens += count
        analytic = float(((grad / tokens) * unit).sum())
        numeric = (plus - minus) / (2 * h)
        tolerance = max(read_json(HERE / "config.json")["finite_difference_abs_tolerance"],
                        read_json(HERE / "config.json")["finite_difference_relative_tolerance"] * abs(analytic))
        result = {"build_window_id": build_id, "target": target, "baseline": baseline,
                  "view": "new_probe_prefix_first4", "step": h,
                  "analytic": analytic, "central_difference": numeric,
                  "absolute_error": abs(analytic - numeric), "tolerance": tolerance,
                  "pass": abs(analytic - numeric) <= tolerance, "token_count": tokens}
        append_jsonl(HERE / "results" / "finite_difference.jsonl", result)
        if not result["pass"]:
            raise RuntimeError(f"P0_FAIL: finite difference mismatch {result}")
        return result
    finally:
        with torch.no_grad(): parameter.copy_(original)


def geometry(model, rows: list[dict], parameter, name: str, a: torch.Tensor,
             ledger: Ledger, *, build_id: int, cfg: dict, vram_limit: int) -> dict:
    """Compute head/rest output energies from real inputs, without materializing Cx."""
    from pilot_core import loss_sum_count
    u, sigma, vh = torch.linalg.svd(a.cpu().double(), full_matrices=False)
    head = sigma[0] * torch.outer(u[:, 0], vh[0])
    rest = a.cpu().double() - head
    module = dict(model.named_modules())[name.removesuffix(".weight")]
    energy = defaultdict(float)
    total_tokens = 0
    for row in rows:
        captured = []
        hook = module.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().cpu().double()))
        try:
            ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
            ledger.forward(row["token_count"])
            with torch.no_grad():
                logits = model(ids[:, :-1], use_cache=False).logits
                _, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device="cuda:0"))
        finally:
            hook.remove()
        x = captured[0].reshape(-1, captured[0].shape[-1])
        hx, rx = x @ head.T, x @ rest.T
        energy["head"] += float(hx.square().sum())
        energy["rest"] += float(rx.square().sum())
        energy["cross"] += float((hx * rx).sum())
        energy["joint"] += float((hx + rx).square().sum())
        total_tokens += count
        append_jsonl(HERE / "results" / "output_energy_documents.jsonl",
                     {"build_window_id": build_id, "target": name, "view": "new_probe_prefix",
                      "doc_id": row["doc_id"], "token_count": count,
                      "head_energy": float(hx.square().sum()), "rest_energy": float(rx.square().sum()),
                      "cross_energy": float((hx * rx).sum()), "joint_energy": float((hx + rx).square().sum())})
        resource_gate(cfg, "after_geometry_document", ledger, vram_limit=vram_limit)
    return {"build_window_id": build_id, "target": name, "view": "new_probe_prefix",
            "head_frobenius_fraction": float(sigma[0].square() / sigma.square().sum()),
            "token_count": total_tokens,
            **{k + "_energy_per_token": v / total_tokens for k, v in energy.items()}}


def execute_replay():
    parent = verify_parent(deep_checkpoints=True)
    manifest_path = HERE / "evaluation_manifest.json"
    child = read_json(manifest_path)
    if child["status"] != "PREPARED_NO_GPU_REPLAY" or child["child_source_hash"] != parent["child_source_hash"]:
        raise RuntimeError("CHILD_SOURCE_OR_STATUS_MISMATCH")
    if child["parent_dependencies_hash"] != sha_object(parent):
        raise RuntimeError("PARENT_DEPENDENCY_MISMATCH")
    for name, field in (("eval_views.jsonl", "eval_views_sha256"),
                        ("legacy_truncation_audit.csv", "legacy_truncation_audit_sha256"),
                        ("selected_probe_docs.csv", "selected_probe_docs_sha256")):
        if sha_file(HERE / name) != child[field]:
            raise RuntimeError(f"CHILD_VIEW_MISMATCH: {name}")
    ledger = Ledger(sha_file(manifest_path), parent["child_source_hash"])
    if ledger.value["replay_started"]:
        raise RuntimeError("REPLAY_ALREADY_STARTED: no automatic rerun or resume")
    cfg = read_json(ROOT / "config" / "local_v1_lowmem.json")
    binding = read_json(ROOT / "bindings.local.json")
    child_cfg = read_json(HERE / "config.json")
    views = load_child_views()
    status_path = HERE / "results" / "replay_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with child_lock():
        ledger.value["replay_started"] = True; ledger.save()
        write_json(status_path, {"status": "RUNNING", "gpu_optimizer_calls": 0})
        try:
            resource_gate(cfg, "before_cuda_model_init", ledger, admission=True)
            if not torch.cuda.is_available():
                raise RuntimeError("RESOURCE_BLOCKED: no bound CUDA")
            free, _ = torch.cuda.mem_get_info(0)
            vram_limit = min(cfg["runtime"]["vram_soft_limit_gib"] * 2 ** 30,
                             free - cfg["runtime"]["vram_free_reserve_gib"] * 2 ** 30)
            if vram_limit <= 0:
                raise RuntimeError("RESOURCE_BLOCKED: VRAM reserve unavailable")
            resource_gate(cfg, "after_cuda_query", ledger, vram_limit=vram_limit)
            torch.set_num_threads(cfg["runtime"]["cpu_threads"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            from pilot_core import load_checkpoint, model_and_optimizer
            model, optimizer, aliases, _ = model_and_optimizer(cfg, binding, child_cfg["peak_lr"], "cuda:0")
            resource_gate(cfg, "after_model_init", ledger, vram_limit=vram_limit)
            data = read_json(ROOT / "artifacts" / "data_manifest.json")
            mem = np.memmap(ROOT / "artifacts" / "train_tokens.u32", dtype=np.uint32, mode="r")
            checkpoint = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003" / "step256.pt"
            output = []
            for window in child["train_build_windows"]:
                resource_gate(cfg, "before_build", ledger, vram_limit=vram_limit)
                first = build_update(model, optimizer, checkpoint, cfg, binding, data["data_hash"],
                                     aliases, mem, window, ledger, vram_limit)
                repeat = build_update(model, optimizer, checkpoint, cfg, binding, data["data_hash"],
                                      aliases, mem, window, ledger, vram_limit)
                repeat_error = max(float((first["updated"][name] - repeat["updated"][name]).abs().max())
                                   for name in first["updated"])
                if (repeat_error > cfg["numerics"]["fp32_atol"] or
                        first["gradient_sha256"] != repeat["gradient_sha256"] or
                        first["optimizer_state_sha256"] != repeat["optimizer_state_sha256"]):
                    raise RuntimeError(f"P0_FAIL: repeated complete build mismatch={repeat_error}")
                load_checkpoint(checkpoint, model, optimizer,
                                config_hash=parent["parent_fingerprints"]["config_hash"],
                                data_hash=data["data_hash"], aliases=aliases)
                resource_gate(cfg, "after_checkpoint_restore_for_eval", ledger, vram_limit=vram_limit)
                for target in child_cfg["targets"]:
                    states = four_states(first["base"], first["delta"], target)
                    f11_error = max(float((states["F11"][name] - first["updated"][name]).abs().max())
                                    for name in first["updated"])
                    if f11_error > cfg["numerics"]["fp32_atol"]:
                        raise RuntimeError(f"P0_FAIL: F11 actual AdamW identity={f11_error}")
                    if not torch.equal(states["F01"][target], first["base"][target]):
                        raise RuntimeError("P0_FAIL: F01 s=0 identity")
                    if any(not torch.equal(first["base"][name], first["updated"][name])
                           for name in first["none_gradient_names"]):
                        raise RuntimeError("P0_FAIL: None-gradient semantics")
                    a = first["directions"][target]
                    head_u, head_s, head_vh = torch.linalg.svd(a.double(), full_matrices=False)
                    head = head_s[0] * torch.outer(head_u[:, 0], head_vh[0])
                    rho_one = first["base"][target].double() - first["eta"] * (head + (a.double() - head))
                    rho_error = float((rho_one.float() - states["F11"][target]).abs().max())
                    if rho_error > cfg["numerics"]["fp32_atol"] + cfg["numerics"]["fp32_rtol"] * float(states["F11"][target].abs().max()):
                        raise RuntimeError(f"P0_FAIL: rho=s=1 identity={rho_error}")
                    summary = {"build_window_id": window["window_id"], "target": target,
                               "repeat_max_abs": repeat_error, "F11_real_step_max_abs": f11_error,
                               "rho_s_one_max_abs": rho_error,
                               "F01_s_zero_exact": True, "real_step_parity_max_abs": first["parity_max_abs"],
                               "repeated_gradient_sha256": first["gradient_sha256"],
                               "repeated_optimizer_state_sha256": first["optimizer_state_sha256"],
                               "build_token_count": first["valid_tokens"],
                               "generation_precision": first["generation_precision"],
                               "evaluation_precision": "fp32_no_tf32", "views": {}}
                    for view_name in child_cfg["views"]:
                        rows = views[view_name]
                        if view_name in ("legacy", "natural_prefix"):
                            rows = [r for r in rows if r["role"] in ("probe_select", "probe_confirm")]
                        nll = {}
                        counts = {}
                        for state_name, parameter_map in states.items():
                            with isolated_state(model, optimizer, {"cursor": window["start_token"]}):
                                apply_state(model, parameter_map)
                                measured, nll[state_name] = evaluate(model, rows, ledger,
                                                                       build_id=window["window_id"], target=target,
                                                                       state_name=state_name, view=view_name,
                                                                       cfg=cfg, vram_limit=vram_limit)
                            counts[state_name] = {(r["doc_id"], r["token_count"]) for r in measured}
                        if len(set(map(frozenset, counts.values()))) != 1:
                            raise RuntimeError("EVAL_VIEW_STATE_MASK_MISMATCH")
                        summary["views"][view_name] = {"nll": nll, **interactions(nll),
                                                       "distinct_items": len(rows),
                                                       "token_count": sum(r["token_count"] for r in rows),
                                                       "descriptive_only": view_name == "new_probe_packed"}
                    seq = cfg["runtime"]["sequence_length"]
                    train_rows = [{"view": "training_build_batch", "doc_id": f"build{window['window_id']}_seq{i}",
                                   "ids": np.asarray(mem[window["start_token"] + i * seq:
                                                         window["start_token"] + (i + 1) * seq + 1],
                                                     dtype=np.int64).tolist(),
                                   "token_count": seq, "source_doc_ids": [],
                                   "cluster_id": f"build{window['window_id']}"}
                                  for i in range(cfg["runtime"]["effective_batch_size"])]
                    train_nll = {}
                    for state_name, parameter_map in states.items():
                        with isolated_state(model, optimizer, {"cursor": window["start_token"]}):
                            apply_state(model, parameter_map)
                            _, train_nll[state_name] = evaluate(model, train_rows, ledger,
                                                                 build_id=window["window_id"], target=target,
                                                                 state_name=state_name, view="training_build_batch",
                                                                 cfg=cfg, vram_limit=vram_limit)
                    summary["training_build_batch"] = {"nll": train_nll, **interactions(train_nll),
                                                       "in_sample_only": True,
                                                       "not_generalization_evidence": True,
                                                       "token_count": cfg["runtime"]["effective_batch_size"] * seq}
                    directions = {"A": a, "M_plus": first["moments_plus"][target],
                                  "current_clipped_gradient": first["grads"][target]}
                    parameter = dict(model.named_parameters())[target]
                    for baseline in ("F00", "F01"):
                        with isolated_state(model, optimizer, {"cursor": window["start_token"]}):
                            apply_state(model, states[baseline])
                            gradient_signal(model, views["new_probe_prefix"], parameter, directions, ledger,
                                            build_id=window["window_id"], target=target, baseline=baseline,
                                            view="new_probe_prefix", cfg=cfg, vram_limit=vram_limit)
                            finite_difference(model, views["new_probe_prefix"], parameter, a, ledger,
                                              cfg=cfg, build_id=window["window_id"], target=target,
                                              baseline=baseline, vram_limit=vram_limit)
                    with isolated_state(model, optimizer, {"cursor": window["start_token"]}):
                        apply_state(model, states["F01"])
                        summary["geometry"] = geometry(model, views["new_probe_prefix"], parameter,
                                                       target, a, ledger, build_id=window["window_id"],
                                                       cfg=cfg, vram_limit=vram_limit)
                    append_jsonl(HERE / "results" / "replay_summary.jsonl", summary)
                    output.append(summary)
                write_json(status_path, {"status": "PARTIAL", "completed_build_windows": window["window_id"] + 1,
                                         **ledger.value})
            write_json(status_path, {"status": "PASS", "completed_build_windows": len(output) // 2,
                                     "summaries": len(output), **ledger.value})
            return output
        except BaseException as exc:
            write_json(status_path, {"status": "STOPPED_BUDGET" if isinstance(exc, BudgetStop) else "FAILED",
                                     "reason": str(exc), **ledger.value})
            raise


def execute_reeval_lr():
    """Optional zero-update LR ranking on the same 32 documents under two explicit views."""
    parent = verify_parent(deep_checkpoints=True)
    manifest_path = HERE / "evaluation_manifest.json"
    manifest = read_json(manifest_path)
    if (manifest["child_source_hash"] != parent["child_source_hash"] or
            manifest["parent_dependencies_hash"] != sha_object(parent) or
            sha_file(HERE / "eval_views.jsonl") != manifest["eval_views_sha256"]):
        raise RuntimeError("CHILD_OR_PARENT_MANIFEST_MISMATCH")
    ledger = Ledger(sha_file(manifest_path), parent["child_source_hash"])
    if ledger.value.get("lr_reeval_started"):
        raise RuntimeError("LR_REEVAL_ALREADY_STARTED: no automatic rerun")
    cfg = read_json(ROOT / "config" / "local_v1_lowmem.json")
    binding = read_json(ROOT / "bindings.local.json")
    views = load_child_views()
    output = {}
    with child_lock():
        ledger.value["lr_reeval_started"] = True; ledger.save()
        try:
            resource_gate(cfg, "lr_reeval_before_cuda", ledger, admission=True)
            if not torch.cuda.is_available():
                raise RuntimeError("RESOURCE_BLOCKED: no bound CUDA")
            free, _ = torch.cuda.mem_get_info(0)
            vram_limit = min(cfg["runtime"]["vram_soft_limit_gib"] * 2 ** 30,
                             free - cfg["runtime"]["vram_free_reserve_gib"] * 2 ** 30)
            if vram_limit <= 0:
                raise RuntimeError("RESOURCE_BLOCKED: VRAM reserve unavailable")
            torch.set_num_threads(cfg["runtime"]["cpu_threads"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            from pilot_core import load_checkpoint, model_and_optimizer
            data = read_json(ROOT / "artifacts" / "data_manifest.json")
            for peak in cfg["baseline"]["peak_lr_grid"]:
                model, optimizer, aliases, _ = model_and_optimizer(cfg, binding, peak, "cuda:0")
                checkpoint = ROOT / "artifacts" / "checkpoints" / f"seed3407_lr{peak:g}" / "step512.pt"
                load_checkpoint(checkpoint, model, optimizer,
                                config_hash=parent["parent_fingerprints"]["config_hash"],
                                data_hash=data["data_hash"], aliases=aliases)
                resource_gate(cfg, "lr_reeval_after_checkpoint", ledger, vram_limit=vram_limit)
                output[str(peak)] = {}
                for view in ("legacy", "natural_prefix"):
                    rows = [row for row in views[view] if row["role"] == "lr_select"]
                    _, nll = evaluate(model, rows, ledger, build_id=-1, target="LR_REEVAL",
                                      state_name=f"checkpoint_lr{peak:g}", view=view,
                                      cfg=cfg, vram_limit=vram_limit)
                    output[str(peak)][view] = nll
                del model, optimizer
            output["ranking"] = {view: sorted((peak for peak in cfg["baseline"]["peak_lr_grid"]),
                                               key=lambda peak: output[str(peak)][view])
                                 for view in ("legacy", "natural_prefix")}
            output["parent_lr_boundary_unresolved"] = True
            output["gpu_optimizer_calls"] = ledger.value["gpu_optimizer_calls"]
            output["token_forward_equivalents_cumulative"] = ledger.value["token_forward_equivalents"]
            write_json(HERE / "results" / "lr_reeval.json", output)
            return output
        except BaseException as exc:
            write_json(HERE / "results" / "lr_reeval_status.json",
                       {"status": "STOPPED_BUDGET" if isinstance(exc, BudgetStop) else "FAILED",
                        "reason": str(exc), **ledger.value})
            raise
