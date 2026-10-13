"""Bounded real GPU readiness measurement from immutable parent checkpoints."""
from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import torch

from .audit import verify_inputs
from .common import (HERE, ROOT, V12, BudgetStop, Ledger, append_jsonl,
                     compare_tensors, isolated_state, optimizer_logical_snapshot,
                     read_json, resource_gate, run_lock, sha_file, sha_object,
                     source_hash, tensor_digest, write_json)

CFG = read_json(HERE / "config.json")
PARENT_CFG = read_json(ROOT / "config" / "local_v1_lowmem.json")
BINDING = read_json(ROOT / "bindings.local.json")
DATA = read_json(ROOT / "artifacts" / "data_manifest.json")
CHECKPOINT = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003" / "step256.pt"
TARGETS = (CFG["primary_target"], CFG["diagnostic_target"])


def verified_run():
    parent, v12, _, _ = verify_inputs()
    manifest = read_json(HERE / "run_manifest.json")
    if (manifest["source_hash"] != source_hash() or
            manifest["parent_dependencies_sha256"] != sha_file(HERE / "parent_dependencies.json") or
            manifest["v12_eval_manifest_sha256"] != sha_file(V12 / "evaluation_manifest.json") or
            manifest["v12_eval_views_sha256"] != sha_file(V12 / "eval_views.jsonl")):
        raise RuntimeError("V13_RUN_IDENTITY_MISMATCH")
    if manifest["fixed_windows"] != v12["train_build_windows"]:
        raise RuntimeError("V13_WINDOW_SELECTION_CHANGED")
    return parent, v12, manifest, Ledger(sha_file(HERE / "run_manifest.json"))


def views():
    groups = defaultdict(list)
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["token_count"] != len(row["ids"]) - 1 or len(row["ids"]) > 257:
                raise RuntimeError("EVAL_VIEW_MASK_INVALID")
            groups[row["view"]].append(row)
    old = {name: [r for r in groups[name] if r["role"] in ("probe_select", "probe_confirm")]
           for name in ("legacy", "natural_prefix")}
    if {r["doc_id"] for r in old["legacy"]} != {r["doc_id"] for r in old["natural_prefix"]}:
        raise RuntimeError("PAIRED_DOCUMENT_IDS_CHANGED")
    groups.update(old)
    return groups


def cuda_session(ledger):
    resource_gate(ledger, "before_cuda_model_init", admission=True)
    if not torch.cuda.is_available():
        raise RuntimeError("RESOURCE_BLOCKED:NO_CUDA")
    free, _ = torch.cuda.mem_get_info(0)
    runtime = PARENT_CFG["runtime"]
    limit = min(runtime["vram_soft_limit_gib"] * 2 ** 30,
                free - runtime["vram_free_reserve_gib"] * 2 ** 30)
    if limit <= 0:
        raise RuntimeError("RESOURCE_BLOCKED:VRAM_RESERVE")
    torch.set_num_threads(runtime["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resource_gate(ledger, "after_cuda_query", vram_limit=limit)
    from pilot_core import model_and_optimizer
    model, optimizer, aliases, count = model_and_optimizer(PARENT_CFG, BINDING, CFG["peak_lr"], "cuda:0")
    resource_gate(ledger, "after_model_init", vram_limit=limit)
    return model, optimizer, aliases, limit


def load_original(model, optimizer, aliases):
    from pilot_core import load_checkpoint
    payload = load_checkpoint(CHECKPOINT, model, optimizer,
                              config_hash=sha_file(ROOT / "config" / "local_v1_lowmem.json"),
                              data_hash=DATA["data_hash"], aliases=aliases)
    if payload["successful_optimizer_step"] != 256:
        raise RuntimeError("PARENT_CHECKPOINT_STEP_MISMATCH")
    return payload


def _stage_digest(grads):
    return tensor_digest(grads)


def _clone_grads(model):
    return {name: None if p.grad is None else p.grad.detach().cpu().clone()
            for name, p in model.named_parameters()}


def _state_fingerprint(model, optimizer, cursor, ids_digest):
    logical = optimizer_logical_snapshot(model, optimizer)
    return {"model": tensor_digest({n: p.detach() for n, p in model.named_parameters()}),
            "buffers": tensor_digest({n: b.detach() for n, b in model.named_buffers()}),
            "optimizer_state": logical["state_digest"], "optimizer_groups": logical["group_digest"],
            "optimizer_aliases": logical["alias_digest"],
            "modes": sha_object({n: m.training for n, m in model.named_modules()}),
            "python_rng": sha_object(str(random.getstate())),
            "numpy_rng": sha_object(str(np.random.get_state())),
            "cpu_rng": tensor_digest({"rng": torch.get_rng_state()}),
            "cuda_rng": tensor_digest({str(i): x for i, x in enumerate(torch.cuda.get_rng_state_all())}),
            "cursor": cursor, "ids_digest": ids_digest,
            "grad_presence": tensor_digest({n: p.grad for n, p in model.named_parameters()})}


def _save_frozen(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(obj, temporary)
    os.replace(temporary, path)


def build(model, optimizer, aliases, window, tag, ledger, vram_limit, *, precision="parent"):
    """One actual isolated full AdamW step; every forward and attempt is counted."""
    from pilot_core import adamw_direction, loss_sum_count, schedule_lr, training_batch
    path = HERE / "frozen" / f"window{window['window_id']}_{tag}.pt"
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta["sha256"] != sha_file(path) or meta["source_hash"] != source_hash():
            raise RuntimeError("FROZEN_UPDATE_IDENTITY_MISMATCH")
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.exists() or meta_path.exists():
        raise RuntimeError("INCOMPLETE_FROZEN_UPDATE_REQUIRES_AUDIT")
    started = time.perf_counter()
    resource_gate(ledger, "before_build_" + tag, vram_limit=vram_limit)
    load_original(model, optimizer, aliases)
    eta = schedule_lr(CFG["peak_lr"], CFG["checkpoint_step"] + 1, PARENT_CFG)
    for group in optimizer.param_groups:
        group["lr"] = eta
    model.train()
    optimizer.zero_grad(set_to_none=True)
    cursor = window["start_token"]
    token_digest = __import__("hashlib").sha256()
    initial = _state_fingerprint(model, optimizer, cursor, "pending")
    mem = np.memmap(ROOT / "artifacts" / "train_tokens.u32", dtype=np.uint32, mode="r")
    valid_total = 0
    batch_ids = []
    if precision == "parent":
        generation_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    elif precision == "fp32_reference":
        generation_dtype = torch.float32
    else:
        raise ValueError(precision)
    raw = normalized = clipped = None
    try:
        for _ in range(PARENT_CFG["runtime"]["accumulation_steps"]):
            resource_gate(ledger, "before_build_forward_" + tag, vram_limit=vram_limit)
            ids, cursor = training_batch(mem, cursor, PARENT_CFG["runtime"]["micro_batch_size"],
                                         PARENT_CFG["runtime"]["sequence_length"], "cuda:0")
            token_digest.update(ids.detach().cpu().contiguous().numpy().tobytes())
            batch_ids.append(ids.detach().cpu().clone())
            count_expected = ids.shape[0] * (ids.shape[1] - 1)
            ledger.forward(count_expected)
            with torch.autocast(device_type="cuda", dtype=generation_dtype,
                                enabled=generation_dtype == torch.bfloat16):
                logits = model(ids[:, :-1], use_cache=False).logits
                loss_sum, count = loss_sum_count(logits, ids,
                                                  torch.full((ids.shape[0],), ids.shape[1], device="cuda:0"))
            if count != count_expected:
                raise RuntimeError("BUILD_MASK_COUNT_MISMATCH")
            ledger.backward(); loss_sum.backward()
            valid_total += count
        if cursor != window["end_token_exclusive"] - 1 or valid_total != window["target_tokens"]:
            raise RuntimeError("BUILD_WINDOW_MISMATCH")
        raw = _clone_grads(model)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(valid_total)
        normalized = _clone_grads(model)
        clip_before = torch.nn.utils.clip_grad_norm_(model.parameters(),
                            PARENT_CFG["baseline"]["global_grad_clip_norm"])
        clipped = _clone_grads(model)
        named = dict(model.named_parameters())
        decay = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
        old = {name: p.detach().cpu().clone() for name, p in named.items()}
        base, a, mplus, moments_before, all_directions = {}, {}, {}, {}, {}
        for name, parameter in named.items():
            if parameter.grad is None:
                base[name] = old[name].clone()
                continue
            base[name] = (parameter.detach() * (1 - eta * decay[id(parameter)])).cpu().clone()
            direction, m_plus = adamw_direction(parameter.detach(), parameter.grad,
                           optimizer.state[parameter], beta1=PARENT_CFG["baseline"]["betas"][0],
                           beta2=PARENT_CFG["baseline"]["betas"][1], eps=PARENT_CFG["baseline"]["eps"])
            all_directions[name] = direction.detach().cpu().clone()
            if name in TARGETS:
                a[name], mplus[name] = direction.detach().cpu().clone(), m_plus.detach().cpu().clone()
                moments_before[name] = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                                        for key, value in optimizer.state[parameter].items()}
        before_step = _state_fingerprint(model, optimizer, cursor, token_digest.hexdigest())
        resource_gate(ledger, "before_optimizer_" + tag, vram_limit=vram_limit)
        ledger.optimizer_attempt()
        optimizer.step()
        ledger.optimizer_complete()
        torch.cuda.synchronize()
        resource_gate(ledger, "after_optimizer_" + tag, vram_limit=vram_limit)
        updated = {name: p.detach().cpu().clone() for name, p in named.items()}
        delta = {name: updated[name] - base[name] for name in named}
        moments_after = {}
        moment_parity = {}
        for name in TARGETS:
            parameter = named[name]
            moments_after[name] = {key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                                   for key, value in optimizer.state[parameter].items()}
            beta1, beta2 = PARENT_CFG["baseline"]["betas"]
            expected_m = moments_before[name]["exp_avg"] * beta1 + clipped[name] * (1 - beta1)
            expected_v = moments_before[name]["exp_avg_sq"] * beta2 + clipped[name].square() * (1 - beta2)
            moment_parity[name] = {
                "m_max_abs": float((moments_after[name]["exp_avg"] - expected_m).abs().max()),
                "v_max_abs": float((moments_after[name]["exp_avg_sq"] - expected_v).abs().max()),
                "step_before": int(moments_before[name]["step"]),
                "step_after": int(moments_after[name]["step"])}
            if (moment_parity[name]["m_max_abs"] > 1e-7 or
                    moment_parity[name]["v_max_abs"] > 1e-7 or
                    moment_parity[name]["step_after"] != moment_parity[name]["step_before"] + 1):
                raise RuntimeError("ADAMW_MOMENT_PARITY_MISMATCH:" + name)
        parity = {name: float((updated[name] - (base[name] - eta * all_directions[name])).abs().max())
                  for name in all_directions}
        for name in named:
            if clipped[name] is None and not torch.equal(updated[name], old[name]):
                raise RuntimeError("NONE_GRADIENT_PARAMETER_CHANGED:" + name)
        target_parity = max(parity.values())
        limit = PARENT_CFG["numerics"]["fp32_atol"] + PARENT_CFG["numerics"]["fp32_rtol"] * max(
            float(updated[name].abs().max()) for name in TARGETS)
        if target_parity > limit:
            raise RuntimeError(f"STEP_FORMULA_MISMATCH:{target_parity}>{limit}")
        frozen = {"window": window, "tag": tag, "source_hash": source_hash(),
                  "checkpoint_sha256": sha_file(CHECKPOINT), "generation_precision": str(generation_dtype),
                  "generation_backend": "MiniMind_parent_SDPA", "evaluation_precision": "fp32_no_tf32",
                  "eta": eta, "valid_tokens": valid_total, "clip_norm_before": float(clip_before),
                  "initial_state": initial, "before_step_state": before_step,
                  "token_ids_sha256": token_digest.hexdigest(), "old": old, "base": base,
                  "delta": delta, "updated": updated, "A": a, "M_plus": mplus,
                  "moments_before": moments_before, "moments_after": moments_after,
                  "moment_parity": moment_parity,
                  "stage": {"raw": raw, "normalized": normalized, "clipped": clipped},
                  "stage_digest": {"raw": _stage_digest(raw), "normalized": _stage_digest(normalized),
                                   "clipped": _stage_digest(clipped)},
                  "optimizer_after": optimizer_logical_snapshot(model, optimizer),
                  "formula_max_abs_all_updated": target_parity, "formula_tolerance": limit,
                  "formula_per_parameter_max_abs": parity,
                  "elapsed_seconds": time.perf_counter() - started}
        _save_frozen(path, frozen)
        write_json(meta_path, {"sha256": sha_file(path), "source_hash": source_hash(),
                               "window_id": window["window_id"], "tag": tag,
                               "generation_precision": str(generation_dtype),
                               "formula_max_abs_all_updated": target_parity,
                               "stage_digest": frozen["stage_digest"],
                               "token_ids_sha256": token_digest.hexdigest()})
        return frozen
    finally:
        if raw is None:
            append_jsonl(HERE / "logs" / "build_failures.jsonl",
                         {"window": window["window_id"], "tag": tag, "phase": "before_gradients",
                          "budget": ledger.data})


def compare_builds(first, second, label):
    if first["window"] != second["window"] or first["token_ids_sha256"] != second["token_ids_sha256"]:
        raise RuntimeError("REPEAT_INPUT_MISMATCH")
    def flatten_moments(which):
        return {name + "/" + field: value
                for name, fields in which.items() for field, value in fields.items()}
    stages = {}
    first_difference = None
    for phase in ("moments_before", "raw", "normalized", "clipped", "M_plus",
                  "A", "moments_after", "base", "delta", "updated", "optimizer_after"):
        if phase in ("moments_before", "moments_after"):
            left, right = flatten_moments(first[phase]), flatten_moments(second[phase])
        elif phase == "optimizer_after":
            left, right = first[phase]["state"], second[phase]["state"]
        else:
            left = first["stage"][phase] if phase in first["stage"] else first[phase]
            right = second["stage"][phase] if phase in second["stage"] else second[phase]
        rows = {name: compare_tensors(left.get(name), right.get(name)) for name in sorted(set(left) | set(right))}
        stages[phase] = rows
        if first_difference is None and any(r.get("different_elements", 0) or r.get("same_shape_dtype") is False
                                             or not r.get("same_presence", True)
                                             or r.get("finite") is False for r in rows.values()):
            first_difference = phase
    result = {"comparison": label, "first_difference_stage": first_difference,
              "input_state": {key: {"first": first["initial_state"].get(key),
                                    "second": second["initial_state"].get(key),
                                    "equal": first["initial_state"].get(key) == second["initial_state"].get(key)}
                              for key in first["initial_state"]},
              "before_step_state": {key: first["before_step_state"].get(key) == second["before_step_state"].get(key)
                                    for key in first["before_step_state"]},
              "optimizer_group_digest_equal": first["optimizer_after"]["group_digest"] == second["optimizer_after"]["group_digest"],
              "optimizer_alias_digest_equal": first["optimizer_after"]["alias_digest"] == second["optimizer_after"]["alias_digest"],
              "stages": stages,
              "digest_order_invariant": True,
              "formula_first": first["formula_max_abs_all_updated"],
              "formula_second": second["formula_max_abs_all_updated"]}
    write_json(HERE / "results" / f"compare_{label}.json", result)
    return result


def _apply_state(model, frozen, target, state):
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            value = frozen["base"][name]
            if state == "F11" or (state == "F10" and name == target) or (state == "F01" and name != target):
                value = value + frozen["delta"][name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def state_identity(frozen, target):
    f11 = max(float((frozen["base"][name] + frozen["delta"][name] - frozen["updated"][name]).abs().max())
              for name in frozen["base"])
    a = frozen["A"][target].double()
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    head = s[0] * torch.outer(u[:, 0], vh[0])
    rho_one = frozen["base"][target].double() - frozen["eta"] * (head + (a - head))
    rho_error = float((rho_one.float() - frozen["updated"][target]).abs().max())
    # Compare the F01 assembly with the original conditional s=0 formula:
    # all non-target parameters take their actual update; target takes decay only.
    conditional = {name: (frozen["base"][name] if name == target else frozen["updated"][name])
                   for name in frozen["base"]}
    f01 = {name: frozen["base"][name] + (frozen["delta"][name] if name != target else 0)
           for name in frozen["base"]}
    f01_error = max(float((f01[name] - conditional[name]).abs().max()) for name in f01)
    limit = frozen["formula_tolerance"]
    if f11 > limit or rho_error > limit or f01_error > limit:
        raise RuntimeError(f"FOUR_STATE_IDENTITY_MISMATCH:F11={f11},F01={f01_error},rho={rho_error},limit={limit}")
    return {"F11_actual_max_abs": f11, "F01_original_conditional_max_abs": f01_error,
            "rho_s_one_max_abs": rho_error, "formula_tolerance": limit}


def _loss_row(model, row, ledger, vram_limit):
    from pilot_core import loss_sum_count
    resource_gate(ledger, "before_eval_forward", vram_limit=vram_limit)
    ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
    ledger.forward(row["token_count"])
    with torch.no_grad():
        logits = model(ids[:, :-1], use_cache=False).logits
        loss, count = loss_sum_count(logits, ids, torch.tensor([len(row["ids"])], device="cuda:0"))
    if count != row["token_count"]:
        raise RuntimeError("EVAL_SHIFT_OR_MASK_MISMATCH")
    return float(loss), count


def _unit_path(tag, target, view, state):
    short = "ffn" if target == CFG["primary_target"] else "attn"
    return HERE / "results" / "losses" / f"{tag}_{short}_{view}_{state}.jsonl"


def evaluate_state(model, optimizer, frozen, target, state, rows, view, tag, ledger, vram_limit):
    path = _unit_path(tag, target, view, state)
    expected = [r["doc_id"] for r in rows]
    if path.exists():
        saved = [json.loads(line) for line in path.open(encoding="utf-8")]
        if ([r["doc_id"] for r in saved] != expected or
                any(r["source_hash"] != source_hash() for r in saved)):
            raise RuntimeError("COMPLETED_LOSS_UNIT_IDENTITY_MISMATCH:" + str(path))
        return saved
    partial = path.with_suffix(".partial")
    saved = [json.loads(line) for line in partial.open(encoding="utf-8")] if partial.exists() else []
    if ([r["doc_id"] for r in saved] != expected[:len(saved)] or
            any(r["source_hash"] != source_hash() for r in saved)):
        raise RuntimeError("PARTIAL_LOSS_UNIT_IDENTITY_MISMATCH:" + str(partial))
    path.parent.mkdir(parents=True, exist_ok=True)
    with isolated_state(model, optimizer, {"cursor": frozen["window"]["start_token"]}):
        _apply_state(model, frozen, target, state)
        model.eval()
        for row in rows[len(saved):]:
            loss, count = _loss_row(model, row, ledger, vram_limit)
            item = {"tag": tag, "window_id": frozen["window"]["window_id"],
                    "target": target, "state": state, "view": view,
                    "doc_id": row["doc_id"], "source_doc_ids": row["source_doc_ids"],
                    "cluster_id": row["cluster_id"], "loss_sum": loss, "token_count": count,
                    "generation_precision": frozen["generation_precision"],
                    "evaluation_precision": "fp32_no_tf32", "source_hash": source_hash()}
            append_jsonl(partial, item)
            saved.append(item)
    os.replace(partial, path)
    write_json(path.with_suffix(".status.json"), {"status": "COMPLETE", "sha256": sha_file(path),
                                                 "rows": len(saved), "source_hash": source_hash()})
    return saved


def _aggregate(rows):
    return sum(row["loss_sum"] for row in rows) / sum(row["token_count"] for row in rows)


def effect_from_rows(state_rows):
    nll = {state: _aggregate(rows) for state, rows in state_rows.items()}
    return {"nll": nll, "I_before": nll["F00"] - nll["F10"],
            "I_after": nll["F01"] - nll["F11"],
            "interaction": nll["F11"] - nll["F10"] - nll["F01"] + nll["F00"],
            "documents": len(state_rows["F00"]),
            "target_tokens": sum(r["token_count"] for r in state_rows["F00"])}


def evaluate_four(model, optimizer, frozen, target, rows, view, tag, ledger, vram_limit):
    identity = state_identity(frozen, target)
    data = {state: evaluate_state(model, optimizer, frozen, target, state, rows, view,
                                  tag, ledger, vram_limit)
            for state in ("F00", "F10", "F01", "F11")}
    masks = [{(r["doc_id"], r["token_count"]) for r in values} for values in data.values()]
    if any(value != masks[0] for value in masks[1:]):
        raise RuntimeError("FOUR_STATE_VIEW_MASK_CHANGED")
    result = {"tag": tag, "window_id": frozen["window"]["window_id"],
              "target": target, "view": view, "identity": identity,
              **effect_from_rows(data), "raw_paths": {state: str(_unit_path(tag, target, view, state))
                                                  for state in data}}
    write_json(HERE / "results" / "summaries" / f"{tag}_{'ffn' if target==TARGETS[0] else 'attn'}_{view}.json", result)
    return result


def paired_label_effect(tag, target):
    short = "ffn" if target == TARGETS[0] else "attn"
    source = {}
    for view in ("legacy", "natural_prefix"):
        source[view] = {state: [json.loads(line) for line in
                               _unit_path(tag, target, view, state).open(encoding="utf-8")]
                        for state in ("F00", "F10", "F01", "F11")}
    ids = [r["doc_id"] for r in source["legacy"]["F00"]]
    if ids != [r["doc_id"] for r in source["natural_prefix"]["F00"]]:
        raise RuntimeError("LABEL_PAIR_IDS_CHANGED")
    def selected(indices, view):
        return effect_from_rows({state: [source[view][state][i] for i in indices]
                                 for state in source[view]})
    full = {view: selected(range(len(ids)), view) for view in source}
    paired = {key: full["natural_prefix"][key] - full["legacy"][key]
              for key in ("I_before", "I_after", "interaction")}
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    samples = {key: [] for key in paired}
    for _ in range(CFG["bootstrap_repetitions"]):
        picks = rng.integers(0, len(ids), size=len(ids))
        a, b = selected(picks, "legacy"), selected(picks, "natural_prefix")
        for key in paired:
            samples[key].append(b[key] - a[key])
    result = {"tag": tag, "target": target, "distinct_documents": len(ids),
              "legacy": full["legacy"], "natural_prefix": full["natural_prefix"],
              "paired_natural_minus_legacy": paired,
              "paired_document_bootstrap_95": {key: list(map(float, np.quantile(value, [0.025, 0.975])))
                                               for key, value in samples.items()},
              "bootstrap_repetitions": CFG["bootstrap_repetitions"],
              "interpretation": "exploratory; old probe_confirm already used"}
    write_json(HERE / "results" / "summaries" / f"{tag}_{short}_paired_labels.json", result)
    return result


def _directions(frozen, target):
    a = frozen["A"][target].double()
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    head = s[0] * torch.outer(u[:, 0], vh[0])
    return {"A": a, "M_plus": frozen["M_plus"][target].double(),
            "current_clipped_gradient": frozen["stage"]["clipped"][target].double(),
            "A_head_rank1": head, "A_rest": a - head}, s


def gradient_diagnostic(model, optimizer, frozen, target, baseline, rows, tag, ledger, vram_limit):
    from pilot_core import loss_sum_count
    path = HERE / "results" / "directions" / f"{tag}_{'ffn' if target==TARGETS[0] else 'attn'}_{baseline}.jsonl"
    if path.exists():
        saved = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(saved) != len(rows) or [r["doc_id"] for r in saved] != [r["doc_id"] for r in rows]:
            raise RuntimeError("DIRECTION_RESULT_IDENTITY_MISMATCH")
        return saved
    directions, spectrum = _directions(frozen, target)
    parameter = dict(model.named_parameters())[target]
    output = []
    sum_grad = torch.zeros_like(directions["A"])
    with isolated_state(model, optimizer, {"cursor": frozen["window"]["start_token"]}):
        _apply_state(model, frozen, target, baseline)
        model.eval()
        for row in rows:
            resource_gate(ledger, "before_gradient_probe", vram_limit=vram_limit)
            ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
            ledger.forward(row["token_count"])
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([len(row["ids"])], device="cuda:0"))
            ledger.backward()
            grad = torch.autograd.grad(loss, parameter)[0].detach().cpu().double()
            if count != row["token_count"]:
                raise RuntimeError("GRADIENT_PROBE_MASK_CHANGED")
            sum_grad += grad
            norm_g = float(torch.linalg.vector_norm(grad))
            norm_a = float(torch.linalg.vector_norm(directions["A"]))
            measures = {}
            for name, direction in directions.items():
                norm_d = float(torch.linalg.vector_norm(direction))
                dot_sum = float((grad * direction).sum())
                measures[name] = {"loss_sum_gradient_dot": dot_sum,
                                  "cosine": dot_sum / (norm_g * norm_d) if norm_g and norm_d else None,
                                  "A_norm_matched_dot": dot_sum * norm_a / norm_d if norm_d else None,
                                  "direction_norm": norm_d}
            output.append({"tag": tag, "target": target, "baseline": baseline,
                           "view": "new_probe_prefix", "doc_id": row["doc_id"],
                           "token_count": count, "loss_sum": float(loss),
                           "gradient_norm_loss_sum": norm_g, "directions": measures})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for item in output:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    total_tokens = sum(r["token_count"] for r in output)
    mean_grad = sum_grad / total_tokens
    norm_g = float(torch.linalg.vector_norm(mean_grad))
    norm_a = float(torch.linalg.vector_norm(directions["A"]))
    summary = {}
    for name, direction in directions.items():
        norm_d = float(torch.linalg.vector_norm(direction))
        dot = float((mean_grad * direction).sum())
        summary[name] = {"token_weighted_dot": dot,
                         "cosine": dot / (norm_g * norm_d) if norm_g and norm_d else None,
                         "A_norm_matched_dot": dot * norm_a / norm_d if norm_d else None,
                         "direction_norm": norm_d}
    write_json(path.with_suffix(".summary.json"), {"tag": tag, "target": target,
               "baseline": baseline, "token_count": total_tokens, "documents": len(output),
               "mean_gradient_norm": norm_g, "directions": summary,
               "source": "independent_new_probe_prefix",
               "generation_precision": frozen["generation_precision"],
               "evaluation_precision": "fp32_no_tf32", "raw_sha256": sha_file(path)})
    return output


def geometry_diagnostic(model, optimizer, frozen, target, rows, tag, ledger, vram_limit):
    from pilot_core import loss_sum_count
    path = HERE / "results" / "geometry" / f"{tag}_{'ffn' if target==TARGETS[0] else 'attn'}.jsonl"
    if path.exists():
        saved = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(saved) != len(rows):
            raise RuntimeError("GEOMETRY_RESULT_IDENTITY_MISMATCH")
        return saved
    directions, spectrum = _directions(frozen, target)
    head, rest = directions["A_head_rank1"], directions["A_rest"]
    module = dict(model.named_modules())[target.removesuffix(".weight")]
    output = []
    with isolated_state(model, optimizer, {"cursor": frozen["window"]["start_token"]}):
        _apply_state(model, frozen, target, "F01")
        model.eval()
        for row in rows:
            resource_gate(ledger, "before_geometry_forward", vram_limit=vram_limit)
            captured = []
            hook = module.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach().cpu().double()))
            try:
                ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
                ledger.forward(row["token_count"])
                with torch.no_grad():
                    logits = model(ids[:, :-1], use_cache=False).logits
                    _, count = loss_sum_count(logits, ids,
                                              torch.tensor([len(row["ids"])], device="cuda:0"))
            finally:
                hook.remove()
            x = captured[0].reshape(-1, captured[0].shape[-1])[:count]
            hx, rx = x @ head.T, x @ rest.T
            output.append({"tag": tag, "target": target, "baseline": "F01",
                           "view": "new_probe_prefix", "doc_id": row["doc_id"],
                           "token_count": count, "input_rows_used": x.shape[0],
                           "head_energy": float(hx.square().sum()),
                           "rest_energy": float(rx.square().sum()),
                           "cross_energy": float((hx * rx).sum()),
                           "joint_energy": float((hx + rx).square().sum())})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for item in output:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(temp, path)
    tokens = sum(row["token_count"] for row in output)
    summary = {name + "_per_token": sum(r[name] for r in output) / tokens
               for name in ("head_energy", "rest_energy", "cross_energy", "joint_energy")}
    write_json(path.with_suffix(".summary.json"), {"tag": tag, "target": target,
               "view": "new_probe_prefix", "baseline": "F01", "token_count": tokens,
               "documents": len(output), "head_frobenius_energy_fraction":
               float(spectrum[0].square() / spectrum.square().sum()),
               **summary, "raw_sha256": sha_file(path)})
    return output


def finite_difference_diagnostic(model, optimizer, frozen, target, baseline, rows, view, ledger, vram_limit):
    """Three preregistered scales on the same token-weighted subset and base."""
    from pilot_core import loss_sum_count
    path = HERE / "results" / "finite_difference" / f"w0_ffn_{baseline}_{view}.json"
    if path.exists():
        return read_json(path)
    direction = frozen["A"][target].double()
    unit = direction / torch.linalg.vector_norm(direction)
    parameter = dict(model.named_parameters())[target]
    with isolated_state(model, optimizer, {"cursor": frozen["window"]["start_token"]}):
        _apply_state(model, frozen, target, baseline)
        model.eval()
        original = parameter.detach().clone()
        sum_grad = torch.zeros_like(unit)
        total_tokens = 0
        for row in rows:
            resource_gate(ledger, "before_fd_autodiff", vram_limit=vram_limit)
            ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
            ledger.forward(row["token_count"])
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([len(row["ids"])], device="cuda:0"))
            ledger.backward()
            sum_grad += torch.autograd.grad(loss, parameter)[0].detach().cpu().double()
            total_tokens += count
        analytic = float(((sum_grad / total_tokens) * unit).sum())
        scales = []
        try:
            for h in CFG["finite_difference_unit_steps"]:
                nlls = []
                applied = []
                for sign in (1.0, -1.0):
                    with torch.no_grad():
                        parameter.copy_(original + sign * h * unit.to(parameter.device, parameter.dtype))
                    applied.append(float(torch.linalg.vector_norm((parameter.detach() - original).cpu().double())))
                    total_loss, total_count = 0.0, 0
                    for row in rows:
                        loss, count = _loss_row(model, row, ledger, vram_limit)
                        total_loss += loss; total_count += count
                    if total_count != total_tokens:
                        raise RuntimeError("FD_TOKEN_WEIGHT_CHANGED")
                    nlls.append(total_loss / total_count)
                numeric = (nlls[0] - nlls[1]) / (2 * h)
                scales.append({"h": h, "actual_applied_l2_plus_minus": applied,
                               "nll_plus": nlls[0], "nll_minus": nlls[1],
                               "nll_difference": nlls[0] - nlls[1],
                               "central_difference": numeric, "autodiff": analytic,
                               "absolute_error": abs(numeric - analytic),
                               "same_sign": math.copysign(1, numeric) == math.copysign(1, analytic)
                               if numeric and analytic else None})
        finally:
            with torch.no_grad():
                parameter.copy_(original)
    result = {"baseline": baseline, "target": target, "view": view,
              "doc_ids": [r["doc_id"] for r in rows], "target_tokens": total_tokens,
              "unit_direction": "A_Frobenius_unit", "autodiff": analytic,
              "scales": scales, "generation_precision": frozen["generation_precision"],
              "evaluation_precision": "fp32_no_tf32",
              "interpretation": "all scales retained; near-zero or rounded differences may be unresolved"}
    write_json(path, result)
    return result


def _calibration_rows(all_views, view):
    return sorted((r for r in all_views[view] if r["role"] == "probe_select"),
                  key=lambda r: r["doc_id"])[:CFG["calibration_old_probe_select_docs"]]


def _calibration_effects(model, optimizer, frozen, tag, all_views, ledger, vram_limit):
    return {view: evaluate_four(model, optimizer, frozen, TARGETS[0],
                                _calibration_rows(all_views, view), view, tag, ledger, vram_limit)
            for view in ("legacy", "natural_prefix")}


def _effect_resolution(first, second):
    differences = {view: {key: abs(first[view][key] - second[view][key])
                          for key in ("I_before", "I_after", "interaction")}
                   for view in ("legacy", "natural_prefix")}
    maximum = max(value for group in differences.values() for value in group.values())
    return {"differences": differences, "observed_max": maximum,
            "engineering_target": CFG["numeric_resolution_nats_per_token"],
            "pass": maximum <= CFG["numeric_resolution_nats_per_token"],
            "scope": "FFN/window0/first8_old_probe_select_paired_labels",
            "strict_mathematical_error_bound": False}


def calibrate_local():
    _, v12, _, ledger = verified_run()
    all_views = views()
    with run_lock():
        model, optimizer, aliases, vram_limit = cuda_session(ledger)
        window = v12["train_build_windows"][0]
        first = build(model, optimizer, aliases, window, "first", ledger, vram_limit)
        second = build(model, optimizer, aliases, window, "repeat", ledger, vram_limit)
        comparison = compare_builds(first, second, "w0_same_process")
        if any(not x["equal"] for x in comparison["input_state"].values()):
            raise RuntimeError("REPEATED_BUILD_INPUT_STATE_POLLUTED")
        first_readout = _calibration_effects(model, optimizer, first, "w0_cal_first", all_views, ledger, vram_limit)
        second_readout = _calibration_effects(model, optimizer, second, "w0_cal_repeat", all_views, ledger, vram_limit)
        resolution = _effect_resolution(first_readout, second_readout)
        write_json(HERE / "results" / "calibration_local.json", {
            "comparison_path": "results/compare_w0_same_process.json",
            "frozen_first_sha256": sha_file(HERE / "frozen" / "window0_first.pt"),
            "frozen_repeat_sha256": sha_file(HERE / "frozen" / "window0_repeat.pt"),
            "first_effects": first_readout, "repeat_effects": second_readout,
            "numeric_resolution": resolution, "budget": ledger.data})
        if not resolution["pass"]:
            raise RuntimeError("NUMERIC_RESOLUTION_FAILED_SAME_PROCESS")
        return resolution


def calibrate_cross():
    _, v12, _, ledger = verified_run()
    if not read_json(HERE / "results" / "calibration_local.json")["numeric_resolution"]["pass"]:
        raise RuntimeError("LOCAL_CALIBRATION_NOT_ACCEPTED")
    all_views = views()
    with run_lock():
        model, optimizer, aliases, vram_limit = cuda_session(ledger)
        window = v12["train_build_windows"][0]
        cross = build(model, optimizer, aliases, window, "cross_process", ledger, vram_limit)
        first = torch.load(HERE / "frozen" / "window0_first.pt", map_location="cpu", weights_only=False)
        comparison = compare_builds(first, cross, "w0_cross_process")
        if any(not x["equal"] for x in comparison["input_state"].values()):
            raise RuntimeError("CROSS_PROCESS_INPUT_STATE_MISMATCH")
        cross_readout = _calibration_effects(model, optimizer, cross, "w0_cal_cross", all_views, ledger, vram_limit)
        local = read_json(HERE / "results" / "calibration_local.json")
        resolution = _effect_resolution(local["first_effects"], cross_readout)
        write_json(HERE / "results" / "calibration_cross.json", {
            "comparison_path": "results/compare_w0_cross_process.json",
            "cross_effects": cross_readout, "numeric_resolution": resolution, "budget": ledger.data})
        if not resolution["pass"]:
            raise RuntimeError("NUMERIC_RESOLUTION_FAILED_CROSS_PROCESS")
        return resolution


def _training_rows(window):
    mem = np.memmap(ROOT / "artifacts" / "train_tokens.u32", dtype=np.uint32, mode="r")
    seq = PARENT_CFG["runtime"]["sequence_length"]
    start = window["start_token"]
    return [{"view": "training_build_batch", "doc_id": f"build{window['window_id']}_seq{i}",
             "ids": np.asarray(mem[start+i*seq:start+(i+1)*seq+1], dtype=np.int64).tolist(),
             "token_count": seq, "source_doc_ids": [], "cluster_id": f"build{window['window_id']}"}
            for i in range(PARENT_CFG["runtime"]["effective_batch_size"])]


def measure():
    _, v12, _, ledger = verified_run()
    if not read_json(HERE / "results" / "calibration_cross.json")["numeric_resolution"]["pass"]:
        raise RuntimeError("CROSS_PROCESS_CALIBRATION_NOT_ACCEPTED")
    all_views = views()
    output = []
    with run_lock():
        model, optimizer, aliases, vram_limit = cuda_session(ledger)
        for window in v12["train_build_windows"]:
            resource_gate(ledger, f"before_window{window['window_id']}", vram_limit=vram_limit)
            tag = f"w{window['window_id']}"
            frozen = build(model, optimizer, aliases, window,
                           "first" if window["window_id"] == 0 else "main", ledger, vram_limit)
            load_original(model, optimizer, aliases)
            for target in TARGETS:
                for view in ("legacy", "natural_prefix", "new_probe_prefix", "new_probe_packed"):
                    result = evaluate_four(model, optimizer, frozen, target, all_views[view],
                                           view, tag, ledger, vram_limit)
                    output.append(result)
                paired_label_effect(tag, target)
                train = evaluate_four(model, optimizer, frozen, target, _training_rows(window),
                                      "training_build_batch", tag, ledger, vram_limit)
                train["in_sample_only"] = True
                for baseline in ("F00", "F01"):
                    gradient_diagnostic(model, optimizer, frozen, target, baseline,
                                        all_views["new_probe_prefix"], tag, ledger, vram_limit)
                geometry_diagnostic(model, optimizer, frozen, target,
                                    all_views["new_probe_prefix"], tag, ledger, vram_limit)
                if window["window_id"] == 0 and target == TARGETS[0]:
                    for baseline in ("F00", "F01"):
                        for view in ("legacy", "natural_prefix"):
                            finite_difference_diagnostic(model, optimizer, frozen, target,
                                 baseline, _calibration_rows(all_views, view), view, ledger, vram_limit)
            if window["window_id"] == 0:
                repeat = torch.load(HERE / "frozen" / "window0_repeat.pt", map_location="cpu", weights_only=False)
                repeated = {}
                for view in ("legacy", "natural_prefix", "new_probe_prefix"):
                    repeated[view] = evaluate_four(model, optimizer, repeat, TARGETS[0],
                            all_views[view], view, "w0_full_repeat", ledger, vram_limit)
                first = {(r["view"]): r for r in output if r["window_id"] == 0 and r["target"] == TARGETS[0]}
                differences = {view: {key: abs(first[view][key] - repeated[view][key])
                                      for key in ("I_before", "I_after", "interaction")}
                               for view in repeated}
                max_diff = max(v for group in differences.values() for v in group.values())
                write_json(HERE / "results" / "w0_full_repeatability.json", {
                    "differences": differences, "observed_max": max_diff,
                    "engineering_target": CFG["numeric_resolution_nats_per_token"],
                    "pass": max_diff <= CFG["numeric_resolution_nats_per_token"],
                    "scope": "FFN/window0/full_legacy_natural_new_probe_prefix"})
                if max_diff > CFG["numeric_resolution_nats_per_token"]:
                    raise RuntimeError("NUMERIC_RESOLUTION_FAILED_FULL_W0")
            write_json(HERE / "results" / f"window{window['window_id']}_complete.json",
                       {"status": "COMPLETE", "window_id": window["window_id"],
                        "source_hash": source_hash(), "budget": ledger.data})
        return output
