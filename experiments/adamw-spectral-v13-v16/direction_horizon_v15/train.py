"""ADAMW_DIRECTION_HORIZON_V15_20261002: temporal horizon test of no-SVD current-direction
controls in real 64-step continuation training. Parent trees are read only.

Per-step semantics (PROMPT_OPENCODE_V15.md section 2), all from the branch's own state:
    A_t = mhat_plus / (sqrt(vhat_plus) + eps)
    S: D = A_t;   F: D = N_A(g_clip);   P: D = N_A(g_clip / (sqrt(vhat_plus)+eps))
    target_next = (1-eta*wd)*target - eta * a * D
    other_next  = (1-eta*wd)*other  - eta * b * A_other
Implemented as native optimizer.step() followed by a verified displacement replacement,
so S,a=b=1 skips every correction and is the untouched native AdamW path.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
V12 = ROOT / "replay_audit_v12"
V14 = ROOT / "mechanism_screen_v14"
V13 = ROOT / "readiness_audit_v13"
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PCFG = json.loads((ROOT / "config" / "local_v1_lowmem.json").read_text(encoding="utf-8"))
TARGET = CFG["target"]
PEAK = 0.003
SOURCE_FILES = ("train.py", "test_v15.py", "config.json", "RUN_VERIFIED.ps1")
VRAM = [None]
# User-authorized amendment 2026-10-02 ("直接运行，不管GPU"): desktop apps hold ~6.4 GiB of the
# 8 GiB GPU; the inherited 1.5 GiB VRAM free reserve is unattainable while actual peak usage is
# ~0.4 GiB. For V15 only, the effective reserve is 0.2 GiB. Parent configs and every other
# resource gate (1.5 GiB RAM admission, 1.0 GiB RAM hard stop, RSS 4 GiB, per-step VRAM cap)
# are unchanged. Evidence: attempts/attempt3_vram_admission_blocked + this note.
VRAM_RESERVE_OVERRIDE_GIB = 0.2


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def append(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def sha(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        left = limit
        while left is None or left > 0:
            block = stream.read(min(left, 1048576) if left is not None else 1048576)
            if not block:
                break
            digest.update(block)
            if left is not None:
                left -= len(block)
    return digest.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_hash():
    return digest({name: sha(HERE / name) for name in SOURCE_FILES})


# ---------------------------------------------------------------- budget ledger

def ledger():
    value = read(HERE / "budget_ledger.json")
    if value["manifest_sha256"] != sha(HERE / "run_manifest.json"):
        raise RuntimeError("LEDGER_MANIFEST_MISMATCH")
    return value


def save_ledger(value):
    write(HERE / "budget_ledger.json", value)


def read_modify_write(fn):
    """All counter mutations go through fresh read-modify-write; no long-lived ledger
    dicts may regress the file (root cause of the attempt-4 accounting race)."""
    value = ledger()
    result = fn(value)
    save_ledger(value)
    return result


def count_forward(tokens, phase=""):
    def mutate(value):
        if value["token_forward_equivalents"] + tokens > CFG["max_token_forward_equivalents"]:
            raise RuntimeError("TOKEN_FORWARD_BUDGET_EXHAUSTED")
        value["token_forward_equivalents"] += tokens
        value["forward_attempts"] += 1
        value["last_phase"] = phase
    read_modify_write(mutate)


def begin_optimizer(phase=""):
    def mutate(value):
        if value["optimizer_step_attempts"] >= CFG["max_gpu_optimizer_step_attempts"]:
            raise RuntimeError("OPTIMIZER_STEP_BUDGET_EXHAUSTED")
        value["optimizer_step_attempts"] += 1
        value["last_phase"] = phase
    read_modify_write(mutate)


def complete_optimizer():
    def mutate(value):
        value["optimizer_step_completions"] += 1
    read_modify_write(mutate)


def bump_ledger(**deltas):
    def mutate(value):
        for key, delta in deltas.items():
            value[key] = value.get(key, 0) + delta
    read_modify_write(mutate)


def resource(phase, admission=False, vram_limit=None):
    value = ledger()
    ram = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    runtime = PCFG["runtime"]
    reserved = torch.cuda.memory_reserved(0) if torch.cuda.is_initialized() else None
    minimum = runtime["system_ram_start_min_gib" if admission else "system_ram_free_reserve_gib"] * 2 ** 30
    reason = ("RAM_BELOW_REQUIRED" if ram < minimum else
              "RSS_ABOVE_LIMIT" if rss > runtime["process_rss_soft_limit_gib"] * 2 ** 30 else
              "VRAM_ABOVE_LIMIT" if vram_limit is not None and reserved is not None and reserved > vram_limit else None)
    row = {"time": time.time(), "phase": phase, "status": "BLOCKED" if reason else
           "WARNING" if not admission and ram < runtime["system_ram_warning_gib"] * 2 ** 30 else "READY",
           "reason": reason, "available_ram_bytes": ram, "rss_bytes": rss,
           "cuda_reserved_bytes": reserved,
           "optimizer_step_attempts": value["optimizer_step_attempts"],
           "token_forward_equivalents": value["token_forward_equivalents"]}
    append(HERE / "logs" / "resources.jsonl", row)
    if reason:
        raise RuntimeError("RESOURCE_BLOCKED:" + reason)
    return row


# ---------------------------------------------------------------- branch specs

def branch_specs():
    rows = []
    for family in CFG["families"]:
        for a in CFG["a_grid"]:
            for b in CFG["b_grid"]:
                if a == 0.0 and family != "S":
                    continue
                rows.append({"id": f"{family}_a{a:g}_b{b:g}", "family": family, "a": a, "b": b})
    return rows


def specs_by_id():
    return {spec["id"]: spec for spec in branch_specs()}


def family_candidates(family):
    rows = [spec for spec in branch_specs() if spec["family"] == family]
    if family != "S":
        rows += [spec for spec in branch_specs() if spec["family"] == "S" and spec["a"] == 0.0]
    return rows


def tie_key(spec):
    return (0 if (spec["a"] == 1.0 and spec["b"] == 1.0) else 1,
            abs(spec["a"] - 1.0) + abs(spec["b"] - 1.0), spec["id"])


# ---------------------------------------------------------------- parent verify

def dependencies():
    binding = read(ROOT / "bindings.local.json")
    if Path(sys.executable).resolve() != Path(binding["python_executable"]).resolve():
        raise RuntimeError("PYTHON_BINDING_MISMATCH")
    data = read(ROOT / "artifacts" / "data_manifest.json")
    gate = read(ROOT / "results" / "local_gate.json")
    if gate["verdict"] == "PASS" or gate["next_action"] != "STOP":
        raise RuntimeError("UNEXPECTED_PARENT_GATE")
    base = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003"
    manifest = read(base / "manifest.json")
    if manifest["status"] != "SUCCESS":
        raise RuntimeError("PARENT_BASELINE_NOT_SUCCESS")
    files = ["artifacts/data_manifest.json", "artifacts/doc_roles.jsonl", "artifacts/eval_blocks.json",
             "artifacts/train_tokens.u32", "config/local_v1_lowmem.json", "bindings.local.json",
             "results/local_gate.json", "replay_audit_v12/core.py", "replay_audit_v12/evaluation_manifest.json",
             "replay_audit_v12/eval_views.jsonl", "mechanism_screen_v14/confirm_views.jsonl",
             "readiness_audit_v13/readiness.json", "readiness_audit_v13/run_manifest.json"]
    files = [p for p in files if (ROOT / p).is_file()]
    checkpoints = {}
    for step in (CFG["aux_checkpoint_step"], CFG["main_checkpoint_step"]):
        path = base / f"step{step}.pt"
        if not path.is_file():
            checkpoints[step] = None
            continue
        measured = sha(path)
        if manifest["checkpoint_hashes"].get(str(step)) != measured:
            raise RuntimeError(f"PARENT_CHECKPOINT_SHA_MISMATCH:{step}")
        checkpoints[step] = measured
    if checkpoints[CFG["main_checkpoint_step"]] is None:
        raise RuntimeError("MAIN_CHECKPOINT_MISSING")
    return {"protocol_id": CFG["protocol_id"], "parent_gate": [gate["verdict"], gate["next_action"]],
            "files": {p: sha(ROOT / p) for p in files},
            "checkpoint_sha256": checkpoints,
            "aux_checkpoint_available": checkpoints[CFG["aux_checkpoint_step"]] is not None,
            "data_window": [data["scanned_bytes"], data["scanned_sha256"]],
            "binding_python": binding["python_executable"]}


def build_confirm():
    from tokenizers import Tokenizer
    from replay_audit_v12.core import natural_prefix, normalized_role
    binding = read(ROOT / "bindings.local.json")
    data = read(ROOT / "artifacts" / "data_manifest.json")
    seq = PCFG["runtime"]["sequence_length"]
    excluded = set()
    for role in ("lr_select", "probe_select", "probe_confirm"):
        excluded.update(row["doc_id"] for row in read(ROOT / "artifacts" / "eval_blocks.json")[role])
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["view"] in ("legacy", "new_probe_prefix"):
                excluded.add(row["doc_id"])
    excluded.update(read(V12 / "evaluation_manifest.json")["new_probe_select_doc_ids"])
    excluded.update(read(V13 / "run_manifest.json").get("fixed_new_doc_ids", []))
    if (V14 / "confirm_views.jsonl").is_file():
        with (V14 / "confirm_views.jsonl").open(encoding="utf-8") as stream:
            excluded.update(json.loads(line)["doc_id"] for line in stream)
    roles = {row["doc_id"]: row["role"] for row in
             (json.loads(line) for line in (ROOT / "artifacts" / "doc_roles.jsonl").open(encoding="utf-8"))}
    source = Path(binding["source_data_path"])
    if sha(source, data["scanned_bytes"]) != data["scanned_sha256"]:
        raise RuntimeError("CONSUMED_DATA_PREFIX_CHANGED")
    seen, available, consumed = set(), {}, 0
    with source.open("rb") as stream:
        while consumed < data["scanned_bytes"]:
            line = stream.readline(data["scanned_bytes"] - consumed + 1)
            if not line or consumed + len(line) > data["scanned_bytes"]:
                raise RuntimeError("SOURCE_SCAN_BOUNDARY_CHANGED")
            consumed += len(line)
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            text, doc_id, role = normalized_role(row["text"])
            if not text or doc_id in seen:
                continue
            seen.add(doc_id)
            if role == "probe_confirm" and doc_id not in excluded:
                if roles.get(doc_id) != role:
                    raise RuntimeError("ROLE_RECORD_MISMATCH")
                available[doc_id] = text
    if consumed != data["scanned_bytes"] or len(available) < CFG["confirmation_documents"]:
        raise RuntimeError(f"CONFIRM_DOCUMENTS_INSUFFICIENT:{len(available)}")
    tokenizer = Tokenizer.from_file(str(Path(binding["tokenizer_path"]) / "tokenizer.json"))
    bos, eos = data["bos_token_id"], data["eos_token_id"]
    views = []
    for doc_id in sorted(available)[: CFG["confirmation_documents"]]:
        content = tokenizer.encode(available[doc_id], add_special_tokens=False).ids
        prefix = natural_prefix(content, bos, eos, seq)
        views.append({"doc_id": doc_id, "role": "probe_confirm", "ids": prefix,
                      "token_count": len(prefix) - 1,
                      "natural_boundary": len(content) <= seq - 1,
                      "exclusion_chain": "V1+V12+V13+V14 used doc IDs excluded"})
    return views, len(available)


def dev_views():
    ids = read(V12 / "evaluation_manifest.json")["new_probe_select_doc_ids"]
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if '"view": "new_probe_prefix"' in line]
    by_id = {row["doc_id"]: row for row in rows}
    if set(by_id) != set(ids) or len(ids) != 128:
        raise RuntimeError("DEV_VIEW_ID_MISMATCH")
    return [by_id[x] for x in ids]


def confirm_views():
    with (HERE / "confirm_views.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != CFG["confirmation_documents"]:
        raise RuntimeError("CONFIRM_VIEW_COUNT_CHANGED")
    return rows


def confirm_allowed():
    path = HERE / "selection_frozen.json"
    if not path.is_file():
        raise RuntimeError("CONFIRM_OPENED_BEFORE_SELECTION_FROZEN")
    frozen = read(path)
    if frozen["manifest_sha256"] != sha(HERE / "run_manifest.json"):
        raise RuntimeError("SELECTION_FROZEN_MANIFEST_MISMATCH")
    for rel, expected in frozen["dev_eval_files"].items():
        if sha(HERE / rel) != expected:
            raise RuntimeError("SELECTION_FROZEN_DEV_HASH_MISMATCH:" + rel)
    return frozen


# ---------------------------------------------------------------- prepare phase

def disk_free_gb():
    try:
        return psutil.disk_usage(str(HERE)).free / 2 ** 30
    except OSError:
        return None


def prepare():
    if (HERE / "run_manifest.json").exists():
        verify_manifest()
        return
    deps = dependencies()
    views, available = build_confirm()
    with (HERE / "confirm_views.jsonl").open("w", encoding="utf-8") as stream:
        for row in views:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    import csv
    with (HERE / "confirm_documents.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["doc_id", "token_count", "natural_boundary"])
        writer.writeheader()
        writer.writerows([{k: row[k] for k in ("doc_id", "token_count", "natural_boundary")} for row in views])
    estimate = 0.097 * (len(branch_specs()) + 8)
    free = disk_free_gb()
    if free is not None and free < estimate:
        raise RuntimeError(f"DISK_SPACE_INSUFFICIENT: need~{estimate:.1f}GB free={free:.1f}GB")
    write(HERE / "run_manifest.json", {
        "protocol_id": CFG["protocol_id"], "source_hash": source_hash(),
        "config_sha256": sha(HERE / "config.json"),
        "parent_dependencies_sha256": digest(deps),
        "confirm_views_sha256": sha(HERE / "confirm_views.jsonl"),
        "confirm_documents_sha256": sha(HERE / "confirm_documents.csv"),
        "confirm_ids_sha256": digest([row["doc_id"] for row in views]),
        "confirm_available_before_seal": available,
        "confirm_selection_rule": "first 128 lexicographic unused probe_confirm doc_ids after V1/V12/V13/V14 exclusion",
        "confirm_seal": "dev and select phases never read confirm_views.jsonl or confirm eval rows",
        "aux_checkpoint_available": deps["aux_checkpoint_available"],
        "aux_timepoint_preregistered": "step128 with frozen configs only; never re-tuned",
        "branch_specs": branch_specs(), "disk_estimate_gb": estimate,
        "budget": {"optimizer_step_attempts": CFG["max_gpu_optimizer_step_attempts"],
                   "token_forward_equivalents": CFG["max_token_forward_equivalents"]},
        "created_at": time.time()})
    write(HERE / "budget_ledger.json", {
        "manifest_sha256": sha(HERE / "run_manifest.json"), "token_forward_equivalents": 0,
        "forward_attempts": 0, "backward_attempts": 0, "optimizer_step_attempts": 0,
        "optimizer_step_completions": 0, "gpu_seconds": 0.0, "train_effective_tokens": 0,
        "train_seconds": 0.0, "eval_seconds": 0.0, "direction_seconds": 0.0,
        "smoke_step_attempts": 0, "started_at": time.time()})


def verify_manifest():
    manifest = read(HERE / "run_manifest.json")
    for key, actual in (("source_hash", source_hash()), ("config_sha256", sha(HERE / "config.json")),
                        ("confirm_views_sha256", sha(HERE / "confirm_views.jsonl")),
                        ("confirm_documents_sha256", sha(HERE / "confirm_documents.csv"))):
        if manifest[key] != actual:
            raise RuntimeError("V15_MANIFEST_CHANGED:" + key)
    if digest(dependencies()) != manifest["parent_dependencies_sha256"]:
        raise RuntimeError("V15_PARENT_DEPENDENCY_CHANGED")
    return manifest


# ---------------------------------------------------------------- direction step core

def n_a(x, a_norm):
    xn = float(torch.linalg.vector_norm(x.double()))
    if xn == 0.0 or a_norm == 0.0:
        return torch.zeros_like(x), {"zero_norm_fallback": True, "input_norm": xn, "A_norm": a_norm}
    out = (x.double() * (a_norm / xn)).to(x.dtype)
    return out, {"zero_norm_fallback": False, "input_norm": xn, "A_norm": a_norm}


def apply_step(model, opt, family, a, b, phase, eta, timing, target_name=TARGET):
    """One optimizer step: native opt.step() then verified displacement replacement."""
    named = dict(model.named_parameters())
    beta1, beta2 = PCFG["baseline"]["betas"]
    eps = PCFG["baseline"]["eps"]
    A_map, meta = {}, {"family_used": family}
    for name, param in named.items():
        if param.grad is None:
            A_map[name] = None
            continue
        state = opt.state[param]
        if "step" in state:
            step_plus = int(state["step"].item() if torch.is_tensor(state["step"]) else state["step"]) + 1
            m_prev, v_prev = state["exp_avg"], state["exp_avg_sq"]
        else:
            step_plus = 1
            m_prev = torch.zeros_like(param)
            v_prev = torch.zeros_like(param)
        m_plus = m_prev * beta1 + param.grad * (1 - beta1)
        v_plus = v_prev * beta2 + param.grad.square() * (1 - beta2)
        A_map[name] = (m_plus / (1 - beta1 ** step_plus)) / (
            (v_plus / (1 - beta2 ** step_plus)).sqrt() + eps)
        if name == target_name:
            meta["target_moments_step_plus"] = step_plus
            meta["_target_v_plus"] = v_plus
    target_A = A_map[target_name]
    if target_A is None:
        raise RuntimeError("TARGET_GRADIENT_MISSING")
    meta["A_norm"] = float(torch.linalg.vector_norm(target_A.double()))
    if family in ("F", "P"):
        g_clip = named[target_name].grad.detach()
        if family == "P":
            step_plus = meta["target_moments_step_plus"]
            x = g_clip / ((meta["_target_v_plus"] / (1 - beta2 ** step_plus)).sqrt() + eps)
        else:
            x = g_clip
        D, direction_meta = n_a(x, meta["A_norm"])
        meta.update(direction_meta)
    else:
        D = target_A
        meta.update({"zero_norm_fallback": False, "input_norm": meta["A_norm"]})
    if not torch.isfinite(D).all() or not torch.isfinite(target_A).all():
        raise RuntimeError("NON_FINITE_DIRECTION")
    meta.pop("_target_v_plus", None)
    resource(f"{phase}_before_optstep", vram_limit=VRAM[0])
    begin_optimizer(phase)
    started = time.perf_counter()
    opt.step()
    complete_optimizer()
    torch.cuda.synchronize()
    timing["optimizer_seconds"] = time.perf_counter() - started
    with torch.no_grad():
        for name, param in named.items():
            A_p = A_map[name]
            if A_p is None:
                continue
            if name == target_name:
                if not (family == "S" and a == 1.0):
                    param.add_(eta * (target_A - a * D))
            elif b != 1.0:
                param.add_(eta * (1.0 - b) * A_p)
    torch.cuda.synchronize()
    meta["D_norm"] = float(torch.linalg.vector_norm(D.double()))
    meta["match_multiplier"] = meta["A_norm"] / meta["input_norm"] if meta["input_norm"] else None
    return meta


def family_for_step(spec, branch_step, one_shot):
    if one_shot:
        return spec["family"] if branch_step == 1 else "S"
    return spec["family"]


# ---------------------------------------------------------------- evaluation

def eval_path(branch_label, horizon, split):
    return HERE / "raw" / "eval" / f"{branch_label}__H{horizon}__{split}.jsonl"


def evaluate_split(model, rows, split, phase, branch_label, horizon):
    from pilot_core import loss_sum_count
    if split == "confirm":
        confirm_allowed()
    path = eval_path(branch_label, horizon, split)
    old = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    if [x["doc_id"] for x in old] != [x["doc_id"] for x in rows[: len(old)]]:
        raise RuntimeError("PARTIAL_EVAL_IDENTITY_MISMATCH:" + str(path))
    if len(old) == len(rows):
        return old
    model.eval()
    started = time.perf_counter()
    for row in rows[len(old):]:
        resource(f"{phase}_eval_doc", vram_limit=VRAM[0])
        ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None, :]
        count_forward(row["token_count"], phase)
        with torch.no_grad():
            logits = model(ids[:, :-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device="cuda:0"))
        if count != row["token_count"]:
            raise RuntimeError("MASK_OR_SHIFT_CHANGED")
        append(path, {"branch": branch_label, "horizon": horizon, "split": split,
                      "doc_id": row["doc_id"], "loss_sum": float(loss), "token_count": int(count),
                      "manifest_sha256": sha(HERE / "run_manifest.json"),
                      "evaluation_precision": "fp32_no_tf32"})
    torch.cuda.synchronize()
    bump_ledger(eval_seconds=time.perf_counter() - started)
    return old


def eval_rows(branch_label, horizon, split):
    return [json.loads(line) for line in eval_path(branch_label, horizon, split).open(encoding="utf-8")]


def token_weighted(rows):
    return sum(r["loss_sum"] for r in rows) / sum(r["token_count"] for r in rows)


# ---------------------------------------------------------------- branch runner

def save_branch_state(path, model, opt, label, spec, branch_step, cursor, total_tokens,
                      one_shot_applied, payload_step):
    state = {"model": model.state_dict(),
             "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
             "optimizer": copy.deepcopy(opt.state_dict()),
             "label": label, "spec": spec, "branch_step": branch_step, "cursor": cursor,
             "total_effective_tokens": total_tokens, "one_shot_applied": one_shot_applied,
             "payload_step": payload_step, "manifest_sha256": sha(HERE / "run_manifest.json"),
             "cpu_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_branch_state(path, model, opt, label, payload_step):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["label"] != label or state["payload_step"] != payload_step or
            state["manifest_sha256"] != sha(HERE / "run_manifest.json")):
        raise RuntimeError("BRANCH_STATE_IDENTITY_MISMATCH")
    model.load_state_dict(state["model"])
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(state["buffers"][name].to(buffer.device))
    opt.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return state


def run_branch(model, opt, aliases, mem, spec, payload_step, splits_at_horizon, one_shot=False,
               phase="dev", steps=None, label_prefix=""):
    from pilot_core import load_checkpoint, loss_sum_count, schedule_lr, training_batch
    steps = steps or CFG["horizon"]
    label = f"{label_prefix}{spec['id']}" + ("_oneshot" if one_shot else "")
    checkpoint = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003" / f"step{payload_step}.pt"
    rolling = HERE / "checkpoints" / f"{label}_rolling.pt"
    final = HERE / "checkpoints" / f"{label}_h64.pt"
    if final.exists():
        return label
    if rolling.exists():
        state = load_branch_state(rolling, model, opt, label, payload_step)
        branch_step = state["branch_step"]
        one_shot_applied = state["one_shot_applied"]
        cursor = state["cursor"]
        total_tokens = state["total_effective_tokens"]
    else:
        payload = load_checkpoint(checkpoint, model, opt,
                                  config_hash=sha(ROOT / "config" / "local_v1_lowmem.json"),
                                  data_hash=read(ROOT / "artifacts" / "data_manifest.json")["data_hash"],
                                  aliases=aliases)
        if payload["successful_optimizer_step"] != payload_step:
            raise RuntimeError("CHECKPOINT_STEP_CHANGED")
        branch_step, one_shot_applied = 0, False
        cursor = payload["data_cursor"]
        total_tokens = payload["total_effective_tokens"]
    step_log = HERE / "raw" / "steps" / f"{label}.jsonl"
    done = ({json.loads(line)["branch_step"] for line in step_log.open(encoding="utf-8")}
            if step_log.exists() else set())
    seq = PCFG["runtime"]["sequence_length"]
    mb = PCFG["runtime"]["micro_batch_size"]
    accum = PCFG["runtime"]["accumulation_steps"]
    bf16 = torch.cuda.is_bf16_supported()
    docs = {"dev": dev_views()}
    while branch_step < steps:
        branch_step += 1
        global_step = payload_step + branch_step
        eta = schedule_lr(PEAK, global_step, PCFG)
        for group in opt.param_groups:
            group["lr"] = eta
        family = family_for_step(spec, branch_step, one_shot)
        resource(f"{phase}_{label}_s{branch_step}_before", vram_limit=VRAM[0])
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss_total, valid_total = 0.0, 0
        for _ in range(accum):
            resource(f"{phase}_{label}_fwd", vram_limit=VRAM[0])
            ids, cursor = training_batch(mem, cursor, mb, seq, "cuda:0")
            lengths = torch.full((ids.shape[0],), ids.shape[1], device="cuda:0", dtype=torch.long)
            count_forward(mb * (ids.shape[1] - 1), phase)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(ids[:, :-1], use_cache=False).logits
                loss_sum, count = loss_sum_count(logits, ids, lengths)
            if not torch.isfinite(loss_sum):
                raise RuntimeError("NON_FINITE_TRAIN_LOSS")
            loss_sum.backward()
            bump_ledger(backward_attempts=1)
            valid_total += count
            loss_total += float(loss_sum.detach())
        for param in model.parameters():
            if param.grad is not None:
                param.grad.div_(valid_total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), PCFG["baseline"]["global_grad_clip_norm"])
        bump_ledger(train_effective_tokens=valid_total)
        timing = {"forward_backward_seconds": time.perf_counter() - t0}
        tdir = time.perf_counter()
        meta = apply_step(model, opt, family, spec["a"], spec["b"],
                          f"{phase}_{label}", eta, timing)
        timing["direction_and_correction_seconds"] = time.perf_counter() - tdir
        bump_ledger(train_seconds=timing["forward_backward_seconds"],
                    direction_seconds=timing["direction_and_correction_seconds"])
        opt.zero_grad(set_to_none=True)
        one_shot_applied = one_shot_applied or (one_shot and branch_step == 1)
        total_tokens += valid_total
        if branch_step not in done:
            append(step_log, {"branch": label, "spec": {"family": spec["family"], "a": spec["a"], "b": spec["b"]},
                              "branch_step": branch_step, "global_step": global_step, "family_used": family,
                              "one_shot": one_shot, "eta": eta,
                              "eta_wd": eta * PCFG["baseline"]["matrix_weight_decay"],
                              "loss_sum": loss_total, "token_count": valid_total,
                              "nll": loss_total / valid_total, "cursor": cursor,
                              "total_effective_tokens": total_tokens,
                              "moments_step_plus": meta["target_moments_step_plus"],
                              "A_norm": meta["A_norm"], "D_norm": meta["D_norm"],
                              "zero_norm_fallback": meta["zero_norm_fallback"],
                              "match_multiplier": meta["match_multiplier"], **timing,
                              "manifest_sha256": sha(HERE / "run_manifest.json")})
        resource(f"{phase}_{label}_s{branch_step}_after", vram_limit=VRAM[0])
        if branch_step in CFG["eval_horizons"]:
            for split in splits_at_horizon.get(branch_step, []):
                rows = docs[split] if split == "dev" else confirm_views()
                evaluate_split(model, rows, split, phase, label, branch_step)
            model.train()
        if branch_step == 1:
            snapshot = HERE / "checkpoints" / f"{label}_h1_snapshot.pt"
            if not snapshot.exists():
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"params": {n: p.detach().cpu().clone() for n, p in model.named_parameters()},
                            "buffers": {n: b.detach().cpu().clone() for n, b in model.named_buffers()},
                            "label": label, "manifest_sha256": sha(HERE / "run_manifest.json")}, snapshot)
        if branch_step % CFG["rolling_checkpoint_interval"] == 0 or branch_step == steps:
            save_branch_state(rolling, model, opt, label, spec, branch_step, cursor, total_tokens,
                              one_shot_applied, payload_step)
            if branch_step == steps and steps == CFG["horizon"] and not phase.startswith("smoke"):
                os.replace(rolling, final)
    return label


# ---------------------------------------------------------------- GPU context

class GpuContext:
    def __enter__(self):
        resource("before_cuda_model_init", admission=True)
        if (ROOT / "artifacts" / "run.lock").exists() or (HERE / "run.lock").exists():
            raise RuntimeError("RUN_LOCK_PRESENT")
        (HERE / "run.lock").write_text(str(os.getpid()), encoding="utf-8")
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA_UNAVAILABLE")
            free, _ = torch.cuda.mem_get_info(0)
            runtime = PCFG["runtime"]
            vram_limit = min(runtime["vram_soft_limit_gib"] * 2 ** 30,
                             free - VRAM_RESERVE_OVERRIDE_GIB * 2 ** 30)
            if vram_limit <= 0:
                raise RuntimeError("VRAM_RESERVE_FAILED")
            VRAM[0] = vram_limit
            torch.set_num_threads(runtime["cpu_threads"])
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            from pilot_core import model_and_optimizer
            binding = read(ROOT / "bindings.local.json")
            self.model, self.opt, self.aliases, _ = model_and_optimizer(PCFG, binding, PEAK, "cuda:0")
            self.mem = np.memmap(ROOT / "artifacts" / "train_tokens.u32", dtype=np.uint32, mode="r")
            resource("after_model_init", vram_limit=vram_limit)
        except BaseException:
            VRAM[0] = None
            (HERE / "run.lock").unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, *exc):
        VRAM[0] = None
        (HERE / "run.lock").unlink(missing_ok=True)
        return False


def load_h1_for_eval(model, label):
    snapshot = torch.load(HERE / "checkpoints" / f"{label}_h1_snapshot.pt",
                          map_location="cpu", weights_only=False)
    if snapshot["label"] != label or snapshot["manifest_sha256"] != sha(HERE / "run_manifest.json"):
        raise RuntimeError("H1_SNAPSHOT_IDENTITY_MISMATCH")
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(snapshot["params"][name].to(param.device, param.dtype))
        for name, buffer in model.named_buffers():
            buffer.copy_(snapshot["buffers"][name].to(buffer.device))


def load_h64_for_eval(model, opt, label, spec, payload_step):
    state = load_branch_state(HERE / "checkpoints" / f"{label}_h64.pt", model, opt, label, payload_step)
    if state["spec"] != spec or state["branch_step"] != CFG["horizon"]:
        raise RuntimeError("H64_ENDPOINT_IDENTITY_MISMATCH")


def snapshot_tree(folder):
    return {p for p in folder.rglob("*") if p.is_file()} if folder.exists() else set()


def remove_new_files(before, folders):
    for folder in folders:
        for path in snapshot_tree(folder) - before.get(folder, set()):
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------- phases

def phase_smoke():
    verify_manifest()
    if (HERE / "smoke_result.json").is_file():
        print(json.dumps(read(HERE / "smoke_result.json")))
        return
    watched = {f: snapshot_tree(HERE / f) for f in ("checkpoints", "raw")}
    docs = {"dev": dev_views()}
    with GpuContext() as ctx:
        from pilot_core import loss_sum_count, schedule_lr, training_batch
        attempts_before = ledger()["optimizer_step_attempts"]
        spec = next(s for s in branch_specs() if s["id"] == "S_a1_b1")
        # 1) branch S,a=b=1 four steps continuous (identity path: no corrections applied).
        run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec, CFG["main_checkpoint_step"],
                   {}, one_shot=False, phase="smoke", steps=4)
        w_continuous = {n: p.detach().cpu().clone() for n, p in ctx.model.named_parameters()}
        # 2) pure native reference loop, no correction code involved.
        payload = torch.load(ROOT / "artifacts/checkpoints/seed3407_lr0.003/step256.pt",
                             map_location="cpu", weights_only=False)
        from pilot_core import load_checkpoint
        load_checkpoint(ROOT / "artifacts/checkpoints/seed3407_lr0.003/step256.pt", ctx.model, ctx.opt,
                        config_hash=sha(ROOT / "config/local_v1_lowmem.json"),
                        data_hash=read(ROOT / "artifacts/data_manifest.json")["data_hash"],
                        aliases=ctx.aliases)
        cursor = payload["data_cursor"]
        for k in range(257, 261):
            eta = schedule_lr(PEAK, k, PCFG)
            for group in ctx.opt.param_groups:
                group["lr"] = eta
            ctx.model.train()
            ctx.opt.zero_grad(set_to_none=True)
            valid_total = 0
            for _ in range(PCFG["runtime"]["accumulation_steps"]):
                resource("smoke_native_fwd", vram_limit=VRAM[0])
                ids, cursor = training_batch(ctx.mem, cursor, PCFG["runtime"]["micro_batch_size"],
                                             PCFG["runtime"]["sequence_length"], "cuda:0")
                lengths = torch.full((ids.shape[0],), ids.shape[1], device="cuda:0", dtype=torch.long)
                count_forward(2 * 256, "smoke_native")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    logits = ctx.model(ids[:, :-1], use_cache=False).logits
                    loss_sum, count = loss_sum_count(logits, ids, lengths)
                loss_sum.backward()
                bump_ledger(backward_attempts=1)
                valid_total += count
            for param in ctx.model.parameters():
                if param.grad is not None:
                    param.grad.div_(valid_total)
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(), PCFG["baseline"]["global_grad_clip_norm"])
            resource("smoke_native_before_optstep", vram_limit=VRAM[0])
            begin_optimizer("smoke_native")
            ctx.opt.step()
            complete_optimizer()
            ctx.opt.zero_grad(set_to_none=True)
        native = {n: p.detach().cpu().clone() for n, p in ctx.model.named_parameters()}
        scale = max(float(v.abs().max()) for v in native.values())
        parity = max(float((w_continuous[n] - native[n]).abs().max()) for n in native)
        tolerance = PCFG["numerics"]["fp32_atol"] + PCFG["numerics"]["fp32_rtol"] * scale
        if parity > tolerance:
            raise RuntimeError(f"SMOKE_PARITY_FAIL:{parity}>{tolerance}")
        # 3) resume equality: 2 steps, reload from rolling state, 2 more.
        run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec, CFG["main_checkpoint_step"],
                   {}, one_shot=False, phase="smoke", steps=2)
        run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec, CFG["main_checkpoint_step"],
                   {}, one_shot=False, phase="smoke", steps=4)
        w_resume = {n: p.detach().cpu().clone() for n, p in ctx.model.named_parameters()}
        resume_err = max(float((w_continuous[n] - w_resume[n]).abs().max()) for n in w_continuous)
        if resume_err > tolerance:
            raise RuntimeError(f"SMOKE_RESUME_FAIL:{resume_err}>{tolerance}")
        # 4) one-shot trigger: family at branch step 1 only.
        spec_f = next(s for s in branch_specs() if s["id"] == "F_a1_b1")
        run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec_f, CFG["main_checkpoint_step"],
                   {}, one_shot=True, phase="smoke", steps=2)
        rows = [json.loads(line) for line in (HERE / "raw/steps/F_a1_b1_oneshot.jsonl").open(encoding="utf-8")]
        if [r["family_used"] for r in sorted(rows, key=lambda r: r["branch_step"])] != ["F", "S"]:
            raise RuntimeError("ONE_SHOT_TRIGGER_WRONG")
        # 5) eval non-mutation of params/RNG/optimizer.
        rng_before = (torch.get_rng_state().clone(), random.getstate())
        params_before = {n: p.detach().clone() for n, p in ctx.model.named_parameters()}
        opt_before = copy.deepcopy(ctx.opt.state_dict())
        evaluate_split(ctx.model, docs["dev"][:3], "dev", "smoke_evalcheck", "smoke_evalcheck", 0)
        for n, p in ctx.model.named_parameters():
            if not torch.equal(p, params_before[n]):
                raise RuntimeError("EVAL_MUTATED_PARAMS")
        if not torch.equal(torch.get_rng_state(), rng_before[0]) or random.getstate() != rng_before[1]:
            raise RuntimeError("EVAL_MUTATED_RNG")
        for key, fields in opt_before["state"].items():
            for field, before_value in fields.items():
                now = ctx.opt.state_dict()["state"][key][field]
                if torch.is_tensor(before_value):
                    if not torch.equal(before_value, now):
                        raise RuntimeError("EVAL_MUTATED_OPTIMIZER")
                elif before_value != now:
                    raise RuntimeError("EVAL_MUTATED_OPTIMIZER")
        # 6) repeated evaluation of the identical state (numeric resolution probe).
        evaluate_split(ctx.model, docs["dev"][:16], "dev", "smoke_repeat1", "smoke_repeat1", 0)
        evaluate_split(ctx.model, docs["dev"][:16], "dev", "smoke_repeat2", "smoke_repeat2", 0)
        n1, n2 = token_weighted(eval_rows("smoke_repeat1", 0, "dev")), token_weighted(eval_rows("smoke_repeat2", 0, "dev"))
        def _finish(value):
            value["smoke_step_attempts"] = value["optimizer_step_attempts"] - attempts_before
            return value["smoke_step_attempts"], value["optimizer_step_attempts"]
        smoke_attempts, attempts_total = read_modify_write(_finish)
        result = {"status": "PASS", "native_parity_max_abs": parity, "parity_tolerance": tolerance,
                  "resume_max_abs": resume_err, "one_shot_families": ["F", "S"],
                  "eval_repeat_nll": [n1, n2], "eval_repeat_spread": abs(n1 - n2),
                  "smoke_step_attempts": smoke_attempts,
                  "optimizer_step_attempts_total": attempts_total}
        write(HERE / "smoke_result.json", result)
        remove_new_files(watched, [HERE / "checkpoints", HERE / "raw"])
        print(json.dumps(result))


def phase_dev():
    verify_manifest()
    if (HERE / "selection_frozen.json").exists():
        raise RuntimeError("DEVELOPMENT_ALREADY_FROZEN")
    with GpuContext() as ctx:
        evaluate_split(ctx.model, dev_views(), "dev", "dev_h0", "H0_step256", 0)
        horizons = {h: ["dev"] for h in CFG["eval_horizons"]}
        for spec in branch_specs():
            run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec,
                       CFG["main_checkpoint_step"], horizons, one_shot=False, phase="dev")


def phase_select():
    verify_manifest()
    if (HERE / "selection_frozen.json").exists():
        raise RuntimeError("SELECTION_ALREADY_FROZEN")
    scores, h1 = {}, {}
    for spec in branch_specs():
        rows = eval_rows(spec["id"], CFG["horizon"], "dev")
        if len(rows) != 128:
            raise RuntimeError("DEV_EVAL_INCOMPLETE:" + spec["id"])
        scores[spec["id"]] = token_weighted(rows)
        h1[spec["id"]] = token_weighted(eval_rows(spec["id"], 1, "dev"))
    selected = {}
    for family in CFG["families"]:
        pool = family_candidates(family)
        best = min(scores[spec["id"]] for spec in pool)
        ties = [spec for spec in pool if scores[spec["id"]] == best]
        selected[family] = sorted(ties, key=tie_key)[0]["id"]
    write(HERE / "dev_scores.json", {"h64_dev_nll": scores, "h1_dev_nll": h1})
    dev_files = {str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "raw" / "eval").glob("*.jsonl"))}
    write(HERE / "selection_frozen.json", {
        "manifest_sha256": sha(HERE / "run_manifest.json"),
        "dev_eval_files": dev_files,
        "dev_scores_sha256": sha(HERE / "dev_scores.json"), "selected": selected,
        "ordinary_adamw": "S_a1_b1", "created_at": time.time(), "confirmation_opened": False})


def phase_confirm():
    manifest = verify_manifest()
    frozen = confirm_allowed()
    if not frozen.get("confirmation_opened"):
        frozen["confirmation_opened"] = True
        write(HERE / "selection_frozen.json", frozen)
    registry = specs_by_id()
    main_ids = list(dict.fromkeys([frozen["selected"]["S"], frozen["selected"]["F"],
                                   frozen["selected"]["P"], "S_a1_b1"]))
    horizons = {1: ["dev", "confirm"], 16: ["dev", "confirm"], 64: ["dev", "confirm"]}
    with GpuContext() as ctx:
        evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_h0", "H0_step256", 0)
        for cid in main_ids:
            load_h1_for_eval(ctx.model, cid)
            evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_main_h1", cid, 1)
            load_h64_for_eval(ctx.model, ctx.opt, cid, registry[cid], CFG["main_checkpoint_step"])
            evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_main_h64", cid, 64)
        for family in ("F", "P"):
            spec = registry[frozen["selected"][family]]
            matched = f"S_a{spec['a']:g}_b{spec['b']:g}"
            load_h64_for_eval(ctx.model, ctx.opt, matched, registry[matched], CFG["main_checkpoint_step"])
            evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_matched_s",
                           matched + "__matchedS", 64)
        for family in ("F", "P"):
            spec = registry[frozen["selected"][family]]
            label = run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, spec,
                               CFG["main_checkpoint_step"], horizons, one_shot=True, phase="oneshot")
            fresh_h1, os_h1 = eval_rows(spec["id"], 1, "dev"), eval_rows(label, 1, "dev")
            if [r["doc_id"] for r in fresh_h1] != [r["doc_id"] for r in os_h1]:
                raise RuntimeError("ONESHOT_H1_PAIRING_MISMATCH")
            delta = abs(token_weighted(fresh_h1) - token_weighted(os_h1))
            append(HERE / "logs" / "oneshot_h1_reuse.jsonl",
                   {"branch": label, "fresh": spec["id"], "h1_dev_nll_delta": delta,
                    "reuse_verified": delta <= PCFG["numerics"]["fp32_atol"]})
        if manifest["aux_checkpoint_available"]:
            evaluate_split(ctx.model, confirm_views(), "confirm", "confirm_h0_aux", "H0_step128", 0)
            evaluate_split(ctx.model, dev_views(), "dev", "dev_h0_aux", "H0_step128", 0)
            for cid in main_ids:
                run_branch(ctx.model, ctx.opt, ctx.aliases, ctx.mem, registry[cid],
                           CFG["aux_checkpoint_step"], horizons, one_shot=False,
                           phase="aux", label_prefix="aux_")
        else:
            write(HERE / "aux_status.json", {"status": "NOT_RUN", "reason": "step128 checkpoint absent"})


def paired_bootstrap(rows_a, rows_b):
    if [r["doc_id"] for r in rows_a] != [r["doc_id"] for r in rows_b]:
        raise RuntimeError("CONFIRM_PAIRING_CHANGED")
    loss_a = np.array([r["loss_sum"] for r in rows_a])
    loss_b = np.array([r["loss_sum"] for r in rows_b])
    tok_a = np.array([r["token_count"] for r in rows_a])
    tok_b = np.array([r["token_count"] for r in rows_b])
    point = loss_a.sum() / tok_a.sum() - loss_b.sum() / tok_b.sum()
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    draws = np.empty(CFG["bootstrap_repetitions"])
    for i in range(CFG["bootstrap_repetitions"]):
        idx = rng.integers(0, len(rows_a), len(rows_a))
        draws[i] = loss_a[idx].sum() / tok_a[idx].sum() - loss_b[idx].sum() / tok_b[idx].sum()
    return {"point": point,
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "ci97p5": [float(np.quantile(draws, 0.0125)), float(np.quantile(draws, 0.9875))]}


def phase_finalize():
    manifest = verify_manifest()
    frozen = confirm_allowed()
    registry = specs_by_id()
    ledger_value = ledger()
    main_ids = list(dict.fromkeys([frozen["selected"]["S"], frozen["selected"]["F"],
                                   frozen["selected"]["P"], "S_a1_b1"]))
    nll_h64 = {cid: token_weighted(eval_rows(cid, 64, "confirm")) for cid in main_ids}
    nll_h1 = {cid: token_weighted(eval_rows(cid, 1, "confirm")) for cid in main_ids}
    s_id = frozen["selected"]["S"]
    effects = {}
    for family in ("F", "P"):
        cid = frozen["selected"][family]
        effects[family] = {"frozen": cid, "delta_h64": nll_h64[s_id] - nll_h64[cid],
                           "delta_h1": nll_h1[s_id] - nll_h1[cid]}
        effects[family].update(paired_bootstrap(eval_rows(s_id, 64, "confirm"),
                                                eval_rows(cid, 64, "confirm")))
    oneshot = {}
    for family in ("F", "P"):
        spec = registry[frozen["selected"][family]]
        label = spec["id"] + "_oneshot"
        matched = f"S_a{spec['a']:g}_b{spec['b']:g}"
        oneshot[family] = {"branch": label, "matched_S": matched,
                           "h64_confirm": token_weighted(eval_rows(label, 64, "confirm")),
                           "h16_confirm": token_weighted(eval_rows(label, 16, "confirm")),
                           "matched_S_h64_confirm": token_weighted(eval_rows(matched + "__matchedS", 64, "confirm")),
                           "sustained_h64_confirm": nll_h64[frozen["selected"][family]],
                           "sustained_h16_dev": token_weighted(eval_rows(frozen["selected"][family], 16, "dev")),
                           "one_shot_h16_dev": token_weighted(eval_rows(label, 16, "dev"))}
    if (HERE / "aux_status.json").is_file():
        aux = read(HERE / "aux_status.json")
    else:
        aux_nll = {}
        for cid in main_ids:
            label = "aux_" + cid
            aux_nll[cid] = {"h64_confirm": token_weighted(eval_rows(label, 64, "confirm")),
                            "h64_dev": token_weighted(eval_rows(label, 64, "dev")),
                            "h1_confirm": token_weighted(eval_rows(label, 1, "confirm"))}
        aux = {"status": "MEASURED", "nll": aux_nll,
               "aux_delta_F_h64": aux_nll[s_id]["h64_confirm"] - aux_nll[frozen["selected"]["F"]]["h64_confirm"],
               "aux_delta_P_h64": aux_nll[s_id]["h64_confirm"] - aux_nll[frozen["selected"]["P"]]["h64_confirm"]}
    smoke = read(HERE / "smoke_result.json") if (HERE / "smoke_result.json").is_file() else {"status": "NOT_RUN"}
    repeat = abs(smoke["eval_repeat_nll"][0] - smoke["eval_repeat_nll"][1]) if smoke.get("eval_repeat_nll") else 0.0
    resolution = max(CFG["numeric_resolution_nats_per_token"], repeat)
    clear_reverse = (aux["status"] == "MEASURED" and
                     min(aux.get("aux_delta_F_h64", 0.0), aux.get("aux_delta_P_h64", 0.0)) <
                     -CFG["delta_branch_nats_per_token"])
    passed = {f: (effects[f]["delta_h64"] > CFG["delta_branch_nats_per_token"] and
                  effects[f]["ci97p5"][0] > resolution) for f in ("F", "P")}
    if any(passed.values()) and not clear_reverse:
        verdict = "BRANCH_SIGNAL_WITHIN_GRID"
    elif any(effects[f]["delta_h1"] > resolution for f in ("F", "P")):
        verdict = "ONE_STEP_ONLY"
    elif all(effects[f]["ci95"][1] < CFG["delta_branch_nats_per_token"] for f in ("F", "P")):
        verdict = "NO_BRANCH_GAIN_WITHIN_GRID"
    else:
        verdict = "INCONCLUSIVE"
    scale_boundary = {"S": registry[s_id]["a"] in (0.0, 2.0) or registry[s_id]["b"] != 1.0,
                      "F": registry[frozen["selected"]["F"]]["a"] in (0.0, 2.0),
                      "P": registry[frozen["selected"]["P"]]["a"] in (0.0, 2.0)}
    result = {"protocol_id": CFG["protocol_id"],
              "parent_gate": ["BASELINE_TUNING_UNRESOLVED", "STOP"],
              "manifest_sha256": sha(HERE / "run_manifest.json"), "verdict": verdict,
              "selection": frozen["selected"], "effects": effects,
              "confirm_nll_h64": nll_h64, "confirm_nll_h1": nll_h1,
              "ordinary_adamw": {"frozen": "S_a1_b1",
                                 "delta_h64_vs_Sbest": nll_h64[s_id] - nll_h64["S_a1_b1"]},
              "h0_confirm": token_weighted(eval_rows("H0_step256", 0, "confirm")),
              "one_shot": oneshot, "aux_step128": aux, "smoke": smoke,
              "numeric_resolution_actual": repeat, "scale_boundary": scale_boundary,
              "passed": passed, "clear_reverse_evidence_at_aux": clear_reverse,
              "budget": ledger_value}
    write(HERE / "findings.json", result)
    write_findings(result)
    package_light()
    print(json.dumps({"verdict": verdict, "selection": frozen["selected"],
                      "optimizer_step_attempts": ledger_value["optimizer_step_attempts"],
                      "token_forward_equivalents": ledger_value["token_forward_equivalents"]},
                     ensure_ascii=False))


def write_findings(result):
    effects = result["effects"]
    lines = ["# V15 当前方向局部优势的时域检验", "",
             f"终点主判断：**{result['verdict']}**。父 gate 保持 STOP（BASELINE_TUNING_UNRESOLVED）。",
             f"协议 `{CFG['protocol_id']}`；manifest `{result['manifest_sha256'][:16]}…`；"
             f"冻结选择 `{result['selection']}`；普通 AdamW 对照 `S_a1_b1` 单列。", "",
             "## 主终点（H64，本轮新确认集 128 个未用 probe_confirm 自然前缀文档，token 加权 NLL）", "",
             "| 分支 | H64 NLL | H1 NLL |", "|---|---:|---:|"]
    for cid, value in result["confirm_nll_h64"].items():
        lines.append(f"| `{cid}` | {value:.9f} | {result['confirm_nll_h1'][cid]:.9f} |")
    lines.append(f"| （H0 共同基点） | {result['h0_confirm']:.9f} | — |")
    lines += ["", "| 主比较 | H64 点估计 | 95% CI | 97.5% Bonferroni CI | H1 点估计 |", "|---|---:|---|---|---:|"]
    for family in ("F", "P"):
        e = effects[family]
        lines.append(f"| Delta_{family}=L_S-L_{family} | {e['delta_h64']:.9g} | "
                     f"[{e['ci95'][0]:.3g}, {e['ci95'][1]:.3g}] | "
                     f"[{e['ci97p5'][0]:.3g}, {e['ci97p5'][1]:.3g}] | {e['delta_h1']:.3g} |")
    lines += ["", f"普通 AdamW 相对冻结 S 的 H64 差：{result['ordinary_adamw']['delta_h64_vs_Sbest']:.9g}"
              "（冻结 S 是独立选择的标量控制，不与普通 AdamW 混同）。",
              f"delta_branch 投入参考阈值 {CFG['delta_branch_nats_per_token']:g} nats/token；"
              f"实测数值分辨率 {result['numeric_resolution_actual']:g}（重复评估同一状态）。", "",
              "## 预设诊断（次要，不替代主 H64）", ""]
    for family in ("F", "P"):
        o = result["one_shot"][family]
        lines += [f"- **one-shot {family}**（`{o['branch']}`，第1步 {family} 后转 S，a/b 不变）："
                  f"H64 确认 {o['h64_confirm']:.9f}；持续 {family} H64 {o['sustained_h64_confirm']:.9f}；"
                  f"匹配系数 S（`{o['matched_S']}`）H64 {o['matched_S_h64_confirm']:.9f}；"
                  f"H16（确认）{o['h16_confirm']:.9f}。"]
    aux = result["aux_step128"]
    if aux["status"] == "MEASURED":
        lines += ["", f"- **step128 辅助起点**（同 seed 时点迁移，非独立 seed 确认）："
                  f"aux Delta_F(H64)={aux['aux_delta_F_h64']:.3g}，aux Delta_P(H64)={aux['aux_delta_P_h64']:.3g}；"
                  f"是否构成明确反向证据：{result['clear_reverse_evidence_at_aux']}。"]
    else:
        lines += ["", f"- step128 辅助起点：{aux['status']}（{aux.get('reason', '')}）。"]
    lines += ["", f"- scale 边界标志：{result['scale_boundary']}；选择落边界仅记录，不扩网格。", "",
              "## 预算与资源（全部实测）", "",
              f"- GPU optimizer.step 尝试 {result['budget']['optimizer_step_attempts']} / 上限 "
              f"{CFG['max_gpu_optimizer_step_attempts']}，完成 {result['budget']['optimizer_step_completions']}；"
              f"其中 smoke {result['budget'].get('smoke_step_attempts', 0)} 步。",
              f"- token-forward-equivalents {result['budget']['token_forward_equivalents']} / "
              f"{CFG['max_token_forward_equivalents']}（训练+评估+失败重复一并累计）；"
              f"训练有效 tokens {result['budget']['train_effective_tokens']}；反向 {result['budget']['backward_attempts']} 次。",
              f"- 计时：训练前向/反向 {result['budget']['train_seconds']:.1f}s、方向构造+位移修正 "
              f"{result['budget']['direction_seconds']:.1f}s、评估 {result['budget']['eval_seconds']:.1f}s。",
              "- 新增从随机初始化完整预训练轨迹：0；新增 64 步短分支训练步数见账本（与 0 区分报告）。",
              "- 训练吞吐 = 训练有效 tokens / train_seconds；资源曲线与每次 RAM/VRAM 门禁见 logs/resources.jsonl。",
              "- 沿用 1.5 GiB 准入 / 1.5 GiB 告警 / 1.0 GiB 硬停、RSS 4 GiB 与父 VRAM 限额；无能耗测量。", "",
              "## 范围与边界", "",
              "单 seed 单时点为主；step128 为同时点迁移诊断。开发集 = V12 的 128 new_probe_prefix（已用于历史开发）；"
              "确认集 = 本轮训练前封存的 128 个未用 probe_confirm 自然前缀文档（V1/V12/V13/V14 排除链后按 doc_id 字典序）。"
              "未调 LR/beta/WD；未测完整预训练终点、其他层/attention/预测器；无跨 seed 结论。正负结果均止步于本网格。"]
    (HERE / "V15_FINDINGS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_light():
    names = ["train.py", "test_v15.py", "config.json", "RUN_VERIFIED.ps1", "run_manifest.json",
             "parent_dependencies.json", "budget_ledger.json", "selection_frozen.json",
             "dev_scores.json", "findings.json", "V15_FINDINGS.md", "smoke_result.json",
             "confirm_views.jsonl", "confirm_documents.csv"]
    for folder in ("raw", "logs"):
        base = HERE / folder
        if base.exists():
            names += [str(p.relative_to(HERE)) for p in sorted(base.rglob("*"))
                      if p.is_file() and p.suffix in (".jsonl", ".json", ".log", ".csv", ".txt")]
    names = [n for n in names if (HERE / n).is_file()]
    with zipfile.ZipFile(HERE / "v15_light.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(HERE / name, name)
    with zipfile.ZipFile(HERE / "v15_light.zip") as archive:
        listed = set(archive.namelist())
    for key in ("budget_ledger.json", "findings.json", "V15_FINDINGS.md", "selection_frozen.json"):
        if key not in listed:
            raise RuntimeError("LIGHT_PACKAGE_INCOMPLETE:" + key)
    return HERE / "v15_light.zip"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "smoke", "dev", "select", "confirm",
                                          "finalize", "status"])
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
        print(json.dumps({"status": "PREPARED"}))
    elif args.phase == "smoke":
        phase_smoke()
    elif args.phase == "dev":
        phase_dev()
    elif args.phase == "select":
        phase_select()
    elif args.phase == "confirm":
        phase_confirm()
    elif args.phase == "finalize":
        phase_finalize()
    else:
        print(json.dumps({"manifest": read(HERE / "run_manifest.json"), "ledger": ledger()},
                         ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
