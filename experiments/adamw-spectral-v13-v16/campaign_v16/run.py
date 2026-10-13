"""V16 serial campaign runner. All state on disk; `python -m campaign_v16.run` resumes."""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.dont_write_bytecode = True

from campaign_v16.core import (ANCHOR_KEYS, CFG, HB, HERE, PCFG, ROOT, TARGET, V12, V14, V15,
                               VRAM, append, apply_step, begin_optimizer, budget_exhausted,
                               bump, count_forward, decision, device_task_hours, digest,
                               dir_size_gib, disk_free_gib, eval_docs, eval_rows, ledger,
                               ledger_txn, model_and_optimizer, n_a, paired_bootstrap,
                               param_fingerprint, parent_schedule_eta, queue_add, queue_rows,
                               queue_set_status, read, resource, schedule_eta, save_state, sha,
                               source_hash, state, token_weighted, wall_hours, write)
import numpy as np
import psutil
import torch

DATA = None


def load_data():
    global DATA
    DATA = read(HERE / "data_manifest.json")
    return DATA


def docs_of(split):
    key = {"development": "development", "screen": "screen", "final_holdout": "final_holdout"}[split]
    return read(HERE / "eval_sets.json")[key]


# ------------------------------------------------------------------ init / data

def verify_parent():
    base = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003"
    manifest = read(base / "manifest.json")
    if manifest["status"] != "SUCCESS":
        raise RuntimeError("PARENT_BASELINE_NOT_SUCCESS")
    gate = read(ROOT / "results" / "local_gate.json")
    if gate["next_action"] != "STOP":
        raise RuntimeError("UNEXPECTED_PARENT_GATE")
    checks = {"gate": [gate["verdict"], gate["next_action"]],
              "step256_sha256": sha(base / "step256.pt"),
              "step128_sha256": sha(base / "step128.pt") if (base / "step128.pt").is_file() else None,
              "parent_data_sha256": read(ROOT / "artifacts" / "data_manifest.json")["data_hash"],
              "v15_manifest_sha256": sha(V15 / "run_manifest.json") if (V15 / "run_manifest.json").is_file() else None}
    for step, key in ((256, "step256_sha256"), (128, "step128_sha256")):
        expected = manifest["checkpoint_hashes"].get(str(step))
        if expected and checks[key] != expected:
            raise RuntimeError(f"PARENT_CHECKPOINT_MISMATCH:{step}")
    return checks


def build_data():
    if (HERE / "data_manifest.json").exists():
        return load_data()
    from tokenizers import Tokenizer
    from replay_audit_v12.core import natural_prefix, normalized_role
    binding = read(ROOT / "bindings.local.json")
    pdata = read(ROOT / "artifacts" / "data_manifest.json")
    seq = CFG["data"]["sequence_length"]
    excluded = set()
    for role in ("lr_select", "probe_select", "probe_confirm"):
        excluded.update(r["doc_id"] for r in read(ROOT / "artifacts" / "eval_blocks.json")[role])
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["view"] in ("legacy", "new_probe_prefix"):
                excluded.add(row["doc_id"])
    with (V14 / "confirm_views.jsonl").open(encoding="utf-8") as stream:
        excluded.update(json.loads(line)["doc_id"] for line in stream)
    with (V15 / "confirm_views.jsonl").open(encoding="utf-8") as stream:
        excluded.update(json.loads(line)["doc_id"] for line in stream)
    old_dev = read(V12 / "evaluation_manifest.json")["new_probe_select_doc_ids"]
    tokenizer = Tokenizer.from_file(str(Path(binding["tokenizer_path"]) / "tokenizer.json"))
    bos, eos = pdata["bos_token_id"], pdata["eos_token_id"]
    cache_path = HERE / "cache" / "train_tokens.u32"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    need_train = CFG["data"]["train_tokens_needed"]
    quotas = {"development": 128, "screen": 256, "final_holdout": CFG["data"]["final_holdout"]["count"]}
    pools = {k: [] for k in quotas}
    n_train, scanned = 0, 0
    out = cache_path.open("wb")
    source = Path(binding["source_data_path"])
    parent_prefix_hash = None
    import hashlib as _hl
    prefix_acc = _hl.sha256()
    with source.open("rb") as stream:
        while n_train < need_train or any(len(pools[k]) < quotas[k] for k in quotas):
            line = stream.readline(CFG["data"]["scan_chunk_bytes"])
            if not line:
                break
            scanned += len(line)
            try:
                row = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            text, doc_id, role = normalized_role(row["text"])
            if not text or doc_id in excluded:
                continue
            ids = [bos] + tokenizer.encode(text, add_special_tokens=False).ids + [eos]
            if role == "train" and n_train < need_train:
                take = min(len(ids), need_train - n_train)
                chunk = np.asarray(ids[:take], dtype=np.uint32).tobytes()
                out.write(chunk)
                if parent_prefix_hash is None:
                    if n_train < pdata["train_token_count"] <= n_train + take:
                        pass
                if n_train + take >= pdata["train_token_count"] and parent_prefix_hash is None:
                    out.flush()
                    with cache_path.open("rb") as f:
                        parent_prefix_hash = _hl.sha256(f.read(pdata["train_token_count"] * 4)).hexdigest()
                n_train += take
            elif role != "train":
                if role == "probe_select" and doc_id not in old_dev and len(pools["development"]) < quotas["development"]:
                    pools["development"].append((doc_id, text))
                elif role == "probe_select" and len(pools["screen"]) < quotas["screen"]:
                    pools["screen"].append((doc_id, text))
                elif role == "final_holdout_reserved" and len(pools["final_holdout"]) < quotas["final_holdout"]:
                    pools["final_holdout"].append((doc_id, text))
    out.close()
    if n_train < CFG["b"]["T"] * 2048 + 1 or any(len(pools[k]) < (CFG["data"]["final_holdout"]["min_count"] if k == "final_holdout" else quotas[k]) for k in quotas):
        raise RuntimeError(f"DATA_BLOCKED:n_train={n_train},pools={ {k: len(v) for k, v in pools.items()} }")
    parent_sha = pdata["train_cache_sha256"]
    if parent_prefix_hash != parent_sha:
        raise RuntimeError("PARENT_PREFIX_MISMATCH")
    sets = {"development": [], "screen": [], "final_holdout": []}
    dev = [{"doc_id": d} for d in old_dev]
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as stream:
        rows = {json.loads(l)["doc_id"]: json.loads(l) for l in stream if '"new_probe_prefix"' in l}
    for d in dev:
        r = rows[d["doc_id"]]
        sets["development"].append({"doc_id": d["doc_id"], "ids": r["ids"], "token_count": r["token_count"]})
    def to_doc(item):
        doc_id, text = item
        content = tokenizer.encode(text, add_special_tokens=False).ids
        ids = natural_prefix(content, bos, eos, seq)
        return {"doc_id": doc_id, "ids": ids, "token_count": len(ids) - 1}
    sets["development"] += [to_doc(x) for x in pools["development"]]
    sets["screen"] = [to_doc(x) for x in pools["screen"]]
    sets["final_holdout"] = [to_doc(x) for x in pools["final_holdout"]]
    write(HERE / "eval_sets.json", sets)
    manifest = {"n_train_tokens": n_train, "scanned_bytes": scanned,
                "train_cache_sha256": sha(cache_path), "cache_tokens": n_train,
                "eval_sets_sha256": {k: digest(v) for k, v in sets.items()},
                "counts": {k: len(v) for k, v in sets.items()},
                "dev_composition": "128 V12 dev + 128 fresh probe_select (scan order)",
                "holdout_rule": "final_holdout_reserved role, scan order; sealed before any tuning",
                "parent_prefix_verified": parent_prefix_hash == parent_sha}
    write(HERE / "data_manifest.json", manifest)
    return manifest


def init_campaign():
    checks = verify_parent()
    data = build_data()
    write(HERE / "run_manifest.json", {
        "protocol_id": CFG["protocol_id"], "source_hash": source_hash(),
        "config_sha256": sha(HERE / "campaign_config.json"),
        "parent": checks, "data_manifest_sha256": sha(HERE / "data_manifest.json"),
        "created_at": time.time()})
    with ledger_txn() as v:
        v.update({"manifest_sha256": sha(HERE / "run_manifest.json"), "wall_started_at": time.time(),
                  "optimizer_step_attempts": 0, "optimizer_step_completions": 0,
                  "token_forward_equivalents": 0, "forward_calls": 0, "backward_attempts": 0,
                  "train_effective_tokens": 0, "train_seconds": 0.0, "direction_seconds": 0.0,
                  "eval_seconds": 0.0, "device_task_seconds": 0.0, "pause_seconds": 0.0,
                  "anchor_svds": 0})
    save_state(stage="A0", created_at=time.time(), selections={}, pauses_total=0.0)
    write(HERE / "STATUS.md", "# V16 status\n\ninitialized\n")
    if not queue_rows():
        queue_add({"id": "a0_h0fix", "kind": "h0_fix"})
        for family in CFG["a1"]["families"]:
            for a in CFG["a1"]["a_grid"]:
                for b in CFG["a1"]["b_grid"]:
                    queue_add({"id": f"a1_{family}_a{a:g}_b{b:g}", "kind": "branch",
                               "stage": "A1", "family": family, "a": a, "b": b,
                               "start": "parent_step256", "T": CFG["a1"]["horizon"],
                               "schedule": "parent", "seed": 3407, "lr": 0.003,
                               "eval_split": "development", "eval_horizons": CFG["a1"]["eval_horizons"],
                               "record_q": True})
        queue_add({"id": "a1_select", "kind": "planner", "planner": "a1_select"})
    print(json.dumps({"initialized": True, "data": data["counts"]}))


# ------------------------------------------------------------------ GPU + branch training

class GpuContext:
    def __enter__(self):
        resource("before_cuda_model_init", admission=True, force=True)
        lock = HERE / "run.lock"
        if (ROOT / "artifacts" / "run.lock").exists():
            raise RuntimeError("RUN_LOCK_PRESENT_PARENT")
        if lock.exists():
            try:
                pid = int(lock.read_text(encoding="utf-8").strip())
                psutil.Process(pid)
                raise RuntimeError("RUN_LOCK_PRESENT")
            except (ValueError, psutil.NoSuchProcess):
                lock.unlink(missing_ok=True)  # stale lock from a dead process
            except psutil.AccessDenied:
                raise RuntimeError("RUN_LOCK_PRESENT")
        (HERE / "run.lock").write_text(str(os.getpid()), encoding="utf-8")
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA_UNAVAILABLE")
            free, _ = torch.cuda.mem_get_info(0)
            if free < HB["vram_admission_min_free_gib"] * 2 ** 30:
                raise RuntimeError("RESOURCE_BLOCKED:VRAM_ADMISSION")
            VRAM[0] = min(HB["vram_own_reserved_cap_gib"] * 2 ** 30,
                          free - HB["vram_min_reserve_gib"] * 2 ** 30)
            torch.set_num_threads(2)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.mem = np.memmap(HERE / "cache" / "train_tokens.u32", dtype=np.uint32, mode="r")
            resource("after_model_init", force=True)
        except BaseException:
            VRAM[0] = None
            (HERE / "run.lock").unlink(missing_ok=True)
            raise
        return self

    def __exit__(self, *exc):
        VRAM[0] = None
        (HERE / "run.lock").unlink(missing_ok=True)
        return False


def make_model(task):
    kwargs = {}
    if task.get("target_beta1") is not None:
        kwargs["target_beta1"] = task["target_beta1"]
    if task.get("all_beta1") is not None:
        kwargs["all_beta1"] = task["all_beta1"]
    if task.get("wd_matrix") is not None:
        kwargs["wd_matrix"] = task["wd_matrix"]
    return model_and_optimizer("cuda:0", peak_lr=task.get("lr", 0.003), **kwargs)


def save_rolling(path, model, opt, task, branch_step, cursor, total_tokens):
    payload = {"model": model.state_dict(),
               "optimizer": copy.deepcopy(opt.state_dict()),
               "task_digest": digest(task), "branch_step": branch_step, "cursor": cursor,
               "total_effective_tokens": total_tokens,
               "cpu_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
               "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def anchor_capture_factory(task):
    anchors = set()
    if task.get("anchor_svd") or task.get("anchor_scalars"):
        T = task["T"]
        anchors = {max(1, int(round(f * T))) for f in CFG["b"]["anchor_fractions"]}
    return anchors


def train_branch(task):
    label = task["id"]
    from pilot_core import load_checkpoint, loss_sum_count, training_batch
    with GpuContext() as ctx:
        mem = ctx.mem
        if task["start"] == "scratch":
            torch.manual_seed(task["seed"])
            torch.cuda.manual_seed_all(task["seed"])
            np.random.seed(task["seed"])
            random.seed(task["seed"])
        model, opt, aliases, _ = make_model(task)
        rolling = HERE / "checkpoints" / f"{label}_latest.pt"
        branch_step, cursor, total_tokens = 0, 0, 0
        if task["start"] == "parent_step256":
            ckpt = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003" / "step256.pt"
            payload = load_checkpoint(ckpt, model, opt,
                                      config_hash=sha(ROOT / "config" / "local_v1_lowmem.json"),
                                      data_hash=read(ROOT / "artifacts" / "data_manifest.json")["data_hash"],
                                      aliases=aliases)
            if payload["successful_optimizer_step"] != 256:
                raise RuntimeError("CHECKPOINT_STEP_CHANGED")
            cursor, total_tokens = payload["data_cursor"], payload["total_effective_tokens"]
            origin = {"state_id": "parent_step256", "loaded_checkpoint_sha": sha(ckpt), "global_step": 256}
        else:
            origin = {"state_id": f"scratch_seed{task['seed']}", "loaded_checkpoint_sha": None,
                      "global_step": 0}
        if rolling.exists():
            st = torch.load(rolling, map_location="cpu", weights_only=False)
            if st["task_digest"] == digest(task):
                model.load_state_dict(st["model"])
                opt.load_state_dict(st["optimizer"])
                torch.set_rng_state(st["cpu_rng"])
                if st["cuda_rng"] is not None:
                    torch.cuda.set_rng_state_all(st["cuda_rng"])
                random.setstate(st["python_rng"])
                np.random.set_state(st["numpy_rng"])
                branch_step, cursor, total_tokens = st["branch_step"], st["cursor"], st["total_effective_tokens"]
        step_log = HERE / "raw" / "steps" / f"{label}.jsonl"
        done = ({json.loads(l).get("branch_step", json.loads(l).get("step"))
                 for l in step_log.open(encoding="utf-8")}
                if step_log.exists() else set())
        q_series = None
        if task.get("q_source"):
            q_series = read(HERE / "raw" / "q_series" / f"{task['q_source']}.json")["q"]
        anchors = anchor_capture_factory(task)
        eval_split = task.get("eval_split", "development")
        docs = docs_of(eval_split)
        T = task["T"]
        bf16 = torch.cuda.is_bf16_supported()
        mb, accum, seq = 2, 4, 256
        while branch_step < T:
            branch_step += 1
            if task["schedule"] == "parent":
                eta = parent_schedule_eta(task["lr"], 256 + branch_step)
            else:
                eta = schedule_eta(task["lr"], branch_step, T, task["warmup"], CFG["b"]["final_lr_ratio"])
            for group in opt.param_groups:
                group["lr"] = eta
            family = task["family"]
            for (wlo, whi, fam) in task.get("windows", []):
                if wlo < branch_step <= whi:
                    family = fam
            resource(f"{label}_s{branch_step}")
            model.train()
            opt.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            loss_total, valid_total, last_ids = 0.0, 0, None
            for _ in range(accum):
                ids, cursor = training_batch(mem, cursor, mb, seq, "cuda:0")
                lengths = torch.full((mb,), ids.shape[1], device="cuda:0", dtype=torch.long)
                count_forward(mb * (ids.shape[1] - 1), label)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                    logits = model(ids[:, :-1], use_cache=False).logits
                    loss_sum, count = loss_sum_count(logits, ids, lengths)
                if not torch.isfinite(loss_sum):
                    raise RuntimeError("NON_FINITE_TRAIN_LOSS")
                loss_sum.backward()
                bump(backward_attempts=1, device_task_seconds=0.0)
                valid_total += count
                loss_total += float(loss_sum.detach())
                last_ids = ids
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(valid_total)
            torch.nn.utils.clip_grad_norm_(model.parameters(), PCFG["baseline"]["global_grad_clip_norm"])
            bump(train_effective_tokens=valid_total)
            tdir = time.perf_counter()
            q_over = q_series[branch_step - 1] if q_series is not None else None
            capture = {}
            meta = apply_step(model, opt, family, task["a"], task["b"], label, eta,
                              q_override=q_over,
                              anchor_recorder=(lambda named, A, tA, m: capture.update(
                                  grab_anchor(named, A, m)) if branch_step in anchors else None),
                              anchor_names=tuple(ANCHOR_KEYS) if branch_step in anchors else ())
            bump(direction_seconds=time.perf_counter() - tdir,
                 train_seconds=time.perf_counter() - t0)
            opt.zero_grad(set_to_none=True)
            total_tokens += valid_total
            if capture:
                record_anchor(label, branch_step, capture, meta, model, last_ids, task)
            if branch_step not in done:
                append(step_log, {"branch": label, "branch_step": branch_step, "step": branch_step,
                                  "eta": eta,
                                  "loss_sum": loss_total, "token_count": valid_total,
                                  "nll": loss_total / valid_total, "family_used": family,
                                  "A_norm": meta["A_norm"], "D_norm": meta["D_norm"],
                                  "q_t": meta["q_t"], "zero_norm_fallback": meta["zero_norm_fallback"]})
            if branch_step in task.get("eval_horizons", []):
                ident = {**origin, "global_step": (256 if task["schedule"] == "parent" else 0) + branch_step,
                         "param_fingerprint": param_fingerprint(model)}
                eval_docs(model, docs, eval_split, label, label, branch_step, ident)
                model.train()
            if branch_step % 128 == 0:
                previous = HERE / "checkpoints" / f"{label}_previous.pt"
                if previous.exists():
                    previous.unlink()
                if rolling.exists():
                    os.replace(rolling, previous)
                save_rolling(rolling, model, opt, task, branch_step, cursor, total_tokens)
        if task.get("save_endpoint"):
            endpoint = HERE / "checkpoints" / f"{label}_endpoint.pt"
            tmp = endpoint.with_name(endpoint.name + ".tmp")
            torch.save({"model": model.state_dict(), "task_digest": digest(task),
                        "origin": origin, "final_step": T, "cursor": cursor,
                        "param_fingerprint": param_fingerprint(model)}, tmp)
            os.replace(tmp, endpoint)
        if rolling.exists():
            rolling.unlink(missing_ok=True)
            (HERE / "checkpoints" / f"{label}_previous.pt").unlink(missing_ok=True)
        enforce_checkpoint_caps()


def grab_anchor(named, A_map, meta):
    out = {}
    m_plus = meta.get("_m_plus", {})
    for key in ANCHOR_KEYS:
        p = named[key]
        out[key] = {"g": p.grad.detach().cpu().clone() if p.grad is not None else None,
                    "A": A_map[key].detach().cpu().clone() if A_map.get(key) is not None else None,
                    "m": m_plus[key].clone() if key in m_plus else None}
    return out


def record_anchor(label, branch_step, capture, meta, model, ids, task):
    row = {"branch": label, "step": branch_step, "q_t": meta["q_t"], "matrices": {}}
    for key, tensors in capture.items():
        g, A, m = tensors["g"], tensors["A"], tensors.get("m")
        if g is None or A is None:
            continue
        gn = float(torch.linalg.vector_norm(g.double()))
        An = float(torch.linalg.vector_norm(A.double()))
        entry = {"g_norm": gn, "A_norm": An,
                 "cos_g_A": float((g.double() * A.double()).sum() / (gn * An)) if gn and An else None}
        if m is not None:
            mn = float(torch.linalg.vector_norm(m.double()))
            entry.update({"m_norm": mn,
                          "cos_g_m": float((g.double() * m.double()).sum() / (gn * mn)) if gn and mn else None})
        if task.get("anchor_svd") and ledger().get("anchor_svds", 0) < CFG["b"]["max_anchor_svds"]:
            s = torch.linalg.svdvals(A.double())
            bump(anchor_svds=1)
            entry.update({"sigma1": float(s[0]),
                          "head_energy_fraction": float(s[0] ** 2 / (s ** 2).sum())})
        row["matrices"][key] = entry
    append(HERE / "raw" / "anchors.jsonl", row)


# ------------------------------------------------------------------ A0: H0 correction

def task_h0_fix(task):
    from pilot_core import load_checkpoint
    with GpuContext() as ctx:
        from pilot_core import model_and_optimizer as parent_mao
        binding = read(ROOT / "bindings.local.json")
        model, opt, aliases, _ = parent_mao(PCFG, binding, 0.003, "cuda:0")
        v15_confirm = [json.loads(l) for l in (V15 / "confirm_views.jsonl").open(encoding="utf-8")]
        v12_dev_ids = read(V12 / "evaluation_manifest.json")["new_probe_select_doc_ids"]
        with (V12 / "eval_views.jsonl").open(encoding="utf-8") as stream:
            rows = {json.loads(l)["doc_id"]: json.loads(l) for l in stream if '"new_probe_prefix"' in l}
        v15_dev = [rows[i] for i in v12_dev_ids]
        results = {}
        for step, docs_dev in ((256, v15_dev), (128, v15_dev)):
            ckpt = ROOT / "artifacts" / "checkpoints" / "seed3407_lr0.003" / f"step{step}.pt"
            load_checkpoint(ckpt, model, opt,
                            config_hash=sha(ROOT / "config" / "local_v1_lowmem.json"),
                            data_hash=read(ROOT / "artifacts" / "data_manifest.json")["data_hash"],
                            aliases=aliases)
            ident = {"state_id": f"parent_step{step}", "loaded_checkpoint_sha": sha(ckpt),
                     "global_step": step, "param_fingerprint": param_fingerprint(model)}
            eval_docs(model, docs_dev, "v15dev", f"a0h0fix_step{step}", f"H0CORR_step{step}", 0, ident)
            eval_docs(model, v15_confirm, "v15confirm", f"a0h0fix_step{step}", f"H0CORR_step{step}", 0, ident)
            results[step] = {"dev": token_weighted(eval_rows(f"H0CORR_step{step}", 0, "v15dev")),
                             "confirm": token_weighted(eval_rows(f"H0CORR_step{step}", 0, "v15confirm")),
                             **ident}
    invalidated = [str(p.relative_to(V15)) for p in sorted((V15 / "raw" / "eval").glob("H0_*__*.jsonl"))]
    v15f = read(V15 / "findings.json")
    v15_ledger_now = read(V15 / "budget_ledger.json")
    write(HERE / "a0_h0_correction.json", {
        "invalid_state_origin_files": invalidated,
        "correction": "V15 H0_step256/H0_step128 rows were evaluated before the intended "
                      "checkpoint was loaded (random init / stale branch state). They are marked "
                      "INVALID_STATE_ORIGIN; corrected zero-training evaluations above. V15 S/F/P "
                      "H1/H64 comparisons are unaffected (they loaded their own snapshots/endpoints).",
        "corrected": results,
        "v15_ledger_note": {
            "reported_backwards": v15f["budget"].get("backward_attempts"),
            "current_backwards": v15_ledger_now.get("backward_attempts"),
            "gpu_seconds_was_unused_field": True,
            "actual_timings_fields": ["train_seconds", "direction_seconds", "eval_seconds"]}})


# ------------------------------------------------------------------ planners

def branch_dev_nll(label, horizon, split="development"):
    return token_weighted(eval_rows(label, horizon, split))


def plan_a1_select(task):
    rows = queue_rows()
    best = {}
    for family in CFG["a1"]["families"]:
        cands = [r for r in rows if r.get("stage") == "A1" and r.get("family") == family and r["status"] == "done"]
        scores = {r["id"]: branch_dev_nll(r["id"], CFG["a1"]["horizon"]) for r in cands}
        top = min(scores.values())
        ties = [r for r in cands if scores[r["id"]] == top]
        chosen = sorted(ties, key=lambda r: (0 if (r["a"] == 1 and r["b"] == 1) else 1,
                                             abs(r["a"] - 1) + abs(r["b"] - 1), r["id"]))[0]
        best[family] = {"id": chosen["id"], "a": chosen["a"], "b": chosen["b"], "dev_h256_nll": top}
    sel = state().get("selections", {})
    sel["a1"] = best
    save_state(selections=sel)
    for r in queue_rows():
        if r.get("stage") == "A1" and r.get("record_q") and r["status"] == "done":
            steps = [json.loads(l) for l in (HERE / "raw" / "steps" / f"{r['id']}.jsonl").open(encoding="utf-8")]
            write(HERE / "raw" / "q_series" / f"{r['id']}.json",
                  {"source": r["id"], "q": [s["q_t"] for s in sorted(steps, key=lambda s: s["step"])]})
    aS, bS = best["S"]["a"], best["S"]["b"]
    for name, family, qsrc in (("a2_Fdir_qS", "F", best["S"]["id"]),
                               ("a2_Pdir_qS", "P", best["S"]["id"]),
                               ("a2_Sdir_qF", "S", best["F"]["id"]),
                               ("a2_Sdir_qP", "S", best["P"]["id"])):
        queue_add({"id": name, "kind": "branch", "stage": "A2", "family": family,
                   "a": aS, "b": bS, "start": "parent_step256", "T": CFG["a2"]["horizon"],
                   "schedule": "parent", "seed": 3407, "lr": 0.003, "q_source": qsrc,
                   "eval_split": "development", "eval_horizons": CFG["a2"]["eval_horizons"]})
    decision("A1 selection complete", json.dumps(best, ensure_ascii=False),
             "freeze per-family (a,b); enqueue A2 common-norm branches; enqueue b_plan")
    queue_add({"id": "b_plan", "kind": "planner", "planner": "b_plan"})


def plan_b(task):
    a1 = state()["selections"]["a1"]
    lr_grid = CFG["b"]["lr_grid"]
    base = {"kind": "branch", "stage": "B", "start": "scratch", "seed": CFG["b"]["seed"],
            "T": CFG["b"]["T"], "warmup": CFG["b"]["warmup"], "schedule": "scratch",
            "eval_split": "development", "eval_horizons": CFG["b"]["eval_horizons"],
            "save_endpoint": True, "anchor_scalars": True}
    for i, lr in enumerate(lr_grid, 1):
        queue_add({**base, "id": f"b_S{i}_plain_lr{lr:g}", "family": "S", "a": 1, "b": 1, "lr": lr,
                   "anchor_svd": True})
    for i, lr in enumerate(lr_grid, 1):
        queue_add({**base, "id": f"b_F{i}_a{a1['F']['a']:g}_lr{lr:g}", "family": "F",
                   "a": min(8, max(1, a1["F"]["a"])), "b": 1, "lr": lr})
    for i, lr in enumerate(lr_grid, 1):
        queue_add({**base, "id": f"b_P{i}_a{a1['P']['a']:g}_lr{lr:g}", "family": "P",
                   "a": min(8, max(1, a1["P"]["a"])), "b": 1, "lr": lr})
    decision("B fixed part", "S/F/P x 3 LRs per frozen plan",
             "9 trials enqueued; SVD anchors on the three S-plain trials only (30/48 budget)")
    queue_add({"id": "b_plan2", "kind": "planner", "planner": "b_plan2"})


def plan_b2(task):
    a1 = state()["selections"]["a1"]
    lr_grid = CFG["b"]["lr_grid"]
    fam_best_lr = {}
    for fam in ("S", "F", "P"):
        rows = [r for r in queue_rows() if r.get("stage") == "B" and r["status"] == "done"
                and r.get("family") == fam and ("_plain_" in r["id"] if fam == "S" else True)]
        scores = {r["lr"]: branch_dev_nll(r["id"], CFG["b"]["T"]) for r in rows}
        fam_best_lr[fam] = min(scores, key=scores.get)
    sel = state().get("selections", {})
    sel["b_best_lr"] = fam_best_lr
    save_state(selections=sel)
    aS, bS = a1["S"]["a"], a1["S"]["b"]
    base = {"kind": "branch", "stage": "B", "start": "scratch", "seed": CFG["b"]["seed"],
            "T": CFG["b"]["T"], "warmup": CFG["b"]["warmup"], "schedule": "scratch",
            "eval_split": "development", "eval_horizons": CFG["b"]["eval_horizons"],
            "save_endpoint": True, "anchor_scalars": True}
    queue_add({**base, "id": f"b_S4_a{aS:g}_b{bS:g}_lr{fam_best_lr['S']:g}", "family": "S",
               "a": aS, "b": bS, "lr": fam_best_lr["S"]})
    queue_add({**base, "id": "b_S5_tb05", "family": "S", "a": aS, "b": bS,
               "lr": fam_best_lr["S"], "target_beta1": 0.5})
    queue_add({**base, "id": "b_S6_tb0", "family": "S", "a": aS, "b": bS,
               "lr": fam_best_lr["S"], "target_beta1": 0.0})
    queue_add({**base, "id": "b_S7_ab05", "family": "S", "a": aS, "b": bS,
               "lr": fam_best_lr["S"], "all_beta1": 0.5})
    s8lr = {0.006: 0.01, 0.001: 0.0003}.get(fam_best_lr["S"])
    if s8lr:
        queue_add({**base, "id": f"b_S8_lr{s8lr:g}", "family": "S", "a": 1, "b": 1, "lr": s8lr})
    else:
        queue_add({**base, "id": "b_S8_wd02", "family": "S", "a": 1, "b": 1,
                   "lr": fam_best_lr["S"], "wd_matrix": 0.2})
    for fam in ("F", "P"):
        rows = [r for r in queue_rows() if r.get("stage") == "B" and r["status"] == "done"
                and r.get("family") == fam]
        if not rows:
            continue
        best_dev = min(branch_dev_nll(r["id"], CFG["b"]["T"]) for r in rows)
        s_best = min(branch_dev_nll(r["id"], CFG["b"]["T"]) for r in queue_rows()
                     if r.get("stage") == "B" and r["status"] == "done" and r["id"].startswith("b_S"))
        if best_dev > s_best + 5 * CFG["stats"]["delta_short"]:
            decision(f"{fam} clearly behind S at H2048",
                     f"{fam}_best={best_dev:.6f} s_best={s_best:.6f}",
                     "skip remaining F/P extension slots (early stop allowed)")
            continue
        a_cur = min(8, max(1, a1[fam]["a"]))
        lr_boundary = {0.006: 0.01, 0.001: 0.0003}.get(fam_best_lr[fam])
        pool = []
        if lr_boundary:
            pool.append({**base, "id": f"b_{fam}4_lr{lr_boundary:g}", "family": fam,
                         "a": a_cur, "b": 1, "lr": lr_boundary})
        nb = a_cur / 2 if a_cur / 2 >= 0.5 else min(8, a_cur * 2)
        pool.append({**base, "id": f"b_{fam}5_a{nb:g}", "family": fam, "a": nb, "b": 1,
                     "lr": fam_best_lr[fam]})
        for t in pool:
            queue_add(t)
    decision("B extension slots", json.dumps(fam_best_lr), "S4-8 and family extensions enqueued per frozen rules")
    queue_add({"id": "b_select", "kind": "planner", "planner": "b_select"})


def plan_b_select(task):
    rows = [r for r in queue_rows() if r.get("stage") == "B" and r["status"] == "done"]
    scores = {r["id"]: branch_dev_nll(r["id"], CFG["b"]["T"]) for r in rows}
    order = sorted(scores.items(), key=lambda kv: kv[1])
    frozen = {"all_dev_h2048": scores, "ranking": [k for k, _ in order]}
    sel = state().get("selections", {})
    sel["b"] = frozen
    save_state(selections=sel)
    write(HERE / "b_selection.json", frozen)
    decision("B selection frozen (dev H2048)", json.dumps(order[:6]), "proceed to screen eval")
    with GpuContext() as ctx:
        from pilot_core import model_and_optimizer as parent_mao
        binding = read(ROOT / "bindings.local.json")
        model, _, _, _ = parent_mao(PCFG, binding, 0.003, "cuda:0")
        screen = docs_of("screen")
        for r in rows[:8]:
            endpoint = HERE / "checkpoints" / f"{r['id']}_endpoint.pt"
            if not endpoint.is_file():
                continue
            payload = torch.load(endpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            ident = {"state_id": f"endpoint:{r['id']}", "loaded_checkpoint_sha": sha(endpoint),
                     "global_step": payload["final_step"],
                     "param_fingerprint": payload["param_fingerprint"]}
            eval_docs(model, screen, "screen", "bscreen", r["id"], CFG["b"]["T"], ident)
    queue_add({"id": "route_b", "kind": "planner", "planner": "route_b"})


def plan_route_b(task):
    frozen = state()["selections"]["b"]
    dev = frozen["all_dev_h2048"]
    screen = {cid: token_weighted(eval_rows(cid, CFG["b"]["T"], "screen"))
              for cid in frozen["ranking"] if (HERE / "raw" / "eval" / f"{cid}__H{CFG['b']['T']}__screen.jsonl").exists()}
    best_s_id = next(cid for cid in frozen["ranking"] if cid.startswith("b_S"))
    best_s = dev[best_s_id]
    fam_best = {}
    for fam in ("F", "P"):
        cands = [cid for cid in frozen["ranking"] if cid.startswith(f"b_{fam}")]
        fam_best[fam] = min(cands, key=lambda c: dev[c]) if cands else None
    simple = [cid for cid in frozen["ranking"] if cid.startswith("b_S") and cid not in (best_s_id,)]
    sel = state().get("selections", {})
    sel["route"] = {"best_s": best_s_id, "fam_best": fam_best, "screen": screen}
    recipes = [best_s_id]
    threshold = CFG["stats"]["delta_short"]
    for fam in ("F", "P"):
        cid = fam_best[fam]
        gain_screen = (screen.get(best_s_id, 0) - screen.get(cid, 0)) if cid and cid in screen else None
        gain_dev = best_s - dev[cid] if cid else None
        if cid and gain_dev is not None and gain_dev >= threshold and (gain_screen is None or gain_screen >= 0):
            recipes.append(cid)
    if len(recipes) == 1 and simple:
        best_simple = min(simple, key=lambda c: dev[c])
        if dev[best_simple] < best_s:
            recipes.insert(0, best_simple)
    plain = next(cid for cid in frozen["ranking"] if "_plain_" in cid)
    if plain not in recipes:
        recipes.append(plain)
    if any(cid.startswith("b_S5") or cid.startswith("b_S6") or cid.startswith("b_S7") for cid in recipes[:2]):
        label = "SIMPLE_CONTROL_EXPLAINS_WITHIN_TESTED_RANGE"
    elif len(recipes) >= 2 and not recipes[1].startswith("b_S"):
        label = "F_OR_P_SUSTAINED"
    else:
        label = "S_ONLY"
    sel["route"]["recipes"] = recipes[:CFG["c"]["max_recipes"]]
    sel["route"]["label"] = label
    save_state(selections=sel)
    decision("route B", json.dumps({"label": label, "recipes": recipes}, ensure_ascii=False),
             "enqueue C finals with frozen recipes")
    c_plan()


def c_plan():
    recipes = []
    for cid in state()["selections"]["route"]["recipes"]:
        row = next(r for r in queue_rows() if r["id"] == cid)
        recipes.append({"name": cid, "family": row["family"], "a": row["a"], "b": row["b"],
                        "lr": row["lr"], "target_beta1": row.get("target_beta1"),
                        "all_beta1": row.get("all_beta1"), "wd_matrix": row.get("wd_matrix")})
    write(HERE / "c_recipes.json", {"recipes": recipes, "seeds": CFG["c"]["seeds"],
                                    "frozen_at": time.time()})
    for recipe in recipes:
        for seed in CFG["c"]["seeds"]:
            queue_add({"id": f"c_{recipe['name']}_s{seed}", "kind": "branch", "stage": "C",
                       **{k: v for k, v in recipe.items() if k != "name"},
                       "start": "scratch", "seed": seed, "T": CFG["c"]["T"],
                       "warmup": CFG["c"]["warmup"], "schedule": "scratch",
                       "eval_split": "development",
                       "eval_horizons": CFG["c"]["eval_horizons_during_training"],
                       "save_endpoint": True})
    queue_add({"id": "holdout_eval", "kind": "planner", "planner": "holdout"})


def plan_holdout(task):
    recipes = read(HERE / "c_recipes.json")["recipes"]
    with GpuContext() as ctx:
        from pilot_core import model_and_optimizer as parent_mao
        binding = read(ROOT / "bindings.local.json")
        model, _, _, _ = parent_mao(PCFG, binding, 0.003, "cuda:0")
        holdout = docs_of("final_holdout")
        for recipe in recipes:
            for seed in CFG["c"]["seeds"]:
                cid = f"c_{recipe['name']}_s{seed}"
                endpoint = HERE / "checkpoints" / f"{cid}_endpoint.pt"
                payload = torch.load(endpoint, map_location="cpu", weights_only=False)
                if payload["final_step"] != CFG["c"]["T"]:
                    raise RuntimeError("ENDPOINT_STEP_MISMATCH:" + cid)
                model.load_state_dict(payload["model"])
                ident = {"state_id": f"endpoint:{cid}", "loaded_checkpoint_sha": sha(endpoint),
                         "global_step": payload["final_step"],
                         "param_fingerprint": payload["param_fingerprint"]}
                eval_docs(model, holdout, "final_holdout", "cfinal", cid, CFG["c"]["holdout_eval_horizon"], ident)
    decision("final holdout opened once", "all C endpoints identity-locked",
             "no further tuning permitted; proceed to finalize")


# ------------------------------------------------------------------ finalize

def finalize():
    st = state()
    recipes = read(HERE / "c_recipes.json")["recipes"] if (HERE / "c_recipes.json").exists() else []
    holdout = {}
    for recipe in recipes:
        for seed in CFG["c"]["seeds"]:
            cid = f"c_{recipe['name']}_s{seed}"
            path = HERE / "raw" / "eval" / f"{cid}__H{CFG['c']['holdout_eval_horizon']}__final_holdout.jsonl"
            if path.exists():
                holdout[cid] = token_weighted(eval_rows(cid, CFG["c"]["holdout_eval_horizon"], "final_holdout"))
    route = st.get("selections", {}).get("route", {})
    recipes_ids = route.get("recipes", [])
    stats = {}
    if len(recipes_ids) >= 2 and holdout:
        base_id = recipes_ids[0]
        for cid in recipes_ids[1:]:
            per_seed = {}
            for seed in CFG["c"]["seeds"]:
                a = f"{base_id}_s{seed}"
                b = f"{cid}_s{seed}"
                if a in holdout and b in holdout:
                    per_seed[seed] = {
                        "delta": holdout[a] - holdout[b],
                        **paired_bootstrap(eval_rows(a, CFG["c"]["holdout_eval_horizon"], "final_holdout"),
                                           eval_rows(b, CFG["c"]["holdout_eval_horizon"], "final_holdout"))}
            if per_seed:
                deltas = [v["delta"] for v in per_seed.values()]
                stats[cid] = {"per_seed": per_seed, "mean_delta": float(np.mean(deltas)),
                              "median_delta": float(np.median(deltas)),
                              "positive_seeds": int(sum(d > 0 for d in deltas))}
    l = ledger()
    verdict = st.get("final_label", "MIXED_OR_LIMITED")
    result = {"protocol_id": CFG["protocol_id"], "verdict": verdict, "route": route,
              "holdout_nll": holdout, "stats": stats, "budget": l,
              "wall_hours": wall_hours(), "device_task_hours": device_task_hours()}
    write(HERE / "findings.json", result)
    write_findings(result)
    package_light()
    return result


def write_findings(r):
    lines = ["# V16 findings", "", f"终态：**{r['verdict']}**（自动生成骨架，最终人工判读见 V16_FINDINGS.md）", ""]
    lines.append(f"route: {json.dumps(r.get('route', {}).get('label'), ensure_ascii=False)}")
    for cid, s in r.get("stats", {}).items():
        lines.append(f"- {cid}: meanΔ={s['mean_delta']:.6g} medianΔ={s['median_delta']:.6g} "
                     f"positive seeds {s['positive_seeds']}/3")
    l = r["budget"]
    lines.append(f"预算：尝试 {l.get('optimizer_step_attempts')} / {HB['max_optimizer_step_attempts']}；"
                 f"token-forwards {l.get('token_forward_equivalents')} / {HB['max_token_forward_equivalents']}；"
                 f"墙钟 {r['wall_hours']:.2f}h；设备任务 {r['device_task_hours']:.2f}h")
    (HERE / "V16_FINDINGS_AUTO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_light():
    names = ["core.py", "run.py", "test_v16.py", "campaign_config.json", "RUN_CAMPAIGN.ps1",
             "PLAN_FROZEN.md", "run_manifest.json", "data_manifest.json", "eval_sets.json",
             "budget_ledger.json", "STATE.json", "QUEUE.jsonl", "DECISIONS.md", "STATUS.md",
             "a0_h0_correction.json", "b_selection.json", "c_recipes.json", "findings.json",
             "V16_FINDINGS.md", "V16_FINDINGS_AUTO.md", "deletions.jsonl", "trial_registry.csv"]
    for folder in ("raw", "logs"):
        base = HERE / folder
        if base.exists():
            names += [str(p.relative_to(HERE)) for p in sorted(base.rglob("*"))
                      if p.is_file() and p.suffix in (".jsonl", ".json", ".log", ".csv", ".txt", ".md")
                      and p.stat().st_size < 20 * 2 ** 20]
    names = [n for n in names if (HERE / n).is_file()]
    with zipfile.ZipFile(HERE / "v16_light.zip", "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(HERE / name, name)
    with zipfile.ZipFile(HERE / "v16_light.zip") as z:
        listed = set(z.namelist())
    for key in ("budget_ledger.json", "findings.json", "trial_registry.csv", "PLAN_FROZEN.md"):
        if key not in listed:
            raise RuntimeError("LIGHT_PACKAGE_INCOMPLETE:" + key)


# ------------------------------------------------------------------ caps, registry, main

def enforce_checkpoint_caps():
    ckpt = HERE / "checkpoints"
    if not ckpt.exists():
        return
    fulls = sorted([p for p in ckpt.glob("*.pt") if "endpoint" not in p.name],
                   key=lambda p: p.stat().st_mtime)
    while len(fulls) > HB["full_checkpoints_max"]:
        victim = fulls.pop(0)
        from campaign_v16.core import log_deletion
        log_deletion(victim, "full-checkpoint cap; regenerable by rerun")
        victim.unlink(missing_ok=True)
    endpoints = sorted(ckpt.glob("*_endpoint.pt"), key=lambda p: p.stat().st_mtime)
    while len(endpoints) > HB["weights_only_max"]:
        victim = endpoints.pop(0)
        from campaign_v16.core import log_deletion
        log_deletion(victim, "weights-only cap; regenerable by rerun")
        victim.unlink(missing_ok=True)


PLANNERS = {"a1_select": plan_a1_select, "b_plan": plan_b, "b_plan2": plan_b2,
            "b_select": plan_b_select, "route_b": plan_route_b, "holdout": plan_holdout}


def write_registry():
    import csv
    rows = queue_rows()
    with (HERE / "trial_registry.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "kind", "stage", "family", "a", "b",
                                          "lr", "seed", "note"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})


def run_task(task):
    if task["kind"] == "planner":
        PLANNERS[task["planner"]](task)
    elif task["kind"] == "h0_fix":
        task_h0_fix(task)
    elif task["kind"] == "branch":
        train_branch(task)
    else:
        raise RuntimeError("UNKNOWN_TASK_KIND:" + task["kind"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    if not (HERE / "run_manifest.json").exists():
        init_campaign()
    deadline = time.time() + args.max_seconds
    pause = __import__("campaign_v16.core", fromlist=["PauseBudget"]).PauseBudget()
    # Recover tasks left in "running" by a killed previous process (single-process runner;
    # the child run.lock is also reclaimed if its pid is dead).
    for row in queue_rows():
        if row["status"] == "running":
            queue_set_status(row["id"], "pending", "recovered-after-process-exit")
    lock = HERE / "run.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text(encoding="utf-8").strip())
            psutil.Process(pid)  # raises if dead
        except (ValueError, psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        else:
            raise RuntimeError("RUN_LOCK_PRESENT_LIVE_PROCESS")
    failures = {}
    while time.time() < deadline:
        if budget_exhausted():
            save_state(stop="STOP_BUDGET")
            break
        st = state()
        if st.get("stop"):
            break
        pending = [r for r in queue_rows() if r["status"] == "pending"]
        if not pending:
            save_state(final_label=derive_final_label())
            break
        task = pending[0]
        queue_set_status(task["id"], "running")
        try:
            resource("task_start:" + task["id"], force=True)
            run_task(task)
            queue_set_status(task["id"], "done")
            write_registry()
        except RuntimeError as exc:
            text = str(exc)
            if "RESOURCE_BLOCKED" in text or "STOP_BUDGET" in text:
                outcome = pause.wait_for_recovery(hard_deadline=deadline)
                if outcome is False:
                    save_state(stop="STOP_RESOURCE")
                    queue_set_status(task["id"], "pending", "paused-no-recovery")
                    break
                if outcome == "deadline":
                    queue_set_status(task["id"], "pending", "runner-deadline-during-pause")
                    break
                queue_set_status(task["id"], "pending", "resumed-after-resource")
            else:
                st = state()
                failures = st.get("task_failures", {})
                failures[task["id"]] = failures.get(task["id"], 0) + 1
                if failures[task["id"]] >= 3:
                    queue_set_status(task["id"], "failed", text[:300])
                    decision("task failed 3x (campaign-wide)", task["id"] + " " + text[:200],
                             "skip task permanently")
                else:
                    queue_set_status(task["id"], "pending", "retry:" + text[:200])
                save_state(task_failures=failures)
        write_registry()
        st = state()
        l = ledger()
        write(HERE / "STATUS.md", f"# V16 status\n\nstage={st.get('stage')} wall={wall_hours():.2f}h "
              f"device={device_task_hours():.2f}h attempts={l.get('optimizer_step_attempts')} "
              f"forwards={l.get('token_forward_equivalents')} queue_done="
              f"{sum(1 for r in queue_rows() if r['status'] == 'done')}/{len(queue_rows())}\n")
    st = state()
    if st.get("stop") or not [r for r in queue_rows() if r["status"] == "pending"]:
        if not (HERE / "findings.json").exists():
            save_state(final_label=st.get("final_label") or derive_final_label())
            finalize()
    print(json.dumps({"status": state().get("stop") or "queue-drained-or-deadline",
                      "attempts": ledger().get("optimizer_step_attempts"),
                      "forwards": ledger().get("token_forward_equivalents")}))


def derive_final_label():
    st = state()
    route = st.get("selections", {}).get("route", {})
    if st.get("stop") == "STOP_RESOURCE":
        return "STOP_RESOURCE"
    if st.get("stop") == "STOP_BUDGET":
        return "STOP_BUDGET"
    label = route.get("label")
    if label == "SIMPLE_CONTROL_EXPLAINS_WITHIN_TESTED_RANGE":
        return "COMPLETE_WITH_SIMPLE_EXPLANATION"
    if label in ("F_OR_P_SUSTAINED",):
        return "COMPLETE_WITH_SIGNAL"
    if label == "S_ONLY":
        return "COMPLETE_WITH_NO_GAIN"
    return "MIXED_OR_LIMITED"


if __name__ == "__main__":
    main()
