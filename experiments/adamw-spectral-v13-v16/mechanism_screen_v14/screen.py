"""V14 frozen-update mechanism screen. Parent trees are read only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.dont_write_bytecode = True
import numpy as np
import psutil
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V13 = ROOT / "readiness_audit_v13"
V12 = ROOT / "replay_audit_v12"
CFG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PCFG = json.loads((ROOT / "config/local_v1_lowmem.json").read_text(encoding="utf-8"))
TARGET = CFG["target"]
FROZEN = [V13 / "frozen" / f"window{i}_{'first' if i == 0 else 'main'}.pt" for i in range(4)]
SOURCE_FILES = ("screen.py", "test_screen.py", "config.json", "RUN_VERIFIED.ps1")


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
            time.sleep(0.1 * (attempt + 1))


def append(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha(path, n=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        left = n
        while left is None or left:
            b = f.read(min(left, 1048576) if left is not None else 1048576)
            if not b:
                break
            h.update(b)
            if left is not None:
                left -= len(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def source_hash():
    return digest({name: sha(HERE / name) for name in SOURCE_FILES})


def candidates():
    rows = []
    for gamma in CFG["gammas"]:
        for s in CFG["scales_scalar"]:
            rows.append({"id": f"scalar_g{gamma:g}_s{s:g}", "family": "scalar", "gamma": gamma, "s": s, "rho": None})
        for rho in CFG["rhos"]:
            for s in CFG["scales_other"]:
                rows.append({"id": f"spectral_r{rho:g}_g{gamma:g}_s{s:g}", "family": "spectral", "gamma": gamma, "s": s, "rho": rho})
        for family in ("fresh_gradient", "fresh_preconditioned"):
            for s in CFG["scales_other"]:
                rows.append({"id": f"{family}_g{gamma:g}_s{s:g}", "family": family, "gamma": gamma, "s": s, "rho": None})
    assert len(rows) == 32 and len({x["id"] for x in rows}) == 32
    return rows


def tie_key(c):
    # Predeclared simplicity: original gamma, original scale, least spectral departure.
    return (c["gamma"] != 1.0, abs(math.log2(c["s"])) if c["s"] else 9,
            c["rho"] != 0.5 if c["rho"] is not None else False, c["id"])


def dependencies():
    from readiness_audit_v13.audit import verify_inputs
    from readiness_audit_v13.common import source_hash as v13_source_hash
    verify_inputs()  # Includes original source/config/data/binding/checkpoint and V12 checks.
    run = read(V13 / "run_manifest.json")
    if run["source_hash"] != v13_source_hash():
        raise RuntimeError("V13_SOURCE_HASH_MISMATCH")
    ready = read(V13 / "readiness.json")
    if ready.get("overall_status") != "READY_FOR_MECHANISM_STUDY":
        raise RuntimeError("V13_NOT_ACCEPTED")
    old = read(ROOT / "results/local_gate.json")
    if old.get("verdict") == "PASS":
        raise RuntimeError("UNEXPECTED_PARENT_GATE")
    accepted = {run["source_hash"], *run.get("accepted_previous_source_hashes", [])}
    frozen = []
    for i, path in enumerate(FROZEN):
        if not path.is_file() or not path.with_suffix(".json").is_file():
            raise RuntimeError(f"FROZEN_UPDATE_MISSING:{path}")
        meta = read(path.with_suffix(".json"))
        actual = sha(path)
        if meta["sha256"] != actual or meta["source_hash"] not in accepted or meta["window_id"] != i:
            raise RuntimeError(f"FROZEN_UPDATE_IDENTITY_MISMATCH:{path}")
        frozen.append({"file": str(path.relative_to(ROOT)), "sha256": actual,
                       "metadata_sha256": sha(path.with_suffix(".json")), "source_hash": meta["source_hash"]})
    paths = ["readiness_audit_v13/readiness.json", "readiness_audit_v13/run_manifest.json",
             "readiness_audit_v13/parent_dependencies.json", "readiness_audit_v13/budget_ledger.json",
             "readiness_audit_v13/results/observed_measurements.json",
             "replay_audit_v12/evaluation_manifest.json", "replay_audit_v12/eval_views.jsonl",
             "artifacts/data_manifest.json", "artifacts/doc_roles.jsonl", "artifacts/eval_blocks.json",
             "config/local_v1_lowmem.json", "bindings.local.json", "results/local_gate.json"]
    paths = [p for p in paths if (ROOT / p).is_file()]
    return {"protocol_id": CFG["protocol_id"], "v13_source_hash": run["source_hash"],
            "v13_accepted_frozen_source_hashes": sorted(accepted),
            "files": {p: sha(ROOT / p) for p in paths}, "frozen": frozen,
            "review": {"path": "E:/goolge down/REVIEW.md", "sha256": sha("E:/goolge down/REVIEW.md")},
            "checkpoint_sha256": sha(ROOT / "artifacts/checkpoints/seed3407_lr0.003/step256.pt")}


def build_confirm():
    from tokenizers import Tokenizer
    from replay_audit_v12.core import normalized_role, natural_prefix
    binding = read(ROOT / "bindings.local.json")
    data = read(ROOT / "artifacts/data_manifest.json")
    old = read(ROOT / "artifacts/eval_blocks.json")
    excluded = {x["doc_id"] for role in ("lr_select", "probe_select", "probe_confirm") for x in old[role]}
    excluded.update(read(V12 / "evaluation_manifest.json")["new_probe_select_doc_ids"])
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as f:
        excluded.update(json.loads(line)["doc_id"] for line in f if '"doc_id"' in line and '"packed_' not in line)
    roles = {row["doc_id"]: row["role"] for row in
             (json.loads(line) for line in (ROOT / "artifacts/doc_roles.jsonl").open(encoding="utf-8"))}
    source = Path(binding["source_data_path"])
    if sha(source, data["scanned_bytes"]) != data["scanned_sha256"]:
        raise RuntimeError("CONSUMED_DATA_PREFIX_CHANGED")
    seen, available, consumed = set(), {}, 0
    with source.open("rb") as f:
        while consumed < data["scanned_bytes"]:
            line = f.readline(data["scanned_bytes"] - consumed + 1)
            if not line or consumed + len(line) > data["scanned_bytes"]:
                raise RuntimeError("SOURCE_SCAN_BOUNDARY_CHANGED")
            consumed += len(line)
            row = json.loads(line)
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
    tok = Tokenizer.from_file(str(Path(binding["tokenizer_path"]) / "tokenizer.json"))
    ids = sorted(available)[:CFG["confirmation_documents"]]
    views, meta = [], []
    for doc_id in ids:
        content = tok.encode(available[doc_id], add_special_tokens=False).ids
        prefix = natural_prefix(content, data["bos_token_id"], data["eos_token_id"], PCFG["runtime"]["sequence_length"])
        natural = len(content) <= PCFG["runtime"]["sequence_length"] - 1
        if not natural and prefix[-1] == data["eos_token_id"]:
            # A naturally occurring EOS ID is possible; the claim is about injection.
            pass
        views.append({"doc_id": doc_id, "role": "probe_confirm", "ids": prefix,
                      "token_count": len(prefix)-1, "natural_boundary": natural})
        meta.append({"doc_id": doc_id, "original_content_token_length": len(content),
                     "token_count": len(prefix)-1, "natural_boundary": natural,
                     "last_target_id": prefix[-1]})
    return views, meta, len(available)


def prepare():
    if (HERE / "run_manifest.json").exists():
        verify_manifest()
        return
    deps = dependencies()
    views, meta, available = build_confirm()
    with (HERE / "confirm_views.jsonl").open("w", encoding="utf-8") as f:
        for row in views:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (HERE / "confirm_documents.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=meta[0].keys()); w.writeheader(); w.writerows(meta)
    write(HERE / "candidate_list.json", candidates())
    write(HERE / "parent_dependencies.json", deps)
    write(HERE / "run_manifest.json", {"protocol_id": CFG["protocol_id"], "source_hash": source_hash(),
         "config_sha256": sha(HERE / "config.json"), "parent_dependencies_sha256": sha(HERE / "parent_dependencies.json"),
         "candidate_list_sha256": sha(HERE / "candidate_list.json"),
         "confirm_views_sha256": sha(HERE / "confirm_views.jsonl"),
         "confirm_documents_sha256": sha(HERE / "confirm_documents.csv"),
         "confirm_ids_sha256": digest([m["doc_id"] for m in meta]),
         "confirm_available_before_selection": available,
         "confirm_selection_rule": "first 128 lexicographic unused probe_confirm SHA doc IDs in consumed parent prefix",
         "confirm_seal": "dev and select phases must not open confirm_views.jsonl",
         "windows": read(V13 / "run_manifest.json")["fixed_windows"],
         "selection_rule": "per candidate mean across 4 windows of each window token-weighted NLL; ties by gamma=1, s near 1, rho near 0.5, ID",
         "budget": CFG["max_token_forward_equivalents"], "formal_steps": 0, "optimizer_steps": 0})
    write(HERE / "budget_ledger.json", {"manifest_sha256": sha(HERE / "run_manifest.json"),
         "token_forward_equivalents": 0, "forward_attempts": 0, "backward_attempts": 0,
         "svd_calls": 0, "svd_seconds": 0.0, "optimizer_step_attempts": 0,
         "gpu_seconds": 0.0, "started_at": time.time()})


def verify_manifest():
    m = read(HERE / "run_manifest.json")
    for key, actual in (("source_hash", source_hash()), ("config_sha256", sha(HERE / "config.json")),
                        ("parent_dependencies_sha256", sha(HERE / "parent_dependencies.json")),
                        ("candidate_list_sha256", sha(HERE / "candidate_list.json")),
                        ("confirm_views_sha256", sha(HERE / "confirm_views.jsonl")),
                        ("confirm_documents_sha256", sha(HERE / "confirm_documents.csv"))):
        if m[key] != actual:
            raise RuntimeError("V14_MANIFEST_CHANGED:" + key)
    if read(HERE / "parent_dependencies.json") != dependencies():
        raise RuntimeError("V14_PARENT_DEPENDENCY_CHANGED")
    if "previous_manifest_sha256" in m:
        archive = HERE / m["previous_attempt_archive"]
        old = read(archive / "run_manifest.json")
        if sha(archive / "run_manifest.json") != m["previous_manifest_sha256"]:
            raise RuntimeError("PREVIOUS_MANIFEST_ARCHIVE_CHANGED")
        if digest({name:sha(archive/name) for name in SOURCE_FILES}) != old["source_hash"]:
            raise RuntimeError("PREVIOUS_SOURCE_ARCHIVE_CHANGED")
        old_ledger = read(archive / "budget_ledger.json")
        if old_ledger["manifest_sha256"] != m["previous_manifest_sha256"]:
            raise RuntimeError("PREVIOUS_LEDGER_ARCHIVE_CHANGED")
    return m


def ledger():
    l = read(HERE / "budget_ledger.json")
    if l["manifest_sha256"] != sha(HERE / "run_manifest.json"):
        raise RuntimeError("LEDGER_MANIFEST_MISMATCH")
    return l


def resource(l, phase, admission=False, vram_limit=None):
    ram = psutil.virtual_memory().available
    rss = psutil.Process().memory_info().rss
    rt = PCFG["runtime"]
    reserved = torch.cuda.memory_reserved(0) if torch.cuda.is_initialized() else None
    min_gib = rt["system_ram_start_min_gib"] if admission else rt["system_ram_free_reserve_gib"]
    reason = ("RAM_BELOW_REQUIRED" if ram < min_gib*2**30 else
              "RSS_ABOVE_LIMIT" if rss > rt["process_rss_soft_limit_gib"]*2**30 else
              "VRAM_ABOVE_LIMIT" if vram_limit is not None and reserved is not None and reserved > vram_limit else None)
    row = {"time": time.time(), "phase": phase, "status": "BLOCKED" if reason else
           "WARNING" if not admission and ram < rt["system_ram_warning_gib"]*2**30 else "READY",
           "reason": reason, "available_ram_bytes": ram, "rss_bytes": rss,
           "cuda_reserved_bytes": reserved, "cuda_allocated_bytes": torch.cuda.memory_allocated(0) if reserved is not None else None,
           "token_forward_equivalents": l["token_forward_equivalents"]}
    append(HERE / "logs/resources.jsonl", row)
    if reason:
        raise RuntimeError("RESOURCE_BLOCKED:" + reason)
    return row


def count_forward(l, n):
    if l["token_forward_equivalents"] + n > CFG["max_token_forward_equivalents"]:
        raise RuntimeError("TOKEN_FORWARD_BUDGET_EXHAUSTED")
    l["token_forward_equivalents"] += n
    l["forward_attempts"] += 1
    write(HERE / "budget_ledger.json", l)


def normalize_to_a(x, a):
    xn = torch.linalg.vector_norm(x.double()).item()
    an = torch.linalg.vector_norm(a.double()).item()
    if xn == 0 or an == 0:
        return torch.zeros_like(a), {"zero_norm_fallback": True, "input_norm": xn, "A_norm": an}
    out = (x.double() * (an/xn)).to(a.dtype)
    return out, {"zero_norm_fallback": False, "input_norm": xn, "A_norm": an,
                 "matched_norm": torch.linalg.vector_norm(out.double()).item()}


def directions(frozen, l):
    a = frozen["A"][TARGET].float()
    start = time.perf_counter()
    u, s, vh = torch.linalg.svd(a.double(), full_matrices=False)
    head = (s[0]*torch.outer(u[:,0], vh[0])).float()
    l["svd_calls"] += 1; l["svd_seconds"] += time.perf_counter()-start
    write(HERE / "budget_ledger.json", l)
    g = frozen["stage"]["clipped"][TARGET].float()
    v = frozen["moments_after"][TARGET]["exp_avg_sq"].float()
    step = int(frozen["moments_after"][TARGET]["step"])
    beta2 = PCFG["baseline"]["betas"][1]
    vhat = v / (1 - beta2**step)
    fresh, fresh_meta = normalize_to_a(g, a)
    pre, pre_meta = normalize_to_a(g / (vhat.sqrt() + PCFG["baseline"]["eps"]), a)
    ds = {"scalar": a, "spectral_0": a-head, "spectral_0.5": a-0.5*head,
          "fresh_gradient": fresh, "fresh_preconditioned": pre}
    formula = (frozen["base"][TARGET].double() - frozen["eta"]*a.double()).float()
    formula_error = float((formula-frozen["updated"][TARGET]).abs().max())
    if formula_error > frozen["formula_tolerance"]:
        raise RuntimeError("SCALAR_FORMULA_VS_ACTUAL_UPDATE_MISMATCH")
    return ds, {"A_norm": torch.linalg.vector_norm(a.double()).item(),
                "head_norm": torch.linalg.vector_norm(head.double()).item(),
                "fresh_gradient": fresh_meta, "fresh_preconditioned": pre_meta,
                "vhat_step": step, "eta": frozen["eta"],
                "scalar_formula_vs_frozen_updated_max_abs": formula_error,
                "formula_tolerance": frozen["formula_tolerance"]}


def direction_for(c, ds):
    return ds[f"spectral_{c['rho']:g}"] if c["family"] == "spectral" else ds[c["family"]]


def assemble(model, frozen, c, ds):
    d = direction_for(c, ds)
    gamma, scale, eta = c["gamma"], c["s"], frozen["eta"]
    with torch.no_grad():
        for name, p in model.named_parameters():
            b = frozen["base"][name]
            if name == TARGET:
                value = b.double() - gamma*eta*scale*d.double()
            else:
                value = b.double() + gamma*frozen["delta"][name].double()
            p.copy_(value.to(device=p.device, dtype=p.dtype))
    return d


def dev_views():
    ids = read(V12 / "evaluation_manifest.json")["new_probe_select_doc_ids"]
    with (V12 / "eval_views.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if '"view": "new_probe_prefix"' in line]
    by_id = {r["doc_id"]: r for r in rows}
    if set(by_id) != set(ids) or len(ids) != 128:
        raise RuntimeError("DEV_VIEW_ID_MISMATCH")
    return [by_id[x] for x in ids]


def confirm_views():
    with (HERE / "confirm_views.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != CFG["confirmation_documents"]:
        raise RuntimeError("CONFIRM_VIEW_COUNT_CHANGED")
    return rows


def _unit_path(phase, window, cid):
    return HERE / "raw" / phase / f"w{window}_{cid}.jsonl"


def evaluate_unit(model, frozen, c, ds, rows, phase, l, vram_limit):
    from pilot_core import loss_sum_count
    w = frozen["window"]["window_id"]
    path = _unit_path(phase, w, c["id"])
    old = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    manifest = read(HERE / "run_manifest.json")
    accepted = {sha(HERE / "run_manifest.json"), manifest.get("previous_manifest_sha256")}
    if [x["doc_id"] for x in old] != [x["doc_id"] for x in rows[:len(old)]] or any(
            x["manifest_sha256"] not in accepted for x in old):
        raise RuntimeError("PARTIAL_RESULT_IDENTITY_MISMATCH:" + str(path))
    if len(old) == len(rows):
        return old
    assemble(model, frozen, c, ds)
    model.eval()
    started = time.perf_counter()
    for row in rows[len(old):]:
        resource(l, f"{phase}_w{w}_{c['id']}", vram_limit=vram_limit)
        ids = torch.tensor(row["ids"], device="cuda:0", dtype=torch.long)[None,:]
        count_forward(l, row["token_count"])
        with torch.no_grad():
            logits = model(ids[:,:-1], use_cache=False).logits
            loss, count = loss_sum_count(logits, ids, torch.tensor([ids.shape[1]], device="cuda:0"))
        if count != row["token_count"]:
            raise RuntimeError("MASK_OR_SHIFT_CHANGED")
        item = {"phase": phase, "window_id": w, "candidate_id": c["id"],
                "doc_id": row["doc_id"], "loss_sum": float(loss), "token_count": int(count),
                "manifest_sha256": sha(HERE / "run_manifest.json"), "evaluation_precision": "fp32_no_tf32",
                "generation_precision": frozen["generation_precision"]}
        append(path, item); old.append(item)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    l["gpu_seconds"] += seconds; write(HERE / "budget_ledger.json", l)
    append(HERE / "logs/timing.jsonl", {"phase": phase, "window": w, "candidate_id": c["id"],
                                        "seconds": seconds, "forward_count": len(rows)})
    return old


def old_calibration(window, cid, current):
    ref_state = "F01" if cid == "scalar_g1_s0" else "F11"
    path = V13 / "results/losses" / f"w{window}_ffn_new_probe_prefix_{ref_state}.jsonl"
    old = [json.loads(line) for line in path.open(encoding="utf-8")]
    if [r["doc_id"] for r in old] != [r["doc_id"] for r in current]:
        raise RuntimeError("V13_CALIBRATION_DOC_MISMATCH")
    delta = sum(r["loss_sum"] for r in current)/sum(r["token_count"] for r in current) - sum(
        r["loss_sum"] for r in old)/sum(r["token_count"] for r in old)
    max_doc = max(abs(a["loss_sum"]/a["token_count"]-b["loss_sum"]/b["token_count"])
                  for a,b in zip(current,old))
    result = {"window": window, "candidate_id": cid, "v13_state": ref_state,
              "delta_nll": delta, "max_doc_nll_abs": max_doc, "v13_sha256": sha(path)}
    append(HERE / "numerical_calibration.jsonl", result)
    if abs(delta) > CFG["numeric_resolution_nats_per_token"]:
        raise RuntimeError("V13_LOSS_CALIBRATION_FAILED")


def run_phase(phase):
    m = verify_manifest()
    if phase == "confirm":
        sel = read(HERE / "selection_frozen.json")
        if sel["manifest_sha256"] != sha(HERE / "run_manifest.json") or sel["dev_raw_sha256"] != digest(
                {str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "raw/dev").glob("*.jsonl"))}):
            raise RuntimeError("SELECTION_FROZEN_IDENTITY_MISMATCH")
        selected = [next(c for c in candidates() if c["id"] == cid)
                    for cid in dict.fromkeys([*sel["selected"].values(), "scalar_g1_s1"])]
        rows = confirm_views()
    else:
        if (HERE / "selection_frozen.json").exists():
            raise RuntimeError("DEVELOPMENT_ALREADY_FROZEN")
        selected = candidates()
        rows = dev_views()
    l = ledger()
    resource(l, "before_cuda_model_init", admission=True)
    if (ROOT / "artifacts/run.lock").exists() or (HERE / "run.lock").exists():
        raise RuntimeError("RUN_LOCK_PRESENT")
    (HERE / "run.lock").write_text(str(os.getpid()), encoding="utf-8")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA_UNAVAILABLE")
        free, _ = torch.cuda.mem_get_info(0)
        rt = PCFG["runtime"]
        vram_limit = min(rt["vram_soft_limit_gib"]*2**30, free-rt["vram_free_reserve_gib"]*2**30)
        if vram_limit <= 0:
            raise RuntimeError("VRAM_RESERVE_FAILED")
        torch.set_num_threads(rt["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from pilot_core import model_and_optimizer, load_checkpoint
        binding = read(ROOT / "bindings.local.json")
        model, opt, aliases, _ = model_and_optimizer(PCFG, binding, 0.003, "cuda:0")
        checkpoint = ROOT / "artifacts/checkpoints/seed3407_lr0.003/step256.pt"
        payload = load_checkpoint(checkpoint, model, opt, config_hash=sha(ROOT / "config/local_v1_lowmem.json"),
                                  data_hash=read(ROOT / "artifacts/data_manifest.json")["data_hash"], aliases=aliases)
        if payload["successful_optimizer_step"] != 256:
            raise RuntimeError("CHECKPOINT_STEP_CHANGED")
        # Each unit starts from all frozen base tensors; optimizer is never invoked.
        original = {n:p.detach().clone() for n,p in model.named_parameters()}
        buffer_before = {n:b.detach().clone() for n,b in model.named_buffers()}
        opt_before = {n:{k:(v.detach().clone() if torch.is_tensor(v) else v) for k,v in opt.state[p].items()}
                      for n,p in model.named_parameters() if p in opt.state}
        rng_cpu = torch.get_rng_state().clone(); rng_cuda = torch.cuda.get_rng_state_all()
        for w, path in enumerate(FROZEN):
            resource(l, f"before_window_{w}", vram_limit=vram_limit)
            frozen = torch.load(path, map_location="cpu", weights_only=False)
            if frozen["checkpoint_sha256"] != m_dep()["checkpoint_sha256"] or frozen["window"]["window_id"] != w:
                raise RuntimeError("FROZEN_CONTENT_IDENTITY_MISMATCH")
            ds, meta = directions(frozen, l)
            write(HERE / f"direction_w{w}.json", meta)
            if w == 0:
                pass
            for c in selected:
                output = evaluate_unit(model, frozen, c, ds, rows, phase, l, vram_limit)
                if phase == "dev" and c["id"] in ("scalar_g1_s0", "scalar_g1_s1"):
                    old_calibration(w, c["id"], output)
            # Check no forward mutated optimizer, buffers or random state.
            for n,p in model.named_parameters():
                if n in opt_before:
                    for k,v in opt_before[n].items():
                        now = opt.state[p][k]
                        if torch.is_tensor(v) and not torch.equal(v,now) or not torch.is_tensor(v) and v != now:
                            raise RuntimeError("OPTIMIZER_STATE_MUTATED")
            for n,b in model.named_buffers():
                if not torch.equal(b,buffer_before[n]):
                    raise RuntimeError("BUFFER_MUTATED")
            if not torch.equal(rng_cpu,torch.get_rng_state()) or any(
                    not torch.equal(a,b) for a,b in zip(rng_cuda,torch.cuda.get_rng_state_all())):
                raise RuntimeError("RNG_MUTATED")
            del frozen, ds
        with torch.no_grad():
            for n,p in model.named_parameters():
                p.copy_(original[n])
        if any(not torch.equal(p, original[n]) for n,p in model.named_parameters()):
            raise RuntimeError("MODEL_RESTORE_FAILED")
        resource(l, "phase_complete", vram_limit=vram_limit)
    finally:
        (HERE / "run.lock").unlink(missing_ok=True)


def m_dep():
    return read(HERE / "parent_dependencies.json")


def raw_hash(phase):
    return digest({str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "raw" / phase).glob("*.jsonl"))})


def aggregate(rows):
    return sum(x["loss_sum"] for x in rows)/sum(x["token_count"] for x in rows)


def select():
    m = verify_manifest()
    if (HERE / "selection_frozen.json").exists():
        raise RuntimeError("SELECTION_ALREADY_FROZEN")
    data = {}
    for c in candidates():
        ns = []
        for w in range(4):
            path = _unit_path("dev",w,c["id"])
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            if len(rows)!=128 or len({r["doc_id"] for r in rows})!=128:
                raise RuntimeError("DEV_INCOMPLETE")
            ns.append(aggregate(rows))
        data[c["id"]] = {"window_nll": ns, "mean_window_nll": sum(ns)/4}
    selected = {}
    for family in ("scalar","spectral","fresh_gradient","fresh_preconditioned"):
        pool = [c for c in candidates() if c["family"] == family]
        minimum = min(data[c["id"]]["mean_window_nll"] for c in pool)
        # Equality is exact here; no outcome-driven tolerance.
        ties = [c for c in pool if data[c["id"]]["mean_window_nll"] == minimum]
        selected[family] = min(ties,key=tie_key)["id"]
    write(HERE / "dev_scores.json", data)
    write(HERE / "selection_frozen.json", {"manifest_sha256": sha(HERE / "run_manifest.json"),
          "dev_raw_sha256": raw_hash("dev"), "dev_scores_sha256": sha(HERE / "dev_scores.json"),
          "selected": selected, "ordinary_adamw": "scalar_g1_s1", "created_at": time.time(),
          "confirmation_opened": False})


def bootstrap(rows_by_cid, selected):
    ids = [r["doc_id"] for r in rows_by_cid[selected["scalar"]][0]]
    for cid, window_rows in rows_by_cid.items():
        if any([r["doc_id"] for r in wr]!=ids for wr in window_rows):
            raise RuntimeError("CONFIRM_PAIRING_CHANGED")
        if any([r["token_count"] for r in wr] != [r["token_count"] for r in window_rows[0]] for wr in window_rows):
            raise RuntimeError("CONFIRM_MASK_CHANGED")
    def nll(cid, indices):
        return sum(sum(rows_by_cid[cid][w][i]["loss_sum"] for i in indices)/
                   sum(rows_by_cid[cid][w][i]["token_count"] for i in indices) for w in range(4))/4
    rng = np.random.default_rng(CFG["bootstrap_seed"])
    n = len(ids)
    variants = {"spectral": selected["spectral"], "fresh_gradient": selected["fresh_gradient"],
                "fresh_preconditioned": selected["fresh_preconditioned"], "ordinary_adamw": "scalar_g1_s1"}
    result = {}
    for name,cid in variants.items():
        point = nll(selected["scalar"],range(n))-nll(cid,range(n))
        draws = []
        for _ in range(CFG["bootstrap_repetitions"]):
            idx = rng.integers(0,n,n)
            draws.append(nll(selected["scalar"],idx)-nll(cid,idx))
        lo,hi = np.quantile(draws,[0.025,0.975])
        result[name] = {"scalar_minus_candidate_nll": point, "ci95_lower":float(lo),"ci95_upper":float(hi)}
    return result


def finalize():
    m = verify_manifest()
    sel = read(HERE / "selection_frozen.json")
    if sel["manifest_sha256"] != sha(HERE / "run_manifest.json") or sel["dev_raw_sha256"] != raw_hash("dev"):
        raise RuntimeError("SELECTION_IDENTITY_MISMATCH")
    chosen = sel["selected"]
    ids = list(dict.fromkeys([*chosen.values(), "scalar_g1_s1"]))
    values = {}
    for cid in ids:
        values[cid] = []
        for w in range(4):
            path = _unit_path("confirm",w,cid)
            rows = [json.loads(line) for line in path.open(encoding="utf-8")]
            if len(rows)!=CFG["confirmation_documents"]:
                raise RuntimeError("CONFIRM_INCOMPLETE:"+str(path))
            values[cid].append(rows)
    nll = {cid:sum(aggregate(wr) for wr in windows)/4 for cid,windows in values.items()}
    effects = bootstrap(values,chosen)
    cal = [json.loads(line) for line in (HERE / "numerical_calibration.jsonl").open(encoding="utf-8")]
    if len({(r["window"],r["candidate_id"]) for r in cal}) != 8:
        raise RuntimeError("CALIBRATION_INCOMPLETE")
    max_error = max(abs(r["delta_nll"]) for r in cal)
    spectral = effects["spectral"]
    scalar = next(c for c in candidates() if c["id"]==chosen["scalar"])
    status = ("SCALE_RANGE_UNRESOLVED" if scalar["s"]==2.0 else
              "EXPLORATORY_SIGNAL" if spectral["scalar_minus_candidate_nll"]>CFG["screen_point_threshold_nats_per_token"]
              and spectral["ci95_lower"]>max(CFG["numeric_resolution_nats_per_token"],max_error) else
              "PARTIAL_UNRESOLVED" if abs(spectral["ci95_lower"])<=max(CFG["numeric_resolution_nats_per_token"],max_error)
              else "NO_RELATIVE_SPECTRAL_SIGNAL")
    l=ledger()
    result={"protocol_id":CFG["protocol_id"],"status":status,"selection":chosen,"confirmation_nll":nll,
            "paired_effects":effects,"numeric_calibration_max_abs_nll":max_error,
            "budget":l,"historical_v13_budget":read(V13/"budget_ledger.json"),
            "manifest_sha256":sha(HERE/"run_manifest.json"),"dev_raw_sha256":raw_hash("dev"),
            "confirm_raw_sha256":raw_hash("confirm"),"parent_dependencies_sha256":sha(HERE/"parent_dependencies.json")}
    write(HERE/"findings.json",result)
    lines=["# V14 谱修正必要性判别", "", f"状态：**{status}**。父 gate 保持 STOP；本轮不是终点训练结论。", "",
           "## 固定设计与身份", "", f"协议 `{CFG['protocol_id']}`；父依赖见 `parent_dependencies.json`；manifest SHA256 `{result['manifest_sha256']}`。",
           f"开发 128 个 V12 `new_probe_prefix` 文档；确认 128 个预先按 doc_id 排序封存的未用 `probe_confirm` 文档。4 个训练窗口只作为固定构造窗口，不作独立 seed。", "",
           "## 选择与确认原始数字", "", "| 家族 | 冻结候选 | 确认 NLL | 标量减候选 | 配对文档 CI95 |", "|---|---|---:|---:|---:|"]
    for family,cid in chosen.items():
        e=effects.get(family)
        lines.append(f"| {family} | `{cid}` | {nll[cid]:.9f} | {e['scalar_minus_candidate_nll']:.9g} | [{e['ci95_lower']:.9g}, {e['ci95_upper']:.9g}] |" if e else
                     f"| {family} | `{cid}` | {nll[cid]:.9f} | 0 | 同一标量参照 |")
    e=effects["ordinary_adamw"]
    lines.append(f"| ordinary AdamW | `scalar_g1_s1` | {nll['scalar_g1_s1']:.9f} | {e['scalar_minus_candidate_nll']:.9g} | [{e['ci95_lower']:.9g}, {e['ci95_upper']:.9g}] |")
    lines += ["", "## 判读", "", "主比较是冻结最优标量减冻结谱候选；正值才表示谱候选更好。两种无 SVD 控制也使用同一确认文档与窗口。区间只描述文档抽样不确定性，不代表训练 seed 不确定性。",
              f"V13 F01/F11 对齐的最大聚合 NLL 偏差 `{max_error:.9g}`；局部参考阈值 `1e-5` nats/token，数值分辨率 `2e-6`。",
              "选择开发结果见 `dev_scores.json`；逐文档原始 loss_sum/count 在 `raw/`。本轮无新的 optimizer.step；标签只用自然文档前缀。",
              "", "## 成本与未测", "", f"新前向 token-equivalents `{l['token_forward_equivalents']}` / 5,000,000；前向 `{l['forward_attempts']}`、反向 `{l['backward_attempts']}`、SVD `{l['svd_calls']}`、GPU optimizer.step `{l['optimizer_step_attempts']}`；计时见 `logs/timing.jsonl`，资源见 `logs/resources.jsonl`。",
              "尚未测跨 seed、完整训练终点、预测器、其他层或注意力；本轮正负均止步。"]
    (HERE/"V14_FINDINGS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    package=["screen.py","test_screen.py","config.json","RUN_VERIFIED.ps1","parent_dependencies.json",
             "run_manifest.json","candidate_list.json","confirm_documents.csv","selection_frozen.json",
             "dev_scores.json","numerical_calibration.jsonl","budget_ledger.json","findings.json","V14_FINDINGS.md"]
    package += [str(p.relative_to(HERE)) for folder in ("raw/dev","raw/confirm","logs") for p in sorted((HERE/folder).glob("*")) if p.is_file()]
    package += [str(p.relative_to(HERE)) for p in sorted(HERE.glob("direction_w*.json"))]
    package += [str(p.relative_to(HERE)) for p in sorted(HERE.glob("tests*.log"))]
    with zipfile.ZipFile(HERE/"v14_light.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in package:
            z.write(HERE/name,name)
    print(json.dumps({"status":status,"selected":chosen,"budget":l["token_forward_equivalents"]},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("phase",choices=["prepare","dev","select","confirm","finalize","status"])
    a=p.parse_args()
    if a.phase=="prepare": prepare()
    elif a.phase=="dev": run_phase("dev")
    elif a.phase=="select": select()
    elif a.phase=="confirm": run_phase("confirm")
    elif a.phase=="finalize": finalize()
    else: print(json.dumps({"manifest":read(HERE/"run_manifest.json"),"ledger":ledger()},ensure_ascii=False))


if __name__=="__main__":
    main()
