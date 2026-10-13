"""Independent, bounded CPU audit of immutable V1.1 artifacts and source window."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHILD_SOURCES = ("__init__.py", "core.py", "replay.py", "cli.py",
                 "test_v12.py", "run_parent_tests_isolated.py", "config.json")
ROLE_NAMES = ("train", "lr_select", "probe_select", "probe_confirm",
              "branch_eval_reserved", "final_holdout_reserved")
ROLE_CUTS = (70, 76, 82, 94, 97, 100)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        left = limit
        while left is None or left > 0:
            chunk = stream.read(min(left, 1024 * 1024) if left is not None else 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if left is not None:
                left -= len(chunk)
    return digest.hexdigest()


def sha_object(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def child_source_hash() -> str:
    return sha_object({name: sha_file(HERE / name) for name in CHILD_SOURCES})


def parent_source_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SOURCE_FILES"
                                                   for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"PARENT_SOURCE_LIST_MISSING: {path}")


def require_equal(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"PARENT_MISMATCH: {label}: actual={actual!r}, expected={expected!r}")


def verify_parent(*, deep_checkpoints: bool = True) -> dict:
    """Verify byte hashes before trusting any parent checkpoint or CSV."""
    cfg_path = ROOT / "config" / "local_v1_lowmem.json"
    cfg = read_json(cfg_path)
    child_cfg = read_json(HERE / "config.json")
    require_equal(cfg["protocol_id"], child_cfg["parent_protocol_id"], "parent protocol")
    binding = read_json(ROOT / "bindings.local.json")
    require_equal(Path(sys.executable).resolve(), Path(binding["python_executable"]).resolve(), "Python binding")
    source_lists = [parent_source_names(ROOT / name) for name in ("pilot_core.py", "prepare_data.py")]
    require_equal(source_lists[0], source_lists[1], "two parent SOURCE_FILES lists")
    source_hash = sha_object({name: sha_file(ROOT / name) for name in source_lists[0]})
    config_hash = sha_file(cfg_path)
    binding_hash = sha_object(binding)
    data = read_json(ROOT / "artifacts" / "data_manifest.json")
    require_equal(data["status"], "READY", "parent data status")
    for field, actual in (("source_hash", source_hash), ("config_hash", config_hash)):
        require_equal(data[field], actual, f"data {field}")
    data_payload = {k: v for k, v in data.items() if k not in
                    ("status", "data_hash", "prepare_seconds", "ram_available_at_end_bytes")}
    require_equal(data["data_hash"], sha_object(data_payload), "data self hash")
    source = Path(binding["source_data_path"])
    if not source.is_file():
        raise RuntimeError("PARENT_DATA_MISSING: bound source absent")
    stat = source.stat()
    require_equal((stat.st_size, stat.st_mtime_ns),
                  (data["source_size"], data["source_mtime_ns"]), "source metadata")
    require_equal(sha_file(source, data["scanned_bytes"]), data["scanned_sha256"], "source consumed window")
    for name, key in (("train_tokens.u32", "train_cache_sha256"),
                      ("eval_blocks.json", "eval_cache_sha256"),
                      ("doc_roles.jsonl", "doc_assignments_sha256")):
        require_equal(sha_file(ROOT / "artifacts" / name), data[key], f"parent {name} SHA")
    tokenizer_files = [Path(binding["tokenizer_path"]) / name for name in
                       ("tokenizer.json", "tokenizer_config.json")]
    require_equal(sha_object({p.name: sha_file(p) for p in tokenizer_files}),
                  binding["tokenizer_fingerprint"], "tokenizer fingerprint")
    repo = Path(binding["repo_root"])
    source_files = (repo / "model" / "model_minimind.py", repo / "trainer" / "train_pretrain.py")
    require_equal(sha_object({str(p.relative_to(repo)): sha_file(p) for p in source_files}),
                  binding["source_fingerprint"], "MiniMind source fingerprint")
    gate = read_json(ROOT / "results" / "local_gate.json")
    require_equal((gate["verdict"], gate["next_action"]),
                  ("BASELINE_TUNING_UNRESOLVED", "STOP"), "frozen parent gate")
    candidate = read_json(ROOT / "results" / "candidate_frozen.json")
    require_equal(candidate["candidate_hash"],
                  sha_object({k: v for k, v in candidate.items() if k != "candidate_hash"}),
                  "candidate self hash")
    require_equal(gate["candidate_hash"], candidate["candidate_hash"], "gate candidate")
    fingerprints = {"source_hash": source_hash, "config_hash": config_hash,
                    "binding_hash": binding_hash, "data_hash": data["data_hash"]}
    for label, obj in (("gate", gate), ("candidate", candidate)):
        for key, value in fingerprints.items():
            require_equal(obj[key], value, f"{label} {key}")
    result_names = ("baseline_runs.csv", "probe_grid.csv", "probe_confirm.csv",
                    "direction_signals.csv", "output_effects.csv", "spectral_metrics.csv",
                    "local_gate.json", "candidate_frozen.json")
    result_hashes = {name: sha_file(ROOT / "results" / name) for name in result_names}
    checkpoint_rows = []
    for peak in cfg["baseline"]["peak_lr_grid"]:
        run_dir = ROOT / "artifacts" / "checkpoints" / f"seed3407_lr{peak:g}"
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"PARENT_CHECKPOINT_MISSING: {manifest_path}")
        manifest = read_json(manifest_path)
        require_equal(manifest["status"], "SUCCESS", f"baseline {peak} status")
        for key, value in fingerprints.items():
            require_equal(manifest[key], value, f"baseline {peak} {key}")
        for step in (256, 512) if peak == child_cfg["peak_lr"] else (512,):
            path = run_dir / f"step{step}.pt"
            if not path.is_file():
                raise RuntimeError(f"PARENT_CHECKPOINT_MISSING: {path}")
            measured = sha_file(path)
            require_equal(measured, manifest["checkpoint_hashes"][str(step)], f"checkpoint {peak}/{step} SHA")
            if deep_checkpoints:
                import torch
                payload = torch.load(path, map_location="cpu", weights_only=False)
                for key in ("source_hash", "config_hash", "data_hash"):
                    require_equal(payload[key], fingerprints[key], f"checkpoint {peak}/{step} {key}")
                require_equal(payload["successful_optimizer_step"], step, f"checkpoint {peak}/{step} step")
                del payload
            checkpoint_rows.append({"peak_lr": peak, "step": step,
                                    "path": str(path), "sha256": measured})
    return {"status": "VERIFIED", "parent_protocol_id": cfg["protocol_id"],
            "parent_fingerprints": fingerprints, "gate_verdict": gate["verdict"],
            "gate_next_action": gate["next_action"], "candidate_hash": candidate["candidate_hash"],
            "checkpoint_rows": checkpoint_rows,
            "parent_result_hashes": result_hashes,
            "parent_review_sha256": sha_file(ROOT / "REVIEW.md"),
            "source_window_bytes": data["scanned_bytes"],
            "source_window_sha256": data["scanned_sha256"],
            "child_protocol_id": child_cfg["protocol_id"],
            "child_source_hash": child_source_hash(),
            "child_config_hash": sha_file(HERE / "config.json")}


def csv_rows(name: str) -> list[dict]:
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _candidate_key(row):
    return (row["kind"], float(row["rho"]), float(row["scale"]),
            None if row["q"] == "" else float(row["q"]))


def _weighted_nll(rows) -> float:
    return sum(float(row["loss_sum"]) for row in rows) / sum(int(row["token_count"]) for row in rows)


def _select_scalar(rows, tolerance: float):
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] == "scalar":
            groups[_candidate_key(row)].append(row)
    values = {key: _weighted_nll(group) for key, group in groups.items()}
    best = min(values.values())
    tied = [key for key, value in values.items() if value <= best + tolerance]
    chosen = sorted(tied, key=lambda key: (abs(math.log(key[1])) if key[1] else 0,
                                           abs(math.log(key[2])) if key[2] else float("inf"),
                                           str(key)))[0]
    return chosen, values[chosen]


def reanalyze_parent() -> dict:
    cfg = read_json(ROOT / "config" / "local_v1_lowmem.json")
    gate = read_json(ROOT / "results" / "local_gate.json")
    candidate = read_json(ROOT / "results" / "candidate_frozen.json")
    grid = csv_rows("probe_grid.csv")
    cells = defaultdict(list)
    for row in grid:
        cells[(int(row["checkpoint_step"]), row["parameter_name"])].append(row)
    cell_rows = []
    for (step, name), rows in sorted(cells.items()):
        best_key, best_nll = _select_scalar(rows, candidate["epsilon_null"])
        docs = {row["doc_id"] for row in rows if row["kind"] == "scalar"}
        reps = {int(row["replicate"]) for row in rows}
        cell_rows.append({"step": step, "parameter_name": name,
                          "best_scalar_scale": best_key[2], "best_scalar_nll": best_nll,
                          "distinct_documents": len(docs), "gradient_build_replicates": len(reps),
                          "document_replicate_rows": len(docs) * len(reps)})
    confirm = csv_rows("probe_confirm.csv")
    def selected_rows(choice):
        key = (choice["kind"], float(choice["rho"]), float(choice["scale"]), choice["q"])
        return [row for row in confirm if _candidate_key(row) == key]
    scalar, spectral = selected_rows(candidate["scalar"]), selected_rows(candidate["spectral"])
    def group_docs(rows):
        grouped = defaultdict(lambda: [0.0, 0])
        for row in rows:
            item = grouped[row["doc_id"]]
            item[0] += float(row["loss_sum"])
            item[1] += int(row["token_count"])
        return grouped
    sg, pg = group_docs(scalar), group_docs(spectral)
    ids = sorted(set(sg) & set(pg))
    if len(ids) != cfg["data"]["probe_confirm_blocks"] or set(sg) != set(pg):
        raise RuntimeError("PARENT_CONFIRM_DOC_MISMATCH")
    delta = sum(sg[i][0] for i in ids) / sum(sg[i][1] for i in ids) - \
            sum(pg[i][0] for i in ids) / sum(pg[i][1] for i in ids)
    rng = np.random.default_rng(cfg["gate"]["bootstrap_seed"])
    samples = []
    for _ in range(cfg["gate"]["paired_bootstrap_repetitions"]):
        picked = rng.choice(ids, size=len(ids), replace=True)
        samples.append(sum(sg[i][0] for i in picked) / sum(sg[i][1] for i in picked) -
                       sum(pg[i][0] for i in picked) / sum(pg[i][1] for i in picked))
    lower, upper = np.quantile(samples, [0.025, 0.975])
    old = gate["primary_statistics"]
    for key, actual in (("delta_nats_per_token", delta), ("ci95_lower", float(lower)),
                        ("ci95_upper", float(upper))):
        if abs(old[key] - actual) > 1e-12:
            raise RuntimeError(f"PARENT_GATE_RECALC_MISMATCH: {key}: {old[key]} vs {actual}")
    baseline = csv_rows("baseline_runs.csv")
    lr = {str(peak): _weighted_nll([row for row in baseline if float(row["peak_lr"]) == peak])
          for peak in cfg["baseline"]["peak_lr_grid"]}
    signals = csv_rows("direction_signals.csv")
    weighted = []
    for key in sorted(cells):
        rs = [r for r in signals if (int(r["checkpoint_step"]), r["parameter_name"]) == key]
        if rs:
            denom = sum(int(r["token_count"]) for r in rs)
            weighted.append({"step": key[0], "parameter_name": key[1],
                             "distinct_documents": len({r["doc_id"] for r in rs}),
                             "gradient_build_replicates": len({r["replicate"] for r in rs}),
                             "total_tokens": denom,
                             "token_weighted_dot_head": sum(float(r["gradient_dot_head"]) * int(r["token_count"]) for r in rs) / denom,
                             "token_weighted_dot_rest": sum(float(r["gradient_dot_rest"]) * int(r["token_count"]) for r in rs) / denom})
    return {"status": "RECOMPUTED", "parent_gate_verdict": gate["verdict"],
            "parent_next_action": gate["next_action"], "cell_best_scalar": cell_rows,
            "lr_select_token_weighted_nll": lr,
            "main_confirm": {"distinct_documents": len(ids), "gradient_build_replicates": 2,
                             "delta_nats_per_token": delta, "ci95_lower": float(lower),
                             "ci95_upper": float(upper), "delta_local": old["delta_local"],
                             "status": "PASS" if delta > old["delta_local"] and lower > 0 else "NO_LOCAL_SIGNAL"},
            "direction_signals_token_weighted": weighted}


def normalized_role(text: str) -> tuple[str, str, str]:
    normalized = " ".join(text.split())
    doc_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    bucket = int(doc_id[:16], 16) % 100
    role = next(name for cut, name in zip(ROLE_CUTS, ROLE_NAMES) if bucket < cut)
    return normalized, doc_id, role


def natural_prefix(content_ids: list[int], bos: int, eos: int, seq: int) -> list[int]:
    return ([bos] + content_ids + [eos])[:seq + 1]


def legacy_window(content_ids: list[int], bos: int, eos: int, seq: int) -> list[int]:
    return [bos] + content_ids[:seq - 1] + [eos]


def training_windows(token_count: int, seq: int, batch: int, count: int) -> list[dict]:
    width = batch * seq + 1
    starts = [((i * (token_count - width)) // (count - 1)) // seq * seq
              for i in range(count)]
    if len(set(starts)) != count or any(b - a < width for a, b in zip(starts, starts[1:])):
        raise RuntimeError("DATA_INSUFFICIENT: nonoverlapping train windows unavailable")
    return [{"window_id": i, "start_token": start, "end_token_exclusive": start + width,
             "target_tokens": batch * seq} for i, start in enumerate(starts)]


def packed_windows(selected: list[dict], bos: int, eos: int, seq: int) -> list[dict]:
    tokens, provenance, true_boundaries = [], [], []
    for doc in selected:
        ids = [bos] + doc["content_ids"] + [eos]
        tokens.extend(ids)
        provenance.extend([doc["doc_id"]] * len(ids))
        true_boundaries.extend([False] * (len(ids) - 1) + [True])
    windows = []
    for start in range(0, len(tokens) - 1, seq):
        ids = tokens[start:start + seq + 1]
        if len(ids) < 2:
            break
        sources = sorted(set(provenance[start:start + len(ids)]))
        windows.append({"view": "new_probe_packed", "doc_id": f"packed_{len(windows):04d}",
                        "ids": ids, "token_count": len(ids) - 1,
                        "source_doc_ids": sources,
                        "natural_boundary": true_boundaries[start + len(ids) - 1],
                        "descriptive_only": True})
    # A shared source document joins windows into the same uncertainty cluster.
    parent = {doc["doc_id"]: doc["doc_id"] for doc in selected}
    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for row in windows:
        lead = row["source_doc_ids"][0]
        for other in row["source_doc_ids"][1:]:
            parent[root(other)] = root(lead)
    groups = defaultdict(list)
    for doc_id in parent:
        groups[root(doc_id)].append(doc_id)
    labels = {key: sha_object(sorted(members))[:16] for key, members in groups.items()}
    for row in windows:
        row["cluster_id"] = labels[root(row["source_doc_ids"][0])]
    return windows


def prepare_views(parent: dict) -> dict:
    """Read exactly the consumed source window; write only child artifacts."""
    from tokenizers import Tokenizer
    binding = read_json(ROOT / "bindings.local.json")
    data = read_json(ROOT / "artifacts" / "data_manifest.json")
    cfg = read_json(ROOT / "config" / "local_v1_lowmem.json")
    child_cfg = read_json(HERE / "config.json")
    tokenizer = Tokenizer.from_file(str(Path(binding["tokenizer_path"]) / "tokenizer.json"))
    bos, eos = data["bos_token_id"], data["eos_token_id"]
    seq = cfg["runtime"]["sequence_length"]
    old = read_json(ROOT / "artifacts" / "eval_blocks.json")
    expected = {role: {row["doc_id"]: row for row in old[role]}
                for role in ("lr_select", "probe_select", "probe_confirm")}
    seen, selected_source, candidates = set(), {}, []
    consumed = 0
    with Path(binding["source_data_path"]).open("rb") as source:
        while consumed < data["scanned_bytes"]:
            line = source.readline(min(cfg["data"]["max_jsonl_line_bytes"] + 1,
                                       data["scanned_bytes"] - consumed + 1))
            if not line or consumed + len(line) > data["scanned_bytes"]:
                raise RuntimeError("PARENT_SOURCE_WINDOW_MISMATCH")
            consumed += len(line)
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                continue
            text, doc_id, role = normalized_role(row["text"])
            if not text or doc_id in seen:
                continue
            seen.add(doc_id)
            if doc_id in expected.get(role, {}):
                selected_source[doc_id] = (role, text)
            if role == "probe_select" and doc_id not in expected["probe_select"]:
                candidates.append((doc_id, text))
    require_equal(consumed, data["scanned_bytes"], "consumed source bytes")
    selected_ids = set().union(*(set(group) for group in expected.values()))
    require_equal(set(selected_source), selected_ids, "all 112 legacy documents found")
    if len(selected_ids) != 112:
        raise RuntimeError(f"PARENT_EVAL_COUNT_MISMATCH: {len(selected_ids)}")
    with (ROOT / "artifacts" / "doc_roles.jsonl").open(encoding="utf-8") as stream:
        roles = {row["doc_id"]: row["role"] for row in (json.loads(line) for line in stream)}
    for doc_id, (role, _) in selected_source.items():
        require_equal(roles.get(doc_id), role, f"legacy role {doc_id}")
    candidates.sort(key=lambda item: item[0])
    wanted = child_cfg["new_probe_select_documents"]
    if len(candidates) < wanted:
        raise RuntimeError(f"DATA_INSUFFICIENT: only {len(candidates)} unused probe_select documents")
    added = candidates[:wanted]
    for doc_id, _ in added:
        require_equal(roles.get(doc_id), "probe_select", f"new probe role {doc_id}")
        if doc_id in selected_ids:
            raise RuntimeError(f"NEW_PROBE_LEGACY_OVERLAP: {doc_id}")
    legacy_audit, views, selected_docs = [], [], []
    for doc_id, (role, text) in sorted(selected_source.items()):
        content = tokenizer.encode(text, add_special_tokens=False).ids
        legacy = legacy_window(content, bos, eos, seq)
        require_equal(legacy, expected[role][doc_id]["ids"], f"legacy eval IDs {doc_id}")
        natural = natural_prefix(content, bos, eos, seq)
        truncated = len(content) > seq - 1
        legacy_audit.append({"doc_id": doc_id, "role": role,
                             "original_content_token_length": len(content),
                             "legacy_window_length": len(legacy),
                             "legacy_target_count": len(legacy) - 1,
                             "legacy_last_target_id": legacy[-1],
                             "legacy_last_target_is_artificial_eos": truncated,
                             "natural_prefix_window_length": len(natural),
                             "natural_prefix_target_count": len(natural) - 1,
                             "natural_prefix_last_target_id": natural[-1],
                             "natural_prefix_contains_true_eos": len(content) <= seq - 1,
                             "natural_prefix_ends_at_document_boundary": len(content) <= seq - 1})
        for view, ids in (("legacy", legacy), ("natural_prefix", natural)):
            views.append({"view": view, "doc_id": doc_id, "role": role, "ids": ids,
                          "token_count": len(ids) - 1, "source_doc_ids": [doc_id],
                          "cluster_id": doc_id, "natural_boundary": not truncated})
    for doc_id, text in added:
        content = tokenizer.encode(text, add_special_tokens=False).ids
        ids = natural_prefix(content, bos, eos, seq)
        selected_docs.append({"doc_id": doc_id, "role": "probe_select", "content_ids": content,
                              "original_content_token_length": len(content),
                              "natural_prefix_contains_true_eos": len(content) <= seq - 1,
                              "natural_prefix_ends_at_document_boundary": len(content) <= seq - 1,
                              "natural_prefix_target_count": len(ids) - 1})
        views.append({"view": "new_probe_prefix", "doc_id": doc_id,
                      "role": "probe_select", "ids": ids, "token_count": len(ids) - 1,
                      "source_doc_ids": [doc_id], "cluster_id": doc_id,
                      "natural_boundary": len(content) <= seq - 1})
    views.extend(packed_windows(selected_docs, bos, eos, seq))
    audit_path = HERE / "legacy_truncation_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(legacy_audit[0]))
        writer.writeheader()
        writer.writerows(legacy_audit)
    selected_path = HERE / "selected_probe_docs.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[k for k in selected_docs[0] if k != "content_ids"])
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k != "content_ids"} for row in selected_docs])
    views_path = HERE / "eval_views.jsonl"
    with views_path.open("w", encoding="utf-8") as stream:
        for row in views:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    offsets = training_windows(data["train_token_count"], seq, cfg["runtime"]["effective_batch_size"],
                               child_cfg["build_window_count"])
    view_counts = {name: sum(row["view"] == name for row in views) for name in child_cfg["views"]}
    view_token_counts = {name: sum(row["token_count"] for row in views if row["view"] == name)
                         for name in child_cfg["views"]}
    manifest = {"status": "PREPARED_NO_GPU_REPLAY", "child_protocol_id": child_cfg["protocol_id"],
                "child_source_hash": child_source_hash(), "child_config_hash": sha_file(HERE / "config.json"),
                "parent_dependencies_hash": sha_object(parent),
                "legacy_truncation_audit_sha256": sha_file(audit_path),
                "selected_probe_docs_sha256": sha_file(selected_path),
                "eval_views_sha256": sha_file(views_path), "view_counts": view_counts,
                "view_token_counts": view_token_counts,
                "legacy_genuinely_truncated_count": sum(row["legacy_last_target_is_artificial_eos"] for row in legacy_audit),
                "new_probe_select_doc_ids": [row["doc_id"] for row in selected_docs],
                "new_probe_select_role": "probe_select",
                "new_probe_select_rule": "lexicographically smallest 128 SHA256 doc_id among unused probe_select in parent consumed source window",
                "packed_source_cluster_count": len({row["cluster_id"] for row in views if row["view"] == "new_probe_packed"}),
                "train_build_windows": offsets,
                "packed_uncertainty": "descriptive_only; if estimated, cluster by overlapping source document groups",
                "evaluation_mask": "one shift, explicit length, same view and mask within comparison",
                "source_window_sha256": data["scanned_sha256"]}
    write_json(HERE / "evaluation_manifest.json", manifest)
    return manifest
