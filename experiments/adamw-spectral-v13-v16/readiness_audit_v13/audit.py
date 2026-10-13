"""Read-only parent/V1.2 evidence verification and child manifests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from replay_audit_v12.core import verify_parent, reanalyze_parent

from .common import HERE, ROOT, V12, read_json, sha_file, sha_object, source_hash, write_json

SOURCE_NAMES = ("__init__.py", "common.py", "audit.py", "engine.py", "cli.py",
                "tests_v13.py", "run_parent_tests_isolated.py", "config.json", "RUN_VERIFIED.ps1")


PROTECTED = (
    "CODEX_RESEARCH_READINESS_AUDIT_V13.md", "bindings.local.json",
    "config/local_v1_lowmem.json", "artifacts/data_manifest.json",
    "artifacts/train_tokens.u32", "artifacts/eval_blocks.json", "artifacts/doc_roles.jsonl",
    "results/local_gate.json", "results/candidate_frozen.json",
    "results/baseline_runs.csv", "results/probe_grid.csv", "results/probe_confirm.csv",
    "results/direction_signals.csv", "results/output_effects.csv", "results/spectral_metrics.csv",
    "replay_audit_v12/core.py", "replay_audit_v12/replay.py", "replay_audit_v12/cli.py",
    "replay_audit_v12/config.json", "replay_audit_v12/evaluation_manifest.json",
    "replay_audit_v12/eval_views.jsonl", "replay_audit_v12/selected_probe_docs.csv",
    "replay_audit_v12/legacy_truncation_audit.csv", "replay_audit_v12/parent_dependencies.json",
    "replay_audit_v12/parent_reanalysis.json", "replay_audit_v12/budget_ledger.json",
    "replay_audit_v12/results/lr_reeval.json", "replay_audit_v12/results/replay_losses.jsonl",
    "replay_audit_v12/results/replay_status.json", "replay_audit_v12/logs/resources.jsonl",
    "replay_audit_v12/logs/diagA_repeat_builds.txt", "replay_audit_v12/logs/diagB_full_pair.txt",
    "replay_audit_v12/logs/diagC_causal.txt", "replay_audit_v12/logs/diagD_loadstate.txt",
)


def protected_hashes():
    result = {}
    for name in PROTECTED:
        path = ROOT / name
        result[name] = sha_file(path) if path.is_file() else None
    return result


def verify_inputs():
    parent = verify_parent(deep_checkpoints=True)
    old = reanalyze_parent()
    v12_parent = read_json(V12 / "parent_dependencies.json")
    if parent != v12_parent:
        raise RuntimeError("V12_PARENT_DEPENDENCIES_MISMATCH")
    evaluation = read_json(V12 / "evaluation_manifest.json")
    if evaluation["child_source_hash"] != parent["child_source_hash"]:
        raise RuntimeError("V12_SOURCE_CHANGED")
    if evaluation["parent_dependencies_hash"] != sha_object(parent):
        raise RuntimeError("V12_PARENT_HASH_CHANGED")
    for name, key in (("eval_views.jsonl", "eval_views_sha256"),
                      ("selected_probe_docs.csv", "selected_probe_docs_sha256"),
                      ("legacy_truncation_audit.csv", "legacy_truncation_audit_sha256")):
        if sha_file(V12 / name) != evaluation[key]:
            raise RuntimeError("V12_EVAL_VIEW_HASH_MISMATCH:" + name)
    losses = [json.loads(line) for line in (V12 / "results" / "replay_losses.jsonl").open(encoding="utf-8")]
    if len(losses) != 192 or any(row["build_window_id"] != -1 for row in losses):
        raise RuntimeError("V12_LR_RAW_ROWS_MISMATCH")
    lr = read_json(V12 / "results" / "lr_reeval.json")
    rebuilt = {}
    for peak in (0.0003, 0.001, 0.003):
        rebuilt[str(peak)] = {}
        for view in ("legacy", "natural_prefix"):
            rows = [r for r in losses if r["state"] == f"checkpoint_lr{peak:g}" and r["view"] == view]
            if len(rows) != 32 or len({r["doc_id"] for r in rows}) != 32:
                raise RuntimeError("V12_LR_ROW_COVERAGE_MISMATCH")
            result = sum(float(r["loss_sum"]) for r in rows) / sum(int(r["token_count"]) for r in rows)
            if abs(result - lr[str(peak)][view]) > 1e-12:
                raise RuntimeError("V12_LR_AGGREGATE_MISMATCH")
            rebuilt[str(peak)][view] = result
    ledger = read_json(V12 / "budget_ledger.json")
    if ledger["gpu_optimizer_calls_attempted"] != 5 or ledger["token_forward_equivalents"] != 46104:
        raise RuntimeError("V12_HISTORY_LEDGER_UNEXPECTED")
    # diagA has three logged 2048-target gradient builds and no V1.2 ledger calls.
    # They are historical reconstruction, not native ledger entries.
    history = {"v12_native_ledger": ledger,
               "v12_lr_raw_rows": 192, "v12_lr_recomputed": rebuilt,
               "v12_four_state_raw_rows": 0,
               "diagA_missing_from_native_ledger": {"builds": 3, "target_tokens_each": 2048,
                   "token_forward_equivalents_reconstructed": 6144,
                   "backward_calls_reconstructed": 12,
                   "source": "replay_audit_v12/logs/diagA_repeat_builds.txt",
                   "status": "INFERRED_FROM_SCRIPT_AND_LOG_NOT_ORIGINAL_COUNTER"},
               "historical_known_minimum_token_forwards": ledger["token_forward_equivalents"] + 6144,
               "diagA_digest_order": "sorted parameter names",
               "diagC_digest_order": "model.named_parameters traversal order",
               "cross_process_sha_difference_is_numeric_evidence": False,
               "diagB_weight_max_abs_reported": 8.191913366317749e-06,
               "diagB_single_build_formula_parity_reported": 2.9802322387695312e-08}
    return parent, evaluation, old, history


def make_claims():
    rows = [
        ("old_gate", "Parent gate remains STOP/NO_LOCAL_SIGNAL", "results/local_gate.json; replay_audit_v12/parent_reanalysis.json", "direct+recomputed", "CONFIRMED"),
        ("lr_rank", "Natural labels preserve ranking among 3 old step512 checkpoints", "replay_audit_v12/results/replay_losses.jsonl:1-192", "recomputed", "CONFIRMED_WITHIN_OLD_GRID"),
        ("label_treatment", "Natural labels preserve step256 s=0 treatment effect", "No V12 four-state rows", "unmeasured", "NOT_MEASURED"),
        ("diagA_C_sha", "Different diagA/diagC whole digest proves numeric gradient change", "replay_audit_v12/logs/diagA_repeat_builds.txt:71-77; diagC_causal.txt:65-76", "refuted", "DIGEST_ORDER_DIFFERS"),
        ("diagB", "Two real builds differ by 8.19e-6 in weights", "replay_audit_v12/logs/diagB_full_pair.txt:43-53; logs/diag_outputs.txt", "direct-report-needs-stage-localization", "REPORTED_NOT_CAUSALLY_LOCATED"),
        ("mismatch0", "replay failure mismatch=0 means all state equal", "replay_audit_v12/replay.py:467-475; results/replay_status.json", "refuted", "WEIGHTS_ONLY_ZERO_SHA_DIFFERED"),
        ("budget", "V12 46104 token forwards includes every diagnostic", "replay_audit_v12/budget_ledger.json; logs/diagA_repeat_builds.txt:67-78", "refuted", "DIAGA_6144_FORWARD_TOKENS_UNCOUNTED"),
        ("research", "Old local NO_LOCAL_SIGNAL invalidates AdamW spectral research", "results/local_gate.json; results/probe_confirm.csv", "unsupported inference", "NOT_ESTABLISHED"),
        ("packed", "105 packed windows are independent documents", "replay_audit_v12/evaluation_manifest.json", "refuted", "ONE_CONNECTED_SOURCE_CLUSTER"),
    ]
    with (HERE / "claim_evidence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("claim_id", "claim", "source_path_or_line", "evidence_class", "finding"))
        writer.writerows(rows)


def prepare():
    parent, evaluation, old, history = verify_inputs()
    if (HERE / "protected_before.json").exists():
        if read_json(HERE / "protected_before.json") != protected_hashes():
            raise RuntimeError("PROTECTED_INPUT_CHANGED_SINCE_V13_START")
    else:
        write_json(HERE / "protected_before.json", protected_hashes())
    write_json(HERE / "parent_dependencies.json", {
        "parent": parent, "v12_evaluation_manifest_sha256": sha_file(V12 / "evaluation_manifest.json"),
        "v12_budget_ledger_sha256": sha_file(V12 / "budget_ledger.json"),
        "v12_view_sha256": sha_file(V12 / "eval_views.jsonl"),
        "v12_reeval_raw_sha256": sha_file(V12 / "results" / "replay_losses.jsonl")})
    write_json(HERE / "history_reconciliation.json", history)
    make_claims()
    run_path = HERE / "run_manifest.json"
    if not run_path.exists():
        write_json(run_path, {
            "protocol_id": read_json(HERE / "config.json")["protocol_id"],
            "source_hash": source_hash(), "config_sha256": sha_file(HERE / "config.json"),
            "parent_dependencies_sha256": sha_file(HERE / "parent_dependencies.json"),
            "v12_eval_manifest_sha256": sha_file(V12 / "evaluation_manifest.json"),
            "v12_eval_views_sha256": sha_file(V12 / "eval_views.jsonl"),
            "new_gpu_optimizer_step_attempts_limit": 16,
            "new_token_forward_equivalents_limit": 5000000,
            "historical_budget_separate": "history_reconciliation.json",
            "generation_mode": "parent_bf16_autocast_else_fp32",
            "evaluation_mode": "fp32_no_tf32", "formal_pretraining_steps": 0,
            "fixed_windows": evaluation["train_build_windows"],
            "fixed_new_doc_ids": evaluation["new_probe_select_doc_ids"]})
    else:
        run = read_json(run_path)
        if run["source_hash"] != source_hash():
            archive = HERE / "attempts" / "attempt1_before_atomic_retry"
            if (not archive.is_dir() or read_json(archive / "run_manifest.json") != run or
                    read_json(archive / "budget_ledger.json") != read_json(HERE / "budget_ledger.json") or
                    sha_object({name: sha_file(archive / name) for name in SOURCE_NAMES}) != run["source_hash"]):
                raise RuntimeError("PRIOR_ATTEMPT_ARCHIVE_MISMATCH")
            previous_manifest_sha = sha_file(run_path)
            previous_ledger_sha = sha_file(HERE / "budget_ledger.json")
            new_run = dict(run)
            new_run.update({"source_hash": source_hash(),
                            "attempt_id": 2,
                            "previous_run_manifest_sha256": previous_manifest_sha,
                            "accepted_previous_source_hashes": [run["source_hash"]],
                            "previous_attempt_archive": str(archive.relative_to(HERE)),
                            "patch_scope": "atomic child JSON replacement retry only; scientific generation/evaluation code unchanged"})
            write_json(run_path, new_run)
            ledger = read_json(HERE / "budget_ledger.json")
            ledger["attempts"].append({"attempt_id": 1, "source_hash": run["source_hash"],
                                        "run_manifest_sha256": previous_manifest_sha,
                                        "budget_snapshot_sha256": previous_ledger_sha,
                                        "archive": str(archive.relative_to(HERE)),
                                        "reason": "WinError 5 during child budget atomic replacement; failed forward was not executed"})
            ledger["run_manifest_sha256"] = sha_file(run_path)
            write_json(HERE / "budget_ledger.json", ledger)
    return {"parent": "VERIFIED", "v12": "VERIFIED", "old_gate": old["parent_gate_verdict"],
            "v12_lr_rows": history["v12_lr_raw_rows"], "v12_four_state_rows": 0,
            "history_known_minimum_token_forwards": history["historical_known_minimum_token_forwards"],
            "run_manifest_sha256": sha_file(run_path)}


def verify_after():
    before = read_json(HERE / "protected_before.json")
    after = protected_hashes()
    changed = [name for name in before if before[name] != after[name]]
    write_json(HERE / "protected_after.json", after)
    if changed:
        raise RuntimeError("PROTECTED_INPUTS_CHANGED:" + ",".join(changed))
    return True
