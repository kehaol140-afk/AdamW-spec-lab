"""V1.2 independent entry point. GPU actions require an explicit --execute flag."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from .core import (HERE, ROOT, child_source_hash, prepare_views, read_json,
                   reanalyze_parent, sha_file, sha_object, verify_parent, write_json)


def prepared_manifest() -> dict:
    parent = verify_parent(deep_checkpoints=True)
    stored = read_json(HERE / "parent_dependencies.json")
    if parent != stored:
        raise RuntimeError("PARENT_DEPENDENCIES_CHANGED")
    child = read_json(HERE / "evaluation_manifest.json")
    if child["child_source_hash"] != child_source_hash() or child["parent_dependencies_hash"] != sha_object(parent):
        raise RuntimeError("CHILD_MANIFEST_CHANGED")
    for path, key in (("eval_views.jsonl", "eval_views_sha256"),
                      ("legacy_truncation_audit.csv", "legacy_truncation_audit_sha256"),
                      ("selected_probe_docs.csv", "selected_probe_docs_sha256")):
        if sha_file(HERE / path) != child[key]:
            raise RuntimeError(f"CHILD_DATA_HASH_CHANGED: {path}")
    return child


def bundle_light() -> Path:
    prepared_manifest()
    path = HERE / "v12_light.zip"
    files = [HERE / name for name in ("__init__.py", "core.py", "replay.py", "cli.py",
                                    "test_v12.py", "run_parent_tests_isolated.py", "config.json", "parent_dependencies.json",
                                    "evaluation_manifest.json", "parent_reanalysis.json",
                                    "legacy_truncation_audit.csv", "selected_probe_docs.csv",
                                    "v12_cpu_tests.log", "v12_cpu_tests.exitcode",
                                    "parent_cpu_tests_isolated.log", "parent_cpu_tests_isolated.exitcode",
                                    "prepare_audit.log", "prepare_audit.exitcode")]
    files.append(ROOT / "reports" / "V12_IMPLEMENTATION_HANDOFF.md")
    for folder in (HERE / "results", HERE / "logs"):
        if folder.exists():
            files.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix in
                         (".json", ".jsonl", ".csv", ".log", ".txt", ".exitcode"))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            if file.exists():
                archive.write(file, file.relative_to(ROOT))
    return path


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-parent")
    commands.add_parser("prepare-audit")
    commands.add_parser("bundle-light")
    for name in ("replay", "reeval-lr"):
        commands.add_parser(name).add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    if args.command == "verify-parent":
        result = verify_parent(deep_checkpoints=True)
    elif args.command == "prepare-audit":
        if (HERE / "budget_ledger.json").exists():
            raise RuntimeError("CHILD_REPLAY_LEDGER_EXISTS: refuse to regenerate active manifests")
        parent = verify_parent(deep_checkpoints=True)
        write_json(HERE / "parent_dependencies.json", parent)
        reanalysis = reanalyze_parent()
        write_json(HERE / "parent_reanalysis.json", reanalysis)
        child = prepare_views(parent)
        result = {"parent": parent["status"], "old_gate": reanalysis["parent_gate_verdict"],
                  "old_next_action": reanalysis["parent_next_action"],
                  "child": child["status"], "view_counts": child["view_counts"],
                  "legacy_genuinely_truncated_count": child["legacy_genuinely_truncated_count"],
                  "gpu_optimizer_calls": 0}
    elif args.command == "bundle-light":
        result = {"bundle": str(bundle_light())}
    else:
        if not args.execute:
            raise RuntimeError("GPU replay requires explicit --execute")
        prepared_manifest()
        from .replay import execute_reeval_lr, execute_replay
        result = execute_replay() if args.command == "replay" else execute_reeval_lr()
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
