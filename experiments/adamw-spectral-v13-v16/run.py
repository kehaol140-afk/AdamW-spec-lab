"""Unified local CLI for the frozen AdamW spectral pilot protocol."""
import os

# The existing CUDA Python also has an incompatible TensorFlow/NumPy install.
# Transformers does not need TensorFlow for this PyTorch-only project.
os.environ["USE_TF"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_MODE"] = "disabled"
os.environ["PYTHONUTF8"] = "1"

import argparse
import json
import sys
import traceback
import zipfile
from pathlib import Path

from prepare_data import ROOT, atomic_json, event, load_config, read_json, source_hash


def report(cfg, cfg_hash):
    paths = {"preflight": ROOT / "artifacts" / "preflight.json",
             "data": ROOT / "artifacts" / "data_manifest.json",
             "smoke": ROOT / "artifacts" / "smoke_run_manifest.json",
             "gate": ROOT / "results" / "local_gate.json"}
    status = {name: read_json(path).get("status", read_json(path).get("verdict"))
              if path.exists() else "NOT_RUN" for name, path in paths.items()}
    if paths["preflight"].exists():
        status["resource"] = read_json(paths["preflight"]).get("resource_status", "READY")
    binding = ROOT / "bindings.local.json"
    py = read_json(binding).get("python_executable") if binding.exists() else None
    lines = ["# Local findings", "", f"Protocol: `{cfg['protocol_id']}`", "",
             f"Implementation status: `IMPLEMENTED_NOT_FORMALLY_RUN`", "",
             "| Stage | Status |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in status.items()]
    lines += ["", "The discovery and confirmation training commands are reserved for later review.",
              "No end-to-end optimizer gain or deployment cost has been measured.", ""]
    out = ROOT / "reports" / "LOCAL_FINDINGS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    event("report", status=status, python=py, source_hash=source_hash(), config_hash=cfg_hash)
    return status


def bundle_light(cfg, cfg_hash):
    dest = ROOT / "artifacts" / "adamw_spectral_light.zip"
    paths = [ROOT / "run.py", ROOT / "pilot_core.py", ROOT / "pilot_workflow.py",
             ROOT / "prepare_data.py", ROOT / "resource_policy.py",
             ROOT / "tests" / "test_numerics.py", ROOT / "tests" / "test_resources.py",
             ROOT / "config" / "local_v1.json", ROOT / "config" / "local_v1_lowmem.json",
             ROOT / "PROTOCOL_LOCAL_V1.md",
             ROOT / "RUN_WINDOWS.md", ROOT / "bindings.local.json",
             ROOT / "RUN_COMMANDS_RESOLVED.ps1", ROOT / "RUN_LOWMEM_SMOKE.ps1",
             ROOT / "data" / "SOURCE.json",
             ROOT / "artifacts" / "preflight.json", ROOT / "artifacts" / "data_manifest.json",
             ROOT / "artifacts" / "protocol_resolved.json",
             ROOT / "artifacts" / "smoke_run_manifest.json"]
    for sub in ("reports", "results", "artifacts/numerics", "logs", "lowmem_revision"):
        base = ROOT / sub
        if base.exists():
            paths += [p for p in base.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".csv", ".jsonl", ".log")]
    paths = [p for p in paths if p.exists()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if path.name == "bindings.local.json":
                b = read_json(path)
                for key in ("repo_root", "python_executable", "tokenizer_path", "source_data_path", "search_root"):
                    b[key] = None
                z.writestr("bindings.portable.json", json.dumps(b, indent=2, ensure_ascii=False))
            else:
                z.write(path, path.relative_to(ROOT))
    event("bundle", path=str(dest), file_count=len(paths), config_hash=cfg_hash)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "prepare", "smoke", "pilot", "report", "bundle"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "preflight":
            p.add_argument("--root", required=True)
        if name == "pilot":
            p.add_argument("--stage", choices=("discover", "confirm"), required=True)
        if name == "bundle":
            p.add_argument("--light", action="store_true", required=True)
    args = parser.parse_args()
    cfg, cfg_hash = load_config(args.config)
    try:
        if args.command == "preflight":
            from pilot_core import preflight
            result = preflight(cfg, cfg_hash, Path(args.root))
            return 0 if result["status"] == "READY" and not result.get("resource_status") else 2
        if args.command == "prepare":
            from prepare_data import prepare
            result = prepare(cfg, cfg_hash)
            return 0 if result["status"] in ("READY", "READY_FOR_SMOKE") else 2
        if args.command == "smoke":
            from pilot_core import smoke
            smoke(cfg, cfg_hash)
        elif args.command == "pilot":
            from pilot_workflow import pilot
            pilot(cfg, cfg_hash, args.stage)
        elif args.command == "report":
            report(cfg, cfg_hash)
        elif args.command == "bundle":
            bundle_light(cfg, cfg_hash)
        return 0
    except Exception as e:
        if args.command == "smoke":
            manifest = ROOT / "artifacts" / "smoke_run_manifest.json"
            existing = read_json(manifest) if manifest.exists() else {}
            if not (existing.get("status") in ("PASS", "FAILED", "BLOCKED") and
                    existing.get("source_hash") == source_hash() and
                    existing.get("config_hash") == cfg_hash and
                    "gpu_optimizer_updates" in existing):
                atomic_json(manifest,
                            {"status": "BLOCKED" if str(e).startswith(("DATA_INSUFFICIENT", "RESOURCE_BLOCKED"))
                             else "FAILED", "reason": str(e), "source_hash": source_hash(),
                             "config_hash": cfg_hash, "gpu_optimizer_updates": 0})
        event("command_failed", command=args.command, error=str(e), traceback=traceback.format_exc())
        print(f"{type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
