"""Portable path resolution for AdamW-spec-lab repro work.

Frozen historical sources keep their original (server) constants untouched.
This layer ONLY resolves external asset locations for the portable wrappers
in portable/run_*.py. Precedence: env var ADAMWSPEC_<KEY> > configs/paths.json
> sane defaults relative to this repository. Read-only inputs are never
written to; all new artifacts must go under OUTPUT_ROOT.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYS = ("DATA_ROOT", "CKPT_ROOT", "V16_ROOT", "MUON_TRANSITION_ROOT", "MINIMIND_ROOT", "OUTPUT_ROOT")


def load_paths(config: str | None = None) -> dict:
    p = {}
    cfg_path = Path(config or os.environ.get("ADAMWSPEC_PATHS", REPO_ROOT / "configs" / "paths.json"))
    if cfg_path.exists():
        p = {k: v for k, v in json.loads(cfg_path.read_text()).items() if k in KEYS}
    for k in KEYS:
        env = os.environ.get(f"ADAMWSPEC_{k}")
        if env:
            p[k] = env
    p.setdefault("V16_ROOT", str(REPO_ROOT / "experiments" / "adamw-spectral-v13-v16"))
    p.setdefault("MINIMIND_ROOT", str(REPO_ROOT / "baselines" / "minimind-base-code"))
    p.setdefault("DATA_ROOT", str(REPO_ROOT / "data_placeholder"))
    p.setdefault("CKPT_ROOT", str(REPO_ROOT / "assets_placeholder"))
    p.setdefault("MUON_TRANSITION_ROOT", "")
    p.setdefault("OUTPUT_ROOT", str(REPO_ROOT / "repro_outputs"))
    return {k: Path(v) if v else "" for k, v in p.items()}


def describe(paths: dict) -> str:
    return "\n".join(f"  {k:22s} = {v}" for k, v in paths.items())
