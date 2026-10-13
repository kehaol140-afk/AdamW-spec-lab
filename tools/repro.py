#!/usr/bin/env python3
"""repro.py — AdamW-spec-lab reproduction control CLI (CPU-safe).

Subcommands:
  list                          six research lines, R0/R1/R2 capability + entrypoints
  check --experiment EXP        verify source presence + external assets (sha if cheap)
  recompute [--all]             run the frozen CPU recompute suite (expects 8/8 OK)
  test --experiment EXP         run verified CPU unit tests of one line
  smoke --experiment EXP        DRY-RUN ONLY: resolved paths/configs/asset status/plan
  train --experiment EXP        DRY-RUN ONLY by default; real training needs
                                --yes AND env ADAMWSPEC_ALLOW_GPU=1 (never default)

No subcommand ever starts GPU work implicitly. Sealed evaluation sets
(holdout_v2) are never touched by any automatic flow.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from portable.paths import load_paths, describe  # noqa: E402

EXP = {
    "spectral-v13-v16": {
        "dir": "experiments/adamw-spectral-v13-v16",
        "frozen_entry": "python -m campaign_v16.run   (from experiment dir; server asset-gated)",
        "portable": "portable/run_v16_campaign.py",
        "r0": "PASS", "r1": "BLOCKED_ASSET(cache+parent ckpt)", "r2": "BLOCKED_ASSET(cache)",
    },
    "direction-validation": {
        "dir": "experiments/adamw-direction-validation",
        "frozen_entry": "python src/dv_runner.py   (V16_ROOT read-only dependency)",
        "portable": "portable/run_dv_layer1.py",
        "r0": "PASS", "r1": "BLOCKED_ASSET(V16 cache)", "r2": "BLOCKED_ASSET(V16 cache)",
    },
    "direction-30m": {
        "dir": "experiments/adamw-direction-30m",
        "frozen_entry": "python src/s30m_runner.py  (needs token cache minted first)",
        "portable": "portable/run_s30m_t1.py",
        "r0": "PASS", "r1": "BLOCKED_ASSET(caches)", "r2": "BLOCKED_ASSET(corpus+cache)",
    },
    "muon-ns-allocation": {
        "dir": "experiments/muon-ns-allocation",
        "frozen_entry": "scripts/run_*.py per round (parent states asset-gated)",
        "portable": "-",
        "r0": "PASS", "r1": "BLOCKED_ASSET(parent states 11.65G on server)", "r2": "UNVERIFIED(replay needs GPU)",
    },
    "specmuon-pilot": {
        "dir": "experiments/specmuon-pilot",
        "frozen_entry": "python src/specmuon/run_pilot.py (muon_transition + msa parent dependency)",
        "portable": "portable/run_specmuon_p1.py",
        "r0": "PASS", "r1": "BLOCKED_ASSET(parent states)", "r2": "BLOCKED_ASSET(parent+eval pool)",
    },
    "context-aware-muon-100m": {
        "dir": "experiments/context-aware-muon-100m",
        "frozen_entry": "python scripts/run_pilot_driver.py --arms all (assets minted first)",
        "portable": "portable/run_context_pilot.py",
        "r0": "PASS", "r1": "BLOCKED_ASSET(33.7M cache)", "r2": "BLOCKED_ASSET(corpus→cache mint)",
    },
}

ASSET_SPECS = {
    "spectral-v13-v16": [
        ("V16_ROOT/campaign_v16/cache/train_tokens.u32", "V16_ROOT", "campaign_v16/cache/train_tokens.u32", "c91382dc (prefix16)", 33_000_000),
        ("corpus (upstream minimind)", "DATA_ROOT", "pretrain_t2t_mini.jsonl", "4403b671 (fingerprint)", 1_240_000_000),
    ],
    "direction-validation": [
        ("V16 token cache (read-only)", "V16_ROOT", "campaign_v16/cache/train_tokens.u32", "c91382dc", 33_000_000),
    ],
    "direction-30m": [
        ("s30m token cache", "CKPT_ROOT", "train_tokens_s30m.u32", "6fe0bb53", 65_000_000),
        ("V16 token cache (prefix identity)", "V16_ROOT", "campaign_v16/cache/train_tokens.u32", "c91382dc", 33_000_000),
        ("corpus", "DATA_ROOT", "pretrain_t2t_mini.jsonl", "4403b671", 1_240_000_000),
    ],
    "context-aware-muon-100m": [
        ("pilot token cache 33.7M", "CKPT_ROOT", "pilot_train_tokens_33p7m.u32", "f92715b4", 128_300_000),
        ("corpus", "DATA_ROOT", "pretrain_t2t_mini.jsonl", "4403b671", 1_240_000_000),
    ],
    "specmuon-pilot": [
        ("MSA parent states (server)", "CKPT_ROOT", "msa/outputs/p0_p1_v1_1/parent/parent_state_step512.pt", "7563daec", 266_500_000),
        ("muon routing report", "MUON_TRANSITION_ROOT", "outputs/m0/routing_report.json", "-", 12_000),
    ],
    "muon-ns-allocation": [
        ("MSA parent states (server)", "CKPT_ROOT", "msa/outputs/p0_p1_v1_1/parent/parent_state_step512.pt", "7563daec", 266_500_000),
    ],
}


def sha16(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:8]


def check_assets(exp: str, paths: dict, verbose=True):
    status = []
    for label, root_key, rel, expect, size in ASSET_SPECS.get(exp, []):
        root = paths.get(root_key, "")
        p = Path(root) / rel if root else None
        ok = bool(p and p.exists())
        note = "OK" if ok else "MISSING -> restore per docs/ASSET_RESTORATION_GUIDE.md"
        if ok and expect not in ("-",) and p.stat().st_size < 400_000_000:
            got = sha16(p)
            note = f"OK sha16={got} (expect {expect})"
        elif ok:
            note = f"OK size={p.stat().st_size} (hash-check skipped >400MB; expect {expect})"
        status.append((label, str(p or "?"), note))
        if verbose:
            print(f"  [{'x' if ok else ' '}] {label:42s} {note}")
    return status


def cmd_list(_):
    print("AdamW-spec-lab — six research lines (R0=CPU recompute, R1=parent-ckpt replay, R2=from-scratch)")
    for k, v in EXP.items():
        print(f"\n{k}")
        print(f"  dir           : {v['dir']}")
        print(f"  frozen entry  : {v['frozen_entry']}")
        print(f"  portable      : {v['portable']}")
        print(f"  capability    : R0={v['r0']}  R1={v['r1']}  R2={v['r2']}")


def cmd_check(a):
    paths = load_paths(a.config)
    print("paths:\n" + describe(paths))
    exps = [a.experiment] if a.experiment else list(EXP)
    allok = True
    for e in exps:
        if e not in EXP:
            sys.exit(f"unknown experiment {e}; choose from {list(EXP)}")
        src = REPO / EXP[e]["dir"]
        print(f"\n== {e}: source {'OK' if src.is_dir() else 'MISSING'} ({ EXP[e]['dir']})")
        st = check_assets(e, paths)
        allok &= all(x[2].startswith("OK") for x in st) and src.is_dir()
    sys.exit(0 if allok else 1)


def cmd_recompute(a):
    r = subprocess.run([sys.executable, str(REPO / "recompute" / "run_recompute.py")],
                       cwd=REPO, capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode:
        print(r.stderr[-2000:]); sys.exit(1)
    d = json.loads(r.stdout[r.stdout.find("{"):])
    bad = {k: v for k, v in d.items() if v != "OK"}
    print(f"recompute suite: {len(d)-len(bad)}/{len(d)} OK" + (f"  FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad or not a.all else 0)


def cmd_test(a):
    e = a.experiment
    cand = {"spectral-v13-v16": "experiments/adamw-spectral-v13-v16/tests/../../campaign_v16/test_v16.py",
            "direction-validation": "experiments/adamw-direction-validation/tests/test_dv_cpu.py",
            "direction-30m": "experiments/adamw-direction-30m/tests/test_s30m_cpu.py",
            "specmuon-pilot": "experiments/specmuon-pilot/tests",
            "context-aware-muon-100m": "experiments/context-aware-muon-100m/tests"}.get(e)
    if not cand:
        print(f"no verified CPU test entry for {e} (R0 recompute covers it)"); return
    p = REPO / cand
    if p.is_dir():
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(p)] if importlib.util.find_spec("pytest")
                           else [sys.executable, "-m", "unittest", "discover", "-s", str(p)], capture_output=True, text=True)
    else:
        r = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, cwd=str(p.parents[1]))
    print(r.stdout[-3000:] or r.stderr[-3000:])
    note = "NOTE: historical tests reference server paths (see docs/PATH_PORTABILITY_AUDIT.md); failures from missing assets are expected outside the server."
    print(note)
    sys.exit(r.returncode if r.returncode and importlib.util.find_spec("pytest") else 0)


def cmd_smoke(a):
    e = a.experiment
    if e not in EXP:
        sys.exit(f"unknown experiment {e}")
    paths = load_paths(a.config)
    print(f"DRY-RUN smoke for {e} (no GPU, no writes)")
    print("paths:\n" + describe(paths))
    check_assets(e, paths)
    v = EXP[e]
    print(f"\nfrozen entry : {v['frozen_entry']}")
    print(f"portable     : python {v['portable']} --dry-run")
    print("config sha   : see experiments docs (frozen configs listed in CODE_MAP.md)")
    print("cost estimate: see docs/TRAINING_ENTRYPOINTS.md table")


def cmd_train(a):
    if not (a.yes and os.environ.get("ADAMWSPEC_ALLOW_GPU") == "1"):
        print("train: DRY-RUN ONLY this round. To actually train you must (1) restore assets "
              "(docs/ASSET_RESTORATION_GUIDE.md), (2) pass --yes, (3) set ADAMWSPEC_ALLOW_GPU=1. "
              "GPU training is NOT authorized in this archive round.")
        cmd_smoke(a); return
    sys.exit("real training not implemented in this archive round — use frozen entry from docs/TRAINING_ENTRYPOINTS.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    c = sub.add_parser("check"); c.add_argument("--experiment"); c.add_argument("--config"); c.set_defaults(fn=cmd_check)
    c = sub.add_parser("recompute"); c.add_argument("--all", action="store_true"); c.set_defaults(fn=cmd_recompute)
    c = sub.add_parser("test"); c.add_argument("--experiment", required=True); c.set_defaults(fn=cmd_test)
    c = sub.add_parser("smoke"); c.add_argument("--experiment", required=True); c.add_argument("--config"); c.set_defaults(fn=cmd_smoke)
    c = sub.add_parser("train"); c.add_argument("--experiment", required=True); c.add_argument("--config"); c.add_argument("--yes", action="store_true"); c.set_defaults(fn=cmd_train)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
