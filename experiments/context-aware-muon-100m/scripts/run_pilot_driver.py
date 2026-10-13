"""Pilot driver: executes the frozen 15-arm queue of 100M_PILOT_EXECUTION_LOCK v1.0.

Order per lock section 2; two selection steps write results/pilot/selection.json
(frozen rules of pilot_arms_v1.json); budget guard stops safely before any arm
that would push cumulative wall past BUDGET_CAP_H. Arms already DONE (lock file
present) are skipped without re-counting. Append-only driver log.
Authorized by user task 2026-10-12 (GPU0, cap 6.0 GPU-h).
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "pilot"
PY = sys.executable
BUDGET_CAP_H = 6.0
EXPECTED_MIN_PER_ARM = 17.0 / 60.0  # conservative planning value (hours)

QUEUE = [
    ("arm", "B_adamw_lr3e-4"),
    ("arm", "B_adamw_lr1e-3"),
    ("arm", "B_adamw_lr3e-3"),
    ("arm", "B_muon_lr5e-3"),
    ("arm", "B_muon_lr1e-2"),
    ("arm", "B_muon_lr2e-2"),
    ("arm", "B_muon_rms_lr1e-3"),
    ("arm", "B_muon_rms_lr3e-3"),
    ("arm", "B_muon_rms_lr1e-2"),
    ("select", "B_muon_best_lr"),
    ("arm", "B_lowmu_0p9"),
    ("arm", "B_lowmu_0p8"),
    ("arm", "C_A_lam0p65"),
    ("arm", "C_A_lam0p85"),
    ("select", "C_A_best_lam"),
    ("arm", "C_A_nat"),
    ("arm", "C_B"),
]


def dev4096(arm):
    return json.load(open(RES / arm / "run.json"))["dev_nll_by_horizon"]["4096"]


def wall_h(arm):
    return json.load(open(RES / arm / "run.json"))["wall_s"] / 3600.0


def done(arm):
    return (RES / arm / "run.lock.json").exists()


def do_selection(which, log):
    sel_path = RES / "selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.exists() else {}
    if which == "B_muon_best_lr":
        cands = {"B_muon_lr5e-3": 0.005, "B_muon_lr1e-2": 0.01, "B_muon_lr2e-2": 0.02}
        scores = {a: dev4096(a) for a in cands}
        best = min(scores, key=scores.get)
        # tie rule: within 1e-4 -> smaller lr
        smaller = min(cands, key=lambda a: (cands[a], scores[a]))
        if abs(scores[best] - scores[smaller]) < 1e-4:
            best = smaller
        sel["B_muon_scores"] = scores
        sel["B_muon_best_lr"] = cands[best]
        sel["B_muon_boundary"] = best in ("B_muon_lr5e-3", "B_muon_lr2e-2")
    elif which == "C_A_best_lam":
        cands = {"C_A_lam0p65": 0.65, "C_A_lam0p85": 0.85}
        scores = {a: dev4096(a) for a in cands}
        best = min(scores, key=scores.get)
        if abs(scores["C_A_lam0p65"] - scores["C_A_lam0p85"]) < 1e-4:
            best = "C_A_lam0p65"  # frozen tie rule: lam 0.65
        sel["C_A_scores"] = scores
        sel["C_A_best_lam"] = cands[best]
    else:
        raise ValueError(which)
    sel_path.write_text(json.dumps(sel, indent=1))
    log.write(f"SELECTION {which} -> {json.dumps(sel.get(which))}\n")
    log.flush()


def main():
    RES.mkdir(parents=True, exist_ok=True)
    log = open(ROOT / "results" / "pilot_driver.log", "a")
    log.write(f"\nDRIVER START {time.strftime('%F %T')} pid={__import__('os').getpid()}\n")
    cum = 0.0
    for kind, name in QUEUE:
        if kind == "select":
            do_selection(name, log)
            continue
        if done(name):
            log.write(f"SKIP {name} (already DONE)\n")
            log.flush()
            cum += wall_h(name)
            continue
        if cum + EXPECTED_MIN_PER_ARM > BUDGET_CAP_H:
            log.write(f"SAFE_STOP before {name}: cumulative {cum:.2f}h + est next "
                      f"{EXPECTED_MIN_PER_ARM:.2f}h exceeds cap {BUDGET_CAP_H}h\n")
            log.flush()
            return
        log.write(f"RUN {name} (cumulative {cum:.2f}h)\n")
        log.flush()
        t0 = time.time()
        r = subprocess.run([PY, str(ROOT / "scripts" / "run_pilot_arm.py"), "--arm", name],
                           capture_output=True, text=True)
        log.write(f"  exit={r.returncode} wall={(time.time()-t0)/60:.1f}m\n")
        if r.stdout.strip():
            log.write(f"  stdout: {r.stdout.strip()[-400:]}\n")
        if r.returncode != 0:
            log.write(f"  stderr: {r.stderr.strip()[-800:]}\n")
            log.write(f"ABORT driver on arm failure {name} (safe stop, no re-count)\n")
            log.flush()
            return
        cum += wall_h(name)
        log.write(f"  cumulative={cum:.2f}h\n")
        log.flush()
    log.write(f"DRIVER DONE {time.strftime('%F %T')} cumulative={cum:.2f}h\n")
    log.flush()


if __name__ == "__main__":
    main()
