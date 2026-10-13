"""Sequential driver for the authorized frozen 15-arm pilot (GPU0 only).

Safety rails (task-mandated):
  - cumulative GPU wall cap 6.0 h (sum of run.json wall_s) -> safe stop before
    launching any arm whose family estimate would exceed the cap
  - per-arm shrink rule: wall > 1.2x family estimate -> safe stop + report
  - non-finite loss -> runner asserts -> driver stops (lock absent)
  - lock present -> arm already done (no double counting); safe resume
Queue per 100M_PILOT_EXECUTION_LOCK.md sec.2 with selection steps interleaved.
Status: results/pilot/DRIVER_STATUS.jsonl (one line per event).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results" / "pilot"
CAP_GPU_S = 6.0 * 3600
FAM_EST = {  # family estimate seconds (LOCK doc sec.3) for cap projection
    "B_adamw": 21 * 60, "B_muon": 19.5 * 60, "B_muon_rms": 19.5 * 60,
    "B_lowmu": 18 * 60, "C_A": 20 * 60, "C_A_nat": 20 * 60, "C_B": 21 * 60,
}
SHRINK = 1.2

QUEUE = [
    ("B_adamw_lr3e-4", "B_adamw"), ("B_adamw_lr1e-3", "B_adamw"), ("B_adamw_lr3e-3", "B_adamw"),
    ("B_muon_lr5e-3", "B_muon"), ("B_muon_lr1e-2", "B_muon"), ("B_muon_lr2e-2", "B_muon"),
    ("B_muon_rms_lr1e-3", "B_muon_rms"), ("B_muon_rms_lr3e-3", "B_muon_rms"), ("B_muon_rms_lr1e-2", "B_muon_rms"),
    ("@select:B_muon", None),
    ("B_lowmu_0p9", "B_lowmu"), ("B_lowmu_0p8", "B_lowmu"),
    ("C_A_lam0p65", "C_A"), ("C_A_lam0p85", "C_A"),
    ("@select:C_A", None),
    ("C_A_nat", "C_A_nat"), ("C_B", "C_B"),
]


def log(evt: dict) -> None:
    PILOT.mkdir(parents=True, exist_ok=True)
    with (PILOT / "DRIVER_STATUS.jsonl").open("a") as f:
        f.write(json.dumps({"ts": time.time(), **evt}) + "\n")
    print(json.dumps(evt), flush=True)


def arm_wall(arm_id: str):
    rp = PILOT / arm_id / "run.json"
    if not rp.exists():
        return None
    return json.load(open(rp))["wall_s"]


def cumulative() -> float:
    return sum(w for _, w in ((a, arm_wall(a)) for a, _ in QUEUE if not a.startswith("@")) if w)


def main():
    log({"event": "driver_start", "cap_gpu_s": CAP_GPU_S})
    for arm_id, fam in QUEUE:
        if arm_id.startswith("@select:"):
            stage = arm_id.split(":")[1]
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "pilot_select.py"), "--stage", stage],
                capture_output=True, text=True,
            )
            log({"event": "select", "stage": stage, "rc": r.returncode,
                 "stdout_tail": r.stdout[-400:] if r.stdout else "", "stderr": r.stderr[-400:]})
            if r.returncode != 0:
                log({"event": "driver_stop", "reason": f"selection failed: {stage}"})
                return
            continue
        lock = PILOT / arm_id / "run.lock.json"
        if lock.exists():
            log({"event": "skip_done", "arm": arm_id})
            continue
        cum = cumulative()
        est = FAM_EST[fam]
        if cum + est > CAP_GPU_S:
            log({"event": "driver_stop", "reason": "GPU cap projection exceeded",
                 "cumulative_s": cum, "next_arm": arm_id, "est_s": est})
            return
        log({"event": "arm_start", "arm": arm_id, "cumulative_s": cum})
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_pilot_arm.py"), "--arm", arm_id],
            capture_output=True, text=True,
        )
        wall_probe = time.time() - t0
        if r.returncode != 0:
            log({"event": "driver_stop", "reason": "arm failed (non-finite/OOM/other)",
                 "arm": arm_id, "rc": r.returncode, "stderr_tail": r.stderr[-1500:],
                 "stdout_tail": r.stdout[-400:]})
            return
        w = arm_wall(arm_id)
        cum = cumulative()
        log({"event": "arm_done", "arm": arm_id, "arm_wall_s": w, "wall_probe_s": round(wall_probe, 1),
             "cumulative_gpu_s": cum})
        if w is not None and w > SHRINK * est:
            log({"event": "driver_stop", "reason": "shrink rule: arm wall > 1.2x estimate",
                 "arm": arm_id, "wall_s": w, "est_s": est})
            return
    log({"event": "driver_done", "cumulative_gpu_s": cumulative()})


if __name__ == "__main__":
    main()
