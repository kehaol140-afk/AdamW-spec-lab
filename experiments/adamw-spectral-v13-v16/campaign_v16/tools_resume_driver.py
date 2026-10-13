"""Light V16 controller. It never changes frozen STATE or budget counters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


LAB = Path(r"E:\大模型预训练\adamw_spectral_pilot")
PY = r"D:\pytorch\anaconda\envs\py39\python.exe"
CAMPAIGN = LAB / "campaign_v16"
WALL_SECONDS = 72 * 3600
PAUSE_SECONDS = 2 * 3600


def budget_status(ledger, state, now):
    """Return a blocking reason before any heavy worker is launched."""
    if now >= ledger["wall_started_at"] + WALL_SECONDS:
        return "BLOCKED_WALL"
    if ledger.get("pause_seconds", 0) >= PAUSE_SECONDS:
        return "BLOCKED_PAUSE_BUDGET"
    if state.get("stop"):
        return "BLOCKED_STATE:" + str(state["stop"])
    if ledger.get("optimizer_step_attempts", 0) >= 120000:
        return "BLOCKED_OPTIMIZER_BUDGET"
    if ledger.get("token_forward_equivalents", 0) >= 400000000:
        return "BLOCKED_FORWARD_BUDGET"
    if ledger.get("device_task_seconds", 0) >= 24 * 3600:
        return "BLOCKED_DEVICE_BUDGET"
    return "READY"


def ram_ok(sample=None, sleep=time.sleep):
    """Three consecutive current available-RAM samples, 10 s apart."""
    sample = sample or (lambda: psutil.virtual_memory().available)
    readings = []
    for i in range(3):
        readings.append(sample())
        if i != 2:
            sleep(10)
    return all(value >= 2.6 * 2**30 for value in readings)


def load_control():
    ledger = json.loads((CAMPAIGN / "budget_ledger.json").read_text(encoding="utf-8"))
    state = json.loads((CAMPAIGN / "STATE.json").read_text(encoding="utf-8"))
    return ledger, state


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    budget_s = int(argv[0]) if argv else 2800
    wall_s = int(argv[1]) if len(argv) > 1 else 3300
    start = time.time()
    env = dict(os.environ)
    env.update(PYTHONUTF8="1", USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    while True:
        ledger, state = load_control()
        reason = budget_status(ledger, state, time.time())
        if reason != "READY":
            print("driver: " + reason, flush=True)
            return 2
        remaining = min(start + wall_s - time.time(),
                        ledger["wall_started_at"] + WALL_SECONDS - time.time())
        if remaining <= 400:
            print("driver: local window complete", flush=True)
            return 0
        if not ram_ok():
            print("driver: BLOCKED_EXTERNAL_RAM; need 2.6 GiB in 3 samples", flush=True)
            return 3
        ledger, state = load_control()
        reason = budget_status(ledger, state, time.time())
        if reason != "READY":
            print("driver: " + reason, flush=True)
            return 2
        remaining = min(start + wall_s - time.time(),
                        ledger["wall_started_at"] + WALL_SECONDS - time.time())
        if remaining <= 400:
            print("driver: local window complete", flush=True)
            return 0
        run_seconds = min(budget_s, int(remaining - 300))
        if run_seconds <= 0:
            return 0
        proc = subprocess.run([PY, "-m", "campaign_v16.run", "--resume", "--max-seconds", str(run_seconds)],
                              cwd=str(LAB), env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=run_seconds + 240)
        out = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        print(f"driver: runner exit={proc.returncode} tail={out[-1] if out else ''}", flush=True)
        if proc.returncode:
            return proc.returncode
        queue = [json.loads(line) for line in (CAMPAIGN / "QUEUE.jsonl").open(encoding="utf-8")]
        if all(row["status"] == "done" for row in queue):
            return 0


if __name__ == "__main__":
    sys.exit(main())
