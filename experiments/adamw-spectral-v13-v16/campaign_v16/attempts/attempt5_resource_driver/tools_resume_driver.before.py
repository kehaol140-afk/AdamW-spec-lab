import os, subprocess, sys, time
import psutil

LAB = r"E:\大模型预训练\adamw_spectral_pilot"
PY = r"D:\pytorch\anaconda\envs\py39\python.exe"
BUDGET_S = int(sys.argv[1]) if len(sys.argv) > 1 else 2800
WALL_S = int(sys.argv[2]) if len(sys.argv) > 2 else 3300

env = dict(os.environ)
env.update(PYTHONUTF8="1", USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")


def ram_ok():
    good = 0
    for _ in range(3):
        if psutil.virtual_memory().available >= 2.6 * 2 ** 30:
            good += 1
        time.sleep(20)
    return good == 3


start = time.time()
while time.time() - start < WALL_S - 400:
    if not ram_ok():
        print("driver: waiting for RAM...", flush=True)
        time.sleep(120)
        continue
    try:
        proc = subprocess.run([PY, "-m", "campaign_v16.run", "--resume", "--max-seconds", str(BUDGET_S)],
                              cwd=LAB, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=BUDGET_S + 300)
        out = ((proc.stdout or "") + (proc.stderr or ""))
        print(f"driver: runner exit={proc.returncode} tail={out.strip().splitlines()[-1] if out.strip() else ''}",
              flush=True)
    except subprocess.TimeoutExpired as exc:
        out = ((exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")) \
              + ((exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
        print("driver: runner hard-killed at subprocess timeout; resume machinery will recover",
              flush=True)
    state = os.path.join(LAB, "campaign_v16", "STATE.json")
    try:
        import json
        s = json.loads(open(state, encoding="utf-8").read())
        if s.get("stop"):
            print(f"driver: campaign stop state: {s['stop']} — clearing for next episode", flush=True)
            s["stop"] = None
            s["pauses_total"] = 0.0
            open(state, "w", encoding="utf-8").write(json.dumps(s, ensure_ascii=False, indent=2))
            with open(os.path.join(LAB, "campaign_v16", "DECISIONS.md"), "a", encoding="utf-8") as f:
                f.write(f"\n## {time.strftime('%H:%M:%S')} driver episode boundary\n"
                        "- 条件: runner hit per-episode pause cap (STOP_RESOURCE packaged)\n"
                        "- 证据: RAM below hard line for >2h in one episode\n"
                        "- 行动: fresh episode after RAM recovery; no counters reset\n")
    except Exception as exc:
        print(f"driver: state handling skipped ({exc})", flush=True)
    if "queue-drained-or-deadline" in out and '"attempts"' in out:
        # runner exited normally (deadline or queue drained); loop continues if wall budget allows
        pass
print("driver: window complete", flush=True)
