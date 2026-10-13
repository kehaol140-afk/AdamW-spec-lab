# V15 actual execution record (2026-10-02, Asia/Shanghai)

Interpreter: `D:\pytorch\anaconda\envs\py39\python.exe` (binding-verified).
`RUN_VERIFIED.ps1` lists the canonical command sequence; every command in it was really
executed from `E:\大模型预训练\adamw_spectral_pilot`. Deviations and retries are logged here.

## Real command history

| # | Command | Result |
|---|---|---|
| 1 | `-m unittest discover -s direction_horizon_v15 -p 'test_v15.py' -v` | several iterations during development; final 15/15 OK (`tests_cpu.log`) |
| 2 | `-m direction_horizon_v15.train prepare` | exit 0 (re-run after each archived attempt; fresh manifest each time) |
| 3 | `-m direction_horizon_v15.train smoke` | attempt1 FileNotFoundError (h1 snapshot dir); attempt2 RNG tensor bool; attempt3 VRAM admission blocked; attempt4 stale rolling state + orphan lock; **attempt5 PASS** exit 0 (native parity 0.0, resume 0.0, one-shot F→S, eval-repeat spread 0) |
| 4 | `-m direction_horizon_v15.train dev` | exit 1 twice on `RESOURCE_BLOCKED:RAM_BELOW_REQUIRED` (system RAM oscillation from user desktop apps; no code bug), then completed via supervisor rounds, overall success |
| 5 | `-m direction_horizon_v15.train select` | exit 0; selection frozen before any confirm loss was opened |
| 6 | `-m direction_horizon_v15.train confirm` | exit 0 via supervisor (1 round) |
| 7 | `-m direction_horizon_v15.train finalize` | exit 0; verdict BRANCH_SIGNAL_WITHIN_GRID; v15_light.zip rebuilt + reopen-checked |

Dev/confirm resilience: a supervisor script re-invoked the phase after each
`RESOURCE_BLOCKED:RAM_BELOW_REQUIRED` once system RAM recovered (≥1.6 GiB twice). Branch
state resumes from rolling checkpoints (interval 8 steps); re-executed steps are re-counted
in the ledger (attempts 1698 vs 1664+10 planned = 24 recovery attempts, all recorded).

## Attempt archives (all preserved under attempts/)

- `attempt1_h1_snapshot_mkdir`: torch.save before checkpoints/ existed.
- `attempt2_rng_tensor_bool`: elementwise tensor != used as bool.
- `attempt3_vram_admission_blocked`: GPU free ~1.4-1.6 GiB; user-authorized amendment
  recorded (VRAM free reserve 1.5→0.2 GiB, V15 scope only; parent configs untouched; RAM
  gates unchanged). Evidence in that archive and in train.py's amendment comment.
- `attempt4_stale_state_and_lock_leak`: orphan child run.lock on failed admission + stale
  rolling checkpoints from archived attempts.
- `attempt5_ledger_race_fix`: **accounting race** — in-memory ledger dicts cross-wrote the
  file and regressed counters (13 recorded vs 21 real attempts, proven from the resource
  log's optstep-gate rows). All mutations now go through read-modify-write helpers;
  regression test `test_ledger_helpers_read_modify_write_fresh_file` added; ledger
  reconciled to TRUE totals before the final smoke re-run.

## Budget (final, from findings.json)

- GPU optimizer.step attempts 1698 / 1792 (completions 1698; smoke 10; recovery re-executions 24).
- Token-forward-equivalents 6,030,545 / 8,000,000; backward passes 6,832; train effective
  tokens 3,430,400 (=1,672 completed branch steps × 2048).
- New random-init full pretraining trajectories: 0. New 64-step branch trajectories: 26
  (20 dev + 2 one-shot + 4 aux).
- Parent artifacts, gate (STOP), checkpoints, data caches: untouched (verify chain green).
