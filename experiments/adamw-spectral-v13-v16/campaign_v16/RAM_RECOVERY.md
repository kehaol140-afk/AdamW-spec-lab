# V16 RAM recovery: evidence and stopped resume

**Status: BLOCKED_BUDGET; concurrent BLOCKED_EXTERNAL_RAM.** This is an engineering/resource stop, not a scientific result. This run added **0** optimizer attempts, **0** completed steps and **0** token forwards. The next trial was not trained, evaluated or saved; no recovery validation is claimed.

## Frozen state and queue

Read on 2026-10-08 (Asia/Shanghai). The original ledger starts at 2026-10-03 00:03:22; the frozen 72 h deadline was **2026-10-06 00:03:22**. The clock is past that deadline. The cumulative `pause_seconds=14,700` (4 h 5 min) also exceeds the original 2 h pause cap by 7,500 s. Neither clock nor pause total was reset.

`QUEUE.jsonl` and `trial_registry.csv` agree by ID and status: **32/38** queued tasks done, comprising A0 and **31/36** A1 branches. A1 has five branches pending, followed by `a1_select`. `STATE.stop=STOP_RESOURCE`; its prose note contains a stale branch count, so the queue and registry counts above take precedence. All 31 done A1 branches have exactly steps 1–256 in their step JSONL and 256 development rows each at H64 and H256. The next task is `a1_P_a4_b1`; its step log contains only step 1, no H64/H256 evaluation, no rolling checkpoint and no endpoint. The three extra real attempts in the ledger are repeated work on this interrupted task, not three completed branch steps. Its only legal starting state currently on disk is the verified parent step256 checkpoint; resuming would have to account for repeated attempts.

Ledger remains **7,939/120,000 optimizer attempts and completions**, **19,171,782/400,000,000 token-forward-equivalents**. `run_manifest` ↔ ledger hash, active runner source hash, config hash, V16 data manifest/cache hash and parent step256 checkpoint hash matched. No checkpoint was copied. `campaign_v16` occupies 0.0462 GiB with 117.752 GiB free on E:; current V16 checkpoint count is zero.

## Resource localization

The original 1,672 resource entries span 2026-10-02 22:31 to 2026-10-03 05:57. A representative final failed attempt is shown in `RAM_CURVE.csv`:

| Phase | Available RAM | Runner RSS | Result |
|---|---:|---:|---|
| before CUDA/model init | 1.71 GiB | 0.36 GiB | admission passed |
| after context/cache setup | 1.63 GiB | 0.46 GiB | passed; model load follows |
| before step 1 | 1.41 GiB | 0.72 GiB | warning |
| before step 2 | **0.80 GiB** | **1.37 GiB** | hard stop at 1.0 GiB |

This step-2 pattern occurred three times (available 0.67, 0.83 and 0.80 GiB). An earlier completed trial's development evaluation reached 1.06 GiB available with 1.45 GiB RSS, but did not cross the hard line. The supplied log has no checkpoint-save boundary sample, so a save-phase peak cannot be quantified. The final failures happened before step 2, prior to that trial's evaluation or save. There is no evidence here of a growing per-step leak: these attempts reached the same early memory level after fresh worker starts. Within some older serial-runner episodes, task starts retained roughly 1.45 GiB RSS after a prior trial; this indicates process-resident memory but does not distinguish allocator retention from a live model. It was not the sole cause of the final fresh-worker failure.

The old driver imports only standard library and `psutil`, not torch/transformers. It starts a serial `campaign_v16.run` subprocess; the runner imports torch and owns model/optimizer. Training tokens are accessed as a read-only `uint32` `np.memmap` (8.5M tokens, cache hash verified); no DataLoader or extra DataLoader workers are present. Per-trial q series and eval views are small, and step JSONL is append-only. `save_rolling` makes a CPU optimizer-state copy, but its actual RAM peak was not logged; changing it without measurement would be speculative. Historical logs lack PID/PPID, so they cannot prove whether an additional unrelated process was present then.

On 2026-10-08, no task-owned `D:\pytorch\anaconda\envs\py39\python.exe` worker or V16 run/ledger lock was present. Three idle system samples 10 s apart had only **0.894, 0.717, 0.457 GiB** available of 15.688 GiB physical RAM. Commit was about 25.7–26.4 GiB of a 44.688 GiB limit. The top current external process was VS Code PID **24268**, 2,504 MiB RSS / 2,731 MiB private in the later snapshot; Chrome, OpenCode and ChatGPT were also substantial. Full top-10 PID/PPID/start/path/working-set/private-byte snapshot is in `RAM_PROCESS_SNAPSHOT.csv`; access-denied paths are marked UNKNOWN. RSS includes shared memory and was not summed as unique usage. At the lowest idle reading the deficit to the requested 2.6 GiB startup condition was **2.14 GiB**; a later snapshot was 0.795 GiB available, deficit 1.81 GiB. The user must free roughly **at least 2.2 GiB** of current system RAM and then verify three ≥2.6 GiB samples 10 s apart. No external process was closed.

## Confirmed control-plane fix

The previous `tools_resume_driver.py` automatically cleared any `STATE.stop` and reset `STATE.pauses_total` after an episode, including terminal stops, without checking the original 72 h deadline. This could falsely present a fresh pause allowance. Only that light driver was changed; `core.py`, `run.py`, `campaign_config.json`, data, checkpoint and scientific thresholds are unchanged. The new driver checks the original wall, cumulative ledger pause, STOP, device/step/forward budgets **before** RAM sampling or spawning, checks again after three 10 s samples, and never edits STATE or ledger. Low RAM returns immediately rather than holding a heavy worker. It does not alter the runtime 1.5 GiB warning, 1.0 GiB hard stop or 4 GiB RSS cap. This is a guard, not a validated training-memory reduction.

Diff: `RAM_RECOVERY.patch`. Before driver SHA256 `DD3F69F2B2DA4450638BEB0AF35EAC02688540199F460FF5DAD090C679C488D2`; new executed driver SHA256 `229977DBBCC3D0398D16E7F6470F63B48E87CDCC8DB729CCD5D6A7223DBEFF14`. The active campaign source fingerprint remains unchanged because this recovery driver is outside the frozen `SOURCE_FILES`; its separate SHA records the control-plane version. A CPU-only red run failed on the old import-time argument handling (`resume_tests_red.log`). Four targeted tests then passed (`resume_tests_green.log`): light import, immutable wall, cumulative pause, three RAM samples. A live invocation of the new driver returned exit **2**, `BLOCKED_WALL`, without starting a worker; SHA256 of STATE, QUEUE and ledger was identical before/after (`resume_guard_live.log`). Original heavy CPU tests were not rerun under <1 GiB available RAM; training code was not modified.

## Resume decision

**No real trial was resumed.** Starting `a1_P_a4_b1` would violate the expired 72 h wall deadline, the exceeded cumulative pause limit and today's startup RAM condition. The requested train→evaluate→save→worker-exit acceptance therefore remains unperformed. A1 selection, A2/B/C and final holdout were not opened; existing completed results and budgets remain intact. Even with more RAM, the current contract still requires an explicit wall/pause-budget extension before any new worker. After authorization and adequate RAM, the next validation is one complete `a1_P_a4_b1` trial from the verified parent checkpoint, with phase-boundary RAM readings including evaluation/save and confirmed worker exit; only then may the original queue continue. A separate per-trial worker lifecycle change may be justified if those readings show retained cross-trial memory; it has not been preemptively applied.

The exact bound interpreter for control-plane checks is `D:\pytorch\anaconda\envs\py39\python.exe`. Current safe commands from project root are:

```powershell
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m unittest campaign_v16.test_resume_driver -v
& 'D:\pytorch\anaconda\envs\py39\python.exe' campaign_v16\tools_resume_driver.py 2800 3300
```

The second command currently exits `BLOCKED_WALL`; it must not be treated as a resume instruction until the budget is explicitly amended and resource conditions are revalidated.
