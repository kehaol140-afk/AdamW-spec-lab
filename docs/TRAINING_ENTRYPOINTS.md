# TRAINING_ENTRYPOINTS — 六线真实训练入口（从历史日志/冻结配置核对，非 README 宣称）

## 入口核对表

| 线 | 正式训练入口（历史实际执行） | 调用证据 | 便携入口 | smoke/dev vs formal |
|---|---|---|---|---|
| spectral V16 | `python -m campaign_v16.run`（自 `adamw_spectral_pilot/` 目录；队列驱动 71 项） | PLAN_FROZEN「执行与状态」节 + logs/server_run_A2BCD.out + watchdog.log | `portable/run_v16_campaign.py` | smoke=resume_tests_{red,green}.log/tests_cpu.log；dev=a1 dev H64/H256；formal=A1/A2/B/C+final_holdout |
| spectral V13/V14/V15 | `readiness_audit_v13/audit.py`、`mechanism_screen_v14/`、`direction_horizon_v15/` 各 runner（run_manifest 记录 source_hash） | 各 run_manifest.json | — | window/dev=screen；confirm=正式 |
| direction-validation | `CUDA_VISIBLE_DEVICES=0 nohup <python> src/dv_runner.py layer1 ...` | DIRECTION_GPU_EXECUTION_READY.md:35 实录命令 | `portable/run_dv_layer1.py` | smoke=test_dv_cpu.py+corpus 验证；formal=layer1 5 臂；layer2 未执行 |
| direction-30m | `python src/s30m_runner.py`（mint→smoke→13 runs→analyze 五阶段） | raw/t1_run.out + mint.out | `portable/run_s30m_t1.py` | smoke=s30m_smoke.py 50 步×2；formal=T1 |
| muon-ns-allocation | `scripts/run_p1_check.py`、`run_r0_gate.py` 等各轮 runner | outputs/*/train.jsonl 逐 run 台账 | — | P1 calibration=dev；check/confirm=正式 |
| specmuon | `python src/specmuon/run_pilot.py` | EXECUTION_LEDGER #13-14 + branch_meta | `portable/run_specmuon_p1.py` | tests=19 项 CPU；formal=14 分支 |
| context-100m | `python scripts/run_pilot_driver.py`（成功链）；`scripts/pilot_driver.py`=并行会话超时 driver；逐臂 `run_pilot_arm.py` | results/pilot_driver.log:80 + PILOT_100M_RUNTIME.md:13 | `portable/run_context_pilot.py` | preflight_100steps/smoke_100m=formal 前置 |

## 成本（历史实测）

| 线 | 预算 | 实测 |
|---|---|---|
| V16 全程 | 120k attempts/400M forwards/24h device/72h wall | 79,882/174.9M/437s/16.84h |
| DV layer-1 | 90min 授权 | ~37min（含重放） |
| 30M T1 | 2.0 GPU-h | 1.45h |
| MSA 各轮 | — | P1 每分支 ~20min |
| SpecMuon P1 | — | 14×~20min |
| 100M pilot | — | driver 4.55h |

## 修复的入口文档错误

仅 1 处：100M REPRODUCE 的 `driver_pilot.py`（不存在）→ `run_pilot_driver.py`（ERRATA §E6）。其余五线入口经日志核对无误。V16-C 3-seed holdout（已执行、已开封）与后续 Layer-2/T2（未执行）的区分见各 RESULTS/EXPERIMENTS。
