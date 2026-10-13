# CODE_MAP — adamw-spectral-v13-v16

## 核心文件（经实际训练验证的版本即当前版本；全目录无 git，身份=SHA256）

| 文件 | 作用 | 核心函数/类 | 阶段 | SHA256（前 16） | 上游关系 |
|---|---|---|---|---|---|
| `campaign_v16/core.py` | V16 数学内核+评估+台账 | `apply_step`(385)、`n_a`(351)、`eval_docs`、`token_weighted`(512)、`paired_bootstrap`(516)、`param_fingerprint`、`source_hash`(86) | V16 全阶段 | `aa094d3e4ea37cd9` | 模型定义经 `pilot_core`→`minimind_repo/model/model_minimind.py`（=上游 minimind vendored）；优化器=PyTorch AdamW 语义逐行复刻，未用上游 trainer |
| `campaign_v16/run.py` | 队列机/单写入者 | `plan_a1/plan_a2/plan_b/plan_c`、`plan_route_b`(633)、`c_plan`、`finalize`(732) | V16 | `b69b486ca59bc624` | 同上 |
| `campaign_v16/test_v16.py` | 单测（含位级复刻） | — | V16 | 见 MANIFEST | — |
| `campaign_v16/campaign_config.json` | 容器预算/统计/目标块 | — | V16 | 见 MANIFEST | — |
| V16 源码 5 文件摘要 | source_hash（core.py 自定义 digest） | — | — | `4afaa78e…`（direction 线复算一致） | — |
| `readiness_audit_v13/audit.py` | V13 四状态测量 | — | V13 | 见 MANIFEST | — |
| `mechanism_screen_v14/`（runner+config） | rank-1 谱筛查 | — | V14 | manifest `74caaaa5` | — |
| `direction_horizon_v15/`（runner+config） | 方向-视界 | — | V15 | manifest `2472baf1` | — |
| `replay_audit_v12/` | 父实验位级重放 | — | 前置 | 见 MANIFEST | — |
| `interpretation/V16_*.md` | 完赛判读（复制自 audits） | — | 10-10 | `SHA256SUMS*.txt` 随附 | — |

## 历史版本保全

- **Windows 期原始数据**：`campaign_v16/raw_win_pilot_20261003/`（66 eval+32 steps）只读保留——旧平台数据集，不参与选优但保历史身份
- **被中断的旧 findings**：`findings.json.windows_interrupted_stale`、`V16_FINDINGS_AUTO.md.windows_interrupted_stale`、`bindings.local.json.windows_orig`（勘误留痕件，不删）
- **删除的 B 段 endpoint**：12 个 checkpoint 按容器上限删除，逐个 sha256 记录于 `campaign_v16/deletions.jsonl`（"regenerable by rerun"）
- 旧报告快照（STOP_RESOURCE 期）：`V16_FINDINGS.md`、`V16_FINAL_REPORT.md`、`V16_MECHANISM_ASSESSMENT.md`、`V16_COMPARISON.csv` 原样保留，权威性声明见 V16_FINAL_RECONCILED.md §1

## 依赖与入口

训练入口（重跑用）：`python -m campaign_v16.run`（按磁盘 QUEUE 串行、崩溃续跑）；评估数据 `campaign_v16/eval_sets.json`；token 缓存 `campaign_v16/cache/train_tokens.u32`（未随仓，见 REPRODUCE）。复算入口（只读）：仓根 `recompute/run_recompute.py`（CK V16C_3seed/V16_A2）。
