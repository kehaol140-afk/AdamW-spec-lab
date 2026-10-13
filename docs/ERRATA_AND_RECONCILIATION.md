# ERRATA_AND_RECONCILIATION — 勘误与对账（2026-10-12 staging 轮）

原则：旧文档一律保留以示历史；本文件登记每处修正的「旧说法→事实→证据」。

## 1. Round-1 审计（audits/adamw_muon_master_archive_20261012，2026-10-12 上午）勘误

| # | 旧说法 | 事实 | 证据 |
|---|---|---|---|
| E1 | 「P1 对 S1 点收益 **+0.1144** ≥1e-4 命中路由规则 1」 | +0.1144 = dev(S1)=4.240565 − screen(P1)=4.126135，**跨 split 混合量，无协议地位**。路由实际：F1 以 gain_dev=+0.1217 + screen 旁路入选；P1 因 screen 缺席基线默认 0 被拒出 C（实现缺陷已披露）。正确引用：dev 口径 P1 −0.1527 / F1 −0.1217 | `campaign_v16/run.py:633-671` 逐行；`interpretation/V16_B_RESULTS.md` §1/§3；本仓 recompute CK |
| E2 | 「V16-A2 跑完但无书面裁决标签（interpretation missing）」 | A2 完整书面裁决早已存在（主表+5 组 CI+边界），只是不在 campaign_v16/ 而在 audits/…/V16_A2_DECISION.md；findings.stats 空 ≠ 无判读 | `interpretation/V16_A2_DECISION.md`（已复制入仓）；findings.json:28 |
| E3 | 「100M 全部 15 臂 dev NLL 从 H1024→H4096 单调上升」 | **14/15 严格单调**；C_A_lam0p85 为 6.1744→6.5859→6.5598（H2048→H4096 微降 0.026，仍远高于 H1024）。核心结论（tokens 不足导致 NLL 大幅抬升）不变 | recompute CK CAM_100M strict_monotone_rise |
| E4 | MASTER_EXPERIMENT_INDEX.csv 10 行列错位（V15/V16_A1_SELECT/V16_D/SM_P0A/P0B/TR×3/MT_M0/OPS）；报告行数口径 33/35 混乱 | 35 数据行（=阶段级实验数；1 表头）。运行级计数=283（六目录 RUN_INDEX 106+5+32+104+17+19，其中含 V16 trial_registry 72 全队列+planner 项）。修正版 = `MASTER_EXPERIMENT_INDEX_V2.csv`（本 staging docs/EXPERIMENT_INDEX.csv 同源，受控枚举：CONFIRMED/DEV_SINGLE_SEED/MEASUREMENT/NEGATIVE_CLOSED/PARTIAL/STAGED/INTERPRETATION_MISSING/DESIGN/ACTIVE/TERMINAL_ARCHIVED） | csv 模块列数直检 17/18/19 混杂→V2 全 19 列 |
| E5 | 报告称「100M AdamW 全场最优与 30M Muon 领先冲突」 | 非冲突：已由 PILOT_100M_DECISION.md:49 登记（尺度/tokens 配比不同）；本仓 100M RESULTS.md §6 沿用 | 同左 |
| E7 | staging 总述出现运行级 284，E4 初版写 278，二者均为加总笔误 | 六条核心研究线 CSV 实际为 106+5+32+104+17+19=**283 行**；另有 Hyperball baseline 4 行，合计 287 行（含设计、历史、重启等非独立训练条目）。35 行是阶段级研究索引，不能与运行行数等同；无证据表明该算术差异对应数据缺失。 | 六个 `experiments/*/RUN_INDEX.csv` + `baselines/hyperball-a1-evidence/RUN_INDEX.csv`，按 csv.DictReader 逐文件计数 |

## 2. V16 内部历史不一致（对账见 experiments/adamw-spectral-v13-v16/V16_FINAL_RECONCILED.md）

- `campaign_v16/V16_FINDINGS.md` 停留 STOP_RESOURCE（10-03）；`V16_FINAL_REPORT.md` 停留 PRELIMINARY 31/36——均为迁移快照，权威=interpretation/ 8 篇 + findings.json(10-10 18:08)
- `findings.json stats:{}` 空：finalize `c_` 前缀 bug；审计层用同一冻结例程补算（CI 见 interpretation/V16_C_SEED_RESULTS.md）
- `STATE.json stage:"A0"` 陈旧装饰字段；权威=final_label+QUEUE 71/71
- Windows vs 服务器 ~1e-5 平台差：两套数据并存，服务器为唯一选优依据（DECISIONS.md:7-10）

## 3. 其他项目已登记勘误（随原报告保留，此处汇总指针）

- R2F wrong-decay 错版：`muon-ns-allocation/outputs/defect_archive/` + `outputs/r2f_corrected_v1/` + R2F_CORRECTED_REPORT_CN.md（归因结论 101.7% 不变）
- continuation_adaptation README 落后于当日执行（旧仓 ERRATA_20261008）
- T0 native 中尺度比值勘误（arm1 62.32%）：muon_temporal ROUND1_ERRATUM（旧仓）
- direction_validation frozen_v1.json 双哈希重冻结（19:39 < 21:01 首跑，账本留痕，旧版字节未存档）
- minimind_base state_bank_report 内迁移前旧绝对路径（瑕疵，不影响数据）

## 4. 复算对账结论（recompute/outputs/，8/8 OK）

| 检查点 | 复算 vs 参照 |
|---|---|
| V16-C 3 seed | 精确一致（6 位小数） |
| DV Layer-1 / 30M T1 点估计 | 精确一致；CI 端点差 ≤1.5e-3（bootstrap 程序差异，已注明非 mismatch） |
| MSA 101.7% / SpecMuon 14 行 / 100M H4096 / Hyperball / A2 四臂 | 一致（A2 舍入 ≤3e-6；100M H4096 与 CSV 0 偏差） |

无 REPRODUCE_MISMATCH 项。

## 5. staging 发布层微调（仅 staging 副本，服务器原件未动）

- 3 份 specmuon 文档（GPU_COST_ESTIMATE / P1_EXECUTION_AUDIT / SPECMUON_FROZEN_PILOT）中他人 GPU 占用用户名已脱敏（原用户名→another-user）；语义不变。
- 5 个全重复证据 zip 省略（DUPLICATE_ELISION_AUDIT.csv 逐成员 SHA 验证）；7 个含唯一成员的 zip 保留。

## 6. 本轮（repro 工程）文档修复

- E6：context-aware-muon-100m/REPRODUCE.md 训练入口由 `scripts/driver_pilot.py`（不存在）更正为 `scripts/run_pilot_driver.py`（成功链证据 `results/pilot_driver.log:80` DRIVER DONE）；同目录 `scripts/pilot_driver.py` 为并行会话超时 driver（PILOT_100M_RUNTIME.md:13）。旧命令在原文中保留为勘误注记。
- 注：docs 与 V16_FINAL_RECONCILED 中「screen[P1](数)」写法改为「screen(P1)=数」以避免 Markdown 误渲染为链接（数值未变）。
