# EXPERIMENTS — specmuon-pilot

## E1 P0-A 动量谱审计（2026-10-10）

6 个 m1 run × 56 矩阵（336 矩阵行，wall 262s CPU 池）；产出 `results/p0a/{p0a_by_module,p0a_per_matrix}.csv` + `p0a_gprofile_points.json`。验证 H1（头部集中）。

## E2 P0-B AdamW bank 两阶段（2026-10-10）

phase1：4 个状态；phase2：56 状态（v07 bank 8 seed×7 步位）× 29 矩阵 = 1624 行（52s）；验证 H4（形状与合并 CV）。**依赖 legacy v07 bank（服务器 4.6G，只读）**。

## E3 P1 Head/Bulk pilot（2026-10-11）

- 目的：谱倾斜能否通过 H5 三重筛查
- 模型/模块：30M MiniMind Muon，全部 56 个 Muon 矩阵逐个施加算子（范数匹配）
- 数据/评估：V_Q H256 筛查（256 docs）；tokenizer vocab 6400；父代状态恢复（s512/s1536，来自 muon-ns-allocation parent）
- 臂（7×2 恢复点=14 分支）：native_lr{0.5,1.0,2.0}、exact_polar、head_lam0.25、bulk_lam0.25、muon_p0.5
- 训练：256 步/分支；重放 gate：与 R2 快照 s512/s1536_N_step1_model.pt 位级 maxdiff 0.0（P1_EXECUTION_AUDIT.md:50-53）
- **H5 门（pilot_frozen_v1.json，FROZEN_BEFORE_ANY_P1_RUN）**：候选须在两个恢复点均比 native 同 LR 好 >2e-3、胜 LR 网格 {0.5×,1×,2×}、超 ExactPolar >1e-3
- 成本：全分支含恢复/评估各 ~20.1/20.0 min GPU（EXECUTION_LEDGER #13-14；P1_RESULTS wall_seconds 为 256 步训练段 45-281s）
- 偏差：P1 成功分支 checkpoint 按协议 prune（EXECUTION_AUDIT.md:43）；/tmp 的 M0 域缓存曾重建一次（台账 #8，109s，断言计数一致）

## E4 未执行（协议内封锁）

P2（梯度时间序列谱）DATA_GAP 永久封锁；AdamSpec 迁移、大规模重跑——裁决后不开。
