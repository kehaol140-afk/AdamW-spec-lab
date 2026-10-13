# EXPERIMENTS — muon-ns-allocation

## 公共设置（全部轮次）

- 模型：30M MiniMind + Muon（m1 配方）；父代状态 = `outputs/p0_p1_v1_1/parent/`（seed 3101 轨迹，恢复点 step 512/1536，3 个 parent_state_*.pt，SHA 索引 AH_ASSET_INDEX.csv）
- 数据/评估：V_Q H256 筛查（256 docs 评测池，muon_freeze 数据审计）；tokenizer vocab 6400；同 LR native 为主对照 + LR 网格 {0.5×,1×,2×} hurdle
- 纪律：config 预注册先于运行（p0_p1_v1_1.json 14:48 早于 16:26 check）；selection_lock/endpoint_lock/evaluation_lock 先于评估锁定；位级重放；逐 run 证书（asset_certificate.json）；成本双卡尺

## E1 P0/P1 单组重分配（2026-09-29）

14 参数组 × 2 状态 × K∈{3,5,7} → 28 个一步检查收益（check_rows.csv 396 行原始）；对照：native + 全局更新缩放 ×{0.5,2}（amplitude_controls.csv）；门 δ_step=1e-5。

## E2 R0 统一时钟 / R1 组倍率与等范数（2026-09-29）

R0：manual 与 step 两种 NS 归一化时序实现对拍（工程 gate）；R1：8 组配对比较（paired_effects.csv）——全局增益、组倍率公平对照、纯组配对、匹配结构 NS3、等搜索结构、反向匹配 + 2 诊断臂（Z_NO_MUON_DATA/P_PRE_STATE）。

## E3 R2 幅度延续（2026-09-29）

N/AH/L/D 四臂自 s512/s1536 恢复 256 步（14 个 .pt 终态+快照）；一步 vs 256 步符号对比；endpoint_lock 先于评估。

## E4 R2F 归因 + 勘误（2026-09-29→30）

40 run（r2f_lrwd_retention_v1）：L/AH/D/N × 2 状态 × 终态+逐步快照；d_trigger τ=1e-3 预注册。**缺陷**：首版 runner decay 版本错误 → `defect_archive/`（错版源码+diff）+ 修正版 4 run（r2f_corrected_v1，evaluation_lock_corrected）+ 勘误报告。LR/WD 语义：B_L = LR×0.5 **且** decay×0.5（两个标量同时减半——归因语义为「同预算标量组合」，引用时不得写成单 LR 减半）。

## E5 R2F-NL（2026-09-30）

非线性尾部变体 ×2 状态（nl_effects.json）；τ=1e-3 实用增量门。

## E6 终局决策（2026-09-30）

NEXT_RESEARCH_DECISION_CN.md：HOLD_NO_NEW_TOPIC（SOURCE_BLOCKED 论文不核对、无实测瓶颈不立项）；hyperball A0 公共基线审计（零训练）。

## 偏差汇总

全部轮次按冻结协议执行；唯一实质偏差 = R2F decay 错码（E4，已勘误闭环）；资产缺口 = /tmp/opencode/p0p1_q_sets.pt（Q 组划分，索引登记 active）灭失，可按 `sha256('P0P1_Q_v1|gid')%3` 确定性重建（待补建）。
