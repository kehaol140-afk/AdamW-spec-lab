# EXPERIMENTS — adamw-spectral-v13-v16

## E0 父实验（LOCAL_V1 lowmem，只读基座）

MiniMind h256/4L/seq256（4,655,616 参数），V12 重放审计通过（`replay_audit_v12/`，192 行重放损失位级一致）；父 gate `results/local_gate.json` = BASELINE_TUNING_UNRESOLVED（Δ=−2.12e-6，NO_LOCAL_SIGNAL，STOP）——即**父基线自身无标量调参残量**，后续任何增量不可归因于父级 LR 未调优。tokenizer vocab=6400；数据=冻结 token 缓存（8.5M tokens，sha `c91382dc…`，minimind t2t 语料子集，V12 data_manifest 封存）。

## E1 V13 readiness（2026-10-02，Windows）

- 目的：四状态（H/A/g/update）谱与夹角测量，建立「谱结构存在」前提
- 设计：4 窗口 × {legacy, natural_prefix, new_probe_packed, new_probe_prefix, training_build_batch} 观测道；预算 3,439,912 token-forwards
- 评估：无训练收益判据；产出 `results/{directions,finite_difference,geometry,losses,summaries}`
- 偏差：无（四窗口全 COMPLETE，`results/window*_complete.json`）

## E2 V14 机制筛查（2026-10-02）

- 目的：有限 rank-1 谱修正 uσvᵀ 是否有独立于标量的增量
- 设计：dev 128 文档筛查 + 20 confirm（冻结于 config，manifest sha `74caaaa5`）；数值校准 max|ΔNLL|=2.51e-8
- baseline：同 LR native；判据：confirm CI 是否与 0 分离
- 结果：`SCALE_RANGE_UNRESOLVED`（谱vs标量点差 2.41e-6 CI 跨零）；fresh_gradient +1.10e-4、fresh_preconditioned +6.89e-5（区间下界为正）→ 指向「方向」而非「谱形」

## E3 V15 方向-视界（2026-10-02）

- 目的：整块方向替换三族 × 视界 H1/H16/H64 的短分支信号
- 设计：S/F/P × (a,b) 小网格 × 256 步短分支；dev+confirm 128；H0 更正轮（h0_confirm=8.9522）
- 结果：`BRANCH_SIGNAL_WITHIN_GRID`；F Δ_H64=+0.0014509 CI[+0.000886,+0.002030]、P Δ=+0.000972 CI[+0.000480,+0.001470]（单 seed 3407）

## E4 V16 战役（2026-10-03 Windows 启动 → 10-10 服务器完赛）

容器（PLAN_FROZEN.md + campaign_config.json）：attempts≤120k、forwards≤400M、device≤24h、wall≤72h、weights-only checkpoints≤12（deletions.jsonl 登记后可再生删除）、anchor SVD≤48。**种子 3407（A/B），3408/3409/3410（C）**。

| 阶段 | 设计 | 关键参数 | 判据/门 |
|---|---|---|---|
| A0 | H0 视界更正 | — | 与 V15 一致性核对 |
| A1 | 三族×(a∈{1,2,4,8}×b∈{0.5,1,2}) 36 条，父 step256 分叉 256 步 | lr0.003（父轨迹） | development H64/H256 选优 |
| a1_select | 三族各冻结 1 条（无平手） | S(8,2)/F(8,2)/P(8,2) | STATE.json:5-24 |
| A2 | 4 条共同范数对照（F@qS/P@qS/S@qF/S@qP） | q 来自 a1 三选 | 方向效应保留率 |
| B | 18 条从头 2048 步：9 固定（S/F/P×LR{0.001,0.003,0.006}）+ 9 预写扩展（S4-S8/F4F5/P4P5，边界外扩 0.0003、a 邻域、β1 变体） | warmup128→cos0.1peak；dev H2048 选优 + screen 256 冻结比较 | 路由规则 1-4（PLAN_FROZEN:38） |
| C | 2 配方（b_S1_plain_lr0.001 + b_F1_a8_lr0.001）× seeds{3408,3409,3410}，4096 步 warmup256 | 同 seed 同初始化同数据序列；LR 从 B 转移 | final_holdout 1024 docs 一次开封；PRELIMINARY_SEED_ROBUST 判据（3 同向/≥2 正/中位≥1e-3/无反 >1e-3） |
| D | B 内嵌锚点 18×4（step 205/512/1024/1536）×3 矩阵 + SVD 36 | 目标/邻接 FFN/o_proj | 机制材料（非门） |

**实际 vs 计划偏差**（全部留痕于 DECISIONS.md）：
1. Windows 72h 墙钟中断（STOP_RESOURCE）→ 迁移服务器全重跑（Option A，旧 31 条只读封存 `raw_win_pilot_20261003/`，排除于选优）
2. 预算容器修订一次（budget_ledger.json:18-27，授权=用户迁移决定）
3. `a2_Fdir_qS` 14 步历史残留按协议重放去重（终态 256 行）
4. B 段 endpoint checkpoint 上限触发删除（12 个，sha 全录 deletions.jsonl）→ 引发 route_b screen 缺席基线缺陷（P1 被误拒出 C；V16_B_RESULTS §3 披露）
5. finalize stats 块 `c_` 前缀 bug → findings.stats 空（审计层以同一冻结例程补算）
6. holdout 开封后冻结：「no further tuning permitted; proceed to finalize」（DECISIONS.md:22）

## 数据划分与评估集

`eval_sets.json`（sha 随 data_manifest 封存）：development 256 / screen 256 / final_holdout 1024 文档，doc_id=SHA256，逐文档 loss_sum/token_count/state_id/param_fingerprint 落盘。三 split 均已在 V16 选择流程中烧毁（final_holdout 10-10 一次开封）。
