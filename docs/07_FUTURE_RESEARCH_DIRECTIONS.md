# 07_FUTURE_RESEARCH_DIRECTIONS — 未来方向（全部未执行；按证据边界排序）

以下均为**建议**，无任何一项已运行；每项标注其前置门与被冻结的边界条件。

## A. AdamW 方向线（有正信号，需补正式确认）

1. **DV Layer-2 + 30M T2 确认层**：seeds 3411-3413（DV）/3422-3424（30M）×4096 步 + holdout_v2 一次开封——协议已冻结（frozen_v1.json:199-209），仅待授权。这是把全部方向结论从 dev 级升到确认级的唯一路径。
2. **P(a=8) 跨 seed 补验**：修复 V16 screen 缺陷后的最小补账（P1 在 C 的缺位）；可与 1 合并。
3. **a>8 / 多块联合 / 层位置扫描**：a=8 仍是角点（B 内 a8>a4>a1 单调未饱和）；多块联合从未测试。
4. **规模外推 100M**：cross_scale G3 三条件（G2=TRANSFERS + readiness 缺口全闭 + 预算获批）未满足——30M_DECISION 明示「不具备」；tokens 配比问题（100M pilot 的 NLL 抬升）须先解决。
5. **机制最小实验**：C 终点谱形状敏感性零训练检查（V16_FINAL_MECHANISM Q5 建议，<1 GPU-h）。

## B. Muon 线（阴性主导，新方向须过 hurdle）

1. **任何新结构候选必须先过同预算 LR/幅度网格**（四次教训）+ 256 步协议（一步筛查不可信）+ 持续配方（状态补丁不保留）。
2. **NorMuon 消融（A2-A6）**：卡 G-2（LR 锚点被 RANGE_UNRESOLVED 悬置）；若推进需先定锚（扩网格或换锚定义）+ 官方仓逐行核验（G-5，网络恢复后）+ 1.5-2.6 GPU-h 授权（G-4）。
3. **100M 重跑的先决条件**：tokens 预算 ≥NLL 不再单调抬升的水平（当前 33.5M/100M 不足）；否则任何位次结论都受 under-training 污染。
4. **AMO/SAMuon/Hyperball 论文核对**：SOURCE_BLOCKED 解除条件=原文可得；解除前禁止二手重建（冻结决议）。

## C. 治理与封存

- holdout_v2：MINTED_SEALED；开封条件=Layer-2 endpoints 身份锁定（与 V16-C 相同纪律）。
- final_holdout（V16）已烧毁：任何后续实验须新评价集。
- 服务器大资产独立备份（[08](08_ASSET_BACKUP_INDEX.md) 的 NOT_YET_BACKED_UP 项）先于任何清理。

## D. 独立线（不在本仓范围）

ITF（信息-热力学基础）：v8 PARTIAL 断点可续，独立仓库 kehaol140-afk/ITF。
