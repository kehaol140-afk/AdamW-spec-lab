# 00_RESEARCH_OVERVIEW — 研究总览

研究服务器 `training_dynamics_lab`（2026-09-26 重组 → 2026-10-12 本轮归档）的 AdamW/Muon 更新几何研究。姊妹归档仓（五条已完结线）：https://github.com/kehaol140-afk/training_dynamics_lab @ `cf1558f`（[RELATED_ARCHIVES](RELATED_ARCHIVES.md)）。

## 两条主线

1. **AdamW 更新几何**（[experiments/adamw-spectral-v13-v16](../experiments/adamw-spectral-v13-v16/README.md) → [direction-validation](../experiments/adamw-direction-validation/README.md) → [direction-30m](../experiments/adamw-direction-30m/README.md)）：AdamW 预条件更新 A 的**方向结构**存在因果收益（三层级验证：A2 范数对照 ≥96% 保留、B 从头 +0.12~0.15、C 跨 seed +0.036~+0.151 3/3 同向）；纯幅度单调有害；幅度轨迹/LR/低β1 三个竞争解释全部排除。证据边界：单目标块、dev 级单 seed（C 除外）、对调优 Muon 落后 0.27 nats（30M）。
2. **Muon 谱/结构分配**（[muon-ns-allocation](../experiments/muon-ns-allocation/README.md) → [specmuon-pilot](../experiments/specmuon-pilot/README.md) → [context-aware-muon-100m](../experiments/context-aware-muon-100m/README.md)）：NS 迭代预算重分配（0/28）、组结构（GLOBAL_ONLY）、谱形倾斜（Head/Bulk NO_EFFECT/SINGLE_POINT）、100M 上下文混合（PILOT_FAIL）**全部阴性**——横断结论「标量幅度/LR 一阶、结构二阶」。

## 方法论（贯穿全部线，来自 GTTR/optimizer_sensitivity 教训）

预注册冻结协议→运行→一次性判读；判据锚定终点 NLL 而非短期 loss；baseline 必须调参（强 baseline）；任何结构候选先过同预算 LR/幅度网格 hurdle；一步筛查不可信（符号翻转实证）；负结果与正结果同权重发布（[05](05_NEGATIVE_RESULTS_AND_LESSONS.md)）。

## 导航

[01 时间线](01_RESEARCH_TIMELINE.md) · [02 方法与公式](02_METHODS_AND_EQUATIONS.md) · [03 实验设计索引](03_EXPERIMENT_DESIGN_INDEX.md) · [04 全结果汇总](04_ALL_RESULTS_SUMMARY.md) · [05 负结果与教训](05_NEGATIVE_RESULTS_AND_LESSONS.md) · [06 基线对比](06_BASELINE_COMPARISON.md) · [07 未来方向](07_FUTURE_RESEARCH_DIRECTIONS.md) · [08 资产备份索引](08_ASSET_BACKUP_INDEX.md) · [勘误对账](ERRATA_AND_RECONCILIATION.md) · [复算指南](REPRODUCE.md) · [封存资产](SEALED_ASSET_EXCLUSIONS.csv) · [去重审计](DUPLICATE_ELISION_AUDIT.csv) · [实验索引 CSV](EXPERIMENT_INDEX.csv)
