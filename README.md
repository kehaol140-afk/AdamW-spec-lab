# AdamW-spec-lab — AdamW/Muon 更新几何研究档案

研究服务器 `training_dynamics_lab` 的 AdamW 谱/方向修正与 Muon 谱分配研究线公开发布仓。
姊妹归档仓（已完结五线）：https://github.com/kehaol140-afk/training_dynamics_lab （FR/GTTR、continuation_adaptation、optimizer_sensitivity、muon_transition、muon_temporal_response，见 docs/RELATED_ARCHIVES.md）。

## 两条研究链

**AdamW 更新几何**（V13→V16→Direction→30M）：AdamW 预条件更新 A 的局部方向修正，在特定匹配幅度对照下呈现独立收益。V16-C 为 3 个 seed × 4096 步的历史 holdout 结果（但未与 S(8,1) 全幅度匹配）；后续 Layer-1 和 30M 属单 seed 开发级匹配幅度验证；新的 Layer-2 / 30M 多 seed 确认未执行。30M 效果仍未接近调优 Muon。

**Muon 谱分配**（NS allocation→SpecMuon→100M Context-Aware）：谱/结构候选四轮全部阴性（0/28 重分配、AH 被 B_L 解释 101.7%、Head/Bulk 无确认效应、100M Group-A/B PILOT_FAIL）——「标量幅度/LR 一阶、结构二阶」为跨线横断结论。

## 目录

```
experiments/  六个实验目录（各含 RESULTS/METHOD/CODE_MAP/REPRODUCE/RUN_INDEX/MANIFEST.sha256）
  adamw-spectral-v13-v16/    V13 测量→V16 战役（含 interpretation/ 判读层 + V16_FINAL_RECONCILED.md）
  adamw-direction-validation/ Layer-1 匹配矩阵（方向 vs 幅度因果分解）
  adamw-direction-30m/        30M 规模迁移 T1
  muon-ns-allocation/         P0/P1→R0/R1→R2→R2F(-NL) 全链阴性 + 勘误链
  specmuon-pilot/             P0 谱审计 + P1 Head/Bulk 负裁决
  context-aware-muon-100m/    100M 15 臂 pilot（PILOT_FAIL）+ NorMuon staging 状态
baselines/    minimind-base-code（vendored 源码）/ modded-nanogpt-provenance（上游 pin+patch）/ hyperball-a1-evidence
recompute/    CPU 独立复算套件（8 检查点，本次输出随附 outputs/）
docs/         索引/时间线/负结果册/勘误/封存清单/存储与备份/复算指南
```

## 证据纪律

预注册冻结协议 → 运行 → 一次性判读；正负结果并列（docs/NEGATIVE_RESULTS.md）；dev 级与确认级严格分级（docs/EVIDENCE_GRADING 由 ERRATA 文档口径承载）；封存评估集不发布（docs/SEALED_ASSET_EXCLUSIONS.csv）；全部文件 SHA256 可验（各目录 MANIFEST.sha256）。

## 一分钟复算

```bash
python3 recompute/run_recompute.py   # 纯标准库 CPU；8/8 检查点应全 OK
```

## CPU 复算 5 分钟入门

```bash
python3 recompute/run_recompute.py   # 8/8 检查点（点估计与发布表精确一致）
python3 tools/repro.py list          # 六线 R0/R1/R2 能力总览
python3 tools/repro.py check         # 外部资产核验（报 BLOCKED_ASSET 即为预期）
```
零第三方依赖（标准库）；无 GPU、无封存集访问。CI（.github/workflows/cpu_ci.yml）在每次 push 自动执行同等检查。

## 训练复现需要什么

R1（parent 重放）/R2（从头）均被外部资产门控：token 缓存与父 checkpoint（文件名/大小/SHA/获取方式见 `docs/ASSET_RESTORATION_GUIDE.md`）+ GPU 环境（`env/`）。统一入口 `tools/repro.py`（dry-run 默认；`--execute` 需 `ADAMWSPEC_ALLOW_GPU=1` 双闸）。**六线当前等级：R0 全 PASS；R1 全 BLOCKED_ASSET；R2 5 线 BLOCKED_ASSET + MSA UNVERIFIED**（详见各 REPRODUCE.md 分级表与 `docs/REPRODUCIBILITY_MATRIX.csv`）。

## 第三方源代码/许可证

minimind（baselines/minimind-base-code，源码+上游清单）；modded-nanogpt（MIT, Keller Jordan；本仓仅 provenance+3 PR patch，不整仓复制）；torch `_muon.py`（PyTorch BSD-style，冻结快照 sha 38e95918）；旧仓 training_dynamics_lab（自有，public @cf1558f，只读引用）。完整表：`docs/EXTERNAL_DEPENDENCIES.md`。

## 历史版本定位

冻结源码身份：`docs/CODE_PROVENANCE.csv`（163 文件逐 SHA+阶段+资产）；同哈希副本映射（dv_core.py 双线固定）与便携层差异：`docs/PORTABLE_VS_FROZEN_DIFF.md`；V13-V16 历史试错/attempts 与勘误链保留于各实验目录（archive 身份未删改），入口区分见 `docs/TRAINING_ENTRYPOINTS.md`。
