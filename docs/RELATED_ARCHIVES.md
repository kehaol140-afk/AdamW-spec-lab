# RELATED_ARCHIVES — 旧仓库与服务器侧归档索引（不复制内容，给准确定位）

## 1. 旧 GitHub 仓库（只读引用）

- 仓库：`https://github.com/kehaol140-afk/training_dynamics_lab`（public）
- 审计时 HEAD：**`cf1558f9f54b91f079c32c9554108e6ac8b9ca7a`**（branch `main`，2026-10-08T14:38:05Z，commit "Add final AdamW research archive and supplementary experiments"）
- 注：`kehaol140-afk/fluctuation-response` 是其旧名（301 重定向）；本地 `legacy_fluctuation_response` 的 remote 指向旧名，历史已分叉，不可 push

| 研究线 | 旧仓路径 | 结果 ID/文件 | 关键裁决（指纹） |
|---|---|---|---|
| FR/GTTR（91 实验） | `archived/fr_gttr/` | `RESULTS.md`、`EXPERIMENT_INDEX.md`（91 条）、`results/server_audit_20260926/`（冻结 39 列 CSV 镜像） | 切线传播 R²=0.9977；FR 普适律失败；GTTR 规则化全部失败（"derivatives correct but not usable"）；服务器 git 冻结 HEAD `6ac4942` |
| continuation_adaptation | `archived/continuation_adaptation/` | `RESULTS.md`+`ERRATA_20261008.md`；SEAL_LOCK.json 448 runs（2026-10-08 复核 byte-identical） | STOP：C1 +1.587 / C2 +0.019（Holm p=1.0）；B−D −0.075 描述性 |
| optimizer_sensitivity | `archived/optimizer_sensitivity/` | `RESULTS.md`、`EXPERIMENT_INDEX.md`、`reviews/2026-10-08/`（复算脚本+输出） | stable primal/unstable sensitivities；SAU NO_METHOD_GAIN；服务器 git HEAD `98ed984`（70 commits）；72G F3 原始转储留服务器 |
| muon_transition (M0/M1/M2) | `archived/muon_transition/` | `RESULTS.md`；服务器 `outputs/m2/m2_summary.json`（48 cells, all ok） | p*=0.5 双侧；R=0.0000；NO_SIGNAL_STOP_LINE |
| muon_temporal_response (T0/T2a/T2b) | `archived/muon_temporal_response/` | `RESULTS.md`（status form）；服务器 `outputs/t2b_target_screen_v1/target_gain_summary.json:104` verdict=NO_USEFUL_SIGNAL_THIS_SETTING | toy 死区不迁移；候选极差低于筛查线 30× |

## 2. 本仓与旧仓的证据分工

- 本仓六实验目录为独立研究（旧仓没有）；Muon 链叙事中 Temporal/Transition 两环**只引用不复制**（上表）
- 跨仓横断结论「标量压倒结构」的证据映射：本仓 MSA 四轮 + SpecMuon P1 + 100M LR 边界；旧仓 muon_transition LR 网格 4/6 边界 + temporal T2b
- `archived/` 内 6039 文件与服务器原目录的对应关系：代码+小证据复制（逐文件 sha 在旧仓 INCREMENTAL_ARCHIVE_AUDIT.md）；大 checkpoint/bank **未**入旧仓（GitHub 归档不等于大资产备份——见 docs/STORAGE_AND_BACKUP.md）

## 3. 服务器侧不入两个仓库的资产（摘要）

ITF 线（活跃，v8 PARTIAL，独立仓库 kehaol140-afk/ITF 已建空）；legacy state bank 18.4G；optimizer_sensitivity F3 转储 72G；全部 checkpoint/token 缓存。详见 docs/STORAGE_AND_BACKUP.md。

## 4. 第三方底座

modded-nanogpt：上游 `https://github.com/KellerJordan/modded-nanogpt`，HEAD pin `4ea6b937337a4889b8cfe3f38a93d120048d8f71`，MIT (c) 2024 Keller Jordan，本地镜像 origin=可校验 bundle（87M，服务器留档）。本仓只发布 provenance+patch（baselines/modded-nanogpt-provenance/）。minimind：上游公开框架，本仓 baselines/minimind-base-code 为 vendored 源码（29 文件，随 audits/minimind_base_file_manifest.sha256 清单，sha 逐条可验）。
