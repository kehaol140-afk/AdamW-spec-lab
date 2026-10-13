# 08_ASSET_BACKUP_INDEX — 资产备份索引

机读版：`FULL_ASSET_BACKUP_MANIFEST.csv`（audits 侧，本轮第四阶段生成）。本页为分级摘要。**「已有 Markdown 总结」不构成删除或免备份理由。**

## 备份状态总览（截至 2026-10-12）

| 状态 | 定义 | 数量级 |
|---|---|---|
| BACKED_UP_GITHUB | 已在本仓/旧仓（内容+SHA 可验） | 源码/协议/小证据（~2200+6039 文件） |
| BACKED_UP_STAGING | 在发布包 tar.gz 内（待上传） | 同本仓 |
| **NOT_YET_BACKED_UP** | 仅服务器一份、无独立副本 | 见下表全部大资产 |

## NOT_YET_BACKED_UP 清单（服务器原件全部保留）

| 资产 | 位置 | 字节 | 可重建 | 说明 |
|---|---|---:|---|---|
| v07 5M bank 56 态 | legacy.../e3_confirmatory/v07/ | 4.6G | 否 | 多项目原料（OPS/CA/ITF/SM-P0B） |
| G1a 26M bank 9 态 | legacy.../generalization_g1a_26m_v01/bank/ | 2.8G | 否 | — |
| G1b 100M bank 9 态 | legacy.../generalization_g1b_100m_v01/bank/ | 14G | 否 | 100M 诊断对照唯一件 |
| G2 数据分布线 | legacy.../generalization_g2_data_distribution_v01/ | 5.3G | 否 | — |
| F3 曲率原始转储 6 件 | optimizer_sensitivity/outputs/f3_curvature_mechanism/raw/ | 72G | 否 | 结论已固化 json；原始 fp64 数据 |
| MSA 训练状态 75 .pt | muon_spectral_allocation/outputs/** | 11.65G | active 5.3G 否 / 诊断 5.4G 可重放 | SpecMuon 父代/temporal 锚点在其中 |
| muon_transition 51 cell | muon_transition_experiment/outputs/ | ~15G | 50 cell 可 GPU 重放 | m1_muon_lr2.0 cell=temporal 只读锚点（必保）；eval_pools.pt 必保 |
| 100M 端点 ×15 | context_aware_muon/results/pilot/ | 6.0G | 是（seed+缓存） | 裁决已落档 |
| 30M/DV 端点 ×18 | adamw_direction_*/raw/checkpoints/ | 1.59G | 是（CRN 位级已证） | — |
| V16 checkpoints ×12 | campaign_v16/checkpoints/ | 214M | C 终点 6 个**否**（holdout 身份锁定）；B 6 个是 | — |
| token 缓存 ×3 | 各项目 artifacts/cache | 226M | 是（mint 配方在档） | 100M v1.1 复用 |
| HB A1 checkpoint ×2 + fineweb | hyperball_reproduction/ | 6.5G | checkpoint 否 / 数据可再取 | — |
| legacy git 4-commit 史 | legacy_fluctuation_response/.git | ~2M | — | 远端已分叉（301 重定向），仅此一份 |
| /tmp/opencode/mte_domain_train_windows.pt | /tmp/opencode/ | 1.39G | 是（109s） | 临时目录，应移出 |
| p0p1_q_sets.pt | 已灭失 | — | 是（确定性规则） | 待补建落盘 |

## 备份动作建议（待批准；本轮零执行）

P0：三 bank+G2+MSA active+temporal 锚点+V16-C 终点+F3 转储（如保留）→ 外部介质/第二服务器；P1：可重建 checkpoint（可让位磁盘）；P2：可再取数据集。SSH 拉取命令见 audits 侧 `BACKUP_AND_UPLOAD_GUIDE.md`。
