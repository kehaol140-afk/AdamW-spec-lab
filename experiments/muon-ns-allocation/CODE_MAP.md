# CODE_MAP — muon-ns-allocation

| 文件/目录 | 作用 | 核心函数 | 阶段 | 身份 | 上游关系 |
|---|---|---|---|---|---|
| `src/p1_engine.py` | P0/P1 探针引擎 | `evaluate_batch`、`kj_schemes`、`names_by_group`、`momenta_from_payload` | P0/P1 | 见 MANIFEST | Muon 更新=torch `_muon.py`（SHA 38e95918，冻结记录 outputs/m0 同款）；组划分 Q 按 sha256('P0P1_Q_v1\|gid')%3 |
| `src/r1_probe.py` / `r1_selection.py` | R0/R1 探针与选择 | — | R0/R1 | 见 MANIFEST | — |
| `src/r2_data_scale.py` / `r2f_recipe.py` / `recipe_registry.py` | R2/R2F 臂配方与数据 | — | R2/R2F | defect_archive 存错版+diff | — |
| `src/group_probe.py` / `state_restore.py` / `cost_meter.py` | 组探针/状态恢复/成本 | — | 公共 | 见 MANIFEST | — |
| `scripts/` | 各轮 runner | — | 全部 | 见 MANIFEST | — |
| `configs/{p0_p1_v1_1,r2f_executed}.json` | 冻结配置 | — | 预注册 | 先于运行（审计核验） | — |
| `references_*/` | 协议链 + r0_r1/r2/r2f contract.json + REVIEW_BUNDLE | — | 预注册 | MANIFEST_SHA256.json | — |
| `outputs/*/evaluation_lock*.json` | 评估锁（逐 run SHA，locked_before_evaluation） | — | 各轮 | 见文件 | — |
| `docs/AH_ASSET_INDEX.csv` | 75 .pt 资产索引（SHA/status） | — | 全部 | 见文件 | parent/R2 快照被 specmuon 只读依赖 |
| `audit_recheck/` | 2026-10-08 独立复算 | REAUDIT_REPORT_CN.md | 复核 | 见文件 | — |

历史版本保全：R2F 错版源码完整保留（defect_archive/run_r2f.py.WRONG_DECAY_VERSION + source_diff.patch）；superseded 旧终态 4 个（索引标 OLD_WRONG）按闭环文档**保留不删**。上游 MiniMind：模型定义；modded-nanogpt：无直接依赖（NS 系数同源 torch 内置实现）。
