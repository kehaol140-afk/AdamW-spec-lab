# CODE_MAP — context-aware-muon-100m

| 文件 | 作用 | 核心函数/类 | 阶段 | 上游关系 |
|---|---|---|---|---|
| `src/context_aware_muon/model_100m.py` | 100M 模型构建（h768/13L） | — | v1.0 | baselines/minimind-base-code（model_minimind.py 骨架） |
| `src/context_aware_muon/group_muon.py` | Group-A/B 混合优化器 | `GroupMuonOptimizer.step`(87)、lam_anchor 位级锚（:151-156） | v1.0 | torch `_muon.py` SHA 38e95918（NS5/系数/eps 同源） |
| `src/context_aware_muon/ns5.py` | NS5 + `adjust_lr` + `lam_anchor` | — | v1.0 | 同上 |
| `src/context_aware_muon/routing.py` | 56 AdamW/34 Muon→分组路由（FFN39/V-O26/QK native） | — | v1.0 | — |
| `src/context_aware_muon/data.py` | TokenStream 8192 tok/步 | — | v1.0 | 缓存前缀字节链（V16+s30m） |
| `scripts/driver_pilot.py` / `run_pilot_arm.py` | 驱动/逐臂 runner | — | v1.0 | — |
| `scripts/`（preflight/smoke/分析/评价） | 全链 | — | v1.0 | — |
| `configs/pilot_arms_v1.json`（sha `8149bcbb…`） | 15 臂冻结 | — | 预注册 | baseline_registry 引用 muon-ns-allocation SOURCE_BLOCKED 纪律 |
| `configs/CODE_FREEZE_V1.json` + `tests/`（25 项） | 代码冻结+单测 | — | 前置 | — |
| `results/pilot/analysis.json`（sha `636cd2e2…`） | 汇总/门 | — | 判读 | — |

历史版本：单轮执行；两起事故留痕（EXPERIMENTS E2）；NorMuon staging 在服务器 `audits/context_muon_baselines_20261012/staging/`（未随本目录，见 docs/RELATED_ARCHIVES.md §4 说明与 docs/08 资产表）。上游 MiniMind：模型骨架；modded-nanogpt：仅 NorMuon staging 适配器的参考来源（PR #276 镜像，非本 pilot 运行依赖）。
