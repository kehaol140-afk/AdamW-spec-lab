# CODE_MAP / REPRODUCE — specmuon-pilot

## CODE_MAP

| 文件 | 作用 | 核心函数 | 阶段 | 上游关系 |
|---|---|---|---|---|
| `src/specmuon/operators.py` | 谱算子库 | `svd_basis`、`exact_polar`(36)、`head_spec_muon`(61)、`bulk_spec_muon`(76)、`muon_power`(91)、`apply_operator`(121) | P0/P1 | — |
| `src/specmuon/native_ns.py` | NS5 vendored | `vendored_zeropower_via_newtonschulz`(33) | 全部 | torch `_muon.py` SHA 38e95918 同源 |
| `src/specmuon/run_p0a.py` / `run_p0b.py` | P0 审计 runner | — | P0 | p0b 依赖 legacy v07 bank（服务器） |
| `src/specmuon/run_pilot.py` / `spec_muon.py` | P1 引擎 | 父态恢复/臂调度/范数匹配 | P1 | 父态=muon-ns-allocation outputs/p0_p1_v1_1/parent（只读） |
| `src/specmuon/analyze_p1.py` + `metrics.py` + `io_utils.py` | 分析/指标 | — | 判读 | — |
| `configs/{p0a_audit_v1,p0b_audit_v1,pilot_frozen_v1}.json` | 冻结配置 | — | 预注册 | pilot_frozen_v1 FROZEN_BEFORE_ANY_P1_RUN |
| `tests/` | 19 项 CPU 数值/语义测试 | — | 前置 | — |
| `audits/EXECUTION_LEDGER.md` | 14+ 条执行台账 | — | 全程 | — |

历史版本：单轮执行无多版本；P1 分支 checkpoint 成功后按协议 prune（台账留身份）；重放 gate 对拍件=R2 快照（SHA 在 P1_EXECUTION_AUDIT.md:50-53）。上游 MiniMind：模型定义；无 modded-nanogpt 依赖。
