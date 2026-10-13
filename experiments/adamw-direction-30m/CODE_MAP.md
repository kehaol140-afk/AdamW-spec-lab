# CODE_MAP — adamw-direction-30m

| 文件 | 作用 | 核心函数 | 阶段 | SHA256 | 上游关系 |
|---|---|---|---|---|---|
| `src/dv_core.py` | 数学内核 | 同 V16（vendored） | 全部 | `b8943249f87599fb` | adamw_spectral_pilot/campaign_v16/core.py（与 direction-validation 同一文件） |
| `src/s30m_core.py` | 30M 适配层 | 模型构建/AdamW+Muon 双路由（56+34 张量）/TokenStream | T1 | 见 MANIFEST | 模型定义依赖 baselines/minimind-base-code（model_minimind.py）；Muon=torch `_muon.py` SHA 38e95918 |
| `src/s30m_runner.py` | 训练入口 | 5 阶段自适应 staging | T1 | 见 MANIFEST | — |
| `src/pilot_shim.py` | 轻依赖垫片 | `loss_sum_count`（16 行逐字节 vendored） | 评估 | 见 MANIFEST | pilot_core.py:531-540 |
| `src/s30m_mint_cache.py` | 缓存铸造 | 17M tokens（前 8.5M 与 V16 逐字节同） | 前置 | 见 MANIFEST | 语料=上游 minimind jsonl（未随仓） |
| `src/s30m_analyze.py` | 分析/门 | 配对 bootstrap | 判读 | 见 MANIFEST | — |
| `src/s30m_smoke.py` | smoke | 50 步×2 路径 | 前置 | 见 MANIFEST | — |
| `tests/test_s30m_cpu.py` | 9 项单测 | — | 前置 | 见 MANIFEST | — |
| `configs/frozen_30m_t1.json` | 冻结协议（引用 DV 门数值） | — | 预注册 | `0156149803ff3310` | — |

历史版本：单轮执行无多版本；父协议链 V16→DV→T1 逐级引用（frozen_30m_t1.json:4 记录 F8-S8/P8-S8 门数值）。上游 MiniMind：模型定义+tokenizer；modded-nanogpt：无依赖。
