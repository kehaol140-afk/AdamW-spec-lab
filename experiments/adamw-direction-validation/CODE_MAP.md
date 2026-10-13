# CODE_MAP / REPRODUCE — adamw-direction-validation

## CODE_MAP

| 文件 | 作用 | 核心函数 | 阶段 | SHA256 | 上游关系 |
|---|---|---|---|---|---|
| `src/dv_core.py` | 数学内核 | `apply_step`/`n_a`/`paired_bootstrap`（同 V16 core.py） | 全部 | `b8943249f87599fb` | vendored 自 adamw_spectral_pilot/campaign_v16/core.py（4 个非语义增量 D1-D4） |
| `src/dv_runner.py` | 训练入口 | layer1 主循环；`sys.path` 只读注入 V16 pilot_core | layer-1 | 见 MANIFEST | 依赖 adamw-spectral-v13-v16（pilot_core/缓存/eval_sets 只读） |
| `src/layer1_analyze.py` | 分析/门 | 配对 bootstrap + 复刻门 | 判读 | 见 MANIFEST | — |
| `src/layer1_screen_eval.py` | 次级评估 | — | screen | 见 MANIFEST | — |
| `tests/test_dv_cpu.py` | 8 项单测 | 含 `test_V16_ledger_untouched`（V16 台账字节不变保护） | 前置 | 见 MANIFEST | — |
| `configs/frozen_v1.json` | 冻结协议 | — | 预注册 | `f8cc6b8e87c1b04a`（重冻结留痕双哈希） | — |
| `layer1_gate.json` | 门输出 | — | 判读 | `5364a0afb8475197` | — |
| `verify_corpus.py` + `verification_out/` | 语料验证链 | — | 前置 | 见 MANIFEST | — |

历史版本：无多版本（单轮执行）；重放事故件（首跑崩溃 log）保留于 `raw/layer1_run.out`。上游 MiniMind：模型定义引用 V16 仓内 `minimind_repo/`（=baselines/minimind-base-code 同源）。
