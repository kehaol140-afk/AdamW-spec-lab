# PATH_PORTABILITY_AUDIT — 硬编码路径审计（历史保留清单 + 便携层映射）

审计范围：repro_work 全部 163 个 .py。方法：grep `/home/b205`（排除 site-packages 行）+ 模块级常量解析。

## 1. 历史源码中的硬编码（**原样保留**，属冻结史实；共 21 文件）

| 文件:行 | 常量 | 指向 | 便携映射 |
|---|---|---|---|
| adamw-direction-validation/src/dv_runner.py:17 | `V16` | …/training_dynamics_lab/adamw_spectral_pilot | `V16_ROOT`（portable/run_dv_layer1.py import 后打补丁） |
| dv_core.py:138（两线同文件） | `sys.path.insert(V16)` | 同上 | 同上（import 前注入新 sys.path） |
| layer1_analyze.py:13 / layer1_screen_eval.py:13 / verify_corpus.py:13-15 / tests/test_dv_cpu.py:11,15,135 | `V16`/`CORPUS` | 同上+语料 | `V16_ROOT`/`DATA_ROOT` |
| adamw-direction-30m/src/s30m_mint_cache.py:16-18 | `CORPUS/V16_CACHE/TOKDIR` | 语料/V16 缓存/tokenizer | `DATA_ROOT/V16_ROOT/MINIMIND_ROOT` |
| s30m_core.py:23-25 | `LAB=minimind_base`（相对布局依赖） | …/training_dynamics_lab/minimind_base | `MINIMIND_ROOT`（布局差异：staging 中为 baselines/minimind-base-code） |
| context…/scripts/mint_pilot_cache.py:28-31, run_pilot_arm.py:31, pilot_eval_docs.py:26, analyze_pilot_gate.py:27 | `CORPUS/V16_CACHE/S30M_CACHE/TOKDIR/DEV` | 同族 | 同名键 |
| context…/src/context_aware_muon/model_100m.py:20 | minimind model 绝对路径 | …/minimind_base/model/model_minimind.py | `MINIMIND_ROOT/model/model_minimind.py` |
| specmuon/src/specmuon/io_utils.py:17 | `LAB_ROOT` | …/training_dynamics_lab | `CKPT_ROOT`+`MUON_TRANSITION_ROOT` 组合 |
| specmuon/tests/test_spec_muon_semantics.py:137 | routing_report 绝对路径 | muon_transition m0 | `MUON_TRANSITION_ROOT` |
| 30m/tests/test_s30m_cpu.py:150,154 | V16 sys.path+memmap | 同上 | `V16_ROOT` |

**原则**：以上文件 SHA 与 staging manifest 一致（见 REPRO_REGRESSION_REPORT §1），永不在原文件上改路径。

## 2. 相对布局依赖（无绝对路径但仍绑定目录结构）

- `campaign_v16/core.py:28` `ROOT=__file__…parent.parent`（自包含 ✓，仅需资产）
- `run_pilot_driver.py` ROOT 相对（✓）；`run_pilot_arm.py` ROOT 相对（✓，唯 DEV 绝对）
- `s30m_core.py` `LAB=HERE.parent`（布局绑定：期望 `experiments/../minimind_base`）

## 3. 便携层（新增，零修改历史文件）

`portable/paths.py`：`configs/paths.json`/环境变量双通道解析六键（见 configs/paths.example.json）。五个 `portable/run_*.py` 包装器：dry-run 打印解析后路径+资产状态+冻结计划；`--execute` 需 `ADAMWSPEC_ALLOW_GPU=1`（本轮一律不执行）。`tools/repro.py` 为统一 CLI。

## 4. 结论

新增便携入口对服务器路径**零强依赖**（全部经 paths 层）；历史硬编码 21 文件全部列入上表保留清单。
