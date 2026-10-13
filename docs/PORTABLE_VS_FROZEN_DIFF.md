# PORTABLE_VS_FROZEN_DIFF — 便携层 vs 冻结源码差异清单

## 原则

冻结历史源码 **0 字节修改**（验收 §1：与 staging manifest SHA 逐一相等，见 REPRO_REGRESSION_REPORT）。便携能力全部由**新增文件**提供：

| 新增 | 性质 | 与冻结版关系 |
|---|---|---|
| `portable/paths.py` | 纯新增 | 无对应冻结件；仅解析配置/环境变量 |
| `portable/run_v16_campaign.py` | 包装器 | 不改 `campaign_v16/run.py`（其 ROOT 本自包含）；dry-run 校验资产后指引冻结入口 |
| `portable/run_dv_layer1.py` | 包装器 | 设计为 import 后对 `dv_runner.V16` 等模块常量打补丁再调 main（本轮未启用 execute 路径）；数学内核 `dv_core.py` 不动 |
| `portable/run_s30m_t1.py` | 包装器 | 同法补 `s30m_core.LAB/MINIMIND_BASE`、`s30m_mint_cache.CORPUS/V16_CACHE/TOKDIR` |
| `portable/run_context_pilot.py` | 包装器 | 补 `run_pilot_arm.DEV`、`mint_pilot_cache.*`、`model_100m` minimind 路径 |
| `portable/run_specmuon_p1.py` | 包装器 | 补 `io_utils.LAB_ROOT` |
| `tools/repro.py` | 新增 CLI | 只读 |
| `env/*`、`configs/paths.example.json`、`.github/workflows/cpu_ci.yml` | 新增 | 只读/CI |

**数学语义**：包装器不含任何算法代码；patch 仅限路径字符串常量（模块属性赋值），不触碰张量运算/随机数/超参。差异清单即本表；无第二个「派生源码副本」——避免分叉。

## 旧版 code copy 的相同性映射（保留冻结副本，不删除）

| 冻结副本 A | 冻结副本 B | 关系 |
|---|---|---|
| experiments/adamw-direction-validation/src/dv_core.py | experiments/adamw-direction-30m/src/dv_core.py | 逐字节相同（sha16 `b8943249f87599fb`，有意跨线固定内核） |
| 各线 vendored NS5（specmuon/native_ns.py 等） | torch `_muon.py` 冻结快照 | 同源（sha 38e95918 锚定） |

两份 dv_core.py 均保留；相同性由 CODE_PROVENANCE.csv 的 sha256 列自证。
