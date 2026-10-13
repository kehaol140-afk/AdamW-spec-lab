# REPRO_REGRESSION_REPORT — 本轮回归验收（2026-10-12 repro 工程）

对象：`/home/b205/projects/AdamW-spec-lab_repro_work`（staging 的隔离副本 + 新增便携层）

## 1. 冻结源码 SHA 核对（验收 §1）

以 staging `manifest.json`（2187 文件）为基线逐一比对：

| 项 | 结果 |
|---|---|
| 缺失文件 | **0/2187** |
| **.py 文件内容变更** | **0**（关键项 PASS） |
| 内容变更文件 | 11 个，全部为计划内：6×REPRODUCE.md（R0/R1/R2 分级块+100M 入口勘误 E6）、README.md（四段新增）、docs/ERRATA（E6+记法）、V16_FINAL_RECONCILED.md（记法 `screen(P1)=数`，数值不变）、manifest.json（重建）、recompute/outputs/S30M_T1_recompute.json（**仅 JSON 键序**——glob 枚举顺序差异，`json equal: True`，数值逐位一致） |
| 原仓（training_dynamics_lab） | 零触碰（本轮无任何写操作） |

## 2. 回归执行记录

| 检查 | 结果 |
|---|---|
| CPU 复算套件 | **8/8 OK**（`python3 tools/repro.py recompute --all`） |
| AST 语法 | 163 个 .py，0 错误 |
| `tools/repro.py list/check/recompute/test/smoke/train` | 全部可用；check 对缺失资产正确报 BLOCKED_ASSET（exit 1）；train 默认 DRY-RUN（双闸说明打印） |
| `portable/run_*.py` ×5 | dry-run 全部打印解析路径+资产状态+冻结计划+`BLOCKED_ASSET`；无 GPU 副作用 |
| GH Actions YAML | yaml.safe_load PASS；CI 逻辑本地等价执行：AST ✓、链接检查（豁免 minimind 上游 README 图片 77 处）真实断链 0、复算 8/8 |
| 封存集 | holdout_v2 三件不在仓内；CLI/包装器无加载路径 |
| 大小写冲突 / .pt/.u32/.pyc/PDF | 0 |

## 3. 入口修复清单

仅 1 处文档错误修复（100M `driver_pilot.py`→`run_pilot_driver.py`，ERRATA §E6）；六线冻结入口全部经历史日志/运行清单复核无误（docs/TRAINING_ENTRYPOINTS.md）。

## 4. 已知边界

- R1/R2 训练级复现全部 BLOCKED_ASSET（缓存/父态在服务器）——按验收 §5 如实展示，不声称训练复现成功
- 历史单测（dv/s30m/specmuon 的 tests）引用服务器绝对路径——在非服务器环境预期失败（PATH_PORTABILITY_AUDIT §1 保留清单 + CI 不执行这些 tests）
- S30M 复算 JSON 键序差异属文件系统枚举顺序，已在 §1 说明
