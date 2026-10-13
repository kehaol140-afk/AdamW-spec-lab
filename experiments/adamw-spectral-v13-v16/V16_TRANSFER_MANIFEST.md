# V16 Transfer Manifest — ADAMW_SPECTRAL_CAMPAIGN_V16_20261002

- 打包日期：2026-10-10（Asia/Shanghai）
- 源机器：Windows 10 / PowerShell 5.1，项目根 `E:\大模型预训练\adamw_spectral_pilot`，解释器 `D:\pytorch\anaconda\envs\py39\python.exe`
- 目标：`/home/b205/projects/training_dynamics_lab/adamw_spectral_pilot`
- 包文件：`E:\大模型预训练\v16_server_transfer_20261010.zip`（单一顶层目录 `adamw_spectral_pilot/`）
- 打包方式：Python `zipfile`，ZIP_DEFLATED；逐文件 SHA256 见包内 `MIGRATION_INFO/SHA256SUMS.txt`（`sha256sum -c` 兼容格式）
- 本轮零删除、零上传、零 GPU 运行；所有历史文件保持原字节

---

## 1. 包含内容（按目录）

### 1.1 V16 活动实验（服务器续跑的直接依赖）

| 路径 | 内容 | 说明 |
|---|---|---|
| `campaign_v16/`（除 `v16_light.zip`、`__pycache__/`） | core.py / run.py / test_v16.py / test_resume_driver.py / tools_resume_driver.py / campaign_config.json / RUN_CAMPAIGN.ps1 / PLAN_FROZEN.md / RAM_RECOVERY.md(+patch) / STATE.json / QUEUE.jsonl / budget_ledger.json / trial_registry.csv / run_manifest.json / data_manifest.json / eval_sets.json / a0_h0_correction.json / findings.json / STATUS.md / a1_paired_stats.json / RAM_CURVE.csv / RAM_PROCESS_SNAPSHOT.csv / resume 测试与守卫日志 / attempts/（4 次工程 attempt 归档） | **冻结 source_hash = `4afaa78e3633c302023c35f926cc1fd866cf9642978213410a751bb79e17aa12`**（core.py+run.py+test_v16.py+campaign_config.json+RUN_CAMPAIGN.ps1，服务器上必须逐字节一致） |
| `campaign_v16/cache/train_tokens.u32` | 8.5M token uint32 训练缓存（34 MB） | SHA256 `c91382dc2a72cdbcf79493e2f2a74ed952b74c067c6af093628a928813fcab2b`；含父 1,052,996 token 前缀（已验证） |
| `campaign_v16/raw/` | steps 32 个 JSONL（31×256 行 + a1_P_a4_b1×1 行）+ eval 66 个 JSONL（31×2 个 A1 H64/H256 各 256 行 + 4 个 H0CORR） | **31 条已完成 A1 的全部逐文档结果与身份字段**（state_id/loaded_checkpoint_sha/global_step/param_fingerprint） |
| `campaign_v16/logs/resources.jsonl` | 资源门禁全量审计（10s 粒度） | 中断证据链 |

### 1.2 父状态（A1/A2 续训起点）

| 路径 | 内容 |
|---|---|
| `artifacts/checkpoints/seed3407_lr0.003/step256.pt`（56 MB） | **唯一必须的 checkpoint**。内含：model state_dict、**完整 optimizer state**、cpu/cuda/python/numpy **四套 RNG 状态**、`data_cursor=524288`、`total_effective_tokens=524288`、scheduler_position、config_hash、data_hash、grad_scaler、buffers、parameter_aliases。SHA256 `ed09a2c80f40aae1a8829a718913a4a40257cdc24fc0e7a2d29bc53a74383000` |
| `artifacts/checkpoints/seed3407_lr0.003/manifest.json` | checkpoint 哈希与 SUCCESS 状态 |
| `artifacts/data_manifest.json` | 父数据身份 `data_hash=fc70632e…`、bos/eos、train_token_count=1,052,996、父缓存哈希（load_checkpoint 校验用） |
| `artifacts/train_tokens.u32`（4.2 MB） | 父 1M token 缓存（V16 缓存前缀的对照原件，用于身份复核） |
| `artifacts/doc_roles.jsonl`、`eval_blocks.json`、`preflight.json`、`protocol_resolved.json`、`smoke_run_manifest.json`、`numerics/*.json` | 父协议数据角色与评估锁、数值校准证据 |

### 1.3 模型定义与 tokenizer（运行时硬依赖）

`pilot_core.model_and_optimizer` 执行 `from model.model_minimind import MiniMind…`（经 `binding["repo_root"]` 注入 sys.path）与 `AutoTokenizer.from_pretrained(binding["tokenizer_path"])`。包内提供：

| 包内路径 | 源（本地） | 说明 |
|---|---|---|
| `minimind_repo/model/model_minimind.py` | `E:\大模型预训练\minimind-master\model\model_minimind.py` | 模型定义（仅依赖 torch/transformers） |
| `minimind_repo/model/__init__.py` | 同上（空文件） | 包标记 |
| `minimind_repo/model/tokenizer.json`（441 KB） | 同上 | 原 binding `tokenizer_fingerprint=f9478a88…`；逐文件 SHA256 见清单 |
| `minimind_repo/model/tokenizer_config.json` | 同上 | AutoTokenizer 离线加载所需 |

### 1.4 根级源码与冻结配置

`pilot_core.py`（模型/checkpoint/训练批/损失，惰性导入）、`config/local_v1_lowmem.json`（**被 checkpoint config_hash 锁定，逐字节不可改**）、`bindings.local.json`（原样保留，服务器需按 §4 改写）、`run.py`、`pilot_workflow.py`、`prepare_data.py`、`resource_policy.py`、`tests/test_numerics.py`、`tests/test_resources.py`、`results/`（local_gate.json 等 8 个正式结果文件）、`README.md`、`PROTOCOL_LOCAL_V1.md`、`REVIEW.md`、`SHA256SUMS.txt`（旧版参考）、`RUN_WINDOWS.md`、`RUN_COMMANDS_RESOLVED.ps1`、`RUN_LOWMEM_SMOKE.ps1`、全部 PROMPT_*.md 合同与 `CODEX_RESEARCH_READINESS_AUDIT_V13.md`、本轮六份交付（`V16_FINAL_REPORT.md`、`V16_COMPARISON.csv`、`V16_MECHANISM_ASSESSMENT.md`、`SERVER_MIGRATION_PLAN.md`、`V16_TRANSFER_MANIFEST.md`、`V16_SERVER_RECOVERY_GUIDE.md`）。

### 1.5 V12–V15 历史证据（源码+正式协议+逐文档结果；不含大 checkpoint）

| 目录 | 包含 | 排除 |
|---|---|---|
| `replay_audit_v12/`（1.1 MB） | 全部 | — |
| `readiness_audit_v13/` | 代码、config、run_manifest、AUDIT_PLAN/READINESS_AUDIT、readiness.json、claim_evidence.csv、history_reconciliation.json、protected_*.json、results/（11 MB）、logs/（7 MB）、patches、tests、attempts | `frozen/`（996 MB 父系深归档）、`__pycache__`、`readiness_v13_light.zip` |
| `mechanism_screen_v14/` | 全部（代码、config、manifest、findings、raw/ 6.7 MB、logs、attempts、方向/校准 json） | `__pycache__`、`v14_light.zip` |
| `direction_horizon_v15/` | 代码、config、manifest、findings、selection_frozen、raw/（5.3 MB）、logs/（6.4 MB）、confirm_views、dev_scores | `checkpoints/`（1.85 GB，26 个分支快照）、`attempts/`（142 MB）、`__pycache__`、`v15_light.zip` |
| `v16_handoff/`（70 KB） | 全部 | — |
| `lowmem_revision/` | 全部（30 KB） | — |

### 1.6 MIGRATION_INFO/（服务器侧辅助，生成本次迁移）

`SHA256SUMS.txt`（全包逐文件哈希）、`requirements_frozen.txt`、`bindings.server_example.json`、`amend_budget_server.py`（一次性预算修订，只动 wall/pause/stop 三个运营字段，科学计数器零改动）、`verify_package.py`（服务器端完整性+身份+A1 清单校验）。

## 2. 明确排除项及理由

| 排除 | 理由 |
|---|---|
| V13 `frozen/`（996 MB）、V15 `checkpoints/`（1.85 GB）、V15 `attempts/`（142 MB） | 历史深归档/分支快照；V16 运行时零引用（唯一加载的 checkpoint 是父 step256）；其结论已有逐文档 raw + findings 保存 |
| `artifacts/checkpoints/seed3407_lr0.003/{step0,step128,step512}.pt`、`seed3407_lr0.001/`、`seed3407_lr0.0003/` | step256 是 A1/A2 唯一起点；step128 仅为已完成的 A0 H0 更正所用（其哈希 `6c839349…` 已在 run_manifest 记录）；step512/其他 LR 为父调参历史 |
| `data/pretrain_t2t_mini.jsonl`（44 MB 原始语料） | 数据身份已由 8.5M token 缓存（SHA256）+ eval_sets.json 封存；仅在需要重建缓存时按 `source_fingerprint=4403b671…` 重新获取 |
| `artifacts/adamw_spectral_light.zip`、`archive/`、`adamw_lowmem_revision.zip`、各 `*_light.zip` | 归档 zip，内容可由包内文件重建；禁止 zip 嵌套 |
| `campaign_v16/v16_light.zip` | `package_light()` 可再生 |
| 全部 `__pycache__/`、`.pyc`、空目录 `work/` | 可再生/空 |
| `E:\大模型预训练\minimind-master` 其余文件（训练脚本、lora 等） | 运行时仅需 `model/model_minimind.py` + tokenizer 两文件（§1.3） |

**排除量合计 ~3.0 GB；包未因此丢失任何复现必需的状态、数据或正式评价资产**（31 条 A1 逐文档结果、A0 更正、全部协议/合同/账本/身份链均在包内）。

## 3. 绝对路径声明（包内存在本地盘符引用的位置）

| 文件 | 字段 | 值 | 服务器处置 |
|---|---|---|---|
| `bindings.local.json` | repo_root / tokenizer_path / source_data_path / python_executable | `E:\大模型预训练\…`、`D:\pytorch\…` | **必改**：用 `MIGRATION_INFO/bindings.server_example.json` 覆盖（该文件不在 source_hash 内，改写合法） |
| `campaign_v16/campaign_config.json` | binding_python | `D:\pytorch\anaconda\envs\py39\python.exe` | 冻结字段，代码运行时不读取（已核实 grep 无引用）；以实际调用解释器为准，字段保留原值作历史记录 |
| `campaign_v16/tools_resume_driver.py` | LAB / PY | `E:\…`、`D:\…` | Windows RAM 门控包装器；服务器 RAM 充裕时**不需要运行**，直接 `python -m campaign_v16.run`；如需保留守卫逻辑须自行改路径（该文件不在 SOURCE_FILES 内） |
| `campaign_v16/RUN_CAMPAIGN.ps1` | 调用路径 | Windows python | 仅归档；Linux 命令见恢复指南 §5 |
| `config/local_v1_lowmem.json` | project.search_root | `E:\大模型预训练` | **元数据字段，运行时不用于路径解析**（实际路径来自 bindings）；该文件被 checkpoint config_hash 锁定，**逐字节保留不改** |
| `campaign_v16/logs/`、`RAM_*` 等 | 历史记录内文 | 含盘符 | 只读历史证据，不改 |

代码自身的路径推导全部为相对 `__file__`（`core.py` ROOT、`pilot_core` 无绝对路径，已核实），无其他未声明依赖。

## 4. 关键身份哈希（包内已验证值）

| 资产 | SHA256 |
|---|---|
| campaign source_hash（5 文件组合） | `4afaa78e3633c302023c35f926cc1fd866cf9642978213410a751bb79e17aa12` |
| campaign_config.json | `f0cf25669c4ee991e5fc56c7676c6eb87fc9459a544ed8cc81a0fa51f6c20982` |
| 父 step256.pt | `ed09a2c80f40aae1a8829a718913a4a40257cdc24fc0e7a2d29bc53a74383000` |
| V16 训练缓存 | `c91382dc2a72cdbcf79493e2f2a74ed952b74c067c6af093628a928813fcab2b` |
| 源语料（未打包，指纹） | `4403b6717352fcefe2b8ce50b47a29248085ac811eff6121e6114cf26e9c3d31` |
| tokenizer（binding 指纹） | `f9478a8852174e4952bddc4143afa8958f6feee6e54c62839f9c5cbe128783fa` |

其余文件（含 tokenizer.json 逐文件值）见 `MIGRATION_INFO/SHA256SUMS.txt`。

## 5. 环境（冻结版本）

Python 3.9；torch 2.6.0+cu124（CUDA 12.4）；transformers 4.49.0；tokenizers 0.21.1；numpy 2.0.2；psutil 6.1.1。GPU 需求：任意 CUDA GPU（本地实测 CUDA reserved 峰值 ~165 MB）；RAM ≥ 8 GiB 空闲（runner+模型 RSS ~1.5 GiB）；磁盘 ≥ 5 GiB（B/C checkpoint 峰值 8×56 MB + endpoints ≤12×14 MB）。

**注意**：服务器 GPU/驱动与本地不同 → 浮点归约顺序不同 → 跨平台逐文档 NLL 不得进入同一统计比较（恢复指南 §6）。

## 6. 密钥与安全声明

包内不含 API Key、密码、`.env`、token 或任何凭据文件（已核查：bindings.local.json 仅含本地路径与语料/分词器指纹，非机密）。

## 7. 打包与校验程序

- 打包：`zipfile.ZipFile(..., ZIP_DEFLATED, compresslevel=9)`，逐文件 `zf.write(src, arcname)`；包内路径全部 ASCII 正斜杠
- 封包后重新打开：`testzip()` CRC 全检 + 逐条目 SHA256 与 `SHA256SUMS.txt` 比对 + 必需文件存在性 + A1 结果行数复核（验证输出随包报告）
