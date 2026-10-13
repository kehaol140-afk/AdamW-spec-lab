# V16 Server Migration Plan

目标服务器项目根：`/home/b205/projects/training_dynamics_lab`
本文件仅为方案，**未执行任何上传**。制定日期 2026-10-10。

---

## 0. 迁移理由（本地无法收尾的实证）

1. 训练中断根因（运行中可用 RAM < 1.0 GiB 硬停线，worker RSS ~1.4 GiB）在 2026-10-03、10-08、10-10 三次复核中持续存在；10-10 实测物理 RAM 仅 0.699/15.688 GiB 可用，低于 2.6 GiB 启动条件
2. 冻结 72h 墙钟已过（2026-10-06 00:03）且累计暂停 14,700 s 超上限 → 本地恢复需显式预算修订，且即便修订也无法提供 B/C 所需 ~19h 无人值守稳定内存（日常使用笔记本）
3. 剩余预算容器充足（~104.5k/120k attempts、~380M forwards 余量）——瓶颈纯属系统资源，非科学预算

## 1. 最小上传资产清单（必须实际上传）

### 1.1 代码与冻结配置（~250 KB）

| 本地路径（相对 `adamw_spectral_pilot/`） | 服务器目标 | 说明 |
|---|---|---|
| `campaign_v16/core.py` | `campaign_v16/core.py` | source_hash 组成部分，**不得改一字节** |
| `campaign_v16/run.py` | `campaign_v16/run.py` | 同上 |
| `campaign_v16/test_v16.py` | `campaign_v16/test_v16.py` | 同上 |
| `campaign_v16/campaign_config.json` | 同名 | config SHA `f0cf2566…` |
| `campaign_v16/RUN_CAMPAIGN.ps1` | 同名（仅归档） | Linux 用 §5 的 POSIX 命令替代 |
| `campaign_v16/PLAN_FROZEN.md`、`RAM_RECOVERY.md`、`tools_resume_driver.py`、`test_resume_driver.py` | 同名 | 协议与恢复历史 |
| `pilot_core.py` | 根目录 | 模型构建/checkpoint 加载/训练批/损失（core.py 惰性导入） |
| `config/local_v1_lowmem.json` | 同名 | PCFG；load_checkpoint 校验 config_hash |
| `bindings.local.json` | 同名 | **路径字段需按服务器改写**（见 §4），改写后 source_hash 不受影响（不在 SOURCE_FILES 内） |

### 1.2 父状态与数据身份（~94 MB）

| 资产 | 大小 | SHA256（已验证） | 作用 |
|---|---|---|---|
| `artifacts/checkpoints/seed3407_lr0.003/step256.pt` | 56 MB | `ed09a2c80f40aae1a8829a718913a4a40257cdc24fc0e7a2d29bc53a74383000` | A1/A2 起点；内含 model+optimizer+**cpu/cuda/python/numpy 四套 RNG**+data_cursor=524,288+total_effective_tokens=524,288+scheduler_position |
| `artifacts/checkpoints/seed3407_lr0.003/manifest.json` | 1 KB | — | step128/256 哈希与 SUCCESS 状态 |
| `artifacts/data_manifest.json` | 2 KB | — | 父 data_hash `fc70632e…`、bos/eos、train_token_count；load_checkpoint 校验用 |
| `results/local_gate.json` | 1.4 KB | — | verify_parent() 读取（gate=STOP）；审计用 |
| `campaign_v16/cache/train_tokens.u32` | 34 MB | `c91382dc2a72cdbcf79493e2f2a74ed952b74c067c6af093628a928813fcab2b` | 8.5M token uint32 memmap；B/C 从头训练同一缓存 |
| `campaign_v16/eval_sets.json` | 4.2 MB | 按 data_manifest 中 digest | development 256 / screen 256 / final_holdout 1024 的 doc_id+ids+token_count（**final_holdout 保持封存不查看**） |
| tokenizer（`model/tokenizer.json`，源自 minimind-master） | ~2 MB | 指纹 `f9478a88…` | 运行时不需要（缓存已建）；上传仅为数据重建能力，记录指纹即可 |
| 源语料 `data/pretrain_t2t_mini.jsonl` | 44 MB | 指纹 `4403b671…` | **不上传**（缓存+eval_sets 已封存身份）；仅当需要在服务器重建缓存时才传，并按指纹核验 |

### 1.3 V16 运行状态与账本（~10 MB）

`campaign_v16/` 下：`STATE.json`、`QUEUE.jsonl`、`budget_ledger.json`、`trial_registry.csv`、`run_manifest.json`、`data_manifest.json`、`a0_h0_correction.json`、`DECISIONS.md`、`STATUS.md`、`raw/`（steps+eval，8.1 MB）、`logs/resources.jsonl`（0.7 MB）、`a1_paired_stats.json`、`RAM_CURVE.csv`、`RAM_PROCESS_SNAPSHOT.csv`。

**上传前必须清理**：删除 `ledger.lock`、`run.lock`（若存在，当前均无）；确认无进程持有。

### 1.4 历史归档（不上传，留本地）

V13/V14/V15 完整目录、`lowmem_revision/`、`archive/`、`replay_audit_v12/`（运行时惰性导入仅在重建数据时需要）、`mechanism_screen_v14/`、`direction_horizon_v15/`、其余 `artifacts/checkpoints/`（step0/128/512、其他 LR）、`*.zip`、`v16_light.zip`。它们是审计证据链，不是服务器执行依赖。

## 2. 环境依赖（服务器需满足）

| 组件 | 本地版本（建议逐项对齐） |
|---|---|
| Python | 3.9（`D:\pytorch\anaconda\envs\py39`） |
| torch | 2.6.0+cu124（CUDA 12.4） |
| transformers | 4.49.0 |
| tokenizers | 0.21.1 |
| numpy | 2.0.2 |
| psutil | 6.1.1 |
| GPU | 任何 CUDA GPU；VRAM 峰值 <1 GiB（本地实测 CUDA reserved ~165 MB），8 GB 充裕 |
| RAM | runner+模型 ~1.5 GiB RSS；建议 ≥8 GiB 空闲 |
| 磁盘 | B/C checkpoint 峰值 ~8×56 MB + endpoints ≤12×14 MB；建议 ≥5 GiB 空闲 |

版本差异（尤其 torch/CUDA）会改变浮点归约顺序 → 见 §6 可比性规则。

## 3. 预算与合同的显式修订（服务器首跑前完成）

冻结合同中三项已无法在本地满足，迁移即视为修订，须写入 `DECISIONS.md` 并由用户批准：

1. **墙钟**：`wall_clock_hours=72` 自 2026-10-03 起已过期。建议改为"服务器首次启动时刻起 72h 新窗口"，写入 DECISIONS.md（条件=迁移；证据=RAM_RECOVERY.md+本计划；行动=新墙钟）。
2. **累计暂停**：`pause_seconds=14,700` 超 2h 上限。随墙钟修订一并重新计数（历史值保留在账本备注，不清零原始记录）。
3. **设备/尝试/前向上限（120k attempts / 400M forwards / 24h device）保持不变**——剩余量充足，无需也不应放宽。

科学阈值（delta_short/long、bootstrap、评估集构成、选择规则）**零修改**。

## 4. 路径改写

`bindings.local.json` 在服务器上改为（示例，按实际路径）：

```json
{
  "status": "READY",
  "repo_root": "/home/b205/projects/training_dynamics_lab/minimind-master",
  "python_executable": "/usr/bin/python3.9",
  "tokenizer_path": "/home/b205/projects/training_dynamics_lab/minimind-master/model",
  "source_data_path": "/home/b205/projects/training_dynamics_lab/data/pretrain_t2t_mini.jsonl",
  "source_fingerprint": "4403b6717352fcefe2b8ce50b47a29248085ac811eff6121e6114cf26e9c3d31",
  "tokenizer_fingerprint": "f9478a8852174e4952bddc4143afa8958f6feee6e54c62839f9c5cbe128783fa"
}
```

`repo_root` 若不上传 minimind-master 仓库，仅 tokenizer 目录按指纹核验即可。`core.py` 的 `ROOT` 由 `Path(__file__).resolve().parent.parent` 推导——服务器目录层级必须保持 `training_dynamics_lab/campaign_v16/`。

## 5. 执行命令（服务器）

```bash
cd /home/b205/projects/training_dynamics_lab
# 1) 校验（见 §7）后，先跑回归测试
python3.9 -m unittest campaign_v16.test_v16 -v
python3.9 -m unittest campaign_v16.test_resume_driver -v
# 2) 串行续跑（单写入者；崩溃后同命令续跑）
nohup python3.9 -m campaign_v16.run --resume --max-seconds 86400 \
  > campaign_v16/logs/server_run.out 2>&1 &
```

队列自动从 pending 任务继续：`a1_P_a4_b1` → 其余 4 条 P → `a1_select` → A2 → B → C → D → holdout → finalize。`--max-seconds` 按 §3 新墙钟设置。

## 6. 跨平台可比性规则（硬约束）

**Windows 与 Linux/GPU 变更后的逐文档 NLL 不得进入同一统计比较。**

两种执行方案：

- **方案 A（推荐）**：服务器上**重跑全部 36 条 A1**（9,216 步 ≈ 1.5h 服务器算力），A1 选择、A2、B/C/D 全部基于服务器数据。Windows 31 条降级为 pilot/存档证据（保留独立身份链，不参与选择与统计）。理由：跨 GPU 浮点差异使 2e-6 nats/token 数值分辨率无法保证，同网格混合会在 1e-3 量级结论（正是本研究的效应量级）中引入不可控平台噪声。
- 方案 B（不推荐）：只补 5 条缺失分支，与 Windows 31 条混合选择；必须先做平台校准（服务器重跑 1 条已完成分支如 `a1_S_a1_b1`，比较 256 步后 development NLL）。若偏差 > 数值分辨率 2e-6（预期会超过），则混合无效，回退方案 A。

B/C 从头训练在服务器上自洽（新 seed、新初始化、同机内配对），不受此约束影响；从 A1 转移的只有配置级量（a\*, b\*），无数据合并。

## 7. 哈希校验与恢复协议

1. **上传前**：生成本地 `SHA256SUMS.txt`（涵盖 §1.1–1.3 全部文件）
2. **服务器启动前**：`sha256sum -c SHA256SUMS.txt` 全绿才允许运行；任何红项重传
3. **关键身份复核**（服务器上执行一次）：
   - `run_manifest.json` 的 `source_hash` 与重算值一致（源码逐字节未改）
   - 父 `step256.pt` 哈希 = `ed09a2c8…`；缓存哈希 = `c91382dc…`
   - `budget_ledger.json.manifest_sha256` 与 `run_manifest.json` 文件哈希一致（ledger() 启动即查，不一致抛 LEDGER_MANIFEST_MISMATCH）
4. **锁与单写入者**：`run.lock`（PID 失效自动回收，Linux psutil 兼容）、`ledger.lock`（120s 过期回收）机制跨平台可用；同时只允许一个 runner 进程
5. **崩溃恢复**：同命令重跑；branch 从 `{label}_latest.pt` 滚动 checkpoint（每 128 步）恢复，RNG/游标/step 完整还原；step JSONL 以 `branch_step` 集合去重，不产生重复曲线
6. **不可恢复判定**：若 parent checkpoint 或缓存哈希校验失败 → 停止并重新上传，禁止跳过校验强行续跑

## 8. 验收清单（服务器收尾完成标准）

- [ ] A1 36/36 done（方案 A）或 平台校准通过（方案 B）
- [ ] `a1_select` 完成，q_series 提取，A2 4 条完成
- [ ] B ≤18 条 + screen 评估 + 路由决策落盘 DECISIONS.md
- [ ] C 配方冻结 → 3 seed × 4096 → **final_holdout 一次性开封**
- [ ] finalize() 产出 findings.json + V16_FINDINGS_AUTO.md + v16_light.zip
- [ ] 终局四态之一 + 五问回答（合同 §8）
- [ ] 训练时间、显存、更新范数、终点 NLL 全部落盘（台账自动）

## 9. 本轮已完成、无需重做

A0（H0 更正、数据封存、台账对账）、Windows 侧 31 条 A1 的原始数据与身份链、全部工程修复（resume driver 预算守卫 + 回归测试）。这些以只读归档形式保留，不参与服务器统计（§6）。
