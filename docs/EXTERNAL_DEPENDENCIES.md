# EXTERNAL_DEPENDENCIES — 外部依赖与恢复

## 1. 第三方代码（来源/版本/许可/获取）

| 依赖 | 用途 | 版本锚点 | 许可 | 合法获取 |
|---|---|---|---|---|
| minimind（框架） | 模型定义 `model_minimind.py`+tokenizer（30M/100M/4.66M 全线） | 服务器副本 50 文件（audits/minimind_base_file_manifest.sha256 逐条） | 上游开源（LICENSE 随 baselines/minimind-base-code） | 上游 minimind 仓库；本仓 baselines/ 已含源码副本 |
| modded-nanogpt | Hyperball A1 底座 + NorMuon staging 参考 | HEAD `4ea6b937`（MIT, Keller Jordan 2024） | MIT | `git clone https://github.com/KellerJordan/modded-nanogpt && git checkout 4ea6b937`；本仓仅 provenance+3 PR patch；离线全史=服务器 modded-nanogpt-full.bundle（87M，`git bundle verify` PASS） |
| torch `_muon.py` | Muon 本体（NS5） | SHA `38e95918…`（冻结快照在各线 vendored/vendor_snapshot） | PyTorch BSD-style | `torch==2.11.0` 内置；快照 sha 已锁 |
| 旧仓 training_dynamics_lab | MSA/SpecMuon 的 muon_transition 资产、五线历史 | HEAD `cf1558f`（public） | 自有 | `git clone https://github.com/kehaol140-afk/training_dynamics_lab`（只读） |

## 2. SpecMuon 的跨仓依赖链（恢复指令）

SpecMuon P1 重放需要：① `muon_transition_experiment`（服务器目录，含 m0/routing_report.json 与 m1 配方）——**不在新仓**；② `muon_spectral_allocation` 的 parent 状态（服务器 11.65G）；③ minimind_base（本仓 baselines/）。恢复顺序：
1. clone 旧仓（上表）获得五线小证据（不含大 checkpoint）
2. 从服务器拷贝 `muon_transition_experiment/outputs/m0/`（86M）与 `muon_spectral_allocation/outputs/p0_p1_v1_1/parent/`（763M，SHA 见 AH_ASSET_INDEX）
3. 配置 `configs/paths.json` 的 `MUON_TRANSITION_ROOT`/`CKPT_ROOT` → `python3 tools/repro.py check --experiment specmuon-pilot`

## 3. 外部大数据

| 数据 | 文件名 | 大小 | SHA256（前 8/指纹） | 获取 |
|---|---|---:|---|---|
| 预训练语料 | pretrain_t2t_mini.jsonl | 1.24GB | `4403b671` | 上游 minimind 数据发布（公开） |
| FineWeb-10B | fineweb_train_*.bin ×20+val | 4.0GB | data_manifest（hyperball a1_preflight） | 上stream modded-nanogpt 数据脚本（公开） |
| V16 token 缓存 | train_tokens.u32 | 33M | `c91382dc` | 服务器；可由语料+tokenizer 按 MIGRATION_INFO 配方铸 |
| 30M 缓存 | train_tokens_s30m.u32 | 65M | `6fe0bb53`（前 8.5M=V16 前缀） | 服务器；`s30m_mint_cache.py` 可再生 |
| 100M 缓存 | pilot_train_tokens_33p7m.u32 | 128M | `f92715b4` | 服务器；`mint_pilot_cache.py` 可再生 |

## 4. 封存评估集（自动流程永不加载）

`holdout_v2.jsonl`（1024 docs）+ 候选/保留 id 清单：仅公开大小/条数/承诺哈希（docs/SEALED_ASSET_EXCLUSIONS.csv）。`tools/repro.py` 与全部 portable 包装器不含任何加载该集的代码路径。

## 5. Python 包

R0：标准库零依赖。R1/R2：`env/requirements-training.txt`（torch==2.11.0 / numpy==2.4.6 / psutil==6.1.1，服务器实测版本）。
