# ASSET_RESTORATION_GUIDE — 资产恢复指南（R1/R2 前置）

每项：文件名 / 大小 / SHA256（前 8 或指纹）/ 合法获取 / 缺失时行为。

## A. 可从公开上游获取

| 资产 | 大小 | 指纹 | 获取 | 缺失时 |
|---|---:|---|---|---|
| pretrain_t2t_mini.jsonl | 1.24GB | `4403b671` | 上游 minimind 数据发布 | `tools/repro.py check` 报 MISSING；mint 脚本给出精确路径需求 |
| modded-nanogpt（如需 hyperball 复跑） | - | HEAD `4ea6b93` | GitHub clone（MIT） | provenance 已随仓；底座仅 hyperball/NorMuon 需要 |
| FineWeb-10B 分片 | 4.0GB | a1_preflight/data_manifest | 上游数据脚本 | 同上 |

## B. 服务器唯一件（scp 拉取；见 audits BACKUP_AND_UPLOAD_GUIDE 命令模板）

| 资产 | 服务器位置 | 大小 | sha16 | 依赖线 |
|---|---|---:|---|---|
| train_tokens.u32 | …/adamw_spectral_pilot/campaign_v16/cache/ | 33M | `c91382dc` | V16/DV/(30M 前缀) |
| step256 父 ckpt | …/adamw_spectral_pilot/artifacts/checkpoints/seed3407_lr0.003/ | ~18M | `ed09a2c8` | V16 R1 |
| train_tokens_s30m.u32 | …/adamw_direction_scale30m/artifacts/ | 65M | `6fe0bb53` | 30M |
| pilot_train_tokens_33p7m.u32 | …/context_aware_muon/artifacts/ | 128M | `f92715b4` | 100M |
| MSA parent ×3 | …/muon_spectral_allocation/outputs/p0_p1_v1_1/parent/ | 763M | `7563daec`(s512)/`014480c7`(s1536) | MSA/SpecMuon R1 |
| muon_transition m0 | …/muon_transition_experiment/outputs/m0/ | 86M | routing_report 内含 | SpecMuon |
| V16-C 终点 ckpt ×6 | campaign_v16/checkpoints/ | 6×18.6M | endpoint/deletions 台账 | （身份审计件，不可重建） |

## C. 可再生（无需备份拉取）

30M/100M 缓存：语料 + `s30m_mint_cache.py`/`mint_pilot_cache.py`（配方+sha 落档）；V16 缓存：MIGRATION_INFO 铸造配方。SpecMuon M0 域缓存：台账 #8 规则（109s）。

## D. 缺失时的统一行为

`tools/repro.py check` / `portable/run_*.py` dry-run 输出 `MISSING -> restore per docs/ASSET_RESTORATION_GUIDE.md` 并以退出码 1 结束；不存在伪实现/静默降级。**不得**以任何占位数据顶替资产（本轮零例外）。
