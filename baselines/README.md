# baselines

## minimind-base-code/

vendored minimind 原生训练框架源码（29 文件，0.57M）。来源：服务器 `training_dynamics_lab/minimind_base/`（只读副本）；逐文件 SHA256 清单见 audits 侧 `minimind_base_file_manifest.sha256`（50 条，含未随仓的数据/文档）。**不含** 1.2G 数据集 `pretrain_t2t_mini.jsonl`（上游 minimind 项目可获取）。用途：30M/100M 实验的模型定义 `model/model_minimind.py` 与训练骨架基线（原生 AdamW lr5e-4/cosine/AMP 约定）。

## modded-nanogpt-provenance/

第三方底座的溯源件（**不整仓复制**）：上游 https://github.com/KellerJordan/modded-nanogpt ，HEAD pin `4ea6b937337a4889b8cfe3f38a93d120048d8f71`，MIT License (c) 2024 Keller Jordan。patches/ 内为我们实际使用的 3 个公开 PR diff（PR267 MuonH / PR323 Tuned Muon / PR324 TunedAux MuonH）+ track3 README + 上游源码快照单文件 + SHA256SUMS（提取自服务器 hyperball_sources.zip；该 zip 内论文 PDF 不随仓发布）。完整离线历史：服务器 `modded-nanogpt-full.bundle`（87M，git bundle verify PASS）。

## hyperball-a1-evidence/

Hyperball A1 复现（124M GPT-2-class，FineWeb-10B，3250 步，单 seed）小证据：双臂 stdout 全程日志、逐 step 诊断（H 臂 total_muon_norm 恒 261.928 vs M 臂 145.71→535.96）、三套冻结配方 json、数据 manifest、来源清点。裁决：tuned Muon 3.27906 vs tuned-aux MuonH 3.27921（Δ+1.5e-4 种子噪声内）——范数保持修正无终点收益。2×1.28G checkpoint 与 4G 数据留服务器（docs/STORAGE_AND_BACKUP.md）。
