# REPRODUCE — adamw-spectral-v13-v16

## 环境（服务器实测）

- Python 3.11（conda `fr_minimind`），torch 2.11.0 + CUDA（训练需要）；numpy/psutil
- CPU 复算：Python ≥3.9 标准库即可（本仓 `recompute/` 不依赖 torch）

## 原始训练入口

`python -m campaign_v16.run`（需 GPU + token 缓存 + eval_sets；本仓不含缓存→训练不可直接重跑，见下）

## 只读结果复算（本仓直接可跑）

```bash
cd <repo-root> && python3 recompute/run_recompute.py
```

| 量 | 预期 | 精度 |
|---|---|---|
| C 三 seed S−F | +0.036276/+0.094018/+0.151494，mean +0.093929 | 精确（无随机性） |
| A2 四臂 Δ | −0.002658/−0.002567/+0.002937/… | ≤3e-6（判读文档舍入） |
| B dev H2048 | P1 4.087912 / F1 4.118871 / S1 4.240565（18 全表见 RESULTS R6） | 精确 |
| CI95 | 与冻结例程差 ≤1.5e-3 | 程序差异（本仓 numpy default_rng 等价实现），非 mismatch |

## 训练复现所需额外资产（未随仓）

| 资产 | 身份哈希 | 位置 |
|---|---|---|
| token 缓存 train_tokens.u32（33M） | sha `c91382dc…`（frozen_v1.json:26 声明、本审计实算一致） | 服务器 campaign_v16/cache/；可按 MIGRATION_INFO 配方从上游语料重建 |
| B/C endpoint checkpoints（12×18.6M） | deletions.jsonl + endpoint 逐个 sha | C 终点 6 个留服务器（holdout 身份锁定，**不可重建**）；B 扩展 6 个可重跑再生 |
| 父 step256 checkpoint | `ed09a2c8…`（V16_A2_DECISION.md:3） | 服务器 artifacts/ |
| eval_sets.json | 随仓（sha 入 data_manifest 链） | 本目录 |

## 评估集身份

development 256 / screen 256 / final_holdout 1024（doc_id=SHA256；三 split 已烧毁）。**不可复现项**：final_holdout 已开封——任何后续实验需新评价集（方向线已铸 holdout_v2，封存未开，见 docs/SEALED_ASSET_EXCLUSIONS.csv）。

## 结论分级声明

「结果可复算」= 本仓 CPU 即可复核全部发布数值；「训练可重跑」= 需上表资产 + GPU（A1/A2/B 可确定性重建，C 的 holdout 身份件除外）。

## 复现能力分级（R0/R1/R2）

| 级 | 状态 | 条件 |
|---|---|---|
| R0 CPU 结果复算 | **PASS** | `python3 recompute/run_recompute.py`（CK V16C/V16_A2，点估计精确） |
| R1 parent ckpt 重放 | **BLOCKED_ASSET** | 需 train_tokens.u32(c91382dc)+父 step256 ckpt(ed09a2c8)+GPU；便携入口 `portable/run_v16_campaign.py`（dry-run） |
| R2 从头训练 | **BLOCKED_ASSET** | 需 token 缓存（MIGRATION_INFO 配方铸造）+GPU；C 终点=holdout 身份锁定件不可再生 |

种子/超参：A/B seed 3407、C seeds 3408-3410；配置哈希见 CODE_MAP；torch 2.11.0。历史层2/T2 未执行——本线指 V16 战役自身（Layer-2 属 direction 线）。
