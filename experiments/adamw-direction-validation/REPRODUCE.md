# REPRODUCE — adamw-direction-validation



环境：同 V16（torch 2.11.0 训练；CPU 复算仅标准库）。训练入口：`python src/dv_runner.py`（需 GPU + V16 缓存 + pilot_core → 依赖 ../adamw-spectral-v13-v16/ 与其未随仓的 cache/train_tokens.u32）。

只读复算：`python3 recompute/run_recompute.py`（CK DV_LAYER1）：五臂 dev H2048 精确（4.240565/4.280268/4.322962/4.118871/4.087912）；F8−S8=−0.204090、P8−S8=−0.235050 点估计精确；CI 端点差 ≤1.5e-3（bootstrap 程序等价差异）。

资产身份：eval 集=V16 eval_sets.json（随 spectral 仓）；holdout_v2 封存未开封（docs/SEALED_ASSET_EXCLUSIONS.csv，承诺哈希在列）；checkpoints 5×18.6M 留服务器（endpoint_hashes.json：逐 sha+param_fingerprint）。**不可复现项**：无（本实验全部判据可由随仓数据复算）；**训练可重跑**：是（seed 3407 + V16 缓存确定性重建，复刻门位级已证）。layer-2 未运行。

## 复现能力分级（R0/R1/R2）

| 级 | 状态 | 条件 |
|---|---|---|
| R0 | **PASS** | CK DV_LAYER1 点估计精确 |
| R1 parent ckpt 重放 | **BLOCKED_ASSET** | V16 token 缓存只读依赖（V16_ROOT）；`portable/run_dv_layer1.py` |
| R2 从头训练 | **BLOCKED_ASSET** | 同上+GPU；Layer-2（seed 3411-3413）**从未执行**，无对应数据 |

seed 3407；frozen_v1.json sha f8cc6b8e。
