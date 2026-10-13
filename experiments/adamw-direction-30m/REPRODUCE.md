# REPRODUCE — adamw-direction-30m

## 环境

torch 2.11.0+CUDA（训练）；CPU 复算仅标准库。模型常量/数据通道见 `src/s30m_core.py` 与 `30M_ENV_AND_MODEL_AUDIT.md`。

## 入口

- 训练：`python src/s30m_runner.py`（5 阶段：mint→smoke→13 runs→analyze；需 GPU+缓存+上游语料）
- 只读复算：`python3 recompute/run_recompute.py`（CK S30M_T1）

## 只读复算预期

| 量 | 预期 | 精度 |
|---|---|---|
| F2−S2@L1 / P2−S2@L1 / P2−S1 | −0.037930 / −0.045737 / −0.013957 | 精确 |
| Muon 最优−P2 / −S1 | −0.272500 / −0.286457 | 精确 |
| 13 臂 H2048 | 见 RESULTS R2 全表 | 精确 |
| CI95 | 冻结例程差 ≤1.5e-3 | 程序差异 |

## 训练重跑资产（未随仓）

| 资产 | 身份 | 位置 |
|---|---|---|
| train_tokens_s30m.u32（65M） | sha `6fe0bb53…`（cache_manifest） | 服务器 artifacts/；可由 mint 配方确定性重建（raw/mint.out 全录） |
| 上游语料 pretrain_t2t_mini.jsonl（1.24G） | fingerprint `4403b671…`（NEW_EVAL_SPLIT_MANIFEST GAP_1） | 服务器/上游 minimind |
| 13×115M checkpoints | endpoint_hashes.json 逐 sha | 服务器（可确定性重建：seed 3421+缓存，CRN 位级已证） |
| V16 tokenizer/eval_sets | 随 ../adamw-spectral-v13-v16 | 本仓 |

## 不可复现项

无硬性不可复现项（本实验判据全部可由随仓数据复算；训练可确定性重跑）。T2/holdout 未运行——不存在对应数据。100M 外推未执行。

## 复现能力分级（R0/R1/R2）

| 级 | 状态 | 条件 |
|---|---|---|
| R0 | **PASS** | CK S30M_T1 |
| R1 | **BLOCKED_ASSET** | s30m 缓存(6fe0bb53)+V16 前缀；`portable/run_s30m_t1.py` |
| R2 | **BLOCKED_ASSET** | 语料(4403b671)→mint→GPU；T2 未执行 |

seed 3421；frozen_30m_t1.json sha 01561498。
