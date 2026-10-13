# REPRODUCE — context-aware-muon-100m

## 环境

torch 2.11.0+CUDA + bf16 autocast（训练）；CPU 复算仅标准库。常量：warmup128+cosine→0.1×peak、wd0.02、无 clip、batch 8192 tok/步。

## 入口

- 训练：`python scripts/run_pilot_driver.py`（真实入口；逐臂 `scripts/run_pilot_arm.py`。**勘误**：本文旧版写 `scripts/driver_pilot.py`，该文件不存在——成功链入口为 `run_pilot_driver.py`（`results/pilot_driver.log`），并行会话的超时 driver 为 `scripts/pilot_driver.py`，见 PILOT_100M_RUNTIME.md:13。需 GPU+缓存；断点续跑按 run.lock.json）
- 只读复算：`python3 recompute/run_recompute.py`（CK CAM_100M）

## 只读复算预期

| 量 | 预期 | 精度 |
|---|---|---|
| 15 臂 H4096（从 eval_docs jsonl） | 与 CSV **0 偏差** | 精确 |
| C_A_lam0p85 − B_muon_rms_lr1e-3 | +0.1367 | 精确 |
| C_B − C_A | +0.0997 | 精确 |
| 严格单调 H1024→H4096 | 14/15（C_A_lam0p85 除外） | 精确 |

（H1024/H2048 无逐文档文件——以冻结 CSV 为准，见 RESULTS R2 注。）

## 训练重跑资产（未随仓）

| 资产 | 身份 | 位置 |
|---|---|---|
| pilot_train_tokens_33p7m.u32（128M） | sha `f92715b4…`（cache_manifest；V16+s30m 前缀字节证明） | 服务器 artifacts/；v1.1 增补臂若获批将复用 |
| 15×422M endpoint_weights | init_fp 33f93a0f…（15 臂全同）/final_fp 逐臂（CSV 列） | 服务器；seed 3471+缓存可确定性重建 |
| 上游语料（同 30M 线） | fingerprint 4403b671… | 服务器/上游 |

## 不可复现项

无硬性不可复现项（判据全部可由随仓数据复算；训练可确定性重跑）。**不存在**的对照：确认层/多 seed/更大 tokens/NorMuon GPU 消融——协议决定不执行，勿引用为证据。封存集：本实验无独立 holdout（dev 级协议）。

## 复现能力分级（R0/R1/R2）

| 级 | 状态 | 条件 |
|---|---|---|
| R0 | **PASS** | CK CAM_100M（H4096 与 CSV 0 偏差） |
| R1 | **BLOCKED_ASSET** | pilot 缓存(f92715b4)；`portable/run_context_pilot.py` |
| R2 | **BLOCKED_ASSET** | 语料→mint 33.7M→GPU 4.55h；确认层协议内不开 |

seed 3471；pilot_arms_v1.json sha 8149bcbb。
