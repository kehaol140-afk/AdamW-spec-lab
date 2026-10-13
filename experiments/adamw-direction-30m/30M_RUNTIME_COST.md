# 30M_RUNTIME_COST — 运行成本实录

日期：2026-10-12。硬件：GPU0（RTX 5090 32G，独占使用；GPU1 vLLM 全程未触碰）。

## 1. 实测（smoke 标定 + T1 全 13 runs）

| 项 | 实测 |
|---|---|
| Smoke 标定（50 步 × 2 路径） | 17.8 s（AdamW 10.2 s + Muon 7.6 s）≈ 0.005 h |
| T1 训练墙钟（Σ 13 runs step-log wall_cum） | **4,488 s = 1.247 h** |
| T1 评估（13 × 3 视界 × 256 docs） | ≈ 10 min（含在 run 墙钟外，eval 分列） |
| **T1 合计 GPU** | **≈ 1.45 h ≤ 2.0 h 授权窗口**（余量 27%） |
| 吞吐 | AdamW 族 39.9–44.6k tok/s；Muon 53.9k tok/s（0.152–0.205 s/步） |
| VRAM 峰值 | 1.27 GiB（smoke 实测 max_memory_allocated） |
| RSS | ~3 GiB |
| 磁盘 | 13 端点 × ~120 MB ≈ 1.56 GB + 步日志 13×~1.4MB + eval ~5MB（148G 空闲内） |
| 数据 | 每 run 16,777,216 tokens，游标 0 CRN 同流；缓存 17M tokens（68MB） |

## 2. 与冻结预算对照

| 项 | 冻结投影 | 实测 | 偏差 |
|---|---|---|---|
| 训练 | 85.4 min | 74.8 min | −12%（5090 实跑快于 smoke 保守值） |
| 评估 | ~13 min | ~10 min | −23% |
| 合计 | 1.64 h | ~1.45 h | −12%，无超支，未触发 shrink rule（+20% 阈） |

## 3. 每臂分解（训练墙钟 / 优化器步累计）

AdamW 族：316–372 s/run，其中 apply_step 优化器+校正 16.1–18.8 s（~5%）；Muon：316–332 s/run，Muon+aux 步 54–59 s（~18%，与 M 线 15.6% 实测同量级）。

## 4. 后续（未执行，仅供授权参考；非本轮授权）

- T2 确认（4 臂 × 3 seeds × 4096 步 = 33.6M tokens/run）：投影 12 × ~2.6 倍 T1 单臂 ≈ **3.1–3.5 h**（同批形状线性外推，smoke 级精度）+ holdout 方案决策 + 语料缓存扩铸（T2 需 ≥ 4×33.6M 独立流）
- 100M（MODEL_100M_READINESS 缺口未闭 + G3 未过）：未立项、未预算
