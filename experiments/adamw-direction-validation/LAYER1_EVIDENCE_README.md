# LAYER1_EVIDENCE_ZIP — 证据包说明与复算指南

协议：ADAMW_DIRECTION_VALIDATION_V1_20261011（configs/frozen_v1.json，PREREGISTERED_FROZEN）
执行日期：2026-10-11。生成者：layer1 证据打包步骤（见 SHA256SUMS.txt）。

## 内容清单

- `eval/` — 原始评价表：5 臂 × H512/H1024/H2048 × development（15 个 jsonl，逐文档 loss_sum/token_count + 身份字段）与 5 个 screen 次级表
- `steps/` — 逐步训练日志（5 臂 × 2048 行：eta/loss_sum/nll/A_norm/D_norm/q_t/opt_step_seconds/zero_norm_fallback）
- `anchors.jsonl` — 锚点观测（204/512/1024/1536 步 × 3 矩阵：g_norm/A_norm/cos_g_A/cos_g_m/sigma1/head_energy_fraction；含确定性重放产生的字节级相同重复行，分析端已去重校验）
- `run_meta_layer1.json` — 每臂墙钟/token/参数指纹
- `endpoint_hashes.json` — 5 个端点 checkpoint 的 sha256 与 param_fingerprint（checkpoint 本体不入包，~18MB/个）
- `configs/` — frozen_v1.json、holdout_v2_manifest.json（状态 MINTED_SEALED，未开封）
- `src/` — dv_core.py（数学内核，哈希 b8943249… 与冻结一致）、dv_runner.py（含 RUNNER_FIX_LOG.md 记录的一行观测层修复）、layer1_analyze.py、layer1_screen_eval.py、tests/test_dv_cpu.py
- `layer1_gate.json`、`raw/layer1_analysis.json`、`raw/layer1_screen_secondary.json` — 判定与全量分析输出
- `SHA256SUMS.txt` — 全部源码/协议/交付文件哈希账本（追加式）

## 复算命令（在 adamw_direction_validation 项目目录、fr_minimind 环境）

```bash
# 1) 从本包恢复 raw/ 后重算全部统计与门判定（CPU 即可）：
python src/layer1_analyze.py        # 重写 raw/layer1_analysis.json 与 layer1_gate.json
# 2) 复刻门核验（dv_S1/dv_F8/dv_P8 vs V16 b_*，需 V16 目录在位，只读）：
#    layer1_analyze.py::replication_gate 已含，输出于 layer1_analysis.json.replication_gate
# 3) screen 次级评估复算（需 GPU + raw/checkpoints/*.pt，本体不在本包）：
python src/layer1_screen_eval.py
```

## 独立锚点

- V16 源码链：source_hash 4afaa78e…（campaign_v16/run_manifest.json，重算一致）
- 训练缓存：c91382dc…（8.5M uint32）
- holdout_v2：3e0f7653…（MINTED_SEALED，本层未开封）
- 复刻门：3/3 臂 2048 步 × 9 字段位级一致
