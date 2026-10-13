# 100M_REPLAY_AND_SMOKE — 受限 100 步训练诊断（任务三）

日期：2026-10-12。脚本 `scripts/preflight_100steps.py`；结果 `results/preflight_100steps.json`。

---

## 1. 执行前置条件核验（逐项）

| 条件 | 状态 | 证据 |
|---|---|---|
| GPU0 空闲 | ✅ | nvidia-smi：GPU0 441 MiB（桌面残留）/0% util；GPU1 27.8 GiB 外部占用（本项目不触碰） |
| 共享资源获准 | ✅ | 本任务书授权 100 步诊断、上限 0.3 GPU-h；未启动 15-run 矩阵 |
| 数据身份一致 | ✅ | 正式 pilot 缓存（`pilot_train_tokens_33p7m.u32`，sha `f92715b4…`）；V16 8.5M 前缀 + s30m 17M 前缀字节校验通过；corpus `6dd6716c…`/tokenizer `71f32c68…` sha 链完整（`artifacts/pilot_cache_manifest.json`） |
| 初始化可复现 | ✅ | 四路径 init fingerprint 全部 `33f93a0f546149a0`（= smoke 同值，跨日可复现） |

## 2. 四路径 100 步实测（seed 3471，micro8×accum4×seq256，bf16，诊断用 warmup8+cosine T=100，peak 1e-2）

| 路径 | 稳态 s/步* | tok/s | VRAM 峰 | opt s/步 | NS5 native/mix s/步 | loss 全有限 | 范数匹配 err |
|---|---|---|---|---|---|---|---|
| muon_native | 0.251 | 28.4k | 3.77 GiB | 0.050 | 0.041 / — | ✅ 100/100 | — |
| group_A λ=0.65 | 0.309 | 25.9k | 3.80 GiB | 0.105 | 0.020 / 0.033 | ✅ | **1.87e-16** |
| group_A λ=0.85 | 0.311 | 26.4k | 3.80 GiB | 0.106 | 0.020 / 0.034 | ✅ | 1.86e-16 |
| lowmu μ=0.9 | 0.238 | 34.3k | 3.82 GiB | 0.041 | 0.035 / — | ✅ | — |

\* 稳态 = 去掉前 10 步的逐步均值（首 10 步含编译/分配：native 0.534→0.251）。F-1 的"100 步复测吞吐"完成：**native 28.4k / mix 25.9–26.4k / lowmu 34.3k tok/s**（smoke 的 23.8–29.1k 口径更新；旧 30 步均值作废）。

- 训练 loss：8.95 → ~6.2–6.45（各路径一致下降；首 3 步四路径位级近似同值 → CRN 数据流一致）。
- dev NLL（V16 development 256 docs，token 加权，step 100，**诊断值，不作任何方法结论**）：native 6.368 / λ=0.65 6.354 / λ=0.85 6.377 / lowmu 6.330。
- 方向诊断（fp64，全程均值）：cos(O_mix,g) λ=0.65→0.182、λ=0.85→0.189；cos(O_mix,O_nat) λ=0.65→**0.516**、λ=0.85→**0.385**（λ 越大离 native 越远，单调符合预期）。

## 3. 模型恢复重放（checkpoint @step50 → 新对象恢复 → 重放 50..99 → 对比）

| 路径 | 位级一致 | max|Δθ| | cursor 恢复 | 重放 loss 末3 |
|---|---|---|---|---|
| muon_native | ✅ **true** | **0.0** | ✅ | 与主行逐位相同 (6.2805, 6.3118, 6.4491) |
| group_A λ=0.65 | ✅ **true** | **0.0** | ✅ | 与主行逐位相同 (6.2851, 6.2889, 6.4299) |

**结论：model + Muon 动量 buffer + aux AdamW m/v + cursor 的完整训练态恢复在本机（RTX 5090 + flash SDPA + bf16 autocast）位级可重放**——pilot 的 CRN 配对与断点续跑能力成立。检查点文件验证后即删（指纹与哈希留档 JSON）。

## 4. 资源消耗

- GPU 墙钟：4 路径主行 115.6 s + 2 次重放（各 ~50 步 + 重建 + 恢复）≈ **总计 ~4 分钟 ≈ 0.066 GPU-h ≪ 0.3 h 上限**。
- 磁盘新增：pilot 缓存 134.7 MB + 结果 JSON（检查点已删）。

## 5. 遗留观察（不构成 gate）

1. 末段 loss 轻微上翘（6.28→6.45，cosine 尾段 + 100 步瞬态；诊断 schedule 与 pilot 不同，pilot T=4096 不受影响）。
2. mix 路径 opt 开销 0.105 s/步（含每步 fp64 cos）；冻结版 diag_every=64 后锚点外步骤省去 cos 计算，预计 ~0.06 s/步（范数匹配的 .double() 必留）。
3. lowmu 34.3k tok/s 高于 native（0.9 动量的 NS5 输入更稀疏？更可能是测量波动；pilot 会给出 4096 步口径）。
