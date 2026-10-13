# V14 谱修正必要性判别

状态：**SCALE_RANGE_UNRESOLVED**。父 gate 保持 STOP；本轮不是终点训练结论。

## 固定设计与身份

协议 `ADAMW_A_SPECTRAL_MECHANISM_SCREEN_V14_20261002`；父依赖见 `parent_dependencies.json`；manifest SHA256 `74caaaa5e45b547b82f3cce7c576c82a1ebf754d2f8a2d412a9b3403dcaa6755`。
开发 128 个 V12 `new_probe_prefix` 文档；确认 128 个预先按 doc_id 排序封存的未用 `probe_confirm` 文档。4 个训练窗口只作为固定构造窗口，不作独立 seed。

## 选择与确认原始数字

| 家族 | 冻结候选 | 确认 NLL | 标量减候选 | 配对文档 CI95 |
|---|---|---:|---:|---:|
| scalar | `scalar_g0.5_s2` | 6.095522278 | 0 | 同一标量参照 |
| spectral | `spectral_r0_g0.5_s2` | 6.095519864 | 2.41393436e-06 | [-1.76979993e-05, 2.1620613e-05] |
| fresh_gradient | `fresh_gradient_g0.5_s2` | 6.095412540 | 0.000109737954 | [7.8845507e-05, 0.000142541355] |
| fresh_preconditioned | `fresh_preconditioned_g0.5_s2` | 6.095453417 | 6.88612574e-05 | [4.47833e-05, 9.46404253e-05] |
| ordinary AdamW | `scalar_g1_s1` | 6.095812910 | -0.000290632193 | [-0.0016100871, 0.000938558546] |

## 判读

主比较是冻结最优标量减冻结谱候选；正值才表示谱候选更好。两种无 SVD 控制也使用同一确认文档与窗口。区间只描述文档抽样不确定性，不代表训练 seed 不确定性。
V13 F01/F11 对齐的最大聚合 NLL 偏差 `2.50630636e-08`；局部参考阈值 `1e-5` nats/token，数值分辨率 `2e-6`。
选择开发结果见 `dev_scores.json`；逐文档原始 loss_sum/count 在 `raw/`。本轮无新的 optimizer.step；标签只用自然文档前缀。

## 成本与未测

新前向 token-equivalents `3393740` / 5,000,000；前向 `18944`、反向 `0`、SVD `9`、GPU optimizer.step `0`；计时见 `logs/timing.jsonl`，资源见 `logs/resources.jsonl`。
尚未测跨 seed、完整训练终点、预测器、其他层或注意力；本轮正负均止步。

## 结论边界与来源核对

最佳标量和谱候选都取 `gamma=0.5,s=2`：目标块数据位移系数 `gamma*s=1`，其他参数数据位移为原步的一半，decoupled decay 基点固定。标量控制独立选择了自己的 `gamma,s`。谱相对标量的点差 `2.41393436e-6` 小于预设 `1e-5`，95% 文档配对区间 `[-1.76979993e-5,2.1620613e-5]` 跨零。因此没有可信的相对谱分配增益；且标量最佳配置位于本轮最大 `s=2` 边界，保留 `SCALE_RANGE_UNRESOLVED`，不扩网格。普通 AdamW `gamma=1,s=1` 不能代替这个标量控制。

无 SVD 的当前梯度与当前预条件梯度控制，相对最佳标量分别为 `+1.09737954e-4` 和 `+6.88612574e-5` nats/token，配对区间下界均为正。本轮局部结果不支持谱结构优于这两种控制；它们的确认效应也不构成新优化器或完整训练终点结论。V13 的头部信号衰减仍可作为未来预测假设，但本次没有训练预测器、跨时刻或跨 seed 验证。

直接配对比较（谱 NLL 减无 SVD 方向 NLL）为：当前梯度 `+1.07324019e-4`，95% 区间 `[7.45242525e-5,1.43188103e-4]`；当前预条件梯度 `+6.64473231e-5`，区间 `[4.42107953e-5,9.07145746e-5]`。两者在这批固定确认文档上的局部损失均低于冻结谱候选；逐文档同窗重采样计算保存在 `posthoc_calibration.json`。

后验只读复核见 `posthoc_calibration.json`：开发 `16,384` 条、`2,961,280` token-equivalents，确认 `2,560` 条、`432,460` token-equivalents，原始行与账本严格相等；确认首次资源记录晚于冻结选择。冻结 `M_plus/Vplus` 按 step257 bias correction 重建 `A` 的最大元素误差 `1.2591e-7`，与冻结公式容差相符。4 窗口标量公式相对实测更新最大元素误差见 `direction_w*.json`，V13 F01/F11 的逐文档与聚合 NLL 校准见 `numerical_calibration.jsonl`。

真实 GPU 评估单元时间合计 `379.44s`，SVD `0.449s`；首次预算建立至最后资源记录的经过时间 `593.66s`，包含中断、重启和选择，不代表纯 GPU 运行时间。运行中最低可用 RAM `1,293,320,192` 字节、最高 RSS `1,507,586,048` 字节、最高 CUDA reserved `165,675,008` 字节；均未触发运行硬停。SVD 计数为 9，包含原子写失败后对 window0 的一次重复分解。首次启动低于 1.5 GiB 的阻断和 WinError 5 续跑证据均保存在 `attempts/`，累计前向预算没有清零；父实验历史账本单列于 `findings.json`。

准确解释器命令见 `RUN_COMMANDS.md`。轻量包保留本轮代码、活动 manifest/配置、逐文档 loss_sum/count、依赖哈希、资源与测试日志；不含原始语料、token 视图或大型 checkpoint。
