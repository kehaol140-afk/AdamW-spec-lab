# V1.3 研究准备审计

总状态：**READY_FOR_MECHANISM_STUDY**。父 gate 保持 `BASELINE_TUNING_UNRESOLVED / STOP / NO_LOCAL_SIGNAL`；正式预训练新增 0 步。

## 证据身份与旧结论边界

父 source/config/data/binding、缓存、CSV 与四份 checkpoint 经 `parent_dependencies.json` 严格核验，运行前后受保护文件 SHA 清单见 `protected_before.json`/`protected_after.json`。旧三份 step512 checkpoint 在 32 个 lr_select 文档上的 192 行原始结果可复算；自然标签保持旧 LR 网格排序，但这不回答 step256 的 s=0 处理效应。diagA 与 diagC 使用不同参数遍历顺序，跨进程整体 SHA 不可作为数值变化证据。diagB 的约 8.19e-6 参数差是旧日志事实，单次 AdamW 公式约 2.98e-8 一致；阶段定位见本轮 `compare_*.json`。

## 数值验收

生成模式 `parent_bf16_autocast_else_fp32`，评估模式 `fp32_no_tf32`。2e-6 nats/token 是本轮工程分辨率目标，不是严格误差上界或新科学门槛。
- local: observed_max=0.0, pass=True
- cross_process: observed_max=8.376279314603607e-08, pass=True
- window0_full: observed_max=0.0, pass=True

## 四状态与配对标签

所有 I 按逐文档 `loss_sum/count` 汇总：I_before=F00−F10，I_after=F01−F11，interaction=I_before−I_after。不同窗口是构造 batch，不是独立 seed；旧 probe_confirm 为已使用探索材料。
- w0 model.layers.1.mlp.down_proj.weight legacy: docs=80, tokens=14788, I_before=6.31937569e-05, I_after=-2.26116516e-05, interaction=8.58054085e-05。
- w0 model.layers.1.mlp.down_proj.weight natural_prefix: docs=80, tokens=14788, I_before=6.35652178e-05, I_after=-2.23722657e-05, interaction=8.59374835e-05。
- w0 model.layers.1.mlp.down_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=5.70709529e-05, I_after=-3.53917067e-05, interaction=9.24626595e-05。
- w0 model.layers.1.mlp.down_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=6.02595204e-05, I_after=-2.67528421e-05, interaction=8.70123625e-05。
- w0 model.layers.1.self_attn.o_proj.weight legacy: docs=80, tokens=14788, I_before=4.89296577e-06, I_after=-5.63010932e-05, interaction=6.1194059e-05。
- w0 model.layers.1.self_attn.o_proj.weight natural_prefix: docs=80, tokens=14788, I_before=5.59461417e-06, I_after=-5.58388307e-05, interaction=6.14334449e-05。
- w0 model.layers.1.self_attn.o_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=-8.89801981e-05, I_after=-0.000142239891, interaction=5.32596925e-05。
- w0 model.layers.1.self_attn.o_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=-0.000109526909, I_after=-0.000172509069, interaction=6.29821606e-05。
- w1 model.layers.1.mlp.down_proj.weight legacy: docs=80, tokens=14788, I_before=6.89452101e-05, I_after=-3.64712712e-05, interaction=0.000105416481。
- w1 model.layers.1.mlp.down_proj.weight natural_prefix: docs=80, tokens=14788, I_before=6.90937945e-05, I_after=-3.64630165e-05, interaction=0.000105556811。
- w1 model.layers.1.mlp.down_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=8.84738249e-05, I_after=-2.02042337e-05, interaction=0.000108678059。
- w1 model.layers.1.mlp.down_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=8.33333685e-05, I_after=-1.77459684e-05, interaction=0.000101079337。
- w1 model.layers.1.self_attn.o_proj.weight legacy: docs=80, tokens=14788, I_before=-1.85111358e-06, I_after=-7.0090548e-05, interaction=6.82394344e-05。
- w1 model.layers.1.self_attn.o_proj.weight natural_prefix: docs=80, tokens=14788, I_before=-1.55394484e-06, I_after=-7.00575292e-05, interaction=6.85035844e-05。
- w1 model.layers.1.self_attn.o_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=1.14902038e-05, I_after=-5.45041448e-05, interaction=6.59943486e-05。
- w1 model.layers.1.self_attn.o_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=-1.84952221e-05, I_after=-8.0753192e-05, interaction=6.22579699e-05。
- w2 model.layers.1.mlp.down_proj.weight legacy: docs=80, tokens=14788, I_before=5.78591654e-05, I_after=-7.03340612e-05, interaction=0.000128193227。
- w2 model.layers.1.mlp.down_proj.weight natural_prefix: docs=80, tokens=14788, I_before=5.84947763e-05, I_after=-6.99626003e-05, interaction=0.000128457377。
- w2 model.layers.1.mlp.down_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=6.55418205e-05, I_after=-6.47075077e-05, interaction=0.000130249328。
- w2 model.layers.1.mlp.down_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=6.25112388e-05, I_after=-6.12528092e-05, interaction=0.000123764048。
- w2 model.layers.1.self_attn.o_proj.weight legacy: docs=80, tokens=14788, I_before=-5.74629404e-05, I_after=-0.00013736212, interaction=7.98991799e-05。
- w2 model.layers.1.self_attn.o_proj.weight natural_prefix: docs=80, tokens=14788, I_before=-5.7314356e-05, I_after=-0.000138228862, interaction=8.09145064e-05。
- w2 model.layers.1.self_attn.o_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=-1.25178436e-05, I_after=-9.09787684e-05, interaction=7.84609248e-05。
- w2 model.layers.1.self_attn.o_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=-5.49184495e-05, I_after=-0.000135269313, interaction=8.03508639e-05。
- w3 model.layers.1.mlp.down_proj.weight legacy: docs=80, tokens=14788, I_before=7.96288389e-05, I_after=-1.6428891e-05, interaction=9.60577298e-05。
- w3 model.layers.1.mlp.down_proj.weight natural_prefix: docs=80, tokens=14788, I_before=8.00333186e-05, I_after=-1.60987035e-05, interaction=9.6132022e-05。
- w3 model.layers.1.mlp.down_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=9.53849027e-05, I_after=-4.02406311e-06, interaction=9.94089658e-05。
- w3 model.layers.1.mlp.down_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=9.44178383e-05, I_after=3.10781993e-06, interaction=9.13100184e-05。
- w3 model.layers.1.self_attn.o_proj.weight legacy: docs=80, tokens=14788, I_before=-4.58754234e-05, I_after=-9.9369923e-05, interaction=5.34944996e-05。
- w3 model.layers.1.self_attn.o_proj.weight natural_prefix: docs=80, tokens=14788, I_before=-4.52480672e-05, I_after=-9.94111965e-05, interaction=5.41631293e-05。
- w3 model.layers.1.self_attn.o_proj.weight new_probe_packed: docs=105, tokens=26638, I_before=-0.000138992572, I_after=-0.000175853276, interaction=3.68607045e-05。
- w3 model.layers.1.self_attn.o_proj.weight new_probe_prefix: docs=128, tokens=23135, I_before=-0.000148502611, I_after=-0.000187752688, interaction=3.92500772e-05。

同一文档 legacy/natural 的处理效应差见 `results/summaries/*_paired_labels.json`，配对文档 bootstrap 同时保留 token 权重；不能只比较绝对 NLL。
- 主 FFN/window0 自然−legacy：I_before=3.71460919e-07, I_after=2.39385925e-07, interaction=1.32074994e-07。

## 方向与实际输入能量

方向单位 16/16，几何单位 8/8，有限差分单位 4/4。方向逐文档贡献、A/M_plus/构造梯度与 A 头/剩余的 token 加权统计见 `results/directions/`；F01 真实输入下头/剩余/交叉/联合输出能量见 `results/geometry/`。谱参数能量与真实输入功能能量分别标注。

## 预算、资源与未解决问题

V1.2 原台账：5 次 GPU step 尝试、46104 token-forward-equivalents；diagA 的 3 次无 step 梯度构建另据脚本/日志补列至少 6144，原账本未改。
V1.3 新台账：6/16 次 GPU step 尝试、3439912/5,000,000 token-forward-equivalents；完整原始台账为 `budget_ledger.json`。资源曲线见 `logs/resources.jsonl`；门禁为启动前 1.5 GiB、运行告警 1.5 GiB、硬停 1.0 GiB、RSS 4 GiB 与父显存约束。
packed 105 窗口共享来源并连成 1 簇，仅描述性；不伪造独立文档区间。

保留 baseline_tuning=UNRESOLVED、method_novelty=NOT_ESTABLISHED、end_to_end_gain=NOT_TESTED、deployment_cost=NOT_TESTED。仍未检验跨 seed 终点收益、强 LR 基线或低成本持续修正。

阻断：无本轮工程阻断

运行命令与解释器固定在 `RUN_VERIFIED.ps1`；测试/运行原始日志在 `logs/`。轻量包含活动代码、配置、依赖、原始数值、最终预算和报告，排除原文、token 正文、大 checkpoint。

## 文档级复核与回到主问题

主 FFN/窗口0/新128自然文档：I_before=6.02595204e-05 (95%文档bootstrap [3.477911366640551e-05, 8.645645789602252e-05]), I_after=-2.67528421e-05 ([-5.263191127842237e-05, -1.0665143079569056e-06]), interaction=8.70123625e-05 ([8.113553996882405e-05, 9.331556668248364e-05])。四窗口共用同一 checkpoint，这些区间只对文档重采样，不把窗口当作独立 seed。

旧80文档同一构建的自然−legacy 处理效应差：I_before=3.71460919e-07，I_after=2.39385925e-07；这直接检验标签对处理效应的影响，不能用三份 LR 排序替代。这些差值小于预设的 2e-6 nats/token 工程分辨率目标，标签处理效应差应记为数值上未分辨，不能宣称 EOS 改变或未改变真实机制。

在新文档上，F00 的 <G,A>=0.0341943033，F01 的 <G,A>=-0.0145808283；M_plus 与当前构造梯度的原范数及匹配A范数信号见逐文档原始记录和 summary。窗口0 A 头部 Frobenius 能量份额=0.151293，真实输入输出能量份额约=0.251059。这是局部机制诊断，不是新优化器的终点收益。

同一冻结更新在校准8文档和完整旧视图中的逐文档重复损失差见 `results/same_frozen_repeatability.json`；文档级效应和方向点积区间见 `results/document_bootstrap.json`、`results/direction_document_bootstrap.json`。有限差分三档均原样保存：最小 h=0.0003 受 FP32 损失求和舍入影响明显，不能用该档近零/符号波动推翻已通过的效应层重复读数，也不能把 h=0.003 的较好一致性冒充所有尺度精确。

真实资源样本 17532 条，最低可用 RAM 1.273 GiB，最高 RSS 1.966 GiB，峰值 CUDA reserved 0.254 GiB；告警 1817 次，阻断 0 次。Windows 换页未测量。按 PID 的采样时间跨度在 `results/resource_timing_summary.json`。

本轮已能可信地继续研究局部“可利用余地”：目标块单独更新与其他参数先更新后的边际效应在固定构造窗口/文档上可区分。关于可预测条件和低成本修正，仍需未来独立 seed、强 LR 基线、完整训练终点与成本验证；本轮没有启动这些工作。

补丁 `patches/attempt2_atomic_resume.patch` 与专门回归 `logs/patch_regression_tests.log`（2/2 通过）证明 Windows 短暂共享冲突重试及累计台账续接。`budget_ledger.json` 的 `elapsed_seconds=0` 为旧字段未填充，不表示实际耗时为零；按 PID 可核对的采样跨度另见时间汇总。最终分析脚本及其哈希见 `analysis_manifest.json`。

## 实际运行命令与 attempt 记录

绑定解释器为 Python 3.9.21。以下命令按顺序串行执行；`calibrate-cross` 是独立新进程。初次 `measure` 在子账本原子替换时遇到 WinError 5，失败前已产生的结果、源码和预算快照保存在 `attempts/attempt1_before_atomic_retry/`；`prepare` 生成 attempt2 后续跑，累计预算未归零。

```powershell
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
$env:USE_TF = '0'
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.cli prepare
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m unittest readiness_audit_v13.tests_v13 -v
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.run_parent_tests_isolated
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m unittest replay_audit_v12.test_v12 -v
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.cli calibrate-local
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.cli calibrate-cross
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.cli measure
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.cli finalize
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m unittest readiness_audit_v13.tests_patch -v
& 'D:\pytorch\anaconda\envs\py39\python.exe' -m readiness_audit_v13.postprocess
```

完整新运行无需重复执行，复核产物时只运行只读校验。`RUN_VERIFIED.ps1` 是本轮冻结测量入口；最后的文档级统计与封包由上面列出的 `postprocess` 命令执行。轻量包已重新打开，196 个 V1.3 原始损失单元逐字节与磁盘文件相等，V1.2 LR 原始结果为 192 行，`budget_ledger.json` 在包内。

本轮规范逐张量对比把差异首次定位在 `raw` 梯度阶段：同进程重复构建有 1 个梯度元素不同、更新后参数 0 个元素不同；跨进程对比有 665 个原始梯度元素不同、更新后参数最大绝对差 `6.668269634246826e-7`。同一冻结构建的校准8文档和完整视图重复评估逐文档 `loss_sum` 最大差均为 0；同/跨进程的效应层最大读数差分别为 0 和 `8.376279314603607e-8 nats/token`。这些事实没有定位到特定 GPU 内核，也没有重现旧 diagB 所报的 `8.191913366317749e-6` 参数差；旧值保留为历史事实，机制记 `NOT_IDENTIFIED`。
