# V1.3 研究准备审计执行计划

协议：`ADAMW_A_SPECTRAL_READINESS_V13_20261002`。父实验、MiniMind、`replay_audit_v12/` 全部只读；新产物只写本目录。正式预训练新增步数 0，隔离 GPU `optimizer.step` 尝试上限 16，新增累计 token-forward-equivalents 上限 5,000,000。

## 已找到的输入

- 绑定解释器 `D:\pytorch\anaconda\envs\py39\python.exe`，Python 3.9.21；本地 RTX 4060 Laptop，8,188 MiB 显存，检查时空闲 6,830 MiB。绑定指向已有 MiniMind、tokenizer 和 JSONL，不下载。
- 父低内存配置、data manifest、gate、三条 step512 checkpoint 与主 step256 checkpoint 均存在。V1.2 的 parent/evaluation manifest、旧标签语义审计、128 新文档、四构造窗口均存在。
- `replay_audit_v12/results/lr_reeval.json` 与 192 行逐文档损失、旧预算台账、资源日志、diagA–D 脚本与输出均存在。V1.2 四状态行数为 0；已有报告见 `reports/V12_FINDINGS.md` 等。
- `v12_independent_review/REVIEW.md` 和 `evidence/SOURCE_EVIDENCE.md` 未在本地项目找到；以任务书疑点和原始产物独立核查，不据缺失材料补造结论。

## 执行次序

1. 建立受保护依赖哈希清单并复核父/V1.2 指纹、checkpoint、原始 CSV、日志与历史预算；制作 `claim_evidence.csv`，裁正未由数据支持的解释。
2. 在本目录复制必要的小型 V1.2 代码并形成补丁：统一逻辑名规范摘要；逐阶段张量/状态比较；清楚区分来源、单次公式、重复构建与损失分辨率；独立台账全前向/反向/尝试计数；逐单元状态、幂等续跑与完整轻量包。
3. 先跑规范摘要、四状态/None/tied、完整恢复、mask、标签配对、预算/续接/打包 CPU 回归，并隔离输出运行旧测试。每次代码变更生成新源码哈希和 attempt，保留原 attempt。
4. 固定 FFN/窗口0/旧 probe_select 前8文档校准：保留原 BF16 生成条件，先定位状态与逐阶段差异；对同一冻结更新反复评估，再对重复构建效应和新进程进行比较。按预设 2e-6 nats/token 工程分辨率判断可否进入全量；有限差分固定多尺度。
5. 若来源、资源与数值可信，自动运行 manifest 固定的4窗口×2目标、配对 legacy/natural、128 新文自然前缀和描述性 packed 视图；输出 F00/F10/F01/F11、方向及实际输入能量。若触发阻断，停止并保留部分原始结果与准确预算。
6. 复核父文件未变、对账新旧预算、重开 ZIP 检查条目与行数；生成 `READINESS_AUDIT.md`、`readiness.json`、`RUN_VERIFIED.ps1`、`readiness_v13_light.zip`。

不会启动父 prepare/discover/confirm，不扩展 LR/rank/rho，不按测量结果挑文档、窗口或精度。
