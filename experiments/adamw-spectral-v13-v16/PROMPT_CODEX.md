# 给 Codex 的实施任务

你负责实现和测试一个边界明确的本地研究 pilot，不负责扩大研究范围或宣称实验成功。用户已同意先进行小规模验证。请先完整读取同目录 `PROTOCOL_LOCAL_V1.md`、`config/local_v1.json`、`RUN_WINDOWS.md`；按其合同实现，不要只返回建议或另写一份空计划。

工作目录是 `E:\大模型预训练\adamw_spectral_pilot`。用户在 `E:\大模型预训练` 下已有 MiniMind GitHub 克隆与可能的数据；当前服务器关闭，本机最多约 15GB 可用显存且内存有限。原仓库与旧实验只读，所有新代码、缓存、输出放本目录。不要修改旧优化器、旧实验或 MiniMind 全局源码，不自动 git pull、装新环境、升级依赖或下载模型/数据。优先适配当前可用 Python 环境与本地源码。不要假设默认 MiniMind 大小、权重名或 tokenizer 版本。

目标只到 P0 数值测试、真实数据 smoke 和可由 OpenCode 执行的 local discover/confirm。不要实现完整新优化器、在线门控、服务器调度、全模型 Hessian 或复杂仪表盘。

## 执行顺序

1. 实际检查本地仓库、现有 Python、torch/CUDA、tokenizer、text 预训练数据和资源，生成 `bindings.local.json`、`artifacts/preflight.json`。检查可自行确定的路径和 API，不要在未检查前追问用户。不确定或缺失就输出明确 BLOCKED 和已发现候选，不假造文件、不抓取远程替代数据。
2. 先写最小测试与真实 AdamW 对齐，再实现。数值参考优先用本机 torch AdamW；probe 路径必须与 baseline 的实际下一步对应。核心要防止 sum/mean、t/t+1、weight decay 和共享参数重复更新错误。
3. 实现小型数据 memmap、MiniMind 适配、完整 checkpoint、流式统计、SVD 诊断、冻结系数选择、局部确认和门禁。模块数保持紧凑，不为每个实验建一套训练器。
4. 运行 CPU 单测和 8 步真实数据 smoke，并检验连续训练/恢复重放。最多 20 个 GPU optimizer steps 用于 smoke/重放，禁止运行三条 512 步发现训练；正式试验交给 OpenCode。
5. 输出 `reports/CODEX_HANDOFF.md`：真实运行命令及退出码、测试误差/结果、依赖与路径、显存/RAM、未完成项、源码/协议 hash。写出本机解释器绝对路径版本的 `RUN_COMMANDS_RESOLVED.ps1`，供用户和 OpenCode审核后执行；不要自动执行整份正式训练脚本。

## 必须提供的统一 CLI

以下是你需要创建的接口，不是 MiniMind 原生命令。每个子命令支持 `--config config/local_v1.json`，幂等性依赖完整 manifest/hash，不可只凭目录存在跳过。

- `python run.py preflight --config config/local_v1.json --root "E:\大模型预训练"`
- `python run.py prepare --config config/local_v1.json`
- `python -m unittest discover -s tests -v`
- `python run.py smoke --config config/local_v1.json`
- `python run.py pilot --config config/local_v1.json --stage discover`
- `python run.py pilot --config config/local_v1.json --stage confirm`
- `python run.py report --config config/local_v1.json`
- `python run.py bundle --config config/local_v1.json --light`

`confirm` 读取并验证 discover gate 和 frozen candidate，不通过则拒绝训练。`report` 可在失败/中断时执行，必须标识部分结果。`bundle --light` 只打包代码、配置、绑定（脱敏可迁移副本）、原始小表、数值测试、manifest与报告，不默认打包原语料、大 checkpoint、密钥、环境目录或整仓库。

## 不得偏离的关键要求

- 方向是实际预条件更新 A，而不是原始 G 或仅 M；候选是 `s*(rho*A_H+A_R)`，主谱划分固定 rank1。
- 目标矩阵外所有参数执行相同普通 AdamW 下一步。目标矩阵始终相同衰减，s 只控制其数据更新。不能用改 lr 的捷径混入衰减变化。
- 主 cell 固定 block1/FFN down/step256；其他层和 step128 仅诊断。发现与确认不得改主 cell、rank 或筛选更漂亮的 seed。
- 系数筛选只用 probe_select；probe_confirm 只评估冻结方案。probe 梯度不写进真实 moments。每个 update-build replicate 都从同一 checkpoint 重新开始。
- M 基底对角捕获率的未解释部分不能全部叫作非对角项；精确 SVD 的输出能量交叉项应为零，但真实损失交叉项可非零。把这些写成数值单测。
- baseline 是本地三 LR pilot，不是充分调优的论文级基线；LR 网格或 scalar 网格未包围有效最优范围时，必须输出 unresolved，不能把它包装成正结论。
- SPECTRA 对照仅为协议指定的 post-soft-clip exact-SVD 方向诊断；不得称完整官方复现。它不需要引入大型外部训练库。
- FP32 参数/状态、BF16或FP32训练，FP32诊断、TF32关闭；无需 FP16。单 GPU、workers0、串行扫描、流式数据；OOM 不得静默改实验配置。
- 缺真实语料时允许单元测试，但不能把随机数据 smoke 标成真实预训练验证。
- NO_LOCAL_SIGNAL 是可接受的科学输出，不是要求你换超参把实验修到通过。
- 不要宣称完成未运行的训练。完成实现只可写 `IMPLEMENTED_NOT_FORMALLY_RUN`；真实发现/确认留空或 NOT_RUN。

代码结构可自主决定。请保持 Windows PowerShell 与后续 Linux 迁移可用：pathlib、绝对解释器记录、相对工作区数据引用、无硬编码用户机器其他路径。生成运行锁，避免两个代理同时启动训练。不要部署长驻服务或修改全局 agent 配置。
