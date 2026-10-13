# AdamW 谱分配 pilot：低系统内存修订 V1.1

日期：2026-10-02。范围：已有实现的资源门禁、内存日志与版本交接；不设计新优化器。
原协议：ADAMW_A_SPECTRAL_LOCAL_V1_20261001。
建议新协议：ADAMW_A_SPECTRAL_LOCAL_V1_1_LOWMEM_20261002。
工作区仍为 `E:\大模型预训练\adamw_spectral_pilot`；外部 MiniMind、数据源与 Python 环境保持原绑定。

本包是修订合同、候选配置和代理指令，不包含已修改的训练代码。未在用户 Windows/CUDA 环境运行。`config/local_v1_lowmem.json` 新增的 start/warning 字段需要 Codex 接入，不能仅复制配置就宣称这些保护已生效。

## 1. 依据与结论边界

上传日志显示，OpenCode 的三次 smoke 均在 `pilot_core.py::resource_check`、模型构建前停止，可用系统 RAM 分别为 2,100,858,880、2,045,362,176、1,907,736,576 字节，即约 1.957、1.905、1.777 GiB。旧门禁要求 2 GiB，第一次仅差 44.465 MiB。Codex 另有初始化模型后的同类阻断。两类情况应分开描述。

没有完成真实 GPU optimizer step，没有证据证明发生实际 CPU/CUDA OOM，也没有证据证明现有机器一定能完成整个 pilot。原 CPU 数值测试通过不等于 GPU smoke 或正式 probe 已验证。

实际绑定 GPU 为 8 GiB 级 RTX 4060 Laptop，模型为 4,655,616 个唯一参数，非 15 GiB 显卡。仅按 FP32 参数、梯度、两个 moments 计算的主张量存储是 71.039 MiB；这不包括激活、logits、框架、CUDA 上下文、临时张量、CPU 副本，不能当作 RAM/VRAM 峰值估计。

本轮采取：先修改人为设置的资源阈值，实测原模型；不先改模型、数据、batch 或科学门槛。

## 2. 资源策略

| 参数/行为 | V1.1 |
|---|---|
| `runtime.system_ram_free_reserve_gib` | 1.0：运行中硬停止线 |
| `runtime.system_ram_start_min_gib` | 1.5：新训练进程首次初始化模型/CUDA前的准入检查 |
| `runtime.system_ram_warning_gib` | 1.5：运行中不足此值、但仍 ≥1.0 时告警，不因此拒绝 |
| `runtime.process_rss_soft_limit_gib` | 4，不提高 |
| GPU 预留/上限 | 原规则不变：预留1.5 GiB，过程上限仍受实际可用显存限制 |
| 执行 | 单 GPU 作业、单模型、无 DataLoader workers、无 pinned memory、2 CPU threads |

这些是新授权的工程运行阈值，不是普适安全定律，也不保证 Windows 完全不换页。

准入线只在新训练/探测进程首次进入重初始化时检查一次，不在训练每步、每个候选或每次 CUDA 查询时反复套用；否则仍会把已运行的作业错误拒绝。CUDA/模型初始化后，以及训练、保存/恢复、probe 等阶段，使用 1.0 GiB 硬线和 1.5 GiB 警告线。preflight 若已经完成 CUDA/模型初始化，其末尾状态按运行硬线判断，不再额外要求初始化前的余量。

首次准入不足时不启动模型、不无限轮询；记录真实数值并退出。低于 1.0 GiB、RSS 超限、CUDA OOM、MemoryError 均不得被转换为 PASS。若已经完成 optimizer step，精确记录实际步数、最后成功 checkpoint 与其后未持久化步数；不要自动从较早 checkpoint 重跑而漏算预算。没有可靠恢复点时如实标记，不能声称断点已保存。

不允许自动杀其他用户进程、修改系统页面文件、升级包、关闭数值检查、改用系统共享内存冒充显存，或再次把阈值下调至0.5/0 GiB。

## 3. 必须保持不变

模型4层/hidden256/FFN768、词表与 tied embedding、seq256、microbatch2×accum4、FP32参数/moments、原训练精度；种子、3个LR、512步预算、checkpoint、数据角色与tokens、AdamW实现、梯度裁剪、eta*wd固定、谱候选/标量/最近邻网格、主cell、统计门槛、确认隔离、状态恢复和smoke强制门禁全部不变。

已有 `smoke_only_fallback` 不得自动启用；本轮不把降 batch 当作系统 RAM 问题的首选解法。出现真实显存不足后才能另行讨论。

## 4. 最小代码范围

统一审查 `pilot_core.py` 的 preflight/resource_check/smoke/train_steps、`pilot_workflow.py` 的 baseline/probe入口与循环边界，以及 `prepare_data.py` 的准入检查。它们要使用同一个运行硬阈值，错误文本必须显示实际阶段、可用字节、对应阈值，不再写死“2 GiB”。不要修改核心张量运算。

资源日志必须包括：protocol/config/source hash、PID、阶段、真实optimizer步数、系统available RAM、过程RSS、CUDA allocated/reserved及可用的峰值、警告/阻断原因。CUDA尚未初始化时记录null，不要为了记录“初始化前显存”提前创建CUDA上下文。每个optimizer步与probe候选边界记录；不以一次瞬时样本代替整轮峰值记录。

日志不能证明“从未换页”。尤其不能把 Windows psutil 的 swap sin/sout 为0解释为没有分页；采不到的量如实标记。

本轮先不重构 `_candidate_directions`：现实现对主矩阵一次物化54个FP64方向，约81 MiB，仅提示后续可能优化。如果资源测量确实显示 probe 压力，再单独实现逐候选生成并检验逐候选数值一致；不要一次混入多个修复。

## 5. 版本、数据和结果保护

不覆盖 `config/local_v1.json`，新活动配置为 `config/local_v1_lowmem.json`。先检查本地没有额外正式结果或项目进程，再把旧代码、配置、报告、日志、轻量manifest、旧smoke证据及其SHA保存到工作区 `archive/v1_resource_blocked_<timestamp>/`。原1.24GB数据、外部MiniMind与Python环境不复制、不删除、不下载。

由于原实现使用全局 `artifacts/` 与 `results/` 路径，需要归档旧证据后重新生成活动manifest，不得把两次运行混写。保留原协议，增加本修订并明确活动protocol_id；原配置及数值科学合同不改。

`source_hash()` 在 `pilot_core.py` 和 `prepare_data.py` 都有 SOURCE_FILES 列表；若新增代码/测试模块，必须两处一致纳入。数据准备在本机用同一源窗口重建小缓存和manifest，比较 train_tokens/eval_blocks/doc_roles 的内容SHA与旧版一致。config_hash/source_hash/data_hash因版本元数据变化可不同，不能要求元数据哈希相同；也不能手工篡改旧manifest跳过验证。

`run.py::bundle_light` 原来写死只包含 local_v1.json；须把活动配置、修订说明、资源辅助代码/测试和新日志纳入。不要打包凭证、原数据、环境或大checkpoint。

## 6. 测试与执行授权

先为资源行为写测试：约1.9 GiB可准入；1.25 GiB运行中告警但允许；0.9 GiB硬阻断；原2 GiB配置仍按原行为阻断；缺失资源读数不得绕过保护；OOM不得被吞掉；新增资源检测不得改变RNG、grad、moments与数据游标。

本地CPU原10项测试继续运行；增加的mock测试不能冒充真实GPU测试。代码冻结并刷新manifest后，Codex只运行真实smoke/replay，总预算仍≤20次实际GPU optimizer step（现有8步连续+4步重放正常消耗12步）。非更新前的资源阻断可在用户释放内存后最多重试2次；发生更新后的失败不得无预算核算整轮重跑。

Codex交付后由OpenCode核验最近一次smoke PASS及完整指纹。匹配的PASS可以复用，不必为“独立监督”重复消耗12步。随后 discover，再仅按gate的RUN_CONFIRM进入confirm，最后STOP_FOR_REVIEW。2048个正式optimizer步总预算与全部科学停止规则不变。

只有达到新实测smoke/replay PASS才能说恢复了GPU最小运行；只有跑完正式阶段才能报告科学信号。1 GiB方案仍被阻断时，先让用户释放后台占用，保留记录，不继续降低保护门槛。
