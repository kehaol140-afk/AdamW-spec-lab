# AdamW 真实更新谱分配：本地机制筛查协议 v1

日期：2026-10-01。状态：待在用户本机实现与执行；不是实验结果。
工作区：`E:\大模型预训练\adamw_spectral_pilot`。
外部 MiniMind、数据和历史实验均只读。研究任务是从随机初始化开始的语言模型预训练，不是微调、数据配比迁移或优化器状态重置研究。

## 1. 本轮问题、权限与边界

唯一主问题：在本协议预先限定的模型、参数块、训练状态和标量搜索范围内，改变实际 AdamW 更新的相对谱分配，是否具有超出标量步幅调节的、在独立数据上可确认的局部收益？

本轮授权：环境与数据检查、数值测试、8 步真实数据 smoke、发现 seed 的三条 512 步 AdamW 轨迹、局部筛查；仅当发现门通过时，运行另一 seed 的一条 512 步确认轨迹与冻结候选确认。所有任务串行。

本轮不做：完整新优化器开发、在线控制器、全层每步 SVD、多模型大扫参、Hessian/HVP、分布式、服务器训练、模型/数据下载、自动环境升级、以负结果为理由自动扩搜索。

最高结论是 `GO_SHORT_BRANCH_ONLY`：有理由另外设计短分支实验。它不是新方法成立、跨 seed 统计显著、完整预算终点改善或训练加速的证明。新方法全程从头预训练的确认留到后续。

## 2. 数学定义与可识别性

### 2.1 真实 AdamW 方向

令 checkpoint 为完成第 t 次 optimizer step 后的完整训练状态。用正常训练数据流的一个有效 batch 构造梯度；先完成 microbatch 累积与有效 token 归一化，再按既定规则 unscale（本轮无 FP16）、全局梯度裁剪。以第 t+1 步实际组学习率 eta 和参数自己的 step 计数构造更新。

`m_plus = beta1*m + (1-beta1)*g`

`v_plus = beta2*v + (1-beta2)*g*g`

`A = (m_plus/(1-beta1**step_plus)) / (sqrt(v_plus/(1-beta2**step_plus)) + eps)`

`W_adam = (1-eta*wd)*W_old - eta*A`

必须与本机实际 `torch.optim.AdamW` 的一步行为核对，不能使用 `-eta*g` 代替 AdamW，不得遗漏 epsilon、bias correction、clip 或每参数 step。`grad=None` 的参数行为也必须忠实；共享参数只能更新一次。基线固定 `amsgrad=False, maximize=False, foreach=False, fused=False`，以本机 API 可用性检查为前提；不要静默换算法。

基线使用 PyTorch AdamW。诊断在可恢复的工作副本上构造一次真实 next step；可从其 next moments 计算 A，但不能只依靠两个 FP32 权重相减除学习率来估计 A（可能有消减误差）。需要同时验证重构的权重与实际 step 一致。

### 2.2 处理变量不是两个独立“原因”

对 A 做精确 reduced SVD，首轮头部固定为最大奇异分量（rank=1）：

`A_H = sigma1 * u1 * v1.T`

`A_R = A - A_H`

`D(rho,s) = s*(rho*A_H + A_R)`

rho 控制相对分配，s 控制目标参数块的数据更新幅度。rho=1 是同一个参数块的标量控制，不是仅给整个模型调一个全局学习率。s=0 是该目标块只做相同衰减的候选，不把其他块冻结。

`0.5*A_H+A_R = 0.5*(A_H+2*A_R)`。允许整体尺度改变后，“压头部”和“抬剩余部分”不是可独立识别的原因。本轮只报告相对分配，不根据名称归因。

主实验不做逐候选范数匹配，以免引入冗余参数化。记录全部范数即可。标量对照采用比候选更密、更宽的预先固定搜索。所有“最佳”只能称为本协议范围内的最佳，不能声称全局线搜索最优。

### 2.3 统一参数基点与权重衰减

对一个目标矩阵，每个梯度构造 replicate 先形成完整普通 AdamW 下一步权重。其他参数固定在这一步的普通 AdamW 结果；目标矩阵的公共基点为：

`W_base_target = (1-eta*wd)*W_old_target`

随后逐一设置：

`W_candidate_target = W_base_target - eta*D(rho,s)`。

所以所有候选的其他参数更新、目标块权重衰减、真实下一步学习率、数据、dropout 模式完全一致。s 只乘数据更新项，绝不通过修改优化器 lr 混入 decay 变化。

在该完整公共基点 theta_base 上测方向梯度与输入激活；在各完整 theta_candidate 上直接计算损失。额外保存 checkpoint 原点损失及普通 AdamW 下一步损失，避免混淆基点。

每个目标块单独干预，不同时选择 FFN 与 attention 的比例。本轮没有跨模块组合优化的结论。

### 2.4 输出能量与损失耦合是两个问题

对固定线性层输入 x，实际数据上的输出变化量可由流式累加 `||D*x||^2` 得到，等于 `tr(D*C_x*D.T)`；无需存储 C_x。乘 eta^2 后才是相应输出变化平方。

对同一精确 SVD 的不重叠分量，`A_H.T @ A_R = 0`，所以对每一个 x 都有 `(A_H*x).T@(A_R*x)=0`。因此：

`E||D*x||^2 = s^2*(rho^2*E||A_H*x||^2 + E||A_R*x||^2)`。

但这不表示最终损失独立：损失 Hessian 的 `vec(A_H).T @ H @ vec(A_R)` 一般不为零。直接评估完整二维候选即可包含这类耦合；不得把各方向分别找到的步长机械拼接。

### 2.5 M 基底与 A 基底诊断的边界

同时记录 m_plus（可使用 bias-corrected M，只是统一标量，不改变非退化奇异方向）和 A 的谱集中度、head gap、头部左右子空间重合度。矩阵近零或首二奇异值接近时，标注方向不稳定；不要把 SVD 符号变化当作物理变化。

另记录：

`paired_diagonal_capture = sum_i (uM_i.T @ A @ vM_i)^2 / ||A||F^2`

`unexplained_fraction = 1 - paired_diagonal_capture`。

必须同时保存分子分母。对矩形矩阵 reduced SVD，这个余量既可含基内非对角项，也可含 thin basis 之外的分量，不能一概命名为“非对角能量”。该比例依赖重根处基底选择，也不是旧文献机制错误或新方法收益的证明。

## 3. 本地资源与工程约束

- 只用一个 GPU 训练进程；同时最多一个模型驻留 GPU。候选顺序执行，禁止为每个候选 deepcopy 一个 GPU 模型。
- 模型配置：dense decoder，4 层，hidden=256，FFN intermediate=768，attention heads=8，KV heads=2，head_dim=32，dropout=0，seq_len=256，use_cache=False。通过本地 MiniMind 配置适配；不另写完整 Transformer。
- 原 tokenizer 与词表保持不变，检查 tokenizer 最大 ID 与词表一致、特殊 token 和权重共享一致。记录真实参数数目；不能把估计的“约 5M”当作测量值。默认参数数目超过 12M 时先报 `MODEL_SIZE_UNRESOLVED`，不裁词表。
- 权重和优化器状态为 FP32。训练优先硬件原生支持的 BF16 autocast，否则 FP32；首轮不使用 FP16。诊断候选前向、探测梯度统一 FP32，关闭 TF32，使用同一注意力后端。
- 初始 microbatch=2、accumulation=4、effective_batch=8。仅在 smoke 阶段允许降到 microbatch=1、accumulation=8，token 目标不变。不能在正式运行中静默改变 batch、序列长度、网络尺寸或精度。
- DataLoader workers=0，pin_memory=False；无 torch.compile、DDP、远程日志、wandb/swanlab。
- 仅离线 CPU FP64 对两个目标矩阵做精确 SVD，诊断用方向转换回权重精度施加。SVD 成本独立计时；不能把它隐藏为零成本，也不能把它当作未来部署成本的准确估计。
- GPU 过程预算上限：min(10 GiB, 启动时空闲显存减 1.5 GiB)，且需记录系统共享显存/其他进程占用。进程 RSS 软上限 4 GiB，系统可用 RAM 至少保留 2 GiB。达不到则停止并记录资源状态，不换页硬撑。
- 原始数据流式扫描；训练缓存用 uint32 token memmap，只转换当前 batch。扫描上限 512 MiB，单行防异常上限 2 MiB。禁止 readlines/load 全量 JSONL 或把所有 tensor/轨迹常驻内存。
- CPU 线程默认 2；所有日志 flush，输出中包括当前阶段、optimizer step、实际有效 tokens、内存、耗时。不存在小时数承诺。

## 4. 环境绑定与数据合同

在 `E:\大模型预训练` 有界查找本地 MiniMind：优先直接子目录 `minimind`，其次 `minimind-master`，再最多深度 4 的候选；排除数据、缓存、输出、虚拟环境及实验归档的深层递归。检查模型类、tokenizer、数据、git origin/commit/dirty 或无 git 时源码哈希，避免用名字猜结构。

根目录存在多个无明确优先级的可用候选时，输出 `AMBIGUOUS_REPO`、候选清单及可编辑绑定文件，不擅自混用；不要下载替代仓库。读取当前 Python 环境和绝对 executable，不默认旧环境一定仍在。发现依赖缺失先输出清单，不自动升级 torch/CUDA/transformers。

绑定记录在 `bindings.local.json`：repo_root、python_executable、tokenizer_path、source_data_path、source 格式、源码指纹、tokenizer 指纹。当前路径与可迁移相对路径分别保存。

只接受现有本地预训练 text 数据或能验证同义 schema 的缓存。不要用 SFT 对话、随机 token 或旧模型生成的假数据冒充预训练。随机数据只用于单元测试。

按文档规范化文本哈希先分配互斥角色，再分词/切块：train、lr_select、probe_select、probe_confirm、branch_eval_reserved、final_holdout_reserved。相同文本进入相同角色；在本轮扫描窗口内去重。需明确这里只保证所消费窗口内的精确重复控制，不声称全库近重复清理。

训练缓存上限 2^21 个有效目标 tokens；一条 512-step 轨迹 nominal tokens=512*8*256=1,048,576。实际有效计数另行记录。可在训练角色内拼接文档，按相同 EOS 规则固定处理；目标 shift 必须恰好一次。probe/validation 使用不跨文档混合的固定块，并保留 doc_id 以便配对统计。不要将同一文档拆到不同角色。

lr_select=32 个块，probe_select=16 个块，probe_confirm=64 个块；每块最多 256 个有效目标 tokens。少于 64 个不同确认文档时应报告有效文档数，不能把同文档多个 chunk 当作独立样本；不足支持预设统计则 `DATA_INSUFFICIENT`。优先每文档仅取一个块。

main loss 为 `sum(有效 token 的 CE) / 有效 token 数`，保存 loss_sum 和 token_count；梯度累积也按整个有效 batch 的总 token 数归一化，不按 unequal-length microbatch 均值再平均。评估必须使用同一有效位置掩码；pad 与 eos 相同时不得按 token ID 无差别屏蔽真实 eos。

缓存清单记录已扫描字节数及其 SHA256、源路径/大小/mtime、缓存 SHA256、doc_id、tokenizer/config SHA256、分区规则及 packing/shift 规则；未扫完整文件时不要伪称完整源文件 SHA256。不足预算时停止，不静默循环小数据或下载补齐。

## 5. 预先固定的小规模设计

### 5.1 AdamW 基线

发现 seed=3407，确认 seed=3408。它们是本轮新实验种子，不继承旧项目裁决。

发现 seed 上训练 3 条匹配初始化与数据顺序的轨迹：peak LR 为 `[3e-4, 1e-3, 3e-3]`；每条独立从随机初始化开始，共 512 个成功 optimizer steps。betas=(0.9,0.95)，eps=1e-8；矩阵权重 WD=0.1，1D/bias/norm WD=0，共享参数去重。全局梯度裁剪阈值为 1.0。warmup=32 steps，随后 cosine 到 peak LR 的 0.1 倍；必须输出精确定义及每一步的实际 lr 与 eta*wd。

以上只是冻结的 pilot 默认配置，不是已证明最优的超参数。使用相同 lr_select 集的第 512 步 token-weighted NLL 选 LR；若数值上并列，优先中间 LR，再选较小 LR。最优 LR 仍在网格边界时记 `BASELINE_TUNING_UNRESOLVED`，不得称强基线或自动通过 gate，不自动加预算扩网格。

保存初始化、step128、step256、step512 的完整状态。确认 seed 只训练所选 LR 一条轨迹，不再次选 LR。总上限为 4*512=2048 个正式 baseline optimizer steps，加最多 20 个 smoke/重放 steps。

### 5.2 主终点与解释性诊断

主终点预先固定：第二个 decoder block（0-based index=1）的 FFN down/output projection，在 step256 checkpoint 上的局部确认收益。根据本地结构解析真实参数名并核对形状，不将 `down_proj`/旧版 `w2` 猜成相同就跳过检查。

辅助诊断：同一块的 attention output projection，以及 step128。四个 cell 不用于“挑最好的一格”过门。它们属于解释性观察，不当作四次独立训练重复。只有 FFN step256 用于候选冻结和通过门。

每个 checkpoint 使用 2 个互不重叠的训练有效 batch 构造更新 replicate。每个 replicate 都从同一完整 checkpoint 开始，只虚拟推进一次 moments；第一批是数据流实际下一批，第二批是同状态上的另一次训练 batch 抽样。不得串行把两个 replicate 的 moments 累计起来。第二批不能伪称真实已发生的下一步轨迹。

### 5.3 搜索与比较

候选 rho=`[0.5,0.75,1,1.5,2]`，候选 s=`[0.25,0.5,1,2,4]`。

标量对照 rho=1，使用 s=0 及从 0.125 到 8、相邻比 sqrt(2) 的 13 个正数。该更密、更宽标量搜索用来排除粗网格人为偏袒谱候选。

最近邻方向控制仅实现 `SPECTRA_POST_SOFT_SVD_DIAGNOSTIC`：`h_c(sigma)=sigma/sqrt(1+(sigma/c)^2)`，c=`q*sigma_max(A)`，q=`[0.5,1,2]`，再乘相同候选 s 网格。q 相对 sigma_max 的设置是本地诊断参数化，不等同原论文完整配置；不得把这个 CPU 精确 SVD 对照写成官方 SPECTRA 复现或部署性能对比。不做 gradient pre-clipping。

所有系数只在 probe_select 上筛选；在主 cell 对两个更新构造 replicate 的 loss_sum/token_count 合并比较。选择一个全局固定 `(rho*,s*)`，并冻结标量 `s_scalar*` 与最近邻 `(q*,s*)`。并列处理优先更接近 rho=1 / s=1 的配置，再按固定字典序；不能用 probe_confirm 解并列。近似并列 tolerance 由 null 重复确定，完整记录。

probe_confirm 只评估冻结的谱候选、冻结标量、普通 AdamW(s=rho=1)及冻结最近邻。不要在该集合上报告 oracle 最小值。

发现 seed 通过后，确认 seed 使用同样的冻结谱 `(rho*,s*)`，不重新扫 rho。标量 baseline 可在该 seed 的 probe_select 集上重新选择 s，给标量对照公平的状态适配能力；最近邻可同样在 select 上重选，均不得读 confirm 决策。确认候选仍严格不重调。额外报告发现 seed 标量系数直接迁移的结果，但不取代更强标量对照。

`rho` 处于候选边界仅说明未定位最优比例，不妨碍有限域内的存在性观察；标量最佳 s 落在最小正边界或最大边界时，记录 `INCONCLUSIVE_SCALE_RANGE`，不把未覆盖的标量范围当作谱机制收益。s=0 是合法非负边界，不等同需要负步长搜索。

### 5.4 两类必要诊断

在 theta_base 的独立探测批次上，以 `autograd.grad` 只取目标参数梯度，计算 `<G_probe,A_H>` 与 `<G_probe,A_R>`；正值表示沿相应负方向的一阶下降倾向。保存逐文档/批次值、均值、波动、符号比例、方向范数与夹角，避免把未经归一化大小直接解释为方向质量。

流式挂钩记录公共输入上的 `||A_H*x||^2`、`||A_R*x||^2`、联合 D 的输出变化，以及上述 SVD 正交能量恒等式误差。只累计必要统计，不保留完整激活库。联合损失扫描为最终局部响应；不得只用输出能量下降替代损失收益。

## 6. 门禁、统计与停止条件

### 6.1 数值可信性 P0

CPU FP64 toy 参数（方阵、长矩阵、向量、共享参数、None/零梯度）与 torch AdamW 单步及 256 步对齐，建议 rtol=1e-10、atol=1e-12。实际 FP32 AdamW 恒等更新检查建议 rtol=1e-5、atol=1e-7，同时输出绝对误差、相对范数误差、near-zero flag。阈值是预设工程容差，不是观察到的结果；不能为通过而放宽。

其余必测：rho=1/s=1 恒等；等价参数化；固定衰减；A_H+A_R=A；SVD 的输出能量正交；一般损失二次型允许非零交叉项；单次标签 shift；变长 token mean；probe 前后状态隔离；共享权重去重；CPU/RNG/数据游标恢复；同环境连续 8 步与 4+恢复4 步对照。SVD 重根测试不要求唯一奇异向量。

真实微型模型 FP32 上重复相同候选、相同输入和状态至少 3 次，测数值/null 波动 epsilon_null。模型/optimizer state、RNG 和数据游标在 probe 结束后与原 checkpoint 恢复一致，原 checkpoint 文件 SHA 不变。不能只检查磁盘文件没改而忽略内存状态。

### 6.2 局部容量门

定义逐文档配对差：`Delta = NLL(best_scalar_selected) - NLL(frozen_spectral)`，正值表示谱候选更好。有效 token 加权。对两个 update-build replicate 先在同一文档内聚合，不把重复文档、replicate 或 checkpoint 伪装成独立样本。

预先约定最小局部实用差 `delta_local=max(1e-5 nats/token, 5*epsilon_null)`。这只是投入决策阈值，不是理论常数；需同时报告 raw Delta，不把小于阈值的结果抹成 0。

在 probe_confirm 文档组上配对 bootstrap（1000 次，固定统计 RNG），输出区间与不同文档数。该区间只描述固定轨迹/候选条件下的评估数据不确定性，不描述跨训练 seed 的总体不确定性。

发现门要求：P0 通过、基线/标量搜索没有 unresolved、冻结候选在主 cell 上 Delta>delta_local，且文档配对 95% 区间下界>0。此时只准进入 seed3408 的冻结确认。

本地确认要求：seed3408 在主 cell 上对更强标量对照同样满足上述条件；报告两个 seed 原始值，不称两个 seed 足够建立普遍规律。满足则 `GO_SHORT_BRANCH_ONLY`；不自动启动短分支、全程谱训练或服务器任务。

### 6.3 状态字典

- `P0_FAIL`：实现或数值对齐失败；不得解释为科学负结果。
- `ENV_BLOCKED` / `DATA_INSUFFICIENT` / `RESOURCE_BLOCKED`：环境、数据、资源阻塞。
- `BASELINE_TUNING_UNRESOLVED` / `INCONCLUSIVE_SCALE_RANGE`：搜索范围不足，本轮不扩。
- `NO_LOCAL_SIGNAL`：主 cell 无超出阈值且独立确认的收益；只否定本次有限配置下的可用信号，不否定整个谱研究。
- `NO_TRANSFER`：发现通过，但冻结候选未在确认 seed 延续；不转为逐 cell oracle 宣称成功。
- `GO_SHORT_BRANCH_ONLY`：本地容量与有限迁移信号值得下一轮短分支。

附加独立字段：`method_novelty=NOT_ESTABLISHED`，`end_to_end_gain=NOT_TESTED`，`deployment_cost=NOT_TESTED`，`nearest_neighbor_comparison=LOCAL_DIRECTION_CONTROL_ONLY`。即使局部门通过也不能改为 ESTABLISHED。

若最近邻相当或更好，允许机制观察继续作为解释工作，但不可声称新候选有方法优势。不得改目标层、rank、seed 或数据以寻找“通过”。任何未来变更必须新 protocol ID、新结果目录、保留旧结果。

## 7. 成本口径

计时必须 CUDA 同步，并区分训练前反向、optimizer、CPU SVD、GPU/CPU传输、probe、validation、数据准备与 I/O。记录 warmup 与稳态计时、实际吞吐、峰值 allocated/reserved VRAM 和进程 RSS。

本轮离线局部探测不等于部署方法，因此不得用它估计真实每步 c 并声称已节能。未来部署若每步开销乘数为 1+c，到目标步数比例 r，只有 r(1+c)<1 才可能净省训练时间；固定初始化/诊断开销需另加。不达目标的 run 视为删失/未达，不插值捏造节省。能量需单独测量，不能用该不等式代替能耗测量。

## 8. 输出、可恢复性与监督接口

必须实现统一入口 `run.py`，见 RUN_WINDOWS.md。所有阶段启动前写 run manifest，退出码非零必须传播。每条轨迹/scan 采用 run_id 与配置+源码+绑定+数据 SHA；只有完全匹配的成功结果才可跳过。FAILED 或部分完成不冒充成功。

完整 checkpoint 包含 model、optimizer states及各 param group、参数名/alias 映射、successful optimizer step、调度位置、CPU/Python/NumPy/CUDA RNG、GradScaler(本轮 null)、数据 stream 游标/排列、累计有效 tokens、配置/数据/源码 hash。只在 accumulation boundary 保存。采用原子写入。

探测只在一个可恢复工作实例上执行；`try/finally` 恢复参数、buffer、moments、RNG、train/eval 模式、梯度及游标。不改变真实训练路径，不消费真实训练全局 RNG；统计 bootstrap 使用独立 RNG。

建议输出：
- `artifacts/preflight.json`, `artifacts/data_manifest.json`, `artifacts/protocol_resolved.json`
- `artifacts/numerics/adamw_parity.json`, `identity_probe.json`, `resume_replay.json`, `memory_smoke.json`
- `results/baseline_runs.csv`, `results/spectral_metrics.csv`, `results/probe_grid.csv`
- `results/probe_confirm.csv`, `results/direction_signals.csv`, `results/output_effects.csv`
- `results/candidate_frozen.json`, `results/local_gate.json`, `reports/LOCAL_FINDINGS.md`
- `logs/events.jsonl`, `logs/runtime.jsonl`, `reports/CODEX_HANDOFF.md`, `reports/OPENCODE_AUDIT.md`

CSV 保存参数名、形状、seed、checkpoint、gradient-build replicate、split、doc_id、rho/s或q、eta、wd、eta*wd、loss_sum/count/NLL、norms、数值误差和耗时。不得只输出聚合最小值。

使用项目级运行锁避免 Codex 与 OpenCode 重复启动训练。只能管理本项目记录的 PID；不能关闭其他 Python、浏览器或 GPU 进程。OpenCode 不得调整科学假设、搜索网格或阈值；非科学性故障最多两次可记录重试，代码变更后必须重新跑相关 P0 并作废受影响旧输出。

## 9. 最近邻与来源边界

这些来源用于定位和实现核对，不意味着本地实验已复现：

- SPECTRA v2，Enhancing LLM Training via Spectral Clipping：基础优化器更新后的 soft spectral clipping，以及可选梯度预裁剪。https://arxiv.org/html/2603.14315v2
- 官方实现：https://github.com/mlolab/llm-spectral-clipping （只查定义；本轮不得自动克隆/运行其全套实验）。
- Muon2 v2：二阶矩预条件后接矩阵正交化。https://arxiv.org/html/2604.09967v2
- SAMuon，附录 B、E：AdamW 轨迹上探测 M，展示 Adam 更新在 M 奇异基上的对角投影，明确不跟踪非对角项。https://arxiv.org/html/2608.25990v1
- PolarAdamW：不能把其视觉/点云实证直接说成语言预训练证据。https://arxiv.org/abs/2605.07067
- MiniMind 当前公开主线只能作参考，本地源码版本才是实现事实。https://github.com/jingyaogong/minimind
- PyTorch AdamW 算法与实现参数：https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html

逐坐标预条件怎样改变 M 与 A 的实际限制方向，仍是待检验的问题；直接在 A 上重新 SVD 并不自动构成创新，也不会自动消除最终损失中的耦合。

## 10. 机器可读交接细节

`results/local_gate.json` 至少包含 `protocol_id, stage, verdict, next_action, source_hash, config_hash, data_hash, candidate_hash, reasons`。discover 通过时 `verdict=DISCOVERY_PASS, next_action=RUN_CONFIRM`；发现未通过则 `next_action=STOP`。确认完成后一律停止：通过时 `verdict=GO_SHORT_BRANCH_ONLY, next_action=STOP_FOR_REVIEW`；其他状态用对应 verdict 和 STOP。确认命令不得绕过这些依赖。

同一源码/配置/数据/环境指纹下，smoke 已通过时默认验证并复用其完整数值产物，不再消耗一套 GPU smoke 步数；不完整或哈希不符不得冒充通过。重新实现后的重测必须遵守剩余授权预算，预算不足就报告，不能无限重跑。

学习率步号合同：成功更新 k=1,...,512；k<=32 时 eta_k=peak*k/32；k>32 时 eta_k=peak*[0.1+0.9*(1+cos(pi*(k-32)/480))/2]。step128 checkpoint 的下一步使用 eta_129，不重启调度。

512 步/约一百万目标 token 只覆盖一个短程训练窗口，不能称充分预训练；step128/256只是本地预算进度，不能据此外推大模型的早中晚期规律。
