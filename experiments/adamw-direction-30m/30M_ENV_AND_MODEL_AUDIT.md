# 30M_ENV_AND_MODEL_AUDIT — 环境与模型核实报告（第一阶段交付）

日期：2026-10-11。全部为只读核验 + 本项目新建产物；未修改 V16/M 线/SpecMuon 任何文件。

## 1. 30M 模型真实架构（CPU 实测，非文档转述）

| 项 | 实测值 | 依据 |
|---|---|---|
| 参数量 | **30,025,216**（与 M0 routing_report.json 一致） | 本机构建 `MiniMindConfig(h512,8L,8Q,4KV,vocab6400,use_moe=False,dropout=0,flash_attn=True)` 实测 |
| 架构 | hidden 512 / 8 层 / FFN 1664（=ceil(512π/64)×64 默认）/ head_dim 64 / QK-Norm / GQA 8Q-4KV / tied head | minimind_base/model/model_minimind.py 默认值 + 构建实测 |
| 初始化确定性 | `torch.manual_seed(3421)` 两次构建全部参数位级相同 | 实测 |
| 目标块 | `model.layers.{1,2}.mlp.down_proj.weight`，shape (512,1664)，各 851,968 参数，占比 **2.84%**（4.66M 为 4.22%） | 实测（§3 层对应） |
| 参数分组 | 90 唯一参数 = 57 个 ≥2D（30,015,488 参数）+ 33 个 1D（9,728）；tied embed/lm_head 单对象 | 实测；M0 路由 56 Muon + 34 aux = 90 ✓ |

## 2. 数据/tokenizer 与 4.66M 的兼容性

| 项 | 核验 |
|---|---|
| tokenizer | `71f32c68…` 同一文件三处在位（V16 minimind_repo、minimind_base、dv 验证链），sha256 实测一致 |
| 语料 | 服务器副本 `/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl`（1,241,043,656 B，与 minimind_base 副本同大小） |
| 新缓存 | `artifacts/train_tokens_s30m.u32`：17,000,000 tokens，sha256 `6fe0bb53…`；**前 8,500,000 tokens 与 V16 campaign 缓存（`c91382dc…`）逐字节相同**（V16 规则重放：normalize→sha256 分区→train 角色→[BOS]+text+[EOS] 拼接） |
| 窗口语义 | `s30m_core.TokenStream` 与 V16 `training_batch` 在共享前缀上输出逐位相同（CPU 测试 test_window_identity_with_v16 PASS） |
| 评价集 | V16 development 256 docs（`campaign_v16/eval_sets.json`；dev 级，历史选优已披露）；本轮不触碰任何 holdout |
| 容量 | T1 需 2048 步 × 8192 tok + 1 = 16,777,217 ≤ 17,000,000 ✓（每臂从游标 0，CRN 同流） |

## 3. 目标块层对应（避免层位置混杂，预注册分组）

- **L-index**：layers.1（索引对齐 4.66M 的"第 2 块"），深度分数 d=1/7≈0.143
- **L-depth**：layers.2（最接近 4.66M d=1/3），d=2/7≈0.286
- 两层分组报告，不合并；符号矛盾 → 记 `LAYER_POSITION_CONFOUNDED`

## 4. 优化器路由 / wd / schedule / AdamW 状态

| 族 | 配方（冻结） |
|---|---|
| AdamW 全族（S/F/P/A0） | 单优化器双参数组：≥2D 矩阵 wd 0.1、1D wd 0；betas (0.9,0.95)、eps 1e-8、foreach=False、fused=False（V16 逐字段同构）；梯度 ÷valid_total → 全局 clip 1.0 → `dv_core.apply_step`（逐字 vendored，sha `b8943249…` 与 dv 冻结一致） |
| F/P 语义 | D=N_A(g_clip) / N_A(g_clip/(√v̂⁺+ε))，fp64 精确匹配 ‖D‖=‖A_t^branch‖；AdamW 二阶状态由原生 opt.step() 正常演化（不动）；其他参数 b=1 原生更新；目标块衰减 (1−ηλ) 保留 |
| Muon 参照臂 | `torch.optim.Muon`（torch 2.11.0，`_muon.py` sha `38e95918…` 实测在位）：momentum 0.95/Nesterov/NS5/系数 (3.4445,−4.775,2.0315)/eps 1e-7，wd 0.02；aux AdamW 同 LR、wd 0.02；56+34 路由实测复现（CPU 测试 test_muon_routing PASS） |
| schedule | 全臂共同时钟：warmup 128 线性 → cosine→0.1×peak，T=2048（dv `schedule_eta` 原文） |
| 批形状 | micro 8 × accum 4 × seq 256 = 8192 tok/步；bf16 autocast（CUDA） |

## 5. Muon 公平基线的可得性

- M 线 M1/M2 提供了同模型（30M）从头的 Muon 网格实测：最优 cell = mult 4（LR 0.02）× wd 0.02（M2 三 mixture 中 2/3 最优在 mult4 边界，LR_RANGE_UNRESOLVED 已披露）
- 本项目 Muon 参照臂 = M 线接线原样移植（56+34 路由、冻结超参）+ 本协议数据/时钟/批形状；LR 网格 {5e-3,1e-2,2e-2}（含 M 线边界值 0.02）；**参照臂不参与 F/P 门判定**，只报告同预算 NLL/wall-clock
- **禁止事项遵守**：不将任何旧 Muon checkpoint 用作共同初始化——全部 13 runs 从头训练（seed 3421 新建初始化，GPU smoke 实测 init_fp `31066e18b1d5` 且 S/M 两路径第一步 NLL 位级相同 8.866115808486938 → CRN 成立）

## 6. 种子与运行环境

- seed 3421：注册表核查（V16 3407–3410、dv 3411–3413、M 线 2030/3101、v07 2030–2037、A0G 2038–2041、G1a/b 2042–2044、SpecMuon 无占用）→ 无冲突
- GPU0：RTX 5090 32G，smoke 实测峰值 VRAM 1.27 GiB；GPU1 vLLM 全程不触碰
- Python env：fr_minimind（torch 2.11.0+cu，bf16 支持）；CPU 9/9 测试 PASS（含 S(1,1) bit-native、S-a 位移数学、F/P 范数匹配、Muon 路由、窗口身份）

## 7. GPU smoke 标定（50 步 × 2 路径，已消耗 ~0.005 h GPU）

| 路径 | s/步 | tok/s | 2048 步投影 | opt 步开销 |
|---|---|---|---|---|
| AdamW S8+apply_step | 0.2048 | 40.0k | 6.99 min | 26.2 ms/步（含解析 A 全参数遍历） |
| Muon(56)+aux AdamW | 0.1518 | 54.0k | 5.18 min | 27.8 ms/步（Muon+aux） |
| **T1 全矩阵投影** | | | **10×6.99 + 3×5.18 = 85.4 min 训练 + ~13 min 评估 ≈ 1.64 h ≤ 2 h 授权窗口** | |
