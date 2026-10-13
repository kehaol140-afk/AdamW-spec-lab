# EXPERIMENTS — context-aware-muon-100m

## E1 v1.0 pilot（2026-10-12 14:06 → 18:49）

- 目的：100M 公平网格下检验 Group-A/B 混合正交化 vs 强基线
- 模型/模块：100,784,000 参数；Group-A=FFN 39 矩阵；Group-B=+V/O 26；Q/K native
- 数据/tokenizer：`artifacts/pilot_train_tokens_33p7m.u32`（128.3MiB，33,685,504 tokens，sha `f92715b4…`；V16 8.5M+s30m 17M 前缀字节证明链）；tokenizer vocab 6400（V16 同源）
- 划分：dev 256 docs × 三视界 H1024/H2048/H4096（V16 development 同源；**无独立 holdout**——pilot 协议即 dev 级）
- 15 臂（`configs/pilot_arms_v1.json` 冻结）：B_adamw_lr{3e-4,1e-3,3e-3}；B_muon_lr{5e-3,1e-2,2e-2}；B_muon_rms_lr{1e-3,3e-3,1e-2}；B_lowmu_{0.9,0.8}；C_A_lam{0.65,0.85}；C_A_nat；C_B（挂 @B_muon_best=5e-3）
- 训练：seed **3471**（15 臂同 init fingerprint `33f93a0f546149a0`）；T=4096 步；33,554,432 tokens/臂（实际 33,685,504）；warmup128+cosine→0.1×peak；batch 8192 tok/步；bf16 autocast、wd 0.02、无 clip
- baseline：B_muon_rms（强 Muon 锚）与 B_adamw（全场参照）
- 指标/门：dev NLL 三视界 token 加权；C_A_pass_vs_best_anchor / momentum_explained / C_B_beats_C_A + RANGE 边界标记（analysis.json gate）
- 成本：driver 累计 4.55h GPU（DRIVER DONE 18:44:17；pilot_driver.log:80）；逐臂 wall/VRAM/tok-s 见 PILOT_100M_RESULTS.csv

## E2 事故与偏差

1. 14:06 首跑 runner 路径 bug exit=1 → 修复重启（run_index RESTARTED）
2. 双 driver 并发 ~90s：B_adamw_lr3e-4 metrics.jsonl 写入竞争 → 逐行审计 **0 损坏**（15 臂完整性 4096 行/0 坏行，analysis.json:147-208；PILOT_100M_RUNTIME.md:14-17）
3. v1.1 增补提案（B_normuon_ns5vr 等）PROPOSAL_ONLY 未执行；第四阶段预算重分配不启动（裁决 §62）

## E3 NorMuon staging（2026-10-12，零 GPU）

适配器（单 GPU 版，源自 modded-nanogpt PR #276 贡献版镜像）、PolarExpress（arXiv:2505.16932 五组系数）、Moonlight 语义对照；18/18 CPU 语义测试 PASS + 动量独立性恒等检查（I1 标度恒等 1.4e-17）。**未跑任何 GPU 消融**；卡 G-2（LR 锚点被 RANGE_UNRESOLVED 悬置）/G-3（30 步吞吐 smoke）/G-4（预算授权）；与官方 NorMuon 仓逐行等价性**未验证**（网络阻断，G-5 待上游上传核对）。
