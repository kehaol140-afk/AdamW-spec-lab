# EXPERIMENTS — adamw-direction-30m

## E1 T1（2026-10-11 22:55 → 10-12 00:22）

- 目的：方向残余规模迁移 + Muon 公平参照（三择一裁决）
- 模型/模块：30.03M MiniMind；目标块 L1/L2 FFN down_proj（各 2.84%）
- 数据/tokenizer：自铸缓存 `train_tokens_s30m.u32` 17M tokens（sha `6fe0bb53…`，mint 配方+out 全录）；tokenizer=V16 同源（vocab 6400）；语料=上游 minimind pretrain_t2t_mini.jsonl（1.24GB，服务器）
- 划分：dev = V16 development 256 docs（只读 ../adamw-spectral-v13-v16 eval_sets）；**零 holdout 触碰**（30M_DECISION.md:34 复核）
- 臂（13）：LR 网格 s30_S1_lr{0.0003,0.001,0.003}_L1；a 网格 s30_S{2,4,8}_lr0.001_L1；F2/P2_lr0.001_L1；层复核 s30_S2/P2_lr0.001_L2；Muon s30_M_lr{0.005,0.01,0.02}。+2×50 步 smoke（双路径）
- 训练：seed **3421**，2048 步/run，**16,777,216 tokens/run**；warmup+cosine→0.1×peak 同 V16
- baseline：同尺度调优 S（S1/S2）与调优 Muon（三 LR）
- 门（frozen_30m_t1.json:31-42 预注册）：选择规则（lr* 内点原则、a* 单调性）+ 主门 F2−S2@L1 与 P2−S2@L1 均 CI95 全负；预算 2.0 GPU-h；**禁止**本轮多 seed/holdout_v2 开封/100M（:44）
- 指标：token 加权 NLL + doc 级 paired bootstrap
- 成本：Σtrain 4,428.6s + eval ≈1.45h（授权 2.0h；30M_RUNTIME_COST.md:12）

## E2 偏差

零事故（13/13 完成、零非有限值、零 fallback；30M_DIRECTION_EFFECT.md:7）。缓存铸造单独留档（`raw/mint.out`）。T2（seeds 3422-3424）按冻结协议禁止本轮执行——计划内未执行，非遗漏。
