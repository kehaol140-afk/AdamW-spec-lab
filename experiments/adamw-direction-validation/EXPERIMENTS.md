# EXPERIMENTS — adamw-direction-validation

## E1 Layer-1 匹配矩阵（2026-10-11 19:39 冻结 → 21:01-21:48 执行）

- 目的：方向 vs 幅度因果分解（补 S(8,1) 缺失格）
- 模型/模块：MiniMind 4,655,616 参数（h256/4L/seq256 tied）；目标块 layers.1.mlp.down_proj（4.22%）
- 数据/tokenizer：V16 冻结缓存 `campaign_v16/cache/train_tokens.u32`（sha c91382dc…，只读 memmap）；tokenizer vocab 6400（V16 同源）
- 划分：主判据 = V16 development 256 docs（H512/H1024/H2048；**dev 级已烧毁**）；次级 = V16 screen 256；holdout_v2（1024 docs，sha 承诺 `3e0f7653…`）MINTED_SEALED 未开封
- 臂：dv_S1/S4/S8/F8/P8（数字=目标块幅度 a；方向族 S/F/P；b=1）
- 训练：seed **3407**，2048 步/臂，4,194,304 tokens/臂；betas/(0.9,0.999)/eps1e-8/wd0.01/clip1.0/warmup+cosine 同 V16-B
- baseline：dv_S1（=V16 b_S1_plain_lr0.001 位级复刻）与 dv_S8（纯幅度对照）
- 门（frozen_v1.json:122-123 预注册）：`F8−S8 ≤ −1e-3 且 CI95 全负` 且 `P8−S8 ≤ −1e-3 且 CI95 全负` → PASS；反转规则（:140）；layer-2 双锁（:199-209，seeds 3411-3413×4096 步）；禁止复用 V16 final_holdout（:222-227）
- 指标：token 加权 NLL + doc 级 paired bootstrap（1000 次，seed 9071）
- 预算：授权 90min；成功轮实测 884.6s + 重放开销 + screen 2.5min ≈ 37min（LAYER1_DIRECTION_EFFECT.md:59-67）

## E2 事故与偏差（全部留痕）

1. 锚点观测设备 bug：一行修复（`RUNNER_FIX_LOG.md`）
2. checkpoints 目录缺失首跑崩溃（`raw/layer1_run.out:1-17` traceback 全录）
3. 进程误杀 ×2；append-only done 机制 + 确定性重放字节级验证（LAYER1_UPDATE_NORM_AUDIT.md:76-78）
4. anchors.jsonl 7 行重放重复（披露并在分析端去重）
5. frozen_v1.json 双哈希重冻结（19:39，holdout_v2 铸成后、首跑前；SHA256SUMS.txt 两行留痕；旧版字节未存档）；frozen_v1.json:206 holdout 注记与 manifest 轻微不同步（以 manifest 为准）

## E3 未执行（计划内、待授权）

Layer-2：seeds 3411/3412/3413 × 4096 步 × {S8, F8, P8}（p_inclusion 强制含 P8，修复 V16 screen 事故）+ holdout_v2 一次开封。冻结协议已就绪，未运行。
