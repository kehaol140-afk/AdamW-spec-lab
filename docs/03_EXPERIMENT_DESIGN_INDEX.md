# 03_EXPERIMENT_DESIGN_INDEX — 实验设计索引

机读版：[EXPERIMENT_INDEX.csv](EXPERIMENT_INDEX.csv)（35 阶段行，受控枚举）；运行级：各目录 RUN_INDEX.csv（六条核心研究线 283 行；Hyperball baseline 另有 4 行）。本页给「设计→门→证据」导航。

| 线/阶段 | 设计要点 | 预注册门/判据 | 证据（相对路径） |
|---|---|---|---|
| V13 测量 | 四状态谱测量 4 窗口 | 完成 | experiments/adamw-spectral-v13-v16/readiness_audit_v13/results/ |
| V14 rank-1 | dev128+confirm20 | CI 与 0 分离 | 同上 mechanism_screen_v14/（findings: SCALE_RANGE_UNRESOLVED） |
| V15 方向-视界 | 三族×(a,b)×H 短分支 | 网格内信号 | 同上 direction_horizon_v15/ |
| V16-A1 | 36 网格（父 step256 分叉 256 步） | development 选优 | campaign_v16/raw/eval/a1_* |
| V16-A2 | 4 臂范数互换 | 方向保留率 | campaign_v16/raw/eval/a2_* + interpretation/V16_A2_DECISION.md |
| V16-B | 18 从头（9 固定+9 预写扩展） | 路由规则 1-4 | campaign_v16/raw/eval/b_* + V16_B_RESULTS.md |
| V16-C | 2 配方×3 seed×4096+holdout 一次开封 | PRELIMINARY_SEED_ROBUST 四条件 | c_*__H4096__final_holdout.jsonl + V16_C_SEED_RESULTS.md |
| V16-D | B 内嵌锚点+SVD36 | 机制材料 | raw/anchors.jsonl + V16_FINAL_MECHANISM.md |
| DV Layer-1 | 5 臂匹配矩阵 2048 步 | F8−S8/P8−S8≤−1e-3 CI 全负 | adamw-direction-validation/raw/ + layer1_gate.json |
| DV Layer-2 | 3 seed×4096+holdout_v2（**未执行**） | 双锁 | 协议 frozen_v1.json:199-209 |
| 30M T1 | 13 臂（LR/a/两层/Muon） | 主门 CI 全负 | adamw-direction-30m/raw/ + frozen_30m_t1.json |
| 30M T2 | 多 seed（**未执行**） | — | 协议 :44 禁止本轮 |
| MSA P0/P1 | 28 单组重分配+幅度对照 | δ=1e-5 | muon-ns-allocation/outputs/p0_p1_v1_1/ |
| MSA R0/R1 | 统一时钟/组倍率 8 比较 | R0 gate；配对 CI | outputs/r0_r1_v1_1/ |
| MSA R2 | N/AH/L/D 256 步 | endpoint_lock | outputs/r2_amplitude_continuation_v1/ |
| MSA R2F(+勘误) | B_L 归因 40 run | d_trigger τ=1e-3 | outputs/r2f_*/ + defect_archive/ |
| MSA R2F-NL | 非线性尾部 | τ=1e-3 | outputs/r2f_nl_tail_v1/nl_effects.json |
| SM P0-A/B | 56+1624 矩阵谱审计 | H1/H2/H4 断言 | specmuon-pilot/results/p0{a,b}/ |
| SM P1 | 7 臂×2 恢复点 | H5 三重门 | specmuon-pilot/results/p1/ |
| CAM 100M | 15 臂 4096 步 | C_A vs 锚/momentum/C_B | context-aware-muon-100m/results/pilot/analysis.json |
| NorMuon | staging（零 GPU） | G-1~G-6 | 服务器 audits/context_muon_baselines_20261012 |
| HB A1 | 3250 步双臂复现 | 终点差 vs 噪声 | baselines/hyperball-a1-evidence/ |
| 旧仓五线 | FR/GTTR、CA、OPS、MT、TR | 各自终局 | [RELATED_ARCHIVES.md](RELATED_ARCHIVES.md)（commit cf1558f 定位） |
