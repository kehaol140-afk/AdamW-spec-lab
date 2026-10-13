# THEORY — adamw-direction-30m

## 1. 科学问题

「方向残余的规模迁移」（30M_FROZEN_PROTOCOL.md:7）：4.66M 上 Layer-1 确认的方向收益（−0.204/−0.235）在 ×6.45 参数规模上是否存续（SCALE_ONLY / DIRECTION_GAIN / NO_TRANSFER 三择一）？工程定位：改进后 AdamW 与调优 Muon 的差距。

## 2. 优化器与更新对象

- 模型：MiniMind 30,025,216 参数（h512/8L/int1664/8Q-4KV/head64/vocab6400），`src/s30m_core.py` 构建（模型定义依赖 baselines/minimind-base-code）
- 目标块：**两处** FFN down_proj `(512,1664)`（L1 与 L2 层，各 851,968 参数，占 2.84%/块）——层位置稳健性首次纳入
- 双路由：AdamW 张量 56 个（含目标块替换）+ Muon 张量 34 个（`s30m_core.py:1-20`；Muon=torch 内置 `_muon.py` SHA `38e95918…`）

## 3. 核心公式

同 V16/DV（`src/dv_core.py` 与 DV 目录逐字节相同，sha `b8943249…`）：

```
S: D=A(target);  F: D=g_clip·‖A‖/‖g_clip‖;  P: D=(g_clip/(√v̂+ε))·‖A‖/‖·‖
有效更新：target=η·a·D（a∈{1,2,4,8} 网格），其余 AdamW 块=η·b·A（b=1）
Muon 参照臂：O=NS5(buf)，momentum 0.95/nesterov，LR 网格 {0.005,0.01,0.02}
```

数据通道：TokenStream 8192 tok/步（micro8×accum4×seq256）；自铸缓存 17M tokens（**前 8.5M 与 V16 缓存逐字节相同**，`cache_manifest.json` v16_prefix_match:true——跨规模数据身份链）。常量逐字段复刻 V16 约定（betas/wd/clip/cosine→0.1×peak）。

## 4. 修改前后差异与动机

同 V16 THEORY §4。本实验增量：(i) 规模 ×6.45；(ii) LR 网格独立重调（0.0003/0.001/0.003）；(iii) a 网格重调（1/2/4/8）；(iv) 层位置（L1/L2）；(v) 调优 Muon 参照（公平基线要求来自 GTTR 教训「必须打败调过参的 baseline」）。

## 5. 已验证 vs 仍假设

已验证：DIRECTION_GAIN（两层、CI 全负）；量级收缩（−0.204→−0.038@L1，禁止跨模型比较绝对值）；幅度有害复现（S8−S1=+0.081）；CRN（13 run 第一步 loss_sum 位级 72631.2207）；Muon 领先 0.2725（CI 全负）。
仍假设：跨 seed（T2 未跑）；更大规模外推（100M 立项门「不具备」）；方向收益的机制解释（→V16-D 材料，未跨规模采集锚点）。
