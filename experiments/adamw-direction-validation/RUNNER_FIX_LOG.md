# RUNNER_FIX_LOG — 运行器热修复记录（仅观测层，零语义改动）

日期：2026-10-11。触发：layer-1 首次启动（dv_S1 step 204，首个锚点步）崩溃。

## 事故

- `dv_runner.py::anchor_scalars` 中 `mdd = md.double()` 的 `md` 来自 `dv_core.apply_step` 的
  `meta["_m_plus"][name] = m_plus.detach().cpu().clone()`（CPU 张量），而 `gd` 在 cuda:0 →
  `torch.dot(gd, mdd)` 设备不匹配 RuntimeError。
- 训练/更新/评估数学**未受影响**：anchor_recorder 在 apply_step 全部位移计算完成后才被调用；
  崩溃前 dv_S1 已完成 1–203 步，与 V16 `b_S1_plain_lr0.001` 对应步
  （eta/loss_sum/token_count/nll/family_used/A_norm/D_norm/q_t/zero_norm_fallback 全字段）
  **203/203 位级一致**（复刻门早期核验通过，见 LAYER1_DIRECTION_EFFECT.md 附录）。

## 修复（唯一改动，一行）

`src/dv_runner.py` anchor_scalars：
`mdd = md.double()` → `mdd = md.double().to(gd.device)`

- 不触碰：S/F/P 位移公式、q_t、schedule、数据流、评估、gate、frozen_v1.json、dv_core.py（数学内核，哈希未变 b8943249…）
- 性质：纯观测层设备搬运（锚点 cos(g,m) 需要动量在 GPU 上与梯度点积）
- 重验：CPU 8/8 测试 PASS；GPU 三矩阵 anchor_scalars 冒烟 PASS（cos_g_m=1.0 自洽）
- 重跑语义：runner 内建 `done` 断点——step 日志 append-only，已写 1–203 步不重写不覆盖；
  scratch 分支从 seed 3407 确定性重放，204 步起仅追加缺失行

## 哈希账本（追加式，原条目保留）

| 文件 | 旧 sha256（SHA256SUMS.txt 第 20 行） | 新 sha256 |
|---|---|---|
| src/dv_runner.py | 5e34b08b47cd7b26bab6b4cdc9e5ffbf7b64f49c8dbbd105c1071b3d36d49b13 | （重算后追加于 SHA256SUMS.txt） |
