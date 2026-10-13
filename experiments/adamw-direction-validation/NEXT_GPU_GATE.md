# NEXT_GPU_GATE — GPU 执行前置门（当前状态：BLOCKED-BY-DESIGN）

本项目本轮**零 GPU 消耗**（仅 CPU 单元测试）；正式训练需以下全部条件满足。

## 1. 资源门

- [ ] SpecMuon P1 完成并释放 GPU0（本轮 GPU0 优先权归 SpecMuon，已遵守）
- [ ] GPU0 空闲确认 + 共享服务器许可（GPU1 vLLM 持续礼让）
- [ ] 连续窗口：layer-1 需 ~1.5h、layer-2 需 ~6h 墙钟

## 2. 数据门（layer-2 硬依赖；layer-1 无此依赖）

- [ ] **语料上传**：`pretrain_t2t_mini.jsonl`（44MB，指纹 `4403b6717352fcefe2b8ce50b47a29248085ac811eff6121e6114cf26e9c3d31`，Windows 源 `E:\大模型预训练\` 下）→ 放 `adamw_direction_validation/_incoming/`
- [ ] 校验指纹 → 重分词 1068 个候选文档（tokenizer `71f32c68…` 在位）→ 验证 doc_id=SHA256(normalized text) 与账本一致
- [ ] 跨账本去污（GAP_2）：1536 个已用 eval 文档 + 5109 个 train 文档按当前归一化重哈希，剔除内容级重叠候选
- [ ] 铸造 holdout_v2 ≤1024 文档 → 封印（写 `holdout_v2_manifest.json{status:MINTED_SEALED}`）
- 若用户决定不上传语料：layer-1 仍可执行（旧 dev/screen 复用已披露）；**layer-2 无法执行**（无独立 holdout，不得伪造）——需明示接受或放弃

## 3. 授权门

- [ ] 用户明确批准 layer-1 开跑（~1.5h GPU）
- [ ] layer-1 gate 结果 + 用户批准后开 layer-2（~6h GPU）
- [ ] 如需延长任何预算/墙钟：报告 + 新授权（不自行续期）

## 4. 通过后的后续建议（本轮不执行，仅登记）

按信息价值排序，每项独立预注册后才可立项：
1. **a-边界解析**（a∈{12,16} 仅 S 臂）：a=8 仍是 V16 遗留边界，测 S 的纯幅度曲线是否饱和（~30 min GPU）
2. 模块外推：attention o_proj / 其他层 down_proj 的 S8/F8/P8 三点（区分"单块偶然收益"vs"该层特异"vs"通用"）
3. 多目标块联合（2-3 块同时 F8）：检验可加性
4. 30M MiniMind 复核（G1a 26M bank 或新训）：规模外推
5. AdamW vs Muon 同模型同预算更新几何比较（cos(g,·)/谱形状面板）——与 SpecMuon 线共享只读基础设施，实验互相独立
