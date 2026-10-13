# NEW_HOLDOUT_READINESS — 新独立确认集就绪报告

日期：2026-10-11。零 GPU；全程只读旧资产；新产物仅写入本项目目录。

## 1. 六项验证结果（verify_corpus.py，全部 PASS）

| # | 验证 | 结果 |
|---|---|---|
| V1 | 用服务器语料前缀重放分区扫描 → 与 `artifacts/doc_roles.jsonl` 逐字节比对 | ✅ **逐字节相同**（7,256 文档；role_for 规范化+哈希+cuts 全复现） |
| V2 | 全语料扫描（1,269,983 文档 / 1.24GB）定位 886 个跨账本 legacy eval id | ✅ **886/886 全部定位**——它们来自早期更大扫描窗口、位于本次未消费区域之外；GAP_2 从"未知风险"降为"已定位且可内容比对" |
| V3 | 1068 个候选文档 id 在前缀中的存在性 | ✅ 1068/1068 在账本内且角色为非 train 的三个评估角色（branch_eval_reserved 202 + probe_confirm 850 + probe_select 16） |
| V4 | tokenizer 链验证：冻结 tokenizer.json（`71f32c68…`）重编码 dev 文档 vs 冻结 token ids | ✅ **16/16 精确**（3 个初判"不匹配"实为协议截断：eval 文档截到 [BOS]+256+[截断]，共同前缀全同——非版本漂移） |
| V5 | **内容级去污**：1068 候选的完整 token 序列 vs 全部 1536 个已曝光 V16 评估文档（development/screen/final_holdout，含已定位的 886 legacy）的 token 序列 | ✅ **0 重复** |
| V6 | 训练缓存去重：每候选前 32 token 在 8.5M train 缓存流中搜索 | ✅ **0/1068 命中**（候选从未进入训练数据） |

## 2. holdout_v2 已铸造并封存

- **文件**：`splits/holdout_v2.jsonl`——1024 文档（1068 验证池的字典序前 1024，规则预冻结）；44 个余量单独列 `holdout_v2_reserve_ids.txt`（不属于 holdout、不得用于选择）
- **SHA256**：`3e0f76536bfeab99c629dfbb4c01f361e9b37daac34f9d39910c61aed17801f1`；共 203,403 token
- **封印**：`holdout_v2_manifest.json{status: MINTED_SEALED}`（含验证链引用与开封规则）
- **开封规则（冻结）**：仅在 layer-2 全部 12 个端点身份锁定后由 `dv_runner.py layer2_holdout` **一次性开封**（代码级：状态非 MINTED_SEALED 即拒绝；哈希不符即拒绝）；开封后不得再调参
- **与旧 holdout 关系**：V16 final_holdout（1024 文档，2026-10-10 已开封）**永久烧毁**，不作为任何新确认依据；holdout_v2 与其零内容交集（V5 实证）

## 3. Layer-1 评估口径（无新依赖）

- 主评估：旧 development 256（开发级，其 V16 B 选优历史已在协议中披露）+ 次级 screen 256；不触碰任何 holdout
- layer-1 的选优/门判定只用 development；screen 仅作一致性观察

## 4. 阻塞清单

- **无数据阻塞**（语料已就位并验证；holdout_v2 已封存）
- 剩余阻塞仅为 GPU 排程与授权（见 DIRECTION_GPU_EXECUTION_READY.md §4）：SpecMuon P1 释放 GPU0 + 用户批准
