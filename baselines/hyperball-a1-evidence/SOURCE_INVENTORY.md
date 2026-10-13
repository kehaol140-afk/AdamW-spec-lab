# SOURCE_INVENTORY — Hyperball/MuonH 离线 A0 来源清单

日期：2026-09-30。zip：`hyperball_sources.zip` SHA256 `6c083cbc…91b9f34e180`；
解压至 `third_party/hyperball_sources/`（Windows 反斜杠路径已规范化）。
内含 `SHA256SUMS.txt` **9/9 全部 PASS**（其中 MuonH log 行初检误报系清单分隔符
解析问题，直接哈希核对 match=True）。

| # | 文件 | 类型 | 大小 | SHA256（前12） | 状态 |
|---|---|---|---|---|---|
| 1 | paper/Hyperball_Optimization_2606.16899.pdf | paper (22页, v1) | 2.88MB | 5d8443867e57 | **VERIFIED** |
| 2 | track3/README.md | README（46条结果史+规则） | 43.8KB | 03d496e8d315 | **VERIFIED** |
| 3 | prs/PR267_MuonH.patch | PR patch（3 commits, 39 diff 段） | 6.89MB | 297083420aef | **VERIFIED** |
| 4 | prs/PR323_Tuned_Muon.patch | PR patch（11 diff 段） | 185KB | c1cc7f5c3f04 | **VERIFIED** |
| 5 | prs/PR324_TunedAux_MuonH.patch | PR patch（12 diff 段） | 222KB | 89992c06a166 | **VERIFIED** |
| 6 | logs/MuonH_3325_capture_full.txt | **capture-full log（含完整可执行脚本+逐步记录）** | 193KB | 2068a4f0137e | **VERIFIED** |
| 7 | logs/TunedMuon_3250.txt | capture-full log（#36） | 17.2KB | 0755b91d9ac3 | **VERIFIED**（#36 主 log 另有官方目录 20260610_tuned_baseline_3250/263ea3c4…，本包为此前版本——首末 val 与 README 一致） |
| 8 | logs/TunedAux_MuonH_3250.txt | capture-full log（#37） | 19.1KB | 30ea192771f8 | **VERIFIED**（官方目录 20260611_muonh_tuned_aux_3250/f6649181…，未随包附——以本 log 内嵌脚本为准） |
| 9 | source/train_gpt_simple_master.py | trainer snapshot（**current master**，非历史版本） | 14.3KB | d39470e5b053 | **VERIFIED** |

## 关键判断

1. **PR patch 是否含足够上下文**：是——PR267 含完整提交头（base commit
   `269edcdd…`、作者 Kaiyue Wen、日期 2026-04-29）、README diff、5 个完整 seed
   log 新增文件；PR323/PR324 结构相同。每个 PR 的修改内容可从 patch 完整确定。
2. **能否识别 base/merge commit**：**能识别 base commit**（patch 头 `From
   <sha>`）；merge commit 与运行时 commit **UNKNOWN**（patch 不含 merge 信息）。
3. **trainer snapshot 是否与三个 PR 运行版本相同**：**否**——`train_gpt_simple_
   master.py` 与 #36 log 脚本**逐字等价（仅 1 空行差）**，即 master=tuned Muon
   #36 版本；但 MuonH(#5) 与 tuned-aux(#37) 的**运行版本内嵌于各自 capture-full
   log**，与 master 有实质差异（优化器/init/schedule 不同）。
   **A1 运行事实应直接取 log 内嵌脚本**（README 明示复现方式=从 log 提取
   `===` 之前的脚本）。
4. **是否只有 current master 而非 historical snapshot**：是（snapshot=master≈
   #36；#5/#37 靠 log 内嵌脚本还原）。
5. **是否拥有运行 canonical benchmark 的完整 repo**：**接近**——训练脚本/
   dataloader/模型/评价全部在单文件 log 脚本内（自包含）；缺 `data/
   cached_fineweb10B.py`（数据下载/预处理脚本）与 benchmark harness 目录结构。
   README quickstart 显示仅需 repo + `pip install torch==2.11 huggingface_hub`
   + `data/cached_fineweb10B.py 20`。
6. **训练数据**：**无**。Track3 需 `data/fineweb10B/fineweb_train_*.bin`/
   `fineweb_val_*.bin`（GPT-2 tokenizer 预分词的二进制 shard）。服务器 HF
   cache 有 fineweb/fineweb-edu 条目但**无 Track3 .bin 格式**（bin 搜索 0 命中）；
   HF hub 网络不可达。
7. **tokenizer**：训练不需要显式 tokenizer（.bin 已是 token id）；生成 .bin
   需 GPT-2 tokenizer（HF 不可达）。
8. **评价脚本**：val 评价内嵌于训练脚本（val_tokens=20×524288，
   每 125 步）；无独立 benchmark harness——**够用**。

## 三层身份当前状态

- ALGORITHM_IMPLEMENTATION：**可从 #5/#37 log 内嵌脚本完整恢复**（见
  HYPERBALL_IMPLEMENTATION_AUDIT.md）。
- PUBLIC_BENCHMARK_REPRODUCTION：模型/优化器/评价可从脚本恢复；**缺数据 .bin**
  ——阻塞点。
- REDUCED_SCALE_TRANSFER：未运行；不称复现。
