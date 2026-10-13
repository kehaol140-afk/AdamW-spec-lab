# CORPUS_RECOVERY_GUIDE — 语料定位与回收结论

日期：2026-10-11。结论先行：**无需上传——语料已在服务器且经密码学验证为同一文件**。本轮零 GPU、零修改（对语料与 V16 资产全程只读）。

## 1. 缺失语料的准确身份（任务书第 1 项）

| 属性 | 值 | 证据 |
|---|---|---|
| 文件名 | `pretrain_t2t_mini.jsonl` | data_manifest `source_path` |
| Windows 相对路径 | `E:\大模型预训练\adamw_spectral_pilot\data\pretrain_t2t_mini.jsonl`（相对 search_root：`adamw_spectral_pilot\data\pretrain_t2t_mini.jsonl`） | `bindings.local.json.windows_orig` |
| 格式 | JSONL，每行 `{"text": <str>}`；含 conversations/messages/instruction/response 键的行按 prepare_data 规则判 invalid（实测 0 行）；普通文本，UTF-8 | `prepare_data.py:166-172` |
| **真实大小** | **1,241,043,656 字节（≈1.24 GB）**——迁移清单所写 "44 MB" **有误** | data_manifest `source_size` |
| 全文件 SHA256 | **从未被直接记录**（见 §2 指纹勘误） | — |
| **已消费前缀 SHA256**（5,429,819 字节） | `2cb92156c151a54826974e57bbffaf0c508433f2e0fd35580d9394c8c5bdf2f7` | data_manifest `scanned_sha256`（prepare_data.py:157,220 逐行累积哈希语义） |
| 扫描状态 | `scan_complete=False`——数据制备在取得足够 train token 后提前停止，**只消费了前 5.4MB**；全部 7,256 个账本文档、8.5M 训练缓存原料与全部评估池文档都出自该前缀 | data_manifest + prepare_data 流程 |

关键含义：**研究实际依赖的语料内容 = 5.4MB 前缀**；前缀之外的 1.23GB 从未被任何 V13–V16 产物读取。

## 2. 指纹勘误（重要更正）

迁移清单/旧审计中的"源语料指纹 `4403b6717352fcefe2b8ce50b47a29248085ac811eff6121e6114cf26e9c3d31`"**不是语料哈希**。源码核实（`pilot_core.py:288`）：

```
source_fingerprint = sha_object({relative_path: sha_file(p) for p in source_files})
# source_files = [repo/model/model_minimind.py, repo/trainer/train_pretrain.py]
```

它是**仓库源码两文件哈希的字典 digest**（绑定时误当作语料指纹转述）。语料真身份锚 = `source_size`（精确字节）+ `scanned_sha256`（前缀逐行哈希）+ `doc_assignments_sha256`（账本哈希）——三者均可独立复核。

## 3. 服务器现有资产排查（任务书第 2 项）——命中

- 全盘按文件名/体积搜索：**`/home/b205/minimind-master/dataset/pretrain_t2t_mini.jsonl`**
- `size = 1,241,043,656`（与 `source_size` **逐字节相等**）
- 前 5,429,819 字节 SHA256 = `2cb92156…` **与 scanned_sha256 完全一致**
- 用服务器副本**重放分区扫描**（prepare_data 的 role_for：规范化空白 → SHA256 → 前 16 hex mod 100 → cuts 70/76/82/94/97/100）：7,256 文档，输出与 `artifacts/doc_roles.jsonl` **逐字节相同**
- 全文件 SHA256（服务器副本）= `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`（原 Windows 全文件哈希无记录可比；已消费区域已证一致，未消费区域与研究无关）

→ **判定：同一语料已在服务器，零上传需求。**

## 4. 若未来仍需从 Windows 上传（任务书第 3 项，模板备用）

源定位（Windows 本地 PowerShell）：

```powershell
Get-Item 'E:\大模型预训练\adamw_spectral_pilot\data\pretrain_t2t_mini.jsonl' | Select-Object FullName, Length
# 必须输出 Length = 1241043656；不符则不是本文件（不得改名顶替）
Get-FileHash 'E:\大模型预训练\adamw_spectral_pilot\data\pretrain_t2t_mini.jsonl' -Algorithm SHA256
```

上传（WSL scp，目标目录只新建不覆盖）：

```powershell
scp 'E:\大模型预训练\adamw_spectral_pilot\data\pretrain_t2t_mini.jsonl' `
    b205@<server>:/home/b205/projects/training_dynamics_lab/adamw_direction_validation/_incoming/pretrain_t2t_mini.jsonl
```

服务器验收门（全部通过才算到位）：

```bash
test "$(stat -c%s _incoming/pretrain_t2t_mini.jsonl)" = "1241043656" || echo REJECT_SIZE
head -c 5429819 _incoming/pretrain_t2t_mini.jsonl | sha256sum   # 必须 = 2cb92156c151a548...
# 然后重跑 verify_corpus.py，V1 必须 PASS（账本逐字节复现）
```

## 5. 复核程序与输出

验证链脚本：`verify_corpus.py`（CPU 只读）；结果：`verification_out/corpus_verification.json`。六项验证见 NEW_HOLDOUT_READINESS.md。
