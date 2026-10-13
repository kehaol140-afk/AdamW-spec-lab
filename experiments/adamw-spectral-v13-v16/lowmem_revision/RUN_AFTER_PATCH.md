# 本机操作

本包不是自动覆盖补丁。将本包文件解压到：
`E:\大模型预训练\adamw_spectral_pilot\lowmem_revision\`
使 `lowmem_revision\PROMPT_CODEX_LOWMEM.md` 真实存在。

## 1. Codex实施

在PowerShell进入现有工作区并启动已安装Codex：

```powershell
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
codex
```

发送：
“读取lowmem_revision/PROMPT_CODEX_LOWMEM.md和LOWMEM_AMENDMENT.md，执行最小资源修订、测试与真实smoke。不要启动discover/confirm，不改科学设置，不重新下载数据。”

不要在Codex测试的同时手动运行另一条GPU作业。

## 2. OpenCode监督

Codex退出/停止GPU任务并交付后：

```powershell
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
opencode
```

发送：
“读取lowmem_revision/PROMPT_OPENCODE_LOWMEM.md，核验LOWMEM_HANDOFF与真实smoke指纹。通过后按新配置串行discover，gate允许才confirm，不重复已经匹配的smoke，不突破原预算。”

## 3. 修订和归档完成后的底层命令

下列命令仍用原CLI；不是要求用户与代理重复执行。以本地bindings.local.json为准，上传报告中的解释器为D:\pytorch\anaconda\envs\py39\python.exe。

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
$Py = 'D:\pytorch\anaconda\envs\py39\python.exe'
$Cfg = 'config/local_v1_lowmem.json'
$env:USE_TF = '0'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

& $Py run.py preflight --config $Cfg --root 'E:\大模型预训练'
if ($LASTEXITCODE -ne 0) { throw 'preflight未通过：保留日志，停止。' }
& $Py run.py prepare --config $Cfg
if ($LASTEXITCODE -ne 0) { throw 'prepare未通过：停止。' }
& $Py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw '测试未通过：停止。' }
& $Py run.py smoke --config $Cfg
if ($LASTEXITCODE -ne 0) { throw '真实smoke未通过：停止，不启动discover。' }
```

smoke通过后、OpenCode审核完成，正式命令为：

```powershell
& $Py run.py pilot --config $Cfg --stage discover
if ($LASTEXITCODE -ne 0) { throw 'discover异常退出：停止并保留日志。' }
$Gate = Get-Content -LiteralPath 'results/local_gate.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Gate.next_action -eq 'RUN_CONFIRM') {
    & $Py run.py pilot --config $Cfg --stage confirm
    if ($LASTEXITCODE -ne 0) { throw 'confirm异常退出：停止并保留日志。' }
}
& $Py run.py report --config $Cfg
if ($LASTEXITCODE -ne 0) { throw 'report失败。' }
& $Py run.py bundle --config $Cfg --light
if ($LASTEXITCODE -ne 0) { throw 'bundle失败。' }
```

CLI自身必须继续校验gate与全部指纹；PowerShell条件不替代程序检查。原report会生成较简报告，OpenCode在report后补全LOCAL_FINDINGS，再重新bundle，避免详细结果被覆盖。

Codex和OpenCode都已运行的步骤不要重复手动执行。不要为了越过门槛关闭整个资源检查、删除smoke门禁或改科学阈值。
