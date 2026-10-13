# Windows PowerShell 操作说明

本包仅含研究协议、配置与代理提示词。**本包不包含 run.py，也没有已运行的实验结果。** 下列 Python 接口需要先由 Codex 实现并测试，不是 MiniMind 原生命令。

## 1. 放置材料并启动 Codex

将 `adamw_spectral_pilot_handoff.zip` 保存到当前用户的“下载”文件夹。压缩包内顶层目录是 `adamw_spectral_pilot`。PowerShell 中执行：

```powershell
$ErrorActionPreference = 'Stop'
$Root = 'E:\大模型预训练'
$Zip = Join-Path $HOME 'Downloads\adamw_spectral_pilot_handoff.zip'
$Lab = Join-Path $Root 'adamw_spectral_pilot'

if (-not (Test-Path -LiteralPath $Root -PathType Container)) {
    throw "工作根目录不存在：$Root"
}
if (-not (Test-Path -LiteralPath $Zip -PathType Leaf)) {
    throw "请先保存压缩包到：$Zip"
}
if (Test-Path -LiteralPath $Lab) {
    throw "目标目录已经存在，请先检查内容，不要覆盖旧实验：$Lab"
}
Expand-Archive -LiteralPath $Zip -DestinationPath $Root
Set-Location -LiteralPath $Lab
codex
```

在 Codex 中发送：

```text
读取当前目录 PROMPT_CODEX.md、PROTOCOL_LOCAL_V1.md、config/local_v1.json 和 RUN_WINDOWS.md，按其中合同实现。
先实际检查 E:\大模型预训练 下的 MiniMind、tokenizer、预训练数据和现有 Python 环境；原仓库和旧实验只读。
本次只做实现、CPU数值测试和受限真实数据smoke，不启动512步正式训练。交付完整日志、CODEX_HANDOFF和绑定解释器的运行命令。
```

用户历史用过的环境名称不保证仍可用，Codex 必须实际检查。不要为了本实验默认重装 CUDA/PyTorch。CLI 已安装并完成登录为前提；未配置 CLI 时使用你已有的 Codex 界面打开同一目录和提示词即可。

## 2. Codex 完成后，由 OpenCode 审核与运行

Codex 不再执行 GPU 任务后，在新的 PowerShell 窗口执行：

```powershell
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
opencode
```

在 OpenCode 中发送：

```text
读取 PROMPT_OPENCODE.md、PROTOCOL_LOCAL_V1.md、config/local_v1.json、RUN_WINDOWS.md 和 reports/CODEX_HANDOFF.md。
实际审核数值产物、代码、状态隔离、数据拆分和资源用量后，按门禁串行运行 discover；只在 next_action=RUN_CONFIRM 时运行 confirm。
不扩网格、不换主指标、不并发训练、不自动启动短分支或服务器实验。结束后生成原始表、LOCAL_FINDINGS、OPENCODE_AUDIT与轻量结果包。
```

不要让 Codex 与 OpenCode 同时运行两个训练进程。OpenCode 应使用 `bindings.local.json` 中已经核实的 Python，而非随意新开一个环境。

## 3. 实现完成后的手动执行接口

此部分仅在 Codex 已完成实现并产生绑定文件后有效。与 OpenCode 执行二选一，不同时运行。

```powershell
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'

if (-not (Test-Path '.\run.py')) { throw 'run.py 尚未由 Codex 实现。' }
if (-not (Test-Path '.\bindings.local.json')) { throw '缺少经核实的环境绑定。' }
$Binding = Get-Content '.\bindings.local.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$Py = [string]$Binding.python_executable
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Python 路径不可用：$Py" }

$env:PYTHONUTF8 = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $script:Py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python阶段失败，退出码=$LASTEXITCODE；停止后续实验。参数：$Arguments"
    }
}

Invoke-CheckedPython -Arguments @('.\run.py','preflight','--config','config\local_v1.json','--root','E:\大模型预训练')
Invoke-CheckedPython -Arguments @('.\run.py','prepare','--config','config\local_v1.json')
Invoke-CheckedPython -Arguments @('-m','unittest','discover','-s','tests','-v')
Invoke-CheckedPython -Arguments @('.\run.py','smoke','--config','config\local_v1.json')
Invoke-CheckedPython -Arguments @('.\run.py','pilot','--config','config\local_v1.json','--stage','discover')

$Gate = Get-Content '.\results\local_gate.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Gate.next_action -eq 'RUN_CONFIRM') {
    Invoke-CheckedPython -Arguments @('.\run.py','pilot','--config','config\local_v1.json','--stage','confirm')
} else {
    Write-Host "未进入确认：$($Gate.verdict) / $($Gate.next_action)"
}

Invoke-CheckedPython -Arguments @('.\run.py','report','--config','config\local_v1.json')
Invoke-CheckedPython -Arguments @('.\run.py','bundle','--config','config\local_v1.json','--light')
```

若任一阶段失败，上述脚本不会继续训练。修复前先保留日志；需要收集失败报告时，可以单独执行绑定 Python 的 report 与 bundle，不能直接跳过数值门。

## 4. 结果应当回答什么

查看 `reports/LOCAL_FINDINGS.md`、`results/local_gate.json` 和 `results/probe_confirm.csv`。最重要的是：主 cell 的冻结谱方案是否超过本协议较强的标量控制，是否超过数值底噪，是否在另一个 seed 上保持，以及最近邻方向控制的结果如何。

此时即使通过，也只能决定是否进入下一轮短分支，不能宣布完整新优化器有效。移到服务器时再绑定路径和环境，复跑 P0；不要自动换最新仓库、tokenizer 或默认大模型。

## 5. CLI 来源

Codex 官方 CLI 文档：https://developers.openai.com/codex/cli/
OpenCode CLI 文档：https://opencode.ai/docs/cli/

上面的启动方式不要求更改全局代理权限、关闭安全审批或启用无人监管模式。
