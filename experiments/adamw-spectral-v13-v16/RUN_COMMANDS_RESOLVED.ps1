$ErrorActionPreference = 'Stop'
$Lab = 'E:\大模型预训练\adamw_spectral_pilot'
$Py = 'D:\pytorch\anaconda\envs\py39\python.exe'
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) { throw "Bound Python missing: $Py" }
Set-Location -LiteralPath $Lab
$env:PYTHONUTF8 = '1'
$env:USE_TF = '0'
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
        throw "Python failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

# Review bindings.local.json, preflight.json, the data manifest, and CPU logs
# before manually running this whole script. It has not been run by Codex.
Invoke-CheckedPython -Arguments @('.\run.py','preflight','--config','config\local_v1.json','--root','E:\大模型预训练')
Invoke-CheckedPython -Arguments @('.\run.py','prepare','--config','config\local_v1.json')
Invoke-CheckedPython -Arguments @('-m','unittest','discover','-s','tests','-v')
Invoke-CheckedPython -Arguments @('.\run.py','smoke','--config','config\local_v1.json')
Invoke-CheckedPython -Arguments @('.\run.py','pilot','--config','config\local_v1.json','--stage','discover')

$Gate = Get-Content -LiteralPath '.\results\local_gate.json' -Raw -Encoding UTF8 | ConvertFrom-Json
if ($Gate.next_action -eq 'RUN_CONFIRM') {
    Invoke-CheckedPython -Arguments @('.\run.py','pilot','--config','config\local_v1.json','--stage','confirm')
}
Invoke-CheckedPython -Arguments @('.\run.py','report','--config','config\local_v1.json')
Invoke-CheckedPython -Arguments @('.\run.py','bundle','--config','config\local_v1.json','--light')
