$ErrorActionPreference = 'Stop'
$Lab = 'E:\大模型预训练\adamw_spectral_pilot'
$Py = 'D:\pytorch\anaconda\envs\py39\python.exe'
$Cfg = 'config\local_v1_lowmem.json'
Set-Location -LiteralPath $Lab
$env:USE_TF = '0'
$env:PYTHONUTF8 = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'

& $Py run.py preflight --config $Cfg --root 'E:\大模型预训练'
if ($LASTEXITCODE -ne 0) { throw 'preflight failed; stop' }
& $Py run.py prepare --config $Cfg
if ($LASTEXITCODE -ne 0) { throw 'prepare failed; stop' }
& $Py -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw 'CPU tests failed; stop' }
& $Py run.py smoke --config $Cfg
if ($LASTEXITCODE -ne 0) { throw 'GPU smoke/replay failed; stop' }
