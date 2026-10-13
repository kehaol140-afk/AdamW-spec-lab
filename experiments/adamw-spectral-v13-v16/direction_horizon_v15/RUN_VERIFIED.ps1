$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
$Py = 'D:\pytorch\anaconda\envs\py39\python.exe'
$env:USE_TF = '0'; $env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'; $env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'

& $Py -m unittest discover -s direction_horizon_v15 -p 'test_v15.py' -v
if ($LASTEXITCODE -ne 0) { throw 'cpu tests failed' }
& $Py -m direction_horizon_v15.train prepare
if ($LASTEXITCODE -ne 0) { throw 'prepare failed' }
& $Py -m direction_horizon_v15.train smoke
if ($LASTEXITCODE -ne 0) { throw 'smoke failed' }
& $Py -m direction_horizon_v15.train dev
if ($LASTEXITCODE -ne 0) { throw 'dev failed' }
& $Py -m direction_horizon_v15.train select
if ($LASTEXITCODE -ne 0) { throw 'select failed' }
& $Py -m direction_horizon_v15.train confirm
if ($LASTEXITCODE -ne 0) { throw 'confirm failed' }
& $Py -m direction_horizon_v15.train finalize
if ($LASTEXITCODE -ne 0) { throw 'finalize failed' }
