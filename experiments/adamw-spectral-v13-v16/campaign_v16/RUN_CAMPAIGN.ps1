$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
$Py = 'D:\pytorch\anaconda\envs\py39\python.exe'
$env:USE_TF = '0'; $env:PYTHONUTF8 = '1'; $env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'; $env:TOKENIZERS_PARALLELISM = 'false'
$env:OMP_NUM_THREADS = '2'; $env:MKL_NUM_THREADS = '2'

& $Py -m unittest campaign_v16.test_v16 -v
if ($LASTEXITCODE -ne 0) { throw 'cpu tests failed' }
# serial persistent runner: each invocation drains as much of the frozen queue as possible
# within its window; re-invoke the same command to resume after interruption/deadline.
& $Py -m campaign_v16.run --resume --max-seconds 5400
