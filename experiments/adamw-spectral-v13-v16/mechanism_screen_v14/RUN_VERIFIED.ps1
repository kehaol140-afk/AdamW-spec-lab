$ErrorActionPreference = 'Stop'
$env:USE_TF = '0'
$env:PYTHONDONTWRITEBYTECODE = '1'
$Python = 'D:\pytorch\anaconda\envs\py39\python.exe'
$Root = 'E:\大模型预训练\adamw_spectral_pilot'
Set-Location -LiteralPath $Root
New-Item -ItemType Directory -Force -Path mechanism_screen_v14\logs | Out-Null
& $Python -m unittest mechanism_screen_v14.test_screen -v 2>&1 | Tee-Object -FilePath mechanism_screen_v14\tests_cpu.log
if ($LASTEXITCODE -ne 0) { throw 'CPU tests failed' }
foreach ($Phase in @('prepare','dev','select','confirm','finalize')) {
  & $Python -m mechanism_screen_v14.screen $Phase 2>&1 | Tee-Object -FilePath "mechanism_screen_v14\logs\$Phase.log"
  if ($LASTEXITCODE -ne 0) { throw "V14 phase failed: $Phase" }
}
