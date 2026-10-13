param([switch]$SkipGpu)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath 'E:\大模型预训练\adamw_spectral_pilot'
$pythonExe = 'D:\pytorch\anaconda\envs\py39\python.exe'
$env:USE_TF = '0'
$out = 'readiness_audit_v13'
New-Item -ItemType Directory -Force -Path "$out\logs" | Out-Null
& $pythonExe -m readiness_audit_v13.cli prepare 2>&1 | Tee-Object -FilePath "$out\logs\prepare.log"
Set-Content -LiteralPath "$out\logs\prepare.exitcode" -Value $LASTEXITCODE
if ($LASTEXITCODE -ne 0) { throw 'V13 prepare failed' }
& $pythonExe -m unittest readiness_audit_v13.tests_v13 -v 2>&1 | Tee-Object -FilePath "$out\logs\cpu_tests.log"
Set-Content -LiteralPath "$out\logs\cpu_tests.exitcode" -Value $LASTEXITCODE
if ($LASTEXITCODE -ne 0) { throw 'V13 CPU tests failed' }
& $pythonExe -m readiness_audit_v13.run_parent_tests_isolated 2>&1 | Tee-Object -FilePath "$out\logs\parent_cpu_tests_isolated.log"
Set-Content -LiteralPath "$out\logs\parent_cpu_tests_isolated.exitcode" -Value $LASTEXITCODE
if ($LASTEXITCODE -ne 0) { throw 'Parent CPU tests failed' }
if (-not $SkipGpu) {
  try {
    & $pythonExe -m readiness_audit_v13.cli calibrate-local 2>&1 | Tee-Object -FilePath "$out\logs\calibrate_local.log"
    Set-Content -LiteralPath "$out\logs\calibrate_local.exitcode" -Value $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'V13 local calibration failed' }
    & $pythonExe -m readiness_audit_v13.cli calibrate-cross 2>&1 | Tee-Object -FilePath "$out\logs\calibrate_cross.log"
    Set-Content -LiteralPath "$out\logs\calibrate_cross.exitcode" -Value $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'V13 cross-process calibration failed' }
    & $pythonExe -m readiness_audit_v13.cli measure 2>&1 | Tee-Object -FilePath "$out\logs\measure.log"
    Set-Content -LiteralPath "$out\logs\measure.exitcode" -Value $LASTEXITCODE
    if ($LASTEXITCODE -ne 0) { throw 'V13 measurement stopped' }
  } finally {
    & $pythonExe -m readiness_audit_v13.cli finalize 2>&1 | Tee-Object -FilePath "$out\logs\finalize.log"
  }
} else {
  & $pythonExe -m readiness_audit_v13.cli finalize 2>&1 | Tee-Object -FilePath "$out\logs\finalize.log"
}
