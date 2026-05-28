$ErrorActionPreference = "Stop"

$TestRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ReleaseRoot = Join-Path (Split-Path -Parent $TestRoot) "test-demo-release"
$ReleaseScript = Join-Path $ReleaseRoot "quick_start_release.ps1"
$TestScript = Join-Path $TestRoot "quick_start_test.ps1"

if (-not (Test-Path $ReleaseScript)) {
    throw "Release script not found: $ReleaseScript"
}

& $ReleaseScript
& $TestScript

Write-Host ""
Write-Host "Release frontend: http://127.0.0.1:5173"
Write-Host "Test frontend:    http://127.0.0.1:5174"
