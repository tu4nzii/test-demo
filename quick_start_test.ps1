$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend\chart-demo-ui"
$LogDir = Join-Path $Root "runtime_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Stop-Port {
    param([int]$Port)
    $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    foreach ($connection in $connections) {
        if ($connection.OwningProcess -and $connection.OwningProcess -ne $PID) {
            Stop-Process -Id $connection.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

Stop-Port 8001
Stop-Port 5174

$BackendLog = Join-Path $LogDir "backend-test.log"
$FrontendLog = Join-Path $LogDir "frontend-test.log"
$BackendQ = $Backend.Replace("'", "''")
$FrontendQ = $Frontend.Replace("'", "''")
$BackendLogQ = $BackendLog.Replace("'", "''")
$FrontendLogQ = $FrontendLog.Replace("'", "''")

$BackendCommand = "& { Set-Location -LiteralPath '$BackendQ'; `$env:BACKEND_PORT='8001'; `$env:CHART_MODEL_PROFILE='dsiclab_gpt54'; python main.py *> '$BackendLogQ' }"
$FrontendCommand = "& { Set-Location -LiteralPath '$FrontendQ'; if (-not (Test-Path 'node_modules')) { npm install }; `$env:VITE_API_URL='http://127.0.0.1:8001'; npx vite --host 127.0.0.1 --port 5174 --strictPort *> '$FrontendLogQ' }"
$BackendEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($BackendCommand))
$FrontendEncoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($FrontendCommand))

Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $BackendEncoded)
Start-Sleep -Seconds 2
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $FrontendEncoded)

Write-Host "Test backend:  http://127.0.0.1:8001"
Write-Host "Test frontend: http://127.0.0.1:5174"
Write-Host "Logs:"
Write-Host "  $BackendLog"
Write-Host "  $FrontendLog"
