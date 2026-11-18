# Start Laptop 1 — PathSnap FastAPI Backend
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PathSnap — Laptop 1 FastAPI Backend Server (Team-01)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Display LAN IPv4 addresses so Laptop 2 knows the exact API URL
$ips = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" } |
    Select-Object -ExpandProperty IPAddress

if ($ips) {
    Write-Host "Laptop 2 should connect to one of these URLs:" -ForegroundColor Green
    foreach ($ip in $ips) {
        Write-Host "  http://${ip}:8000" -ForegroundColor Yellow
    }
}
Write-Host "Local health check: http://127.0.0.1:8000/health" -ForegroundColor Gray
Write-Host "Swagger API docs  : http://127.0.0.1:8000/docs" -ForegroundColor Gray
Write-Host "------------------------------------------------------------" -ForegroundColor Cyan

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
