# Start Laptop 2 — PathSnap Evaluator UI (Streamlit)
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PathSnap — Laptop 2 Evaluator UI (Team-01)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "In the sidebar, enter Laptop 1's URL (e.g. http://<LAPTOP1_IP>:8000)" -ForegroundColor Yellow

python -m streamlit run ui/app.py --server.port 8501
