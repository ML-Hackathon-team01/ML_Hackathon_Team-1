# Start BOTH Backend (FastAPI) and Frontend (Streamlit UI) on the SAME Laptop
# Usage (default):
#   .\deployment\start_all_local.ps1
# Usage (with evaluator demonstration dataset):
#   .\deployment\start_all_local.ps1 -EvalDir "C:\path\to\evaluator_demo_dataset"
param(
    [string]$EvalDir = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  PathSnap — Single-Laptop Full Stack Launcher (Team-01)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Start FastAPI Backend in its own visible PowerShell window (for live CMD logs)
$apiArgs = "-NoExit -Command `"Set-Location '$RepoRoot'; "
if ($EvalDir -ne "") {
    $apiArgs += "`$env:PATHSNAP_EVAL_DIR='$EvalDir'; "
} else {
    $apiArgs += "Remove-Item Env:\PATHSNAP_EVAL_DIR -ErrorAction SilentlyContinue; "
}
$apiArgs += ".\deployment\start_laptop1_api.ps1`""

Start-Process powershell -ArgumentList $apiArgs
Write-Host "Started FastAPI Backend window -> http://127.0.0.1:8000" -ForegroundColor Green

Start-Sleep -Seconds 3

# 2. Start Streamlit Evaluator UI in this window (defaults to http://127.0.0.1:8000)
Write-Host "Starting Evaluator UI -> http://localhost:8501" -ForegroundColor Yellow
python -m streamlit run ui/app.py --server.port 8501
