# UNINA-YOLO-DLA: Virtual Environment Setup (PowerShell)
# =============================================================================

$ErrorActionPreference = "Stop"

Write-Host ">>> Creating virtual environment in .venv..." -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "    Virtual environment created." -ForegroundColor Green
}
else {
    Write-Host "    .venv already exists, skipping creation." -ForegroundColor Gray
}

Write-Host ">>> Activating virtual environment..." -ForegroundColor Cyan
# PowerShell activation
& .\.venv\Scripts\Activate.ps1

Write-Host ">>> Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host ">>> Installing core dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ">>> Installing NVIDIA Quantization tools..." -ForegroundColor Cyan
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ">>> SETUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "To start working, ensure the environment is active."
Write-Host "Your shell should show (.venv)."
Write-Host ""
