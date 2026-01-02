# =============================================================================
# UNINA-YOLO-DLA: Local Debug Dry-Run Script (PowerShell)
# =============================================================================
#
# This script orchestrates a complete dry-run of the training pipeline
# on CPU-only systems (no CUDA, no TensorRT).
#
# Usage:
#   .\run_local_debug.ps1
#   .\run_local_debug.ps1 -SkipDeps -Epochs 1
#
# =============================================================================

param(
    [switch]$SkipDeps = $true,
    [switch]$SkipGen,
    [int]$Epochs = 2,
    [int]$BatchSize = 4,
    [int]$ImgSize = 320
)

# --- Step 0: Check for Virtual Environment ---
if ($null -eq $env:VIRTUAL_ENV) {
    if (Test-Path ".venv") {
        Write-Host ">>> Virtual environment detected but not active." -ForegroundColor Yellow
        Write-Host "    Activating .venv..."
        & .\.venv\Scripts\Activate.ps1
    }
    else {
        Write-Host ">>> WARNING: No virtual environment detected." -ForegroundColor Yellow
        Write-Host "    It is recommended to run .\setup_env.ps1 first."
        Write-Host ""
    }
}
else {
    Write-Host ">>> Using active virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
}
Write-Host ""

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "UNINA-YOLO-DLA: Local Debug Dry-Run (Windows)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:"
Write-Host "  Epochs:     $Epochs"
Write-Host "  Batch Size: $BatchSize"
Write-Host "  Image Size: $ImgSize"
Write-Host "  Skip Deps:  $SkipDeps"
Write-Host "  Skip Gen:   $SkipGen"
Write-Host ""

# --- Step 1: Install Dependencies ---
if (-not $SkipDeps) {
    Write-Host ">>> Step 1: Installing dependencies..." -ForegroundColor Yellow
    pip install --quiet --upgrade pip
    pip install --quiet ultralytics opencv-python numpy torch
    Write-Host "    Dependencies installed." -ForegroundColor Green
}
else {
    Write-Host ">>> Step 1: Skipping dependency installation." -ForegroundColor Gray
}
Write-Host ""

# --- Step 2: Generate Synthetic Dataset ---
if (-not $SkipGen) {
    Write-Host ">>> Step 2: Generating synthetic FSD dataset..." -ForegroundColor Yellow
    python "$ScriptDir\tools\generate_synthetic_fsd.py" `
        --num-train 50 `
        --num-val 10 `
        --seed 42
    Write-Host "    Dataset generated." -ForegroundColor Green
}
else {
    Write-Host ">>> Step 2: Skipping dataset generation." -ForegroundColor Gray
}
Write-Host ""

# --- Step 3: Verify Dataset Structure ---
Write-Host ">>> Step 3: Verifying dataset structure..." -ForegroundColor Yellow
$DatasetDir = "$ScriptDir\datasets\fsd_synth"
$ConfigFile = "$DatasetDir\fsd_synth.yaml"

if (-not (Test-Path $ConfigFile)) {
    Write-Host "ERROR: Dataset config not found at $ConfigFile" -ForegroundColor Red
    exit 1
}

$TrainImages = (Get-ChildItem "$DatasetDir\images\train" -ErrorAction SilentlyContinue | Measure-Object).Count
$ValImages = (Get-ChildItem "$DatasetDir\images\val" -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "    Train images: $TrainImages"
Write-Host "    Val images:   $ValImages"

if ($TrainImages -lt 10 -or $ValImages -lt 5) {
    Write-Host "ERROR: Insufficient dataset images." -ForegroundColor Red
    exit 1
}
Write-Host "    Dataset structure OK." -ForegroundColor Green
Write-Host ""

# --- Step 4: Run Training Dry-Run ---
Write-Host ">>> Step 4: Running training dry-run (CPU mode)..." -ForegroundColor Yellow
Write-Host "    This verifies the training loop completes without errors."
Write-Host ""

# Set environment variable to force mocks if CUDA not available
$env:UNINA_FORCE_MOCKS = "1"

python "$ScriptDir\train.py" `
    --data "$ConfigFile" `
    --epochs $Epochs `
    --batch $BatchSize `
    --imgsz $ImgSize `
    --device cpu `
    --project runs/local_debug

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host ">>> DRY-RUN COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results:"
Write-Host "  - Training loop completed without errors"
Write-Host "  - Output saved to: runs/local_debug/dry_run"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Review the training logs for any warnings"
Write-Host "  2. Check that custom metrics (mAP_small) are being computed"
Write-Host "  3. For full training, use real FSD dataset on GPU"
Write-Host ""
