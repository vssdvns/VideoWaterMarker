# Setup GPU for RTX 50-series (Blackwell / sm_120)
# Standard PyTorch cu121 does NOT support sm_120 - use nightly cu128
# Run from project root: powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_blackwell.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $root "venv_gpu"
$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host "=== RTX 50-series / Blackwell (sm_120) GPU Setup ===" -ForegroundColor Cyan
Write-Host "Project root: $root"

if (-not (Test-Path $venv)) {
    Write-Host "venv_gpu not found. Creating with Python 3.11..."
    py -3.11 -m venv $venv
    if (-not $?) { throw "Python 3.11 not found. Install from python.org" }
}

Write-Host "`nUninstalling old PyTorch (cu121) if present..."
& $pip uninstall -y torch torchvision torchaudio 2>&1 | Out-Null

Write-Host "`nInstalling PyTorch nightly with CUDA 12.8 (Blackwell support)..."
& $pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

Write-Host "`nEnsuring project dependencies..."
& $pip install opencv-python scikit-image lpips pycryptodome reedsolo

Write-Host "`nVerifying GPU..."
& $python -c @"
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print('CUDA: True')
    print('Device:', name)
    print('Compute capability:', cap)
else:
    print('CUDA: False - GPU not detected')
"@

Write-Host "`n=== Done! Run training with: ===" -ForegroundColor Green
Write-Host "  train_gpu.bat"
Write-Host "  OR: venv_gpu\Scripts\activate; python -m src.neural_watermark.train --device cuda"
