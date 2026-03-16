# Setup GPU training environment (Python 3.11 + PyTorch CUDA)
# Run from project root: powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$venv = Join-Path $root "venv_gpu"

Write-Host "Project root: $root"
Write-Host "GPU venv: $venv"

# Create venv with Python 3.11
Write-Host "`nCreating venv with Python 3.11..."
py -3.11 -m venv $venv
if (-not $?) { throw "Python 3.11 not found. Install from python.org" }

# Activate and install
$pip = Join-Path $venv "Scripts\pip.exe"
$python = Join-Path $venv "Scripts\python.exe"

Write-Host "Installing PyTorch with CUDA 12.1..."
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "Installing project dependencies..."
& $pip install opencv-python numpy pandas streamlit scikit-image lpips pycryptodome reedsolo fastapi uvicorn python-multipart

# Verify CUDA
Write-Host "`nVerifying CUDA..."
& $python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "`n=== Done! To train on GPU, run: ==="
Write-Host "  venv_gpu\Scripts\activate"
Write-Host "  python -m src.neural_watermark.train --data_dir data/input --epochs 100 --batch 4 --size 128 --device cuda"
