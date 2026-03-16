# Install PyTorch with CUDA support (for GPU training)
# NOTE: Python 3.14 has NO CUDA wheels. Use Python 3.11 or 3.12:
#   py -3.12 -m venv venv_gpu
#   venv_gpu\Scripts\activate
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# See scripts/setup_gpu_training.md for full instructions.

$py = "py"
if (Get-Command py -ErrorAction SilentlyContinue) {
    $ver = & py -c "import sys; print(sys.version_info.minor)" 2>$null
    if ($ver -eq 14) {
        Write-Host "Python 3.14 detected - no CUDA wheels available. Use Python 3.12. See setup_gpu_training.md"
        exit 1
    }
}
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
Write-Host "Done. Verify with: py -c `"import torch; print('CUDA:', torch.cuda.is_available())`""
