@echo off
REM Run neural watermark training on GPU (uses Python 3.11 venv with CUDA)
cd /d "%~dp0"

if not exist "venv_gpu\Scripts\activate.bat" (
    echo venv_gpu not found. Run: powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1
    pause
    exit /b 1
)

call venv_gpu\Scripts\activate.bat
python -m src.neural_watermark.train --data_dir data/input --epochs 100 --batch 4 --size 128 --device cuda --max_videos 25 --max_frames_per_video 30
pause
