# GPU Training Setup

## Issue: Python 3.14 + CUDA

**PyTorch does not yet provide CUDA wheels for Python 3.14.** The `cu121` index has no matching distribution.

## Issue: RTX 50-series (Blackwell / sm_120)

**Standard PyTorch cu121 does NOT support RTX 5050/5080/5090** (compute capability sm_120). You must use PyTorch nightly with CUDA 12.8.

---

## Option A: Use CPU ✓

Training works on CPU. It will auto-fallback when CUDA isn't available.

```cmd
py -m src.neural_watermark.train --data_dir data/input --epochs 100 --batch 4 --size 128
```

---

## Option B: GPU – RTX 50-series (Blackwell)

For RTX 5050, 5080, 5090 or any GPU with sm_120:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_blackwell.ps1
```

Then run:

```cmd
train_gpu.bat
```

---

## Option C: GPU – RTX 30/40 series and older

1. **Install Python 3.11** from [python.org](https://www.python.org/downloads/)

2. **Run standard GPU setup:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1
   ```

3. **Run training:**
   ```cmd
   train_gpu.bat
   ```

---

## Verify GPU

```cmd
venv_gpu\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

- `CUDA: False` = CPU only
- `CUDA: True` = GPU available
