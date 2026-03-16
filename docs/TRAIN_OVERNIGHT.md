# Overnight Neural Watermark Training

## Step 1: Run Training (Before Bed)

Use **overnight mode** so your PC stays responsive:

```bash
py -m src.neural_watermark.train --overnight --epochs 100
```

This sets:
- **batch=2** — uses less memory
- **size=128** — smaller images, faster
- **CPU only** — avoids GPU memory spikes
- **20 videos, 25 frames each** — ~500 frames (enough for learning)

**If it still freezes**, use even lighter settings:

```bash
py -m src.neural_watermark.train --overnight --epochs 50 --batch 1 --max_videos 10 --max_frames_per_video 15
```

**Windows tip:** Run in a separate terminal and close other apps. You can also lower process priority:
```cmd
start /low py -m src.neural_watermark.train --overnight --epochs 100
```

---

## Step 2: What to Do After Training

When the command finishes, you'll see:
```
[TRAIN] Done. Saved to data/models/neural_wm
```

### A) Quick test (check if it worked)

```bash
py -m src.neural_watermark.embed
```

This embeds and extracts random bits. Low BER (%) = model learned.

### B) Use in Python code

```python
from src.neural_watermark.embed import NeuralWatermarker
import cv2

wm = NeuralWatermarker(
    encoder_path="data/models/neural_wm/encoder.pt",
    decoder_path="data/models/neural_wm/decoder.pt",
)

# Load a frame
frame = cv2.imread("some_image.jpg")

# Embed 48 bits (e.g. user ID encoded as bits)
payload = [0, 1, 0, 1, 1, 0, ...]  # 48 bits
watermarked = wm.embed(frame, payload)

# Extract from watermarked frame
extracted = wm.extract(watermarked)
```

### C) Watermark a video (custom script)

You can loop over video frames, embed a payload per frame, and write a new video using OpenCV.

---

## Troubleshooting

| Problem | Fix |
|--------|-----|
| **"Torch not compiled with CUDA enabled"** | Auto-falls back to CPU now. To use GPU: `py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| PC freezes | Use `--batch 1`, `--max_videos 5`, `--max_frames_per_video 10` |
| "No images in data/input" | Ensure videos are in `data/input/` or use `--synthetic` |
| Out of memory | Add `--device cpu` and `--batch 1` |
| Want faster training | Install PyTorch with CUDA, then use `--batch 4 --device cuda` |
