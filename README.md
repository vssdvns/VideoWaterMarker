# VideoWaterMarker

**Deep Learning–Based Video Watermarking for OTT Piracy Prevention** — Master's Project, CSU Sacramento

Model-guided adaptive video watermarking combining:
- **Saliency detection** (Laplacian + DeepLab) for low-attention placement
- **Visible + invisible (DCT)** watermarks per proposal
- **AES-GCM + Reed-Solomon** encrypted payload for forensic traceability
- **Optical flow** (Farneback or RAFT) for temporal consistency

---

## What This Does

1. **Watermarks videos** — Adds semi-transparent text + optional **invisible DCT watermark** (proposal-aligned).
2. **Smart placement** — Chooses low-complexity, low-saliency regions (avoids faces, objects, text).
3. **Forensic payload** — AES-GCM encryption + Reed-Solomon ECC for session ID robustness.
4. **Detects watermarks** — Template matching (visible) + DCT extraction (invisible).
5. **Traceability** — Embeds user/session ID; leaked copies can be traced back.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the interactive app
streamlit run src/app_watermark_ui.py
```

Open the URL (usually http://localhost:8501) in your browser.

**Optional:** Install [ffmpeg](https://ffmpeg.org/) for in-browser video preview and attack testing.

---

## App Overview

The Streamlit app has **6 tabs**:

### 1. Generate
- **Upload** a video (mp4, avi, mov, mkv).
- **Choose method:**
  - **Fixed** — Bottom-right corner (baseline).
  - **Heuristic** — Laplacian complexity (smooth regions).
  - **DeepLab** — Semantic saliency (avoids foreground).
  - **Hybrid** — Laplacian + DeepLab combined (recommended).
- **Traceability (optional):** Enable and enter User ID, Location, Device; optionally embed **invisible DCT** with Reed-Solomon.
- **Placement:** Prefer edges/corners (default) to reduce intrusion and improve crop resilience.
- Click **Generate** to create the watermarked video.

### 2. Preview
- Watch the watermarked video.
- **Compare:** Slider to view Original vs Watermarked frame-by-frame.
- **Download** the watermarked video.
- **Export summary** — Text file with settings (method, text, opacity, etc.).

### 3. Manual Adjust
- View the **saliency/complexity heatmap** (blue = less intrusive).
- **Preset buttons:** Top-left, Top-right, Bottom-left, Bottom-right, Bottom-center.
- **Sliders:** Fine-tune X/Y position.
- **Keyframes:** Set position at multiple frames → interpolate → re-export.
- Click **Re-export** to apply your changes.

### 4. Detect (No Generate)
- Upload a **watermarked video** and its **positions.json** (from when it was created).
- Run detection to get:
  - **Detection rate** — % of frames where watermark was found.
  - **Embedded fingerprint** — The text in the watermark (e.g. `ID:user_001`).
  - **DCT payload** — Invisible payload (user/session ID) when enabled.
- Works with **previous files** — no need to generate in this session.

### 5. Attack Test
- Upload a watermarked video + positions.
- **Select attacks:** blur, crop, reencode, grayscale, etc.
- Run attacks → Run detection → View results per attack.
- Shows how robust the watermark is to distortions.

### 6. Traceability Info
- Explains user-specific fingerprint flow.
- Format examples: `ID:user_001`, `ID:a1b2c3d4`, `ID:a1b2.US-CA.ios`.

---

## Watermarking Methods

| Method   | Description                                                                 |
|----------|-----------------------------------------------------------------------------|
| **Fixed** | Constant bottom-right position. Simple baseline.                            |
| **Heuristic** | Laplacian edge map → low-complexity (smooth) regions.                       |
| **DeepLab** | DeepLabV3-ResNet50 segmentation → avoids foreground objects.              |
| **Hybrid** | Combines Laplacian + DeepLab (or U²-Net) with weighted fusion. Best balance. |

All adaptive methods use temporal smoothing (EMA) to avoid jitter between frames.

---

## Detection

- **Template matching** with normalized cross-correlation (NCC).
- **Multi-scale** — Handles slight size changes (0.7×–1.4×).
- **Global fallback** — When local search fails, searches the full frame (downscaled).
- **Threshold** — Default 0.40; higher = stricter.

The `positions.json` file (saved when watermarking) tells the detector where to look and what template to use.

---

## Benchmark Results

Tested on 26 HD clips under 17 attacks (blur, crop, reencode, rescale, etc.):

- **Overall detection:** 94.6% (threshold 0.40)
- **Severe crop (80px):** 52.4%
- **Down-up rescale:** 94.7%

**Ablation (no global fallback):** Detection drops to 81.6% overall, 0.7% on crop80, 25% on rescale — global fallback is essential.

---

## CLI Usage

Run from project root (`cd VideoWaterMarker`).

**Watermark a single video:**
```bash
python src/watermark_video.py --input video.mp4 --output out.mp4 --positions_out positions.json
```

**Generate videos with different user fingerprints:**
```bash
python -m src.demo_fingerprint --input video.mp4 --users user_001 user_002 alice --hybrid
```

**Run full benchmark** (see `src/runner/run_benchmark.py`).

**Run FastAPI microservice** (OTT pipeline):
```bash
uvicorn src.api_watermark:create_app --factory --host 0.0.0.0 --port 8000
```

**Baseline comparison** (VideoSeal / ItoV):
```bash
python -m src.baselines.run_baseline_comparison --clean video.mp4 --ours out/watermarked.mp4 --out_csv baseline.csv
```

**Neural encoder-decoder training**:
```bash
python -m src.neural_watermark.train --data_dir data/input --epochs 50
```

**HLS segment watermarking**:
```bash
python -m src.hls_watermark --manifest path/to/playlist.m3u8 --output data/hls_watermarked
```

---

## Project Structure

```
VideoWaterMarker/
├── README.md
├── PROJECT_PLAN.md
├── requirements.txt
├── docs/
│   ├── REPORT.md          # Master's project report
│   └── README.md          # How to convert report to PDF
├── data/
│   ├── input/             # Place input videos here (gitignored)
│   └── output/            # Watermarked outputs (gitignored)
└── src/
    ├── app_watermark_ui.py    # Main Streamlit app
    ├── api_watermark.py       # FastAPI microservice (OTT)
    ├── video_watermark_demo.py# Watermarking logic
    ├── detect_watermark.py    # Detection logic
    ├── fingerprint.py         # User ID encoding
    ├── crypto_payload.py     # AES-GCM + Reed-Solomon payload
    ├── dct_watermark.py       # Invisible DCT embedding
    ├── evaluate_ber.py       # BER / recovery metrics
    ├── run_attacks.py        # Apply distortion attacks
    ├── run_attacks_ui.py     # Attack runner for UI
    ├── demo_fingerprint.py   # CLI fingerprint demo
    ├── models/
    │   └── saliency_deeplab.py  # DeepLabV3-ResNet50 wrapper
    ├── models/
    │   ├── saliency_deeplab.py
    │   ├── saliency_u2net.py   # U²-Net saliency (proposal)
    │   └── u2net.py
    ├── neural_watermark/      # Neural encoder-decoder (proposal)
    │   ├── models.py
    │   ├── train.py
    │   └── embed.py
    ├── hls_watermark.py        # HLS/DASH segment watermarking
    └── baselines/             # VideoSeal / ItoV comparison
        └── run_baseline_comparison.py
```

---

## Report

The Master's project report is in **`docs/REPORT.md`** (~15 pages). To convert to PDF:

```bash
pandoc docs/REPORT.md -o docs/REPORT.pdf --pdf-engine=xelatex -V geometry:margin=1in
```

---

## Requirements

- Python 3.10+
- OpenCV, NumPy, PyTorch, torchvision, Streamlit
- pycryptodome, reedsolo (AES-GCM + Reed-Solomon)
- fastapi, uvicorn (for API microservice)
- ffmpeg (optional, for video preview and attack testing)

---

## License

Academic project — CSU Sacramento.
