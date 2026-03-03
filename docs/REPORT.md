# Hybrid Video Watermarking with Global Fallback Detection

**Master's Project Report**  
California State University, Sacramento  
Computer Science

---

## Abstract

We present a hybrid video watermarking system that places semi-transparent text watermarks in less intrusive regions of each frame. The placement strategy combines Laplacian-based structural complexity with DeepLab semantic saliency through weighted fusion, temporal smoothing (EMA), and optional optical flow stabilization. For detection, we implement a multi-scale normalized cross-correlation (NCC) template matcher with a local ROI search and a global fallback when local detection fails. We benchmark the system on 26 HD video clips under 17 distortion attacks, achieving 94.6% mean detection rate at threshold 0.40. Ablation studies demonstrate that the global fallback is critical for robustness under severe cropping (52.4% with fallback vs. 0.7% without) and rescaling (94.7% vs. 25%). We also introduce user-specific fingerprint encoding for traceability in OTT/streaming scenarios and provide an interactive UI for watermarking, manual position adjustment, and attack testing.

---

## 1. Introduction

### 1.1 Motivation

Video piracy remains a significant challenge for content distributors, especially over-the-top (OTT) streaming platforms. Visible watermarks can deter casual copying and, when combined with per-user encoding, enable traceability of leaked content. A key concern is user experience: watermarks placed over important regions (faces, text, action) distract viewers. We therefore seek to place watermarks in *less intrusive* regions while maintaining robustness to common distortions (compression, cropping, blur, rescaling).

### 1.2 Contributions

1. **Hybrid placement model** combining Laplacian structural complexity and DeepLab semantic saliency with weighted fusion, temporal smoothing, and optional optical flow.
2. **Edge-biased placement** to prefer corners/edges for reduced intrusiveness and improved crop resilience.
3. **Multi-scale NCC detection** with local ROI and global fallback, significantly improving robustness under severe attacks.
4. **User-specific fingerprint encoding** for traceability (user ID, location, device).
5. **Interactive application** for watermarking, manual adjustment, detection, and attack testing.

### 1.3 Report Outline

Section 2 reviews related work. Section 3 describes the watermarking and detection methods. Section 4 presents experiments and results. Section 5 discusses the implementation and UI. Section 6 concludes and outlines future work.

---

## 2. Background and Related Work

### 2.1 Visible vs. Invisible Watermarking

Visible watermarks (logos, text) are commonly used for branding and piracy deterrence. Invisible or semi-transparent watermarks can be more resistant to removal but require different detection approaches. Our system uses semi-transparent visible text, which is straightforward to detect and can encode user-specific information.

### 2.2 Saliency and Content-Aware Placement

Content-aware placement aims to avoid salient regions (faces, objects, text). Prior work uses:
- **Edge-based metrics** (e.g., Laplacian) to identify low-complexity, smooth regions.
- **Semantic segmentation** (e.g., DeepLab) to distinguish foreground from background.
- **Eye-tracking or attention models** for more refined placement.

We combine structural (Laplacian) and semantic (DeepLab) cues in a hybrid model.

### 2.3 Watermark Detection

Template matching with normalized cross-correlation (NCC) is standard for visible watermarks. Challenges include scale changes (rescaling), geometric transforms (rotation, crop), and compression. We address these with multi-scale matching and a global fallback search when local ROI detection fails.

---

## 3. Method

### 3.1 Watermark Placement

#### 3.1.1 Laplacian Complexity Map

For each frame, we compute a *complexity map* based on edge magnitude:

1. Convert to grayscale and apply Gaussian blur.
2. Compute the Laplacian.
3. Take absolute value and normalize to [0, 1].

Low values indicate smooth regions (e.g., sky, walls) where a watermark is less distracting.

#### 3.1.2 DeepLab Saliency Map

We use a pretrained DeepLabV3-ResNet50 (COCO) model to obtain per-pixel class probabilities. Background (class 0) is treated as low-saliency; we invert this to produce a map where low values indicate suitable placement regions.

#### 3.1.3 Hybrid Fusion

The Laplacian map \( L \) and DeepLab map \( D \) are fused with weight \( w_D \):

\[
G = (1 - w_D) \cdot L + w_D \cdot D
\]

We use \( w_D = 0.6 \) by default. The combined map \( G \) is normalized to [0, 1].

#### 3.1.4 Sliding-Window Selection

A fixed-size window (watermark dimensions plus padding) slides over \( G \) with stride 32. The window with the lowest average value is selected.

#### 3.1.5 Edge Preference

To reduce intrusiveness and improve crop resilience, we restrict placement to an *edge zone*: the outer 12% margin from each frame border. Windows whose center falls outside this zone are excluded.

#### 3.1.6 Temporal Smoothing

Watermark position is smoothed over time with exponential moving average (EMA):

\[
(x_t, y_t) = \alpha \cdot (x_{t-1}, y_{t-1}) + (1 - \alpha) \cdot (x_{\text{cand}}, y_{\text{cand}})
\]

with \( \alpha = 0.8 \). Large jumps (e.g., at scene cuts) are allowed beyond a threshold (max jump 150 px).

#### 3.1.7 Optional Optical Flow

Optical flow (Farneback) can warp the previous frame's guide map to the current frame, blending it with the instantaneous map for smoother transitions.

### 3.2 Watermark Rendering

The watermark text is rendered with OpenCV's `putText`, scaled by resolution (relative to 1280px width). Semi-transparent overlay uses `addWeighted` with user-defined opacity \( \alpha \) (default 0.5).

### 3.3 Detection

#### 3.3.1 Template Construction

A grayscale template is built from the known watermark text, font, and dimensions (box_w × box_h).

#### 3.3.2 Local ROI Search

For each sampled frame (step 5), we use the saved position from the watermarking phase. A search ROI is defined as the predicted region plus padding (60 px). Multi-scale NCC (scales 0.70–1.40) is applied. A frame is *detected* if max NCC ≥ threshold (default 0.40).

#### 3.3.3 Global Fallback

When local NCC < threshold, we perform a global search on the downscaled frame (0.5×) with the same multi-scale NCC. This handles cases where the watermark has shifted (e.g., crop, rescale) and is no longer at the predicted position.

### 3.4 User-Specific Fingerprint

For traceability, the watermark text can encode user ID and optionally location and device:

- **Direct mode**: Short IDs shown as-is, e.g., `ID:user_001`.
- **Hashed mode**: SHA-256 truncated to 8 chars, e.g., `ID:a1b2c3d4`, with a lookup table for reverse mapping.

A pirated copy's watermark text can thus be traced back to the source account.

---

## 4. Experiments and Results

### 4.1 Setup

- **Dataset**: 26 HD clips (720p/1080p) from DAVIS and Xiph.
- **Attacks**: 17 distortion types implemented with ffmpeg:
  - Re-encode (CRF 28, 35)
  - Down-up rescale (854×480 → 1280×720)
  - Gaussian blur (σ=2, 4)
  - Noise (levels 20, 40)
  - Crop (40 px, 80 px from each side)
  - FPS reduction (15 fps)
  - Gamma/contrast adjustments
  - Saturation, grayscale
  - Rotation (2°)
  - Unsharp, denoise

### 4.2 Threshold Selection

We swept NCC thresholds 0.40, 0.45, 0.50. Threshold 0.40 was selected for the best balance between robustness and avoiding false negatives.

### 4.3 Main Results (Threshold = 0.40)

| Metric | Value |
|--------|-------|
| **Overall mean detection** | **94.6%** |
| Severe crop (80 px) | 52.4% |
| Down-up rescale | 94.7% |
| Blur, reencode, grayscale | High (>90%) |

### 4.4 Ablation: Global Fallback

| Configuration | Overall | Crop 80 | Down-up |
|---------------|---------|--------|---------|
| With global fallback | 94.6% | 52.4% | 94.7% |
| Without global fallback | 81.6% | 0.7% | 25% |

The global fallback is essential for robustness under severe cropping and rescaling, where the watermark moves significantly from its predicted position.

---

## 5. Implementation

### 5.1 Technical Stack

- **Watermarking**: OpenCV, NumPy, PyTorch (DeepLabV3-ResNet50 via torchvision)
- **Detection**: OpenCV NCC
- **Attacks**: ffmpeg
- **UI**: Streamlit

### 5.2 Application Features

The interactive UI provides:

1. **Generate**: Upload video, select method (Fixed, Heuristic, DeepLab, Hybrid), set traceability options, generate watermarked video.
2. **Preview**: Watch output, compare original vs. watermarked frame-by-frame, export summary.
3. **Manual adjust**: View saliency heatmap, use preset buttons (corners, edges) or sliders, add keyframes for interpolation, re-export.
4. **Detect**: Upload watermarked video and positions.json, run detection, view fingerprint and rate.
5. **Attack test**: Apply selected attacks (blur, crop, reencode, grayscale, etc.), run detection, view per-attack results and average.

### 5.3 Reproducibility

- Install: `pip install -r requirements.txt`
- Run UI: `streamlit run src/app_watermark_ui.py`
- Benchmark: See `src/runner/run_benchmark.py` and `PROJECT_PLAN.md`

---

## 6. Conclusion and Future Work

### 6.1 Summary

We implemented a hybrid video watermarking system that places semi-transparent text in less intrusive regions using Laplacian + DeepLab fusion, edge preference, and temporal smoothing. Multi-scale NCC detection with global fallback achieves 94.6% mean detection under 17 attacks. User-specific fingerprint encoding enables traceability for piracy forensics. An interactive UI supports the full workflow: watermarking, adjustment, detection, and attack testing.

### 6.2 Limitations

- Severe crop (80 px) remains challenging (52.4% detection).
- Optical flow adds compute cost; we use it optionally.
- Visible watermark is removable by inpainting; invisible component could augment robustness.

### 6.3 Future Work

- **Invisible component**: DCT/DWT embedding alongside visible text for redundancy.
- **Collusion resistance**: Encoding that survives averaging of multiple copies.
- **ROI exclusion**: Face detection to avoid watermarking over faces.
- **Batch processing**: Pipeline for watermarking large catalogs.
- **Quality metrics**: SSIM, VMAF for perceptual impact assessment.

---

## References

1. Chen, B., & Wornell, G. W. (2001). Digital watermarking and information embedding using dither modulation. *IEEE Transactions on Signal Processing*.
2. Farnebäck, G. (2003). Two-frame motion estimation based on polynomial expansion. *SCIA*.
3. Garcia-Diaz, A., et al. (2012). On the relationship between visual attention and saliency models. *Journal of Vision*.
4. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
5. Lewis, J. P. (1995). Fast normalized cross-correlation. *Vision Interface*.

---

*Report generated for CSU Sacramento Master's Project.*
