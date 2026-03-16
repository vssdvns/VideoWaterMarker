# Neural Watermark Encoder–Decoder: Workflow & Why It Matters

## What Are These Trained Models?

The neural watermark system uses **two trained networks**:

| Model | Role | Input | Output |
|-------|------|-------|--------|
| **Encoder** | Embeds the secret payload into the image | Cover image + payload bits | Watermarked image |
| **Decoder** | Recovers the payload from the watermarked image | Watermarked image | Predicted payload bits |

They are trained together so the encoder learns to hide information in a way the decoder can recover.

---

## Where They Fit in the Overall Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     VIDEO WATERMARKING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. CONTENT OWNER (e.g. OTT platform)                                       │
│     • Has video frames to protect                                            │
│     • Wants to embed user/session ID for piracy traceability                  │
│                                                                             │
│  2. ENCODER (neural model)                                                   │
│     • Input: clean frame + 48 bits (e.g. user ID)                            │
│     • Output: frame with invisible watermark (tiny pixel changes)             │
│     • Trained to: (a) embed payload (b) keep changes imperceptible           │
│                                                                             │
│  3. WATERMARKED VIDEO                                                       │
│     • Sent to user or stored                                                  │
│     • Looks identical to human eye                                           │
│                                                                             │
│  4. IF PIRACY OCCURS (leaked copy found)                                     │
│     • DECODER (neural model)                                                 │
│       • Input: leaked frame (possibly attacked: blur, crop, re-encode)      │
│       • Output: recovered payload bits (user ID)                             │
│     • Traceability: identify which user leaked the content                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

The neural encoder/decoder provide **invisible, data-carrying** watermarks. They sit alongside your visible watermarks (text overlay) and DCT-based invisible watermarks, adding a learning-based option that can be robust to some attacks.

---

## What the Encoder Does

- **Input:** Cover image (B, 3, H, W) + payload bits (B, 48)
- **Process:**
  1. Expands payload to spatial map and concatenates with cover
  2. Runs U-Net to predict a small **delta** (perturbation)
  3. Outputs `watermarked = cover + delta`, with delta clamped for invisibility
- **Goal:** Change pixels just enough so the decoder can read the bits, but not enough for humans to notice.

---

## What the Decoder Does

- **Input:** Watermarked image (B, 3, H, W)
- **Process:**
  1. CNN backbone (conv blocks + pooling)
  2. Global pooling → MLP → 48 logits
  3. Sigmoid → bit predictions (0 or 1)
- **Goal:** Recover the same bits that the encoder embedded, even after mild attacks (blur, noise).

---

## Why Good Loss Values Matter

Training minimizes:

- **BCE (Binary Cross-Entropy):** How well the decoder predicts the payload.
- **MSE:** How small the pixel changes are (invisibility).

| Loss | Random | Good | Very Good |
|------|--------|------|-----------|
| **BCE** | ≈ 0.693 | 0.1–0.3 | < 0.05 |
| **MSE** | 0 | 0.001–0.01 | < 0.005 |

- **BCE ≈ 0.693:** Random guessing, no useful embedding.
- **BCE < 0.1:** Decoder can recover most bits; traceability is viable.
- **BCE < 0.05:** High-quality embedding and extraction.
- **Low MSE:** Watermark remains invisible.

---

## What Each Component Enables

| Component | Purpose |
|-----------|---------|
| **Encoder (trained)** | Invisible embedding of IDs/bits into frames |
| **Decoder (trained)** | Extracting those bits from watermarked (or attacked) frames |
| **Low BCE** | Reliable traceability (few bit errors) |
| **Low MSE** | Imperceptible changes (quality preserved) |

---

## Using the Trained Models

```python
from src.neural_watermark.embed import NeuralWatermarker

wm = NeuralWatermarker(
    encoder_path="data/models/neural_wm/encoder.pt",
    decoder_path="data/models/neural_wm/decoder.pt"
)

# Embed user ID (48 bits or fewer)
payload = [0, 1, 0, 1, 1, 0, ...]  # e.g. encoded user ID
watermarked_frame = wm.embed(clean_frame_bgr, payload)

# Later: extract from leaked copy
recovered_bits = wm.extract(leaked_frame_bgr)
```

---

## Summary

- **Encoder:** Hides payload in the image with minimal visible change.
- **Decoder:** Recovers payload from watermarked (and possibly attacked) frames.
- **Good training (BCE < 0.1, low MSE):** Makes traceability reliable and the watermark invisible.
- **Role in workflow:** Adds neural invisible watermarking for OTT traceability and piracy attribution.
