# Tracing and Watermark Types Explained

## What is Tracing?

**Tracing** (forensic traceability) means identifying *who* distributed a leaked copy. When piracy occurs, you recover the user/session ID from the leaked video to attribute responsibility.

You trace using whichever watermark(s) survived the attack and recovered payload.

---

## 1. Visible Watermark (Text Overlay)

| Aspect | Detail |
|--------|--------|
| **What** | Semi-transparent text (e.g. `User_123`) overlaid on the video |
| **Detection** | NCC template matching – find where the text appears; if NCC > threshold → detected |
| **Tracing** | The text *is* the trace – you see "User_123" directly in the video |
| **Robustness** | Tested under re-encode, blur, noise, crop, etc. (see `robustness_results.csv`) |

**Flow:** Embed text → positions.json stores (x,y) per frame → On leaked video: search for template at expected positions → if found, you’ve traced (the text = user ID).

---

## 2. DCT Watermark (Invisible)

| Aspect | Detail |
|--------|--------|
| **What** | Payload bits embedded in DCT coefficients of a region (invisible to the eye) |
| **Placement** | **Saliency-guided**: same logic as visible text – placed in **low saliency / low complexity** areas (background, edges). The DCT ROI is positioned to the *right* of the visible text box (or below/corner if no space) so it does not overlap with the overlay. Both visible and DCT avoid salient regions. |
| **Detection** | `extract_dct_watermark(frame, roi)` – extract bits from the known ROI |
| **Tracing** | Recovered bytes → `decode_payload()` → `"user_123|sess_xyz"` (user/session ID) |
| **positions.json** | `dct_roi` stores (x, y, w, h) of the region for extraction |

**Flow:** Encode payload (AES-GCM + Reed-Solomon) → embed in DCT ROI → positions.json stores `dct_roi` → On leaked video: extract from ROI → decode → trace.

**BER:** Bit Error Rate between original and extracted payload. Lower BER = more reliable trace. Reed-Solomon helps correct some bit errors.

---

## 3. Neural Watermark (Invisible, Learning-Based)

| Aspect | Detail |
|--------|--------|
| **What** | Encoder/decoder neural networks that hide and recover bits in images |
| **Encoder** | Cover + payload bits → watermarked image (tiny pixel changes) |
| **Decoder** | Watermarked image → predicted payload bits |
| **Tracing** | Recovered bits → decode to user ID (same idea as DCT) |

**Current status:** BER ~25–37% on your setup → not reliable for traceability; visible + DCT preferred for production.

Neural watermarking is an **invisible, data-carrying** alternative. It learns to embed robustly but requires more training to reach low BER.

---

## Evaluation Under Attacks

Like the visible watermark, we can run attacks (re-encode, blur, noise, crop, etc.) and evaluate:

| Metric | Visible | DCT |
|--------|---------|-----|
| **Detection rate** | % of frames where NCC > threshold | % of videos where payload extracted |
| **BER** | N/A (text is read directly) | Bit error rate vs original payload |
| **Trace success** | Text recovered → decode user ID | Payload recovered → decode user ID |

Run `scripts/evaluate_attacks.py` to evaluate both visible and DCT under the same attack suite.
