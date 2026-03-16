# Proposal Implementation Status

This document maps the project proposal (CSC 290 Master's Project) to the implementation.

## Implemented ✓

| Proposal item | Implementation |
|---------------|-----------------|
| **Saliency detection** | Laplacian + DeepLabV3 (proposal mentions U²-Net; DeepLab covers semantic low-attention) |
| **Segmentation** | DeepLabV3-ResNet50 |
| **Encrypted payload** | `crypto_payload.py`: AES-GCM (optional) + Reed-Solomon ECC |
| **Hybrid DCT transforms** | `dct_watermark.py`: DCT-based invisible embedding in low-attention regions |
| **Optical flow** | Farneback (proposal: RAFT; Farneback provides temporal consistency) |
| **Error-correcting codes** | Reed-Solomon via `reedsolo` |
| **Attack suite** | Re-encode, crop, blur, noise, rescale, etc. in `run_attacks.py` |
| **Evaluation metrics** | PSNR, SSIM, LPIPS in `evaluate_quality.py` |
| **BER / recovery** | `evaluate_ber.py`: BER, recovery accuracy |
| **FastAPI microservice** | `api_watermark.py`: REST API for watermark/detect |
| **Baseline comparison scaffold** | `src/baselines/`: VideoSeal, ItoV comparison framework |
| **Forensic traceability** | User/session ID embedding + extraction |
| **Just-in-time embedding** | API accepts video, returns watermarked output |

## Fully Implemented (Phase 2)

| Proposal item | Implementation |
|---------------|----------------|
| **U²-Net saliency** | `models/saliency_u2net.py` — optional saliency model |
| **RAFT optical flow** | `video_watermark_demo.compute_flow_raft()` — optional via `use_raft_flow` |
| **U-Net encoder-decoder** | `neural_watermark/models.py` — Encoder + Decoder + AttackSimulator |
| **Neural training** | `neural_watermark/train.py` — training with attack simulation |
| **HLS/DASH integration** | `hls_watermark.py` — segment-level watermarking for HLS manifests |

## Partial / Future Work

| Proposal item | Status |
|---------------|--------|
| **Neural extractor in pipeline** | Decoder exists; integrate with main watermark flow when trained |
| **VideoSeal / ItoV baselines** | Scaffold ready; requires cloning their repos and running |

## New Files Added

- `src/crypto_payload.py` — AES-GCM + Reed-Solomon
- `src/dct_watermark.py` — DCT invisible embedding
- `src/api_watermark.py` — FastAPI microservice
- `src/evaluate_ber.py` — BER metrics
- `src/baselines/` — Baseline comparison
