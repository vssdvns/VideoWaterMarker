"""
Watermark video with visible + DCT + neural.

1. Visible text + DCT (saliency-guided)
2. Neural watermark on top (full-frame)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.crypto_payload import encode_payload
from src.video_watermark_demo import add_text_watermark_to_video

ROOT = Path(__file__).resolve().parents[1]
NEURAL_MODEL = ROOT / "data" / "models" / "neural_wm"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument("--positions", "-p", type=Path, default=None)
    parser.add_argument("--neural_payload", type=str, default=None,
        help="Comma-separated bits, e.g. 1,0,1,0,1,0,1,0 (default: derive from user_001)")
    args = parser.parse_args()

    out = args.output
    pos = args.positions or (out.parent / (out.stem + ".positions.json"))
    pos.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: visible + DCT
    payload_bytes, _ = encode_payload("user_001", "sess_xyz", use_aes=False, use_ecc=True)
    tmp_vid = out.parent / (out.stem + "_tmp_vis_dct.mp4")
    add_text_watermark_to_video(
        input_path=args.input,
        output_path=tmp_vid,
        text="user_001",
        alpha=0.5,
        use_hybrid=True,
        w_deeplab=0.5,
        use_optical_flow=True,
        flow_beta=0.7,
        temporal_smoothing=True,
        temporal_alpha=0.8,
        max_jump=150,
        save_positions=True,
        positions_path=pos,
        embed_dct_payload=payload_bytes,
    )

    # Step 2: neural on top
    enc_path = NEURAL_MODEL / "encoder.pt"
    dec_path = NEURAL_MODEL / "decoder.pt"
    if not enc_path.exists() or not dec_path.exists():
        print("[WARN] Neural models not found, skipping neural. Output = visible+DCT only.")
        tmp_vid.rename(out)
        _save_neural_meta(pos, None)
        return 0

    if args.neural_payload:
        neural_bits = [int(x.strip()) for x in args.neural_payload.split(",")]
    else:
        # Encode user_001 as 8 bits: hash-based
        h = hash("user_001") & 0xFF
        neural_bits = [(h >> i) & 1 for i in range(8)]

    import cv2
    from src.neural_watermark.embed import NeuralWatermarker
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wm = NeuralWatermarker(encoder_path=enc_path, decoder_path=dec_path, device=device)

    cap = cv2.VideoCapture(str(tmp_vid))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = wm.embed(frame, neural_bits)
        writer.write(frame)
        n += 1
    cap.release()
    writer.release()
    tmp_vid.unlink(missing_ok=True)

    _save_neural_meta(pos, neural_bits)
    print("Saved (visible+DCT+neural):", out, pos)
    return 0


def _save_neural_meta(pos_path: Path, bits: list[int] | None):
    if not pos_path.exists() or bits is None:
        return
    data = json.loads(pos_path.read_text(encoding="utf-8"))
    if data:
        data[0]["neural_payload_bits"] = bits
        pos_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
