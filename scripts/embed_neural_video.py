"""
Embed neural watermark on a video (frame by frame).

Usage:
  python scripts/embed_neural_video.py input.mp4 output.mp4
  python scripts/embed_neural_video.py input.mp4 output.mp4 --payload 0,1,0,1,1,0,...

Payload defaults to a simple fixed bit pattern; you can pass custom bits (e.g. user ID).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.neural_watermark.embed import NeuralWatermarker


def main():
    ap = argparse.ArgumentParser(description="Embed neural watermark on video")
    ap.add_argument("input", help="Input video path")
    ap.add_argument("output", help="Output video path")
    ap.add_argument("--payload", default=None,
        help="Comma-separated bits, e.g. 0,1,0,1,1,0 (uses [1,0,1,0,...] if omitted)")
    ap.add_argument("--model_dir", default=str(ROOT / "data" / "models" / "neural_wm"),
        help="Directory with encoder.pt, decoder.pt, config.json")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device for model inference")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    enc_path = model_dir / "encoder.pt"
    dec_path = model_dir / "decoder.pt"

    if not enc_path.exists() or not dec_path.exists():
        print(f"Models not found: {model_dir}")
        print("Run training first: python -m src.neural_watermark.train --data_dir data --epochs 50")
        sys.exit(1)

    wm = NeuralWatermarker(
        encoder_path=enc_path,
        decoder_path=dec_path,
        device=args.device,
    )

    if args.payload:
        payload = [int(x.strip()) for x in args.payload.split(",")]
    else:
        payload = [1, 0, 1, 0] * 4  # simple pattern, padded to payload_bits
    payload = payload[:wm.payload_bits]
    if len(payload) < wm.payload_bits:
        payload = payload + [0] * (wm.payload_bits - len(payload))

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Cannot open input: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))
    if not out.isOpened():
        print(f"Cannot create output: {args.output}")
        cap.release()
        sys.exit(1)

    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        watermarked = wm.embed(frame, payload)
        out.write(watermarked)
        n += 1
        if n % 100 == 0:
            print(f"  Frame {n}...")

    cap.release()
    out.release()
    print(f"Done. Wrote {n} frames to {args.output}")


if __name__ == "__main__":
    main()
