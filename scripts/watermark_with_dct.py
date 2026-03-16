"""Watermark a video with visible + DCT for evaluation testing."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.crypto_payload import encode_payload
from src.video_watermark_demo import add_text_watermark_to_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=Path, default=None)
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--positions", "-p", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    inp = args.input or root / "data" / "input" / "sample.mp4"
    out = args.output or root / "data" / "output" / "watermarked_with_dct.mp4"
    pos = args.positions or out.with_suffix(".positions.json")

    if not inp.exists():
        print("Input not found:", inp)
        return 1

    payload, _ = encode_payload("user_001", "sess_xyz", use_aes=False, use_ecc=True)
    add_text_watermark_to_video(
        input_path=inp,
        output_path=out,
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
        embed_dct_payload=payload,
    )
    print("Saved:", out, pos)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
