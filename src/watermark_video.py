from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]  # project root
    sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--positions_out", type=str, required=True)

    parser.add_argument("--text", type=str, default="VideoWaterMarker")
    parser.add_argument("--alpha", type=float, default=0.5)

    # Force hybrid for benchmark (recommended)
    parser.add_argument("--w_deeplab", type=float, default=0.6)
    parser.add_argument("--use_optical_flow", action="store_true")
    parser.add_argument("--flow_beta", type=float, default=0.7)
    parser.add_argument("--temporal_alpha", type=float, default=0.8)
    parser.add_argument("--max_jump", type=int, default=150)

    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    pos_out = Path(args.positions_out)

    out.parent.mkdir(parents=True, exist_ok=True)
    pos_out.parent.mkdir(parents=True, exist_ok=True)

    from src.video_watermark_demo import add_text_watermark_to_video

    add_text_watermark_to_video(
        input_path=inp,
        output_path=out,
        text=args.text,
        alpha=float(args.alpha),

        # ✅ HYBRID pipeline
        use_hybrid=True,
        use_saliency_model=True,
        w_deeplab=float(args.w_deeplab),

        # optional stabilization
        use_optical_flow=bool(args.use_optical_flow),
        flow_beta=float(args.flow_beta),
        temporal_smoothing=True,
        temporal_alpha=float(args.temporal_alpha),
        max_jump=int(args.max_jump),

        # ✅ positions output (correct names)
        save_positions=True,
        positions_path=pos_out,
    )

    print("[WATERMARK] out       ->", out)
    print("[WATERMARK] positions ->", pos_out)


if __name__ == "__main__":
    main()