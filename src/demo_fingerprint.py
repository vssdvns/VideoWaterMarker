"""
Phase 2: Demo – watermarks with different user fingerprints.

Generates watermarked videos for multiple user IDs to show traceability.
Usage: python -m src.demo_fingerprint --input data/input/sample.mp4 --output_dir data/output/fingerprint_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fingerprint import encode_fingerprint
from src.video_watermark_demo import add_text_watermark_to_video


def main():
    ap = argparse.ArgumentParser(description="Generate watermarked videos with different user fingerprints")
    ap.add_argument("--input", type=str, required=True, help="Input video path")
    ap.add_argument("--output_dir", type=str, default="data/output/fingerprint_demo")
    ap.add_argument("--users", type=str, nargs="+", default=["user_001", "user_002", "alice"])
    ap.add_argument("--hybrid", action="store_true", help="Use hybrid method (default: fixed for speed)")
    args = ap.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    if not inp.exists():
        print(f"Input not found: {inp}")
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[FINGERPRINT DEMO] Generating watermarked videos for multiple users...")

    for user_id in args.users:
        display_text = encode_fingerprint(user_id, use_hash=False)
        out_path = out_dir / f"watermarked_{user_id.replace(' ', '_')}.mp4"
        pos_path = out_dir / f"positions_{user_id.replace(' ', '_')}.json"

        print(f"  {user_id} -> {display_text} -> {out_path.name}")

        if args.hybrid:
            add_text_watermark_to_video(
                input_path=inp,
                output_path=out_path,
                text=display_text,
                alpha=0.5,
                use_hybrid=True,
                use_saliency_model=True,
                w_deeplab=0.6,
                temporal_smoothing=True,
                temporal_alpha=0.8,
                max_jump=150,
                save_positions=True,
                positions_path=pos_path,
            )
        else:
            from src.video_watermark_demo import add_text_watermark_fixed
            add_text_watermark_fixed(
                input_path=inp,
                output_path=out_path,
                text=display_text,
                alpha=0.5,
                save_positions=True,
                positions_path=pos_path,
            )

    print(f"\n[FINGERPRINT DEMO] Done. Outputs in {out_dir.resolve()}")
    print("Each video has a unique watermark. Traceability: watermark text -> user ID.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
