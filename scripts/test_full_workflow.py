"""
Test full workflow: visible + DCT + neural, Hybrid, optical flow + RAFT.
Run: py scripts/test_full_workflow.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def main():
    from src.crypto_payload import encode_payload
    from src.video_watermark_demo import add_text_watermark_to_video
    from src.app_watermark_ui import run_watermark

    input_path = ROOT / "data" / "app" / "input" / "sample_test.mp4"
    output_path = ROOT / "data" / "app" / "output" / "test_full.mp4"
    positions_path = ROOT / "data" / "app" / "output" / "test_full_positions.json"

    if not input_path.exists():
        print(f"ERROR: Sample video not found: {input_path}")
        return 1

    payload_bytes, _ = encode_payload("user_001", "sess_xyz", use_aes=False, use_ecc=True)
    neural_bits = [(hash("user_001") >> i) & 1 for i in range(8)]
    text = "ID:user_001"

    print("Running: Hybrid + visible + DCT + neural + optical flow + RAFT")
    print("Input:", input_path, "| Output:", output_path)
    print("---")

    run_watermark(
        method="Hybrid",
        input_path=input_path,
        output_path=output_path,
        positions_path=positions_path,
        text=text,
        alpha=0.5,
        w_deeplab=0.6,
        prefer_edges=True,
        edge_margin=0.12,
        embed_dct_payload=payload_bytes,
        use_visible=True,
        use_neural=True,
        neural_payload_bits=neural_bits,
        neural_color_mode="bias_corrected",
        saliency_type="deeplab",
        use_optical_flow=True,
        use_raft_flow=True,
    )

    print("---")
    print("SUCCESS: Workflow completed.")
    print("Output:", output_path)
    print("Positions:", positions_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
