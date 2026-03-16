"""
Evaluate visible, DCT, and neural watermark detection under attack suite.

Runs detection on each attacked video and reports:
- Visible: detection rate (% frames), NCC stats
- DCT: extracted (yes/no), BER when original payload available
- Neural: BER when original payload bits available (--with_neural)

Usage:
  python scripts/evaluate_attacks.py --attacks_dir data/attacks --pos_json data/output/final_watermarked.positions.json
  python scripts/evaluate_attacks.py ... --with_neural
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.detect_watermark import build_text_template, detect_video
from src.evaluate_ber import compute_ber, compute_payload_ber


def extract_neural_ber(video_path: Path, neural_wm, original_bits: list[int], frame_step: int = 5) -> tuple[bool, float]:
    """Extract neural bits from video, return (extracted, ber)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 1.0
    preds = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_step == 0:
            ext = neural_wm.extract(frame)
            preds.append([int(round(x)) for x in ext[:len(original_bits)]])
        idx += 1
    cap.release()
    if not preds:
        return False, 1.0
    # Majority vote across frames
    n_bits = len(original_bits)
    voted = []
    for i in range(n_bits):
        vals = [p[i] for p in preds if len(p) > i]
        voted.append(1 if sum(vals) > len(vals) / 2 else 0)
    ber = compute_ber(original_bits, voted)
    return True, ber


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks_dir", type=str, default=str(root / "data" / "attacks"))
    parser.add_argument("--pos_json", type=str, default=str(root / "data" / "output" / "final_watermarked.positions.json"))
    parser.add_argument("--csv_out", type=str, default=str(root / "data" / "output" / "robustness_results.csv"))
    parser.add_argument("--original_payload_hex", type=str, default=None, help="Override: original DCT payload as hex (for BER)")
    parser.add_argument("--with_neural", action="store_true", help="Evaluate neural watermark BER")
    parser.add_argument("--neural_model_dir", type=str, default=str(root / "data" / "models" / "neural_wm"))
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--search_pad", type=int, default=60)
    parser.add_argument("--thr", type=float, default=0.35)
    parser.add_argument("--global_fallback", action="store_true", help="Enable global search if local ROI fails")
    parser.add_argument("--global_downscale", type=float, default=0.5)

    args = parser.parse_args()

    attacks_dir = Path(args.attacks_dir)
    pos_json = Path(args.pos_json)

    positions = json.loads(pos_json.read_text(encoding="utf-8"))
    if not positions:
        print("No positions found in", pos_json)
        return

    # Original DCT payload for BER (from positions or CLI)
    original_payload: bytes | None = None
    if args.original_payload_hex:
        try:
            original_payload = bytes.fromhex(args.original_payload_hex)
        except ValueError:
            print("[WARN] Invalid --original_payload_hex, BER will be skipped")
    else:
        for p in positions:
            h = p.get("original_dct_payload_hex")
            if h:
                try:
                    original_payload = bytes.fromhex(h)
                    break
                except ValueError:
                    pass

    has_dct_roi = bool(positions and positions[0].get("dct_roi"))
    if has_dct_roi and original_payload is None:
        print("[INFO] DCT ROI present but no original payload (no BER). Use --original_payload_hex or watermark with DCT to save it.")

    neural_wm = None
    neural_bits = None
    if args.with_neural:
        model_dir = Path(args.neural_model_dir)
        enc_p, dec_p = model_dir / "encoder.pt", model_dir / "decoder.pt"
        neural_bits = positions[0].get("neural_payload_bits") if positions else None
        if enc_p.exists() and dec_p.exists() and neural_bits:
            from src.neural_watermark.embed import NeuralWatermarker
            neural_wm = NeuralWatermarker(encoder_path=enc_p, decoder_path=dec_p)
            print(f"[EVAL] Neural: payload_bits={len(neural_bits)}")
        else:
            print("[INFO] Neural: models or payload not found, skipping")

    text = positions[0].get("text", "VideoWaterMarker")
    box_w = int(positions[0]["box_w"])
    box_h = int(positions[0]["box_h"])
    font_scale = float(positions[0].get("font_scale", 1.0))
    thickness = int(positions[0].get("thickness", 2))

    template_g = build_text_template(text, box_w, box_h, font_scale, thickness)

    vids = sorted(attacks_dir.glob("*.mp4"))
    if not vids:
        print("No attacked videos found in:", attacks_dir)
        return

    print(f"[EVAL] Visible: threshold={args.thr}, frame_step={args.frame_step}")
    print(f"[EVAL] DCT: has_roi={has_dct_roi}, original_payload={'yes' if original_payload else 'no'}")
    print(f"[EVAL] {len(vids)} attack(s)")
    print()

    results = []
    for vp in vids:
        r = detect_video(
            vp,
            positions,
            template_g,
            frame_step=args.frame_step,
            search_pad=args.search_pad,
            thr=args.thr,
            enable_global_fallback=bool(args.global_fallback),
            global_downscale=args.global_downscale,
        )

        row = {
            "attack": vp.name,
            "detected_rate_percent": round(r["rate"], 2),
            "mean_ncc": round(r["mean"], 4),
            "p10_ncc": round(r["p10"], 4),
            "p50_ncc": round(r["p50"], 4),
            "p90_ncc": round(r["p90"], 4),
        }

        # DCT metrics
        dct_extracted = r.get("dct_payload") is not None and len(r.get("dct_payload", b"")) > 0
        row["dct_extracted"] = "yes" if dct_extracted else "no"
        if has_dct_roi:
            if dct_extracted and original_payload is not None:
                ber = compute_payload_ber(original_payload, r["dct_payload"])
                row["dct_ber"] = round(ber, 4)
            else:
                row["dct_ber"] = "" if dct_extracted else "N/A (no extract)"
        else:
            row["dct_ber"] = "N/A (no DCT)"

        # Neural metrics
        if neural_wm is not None and neural_bits:
            neural_ok, neural_ber = extract_neural_ber(vp, neural_wm, neural_bits, args.frame_step)
            row["neural_extracted"] = "yes" if neural_ok else "no"
            row["neural_ber"] = round(neural_ber, 4) if neural_ok else "N/A"
        else:
            row["neural_extracted"] = "N/A"
            row["neural_ber"] = "N/A"

        results.append(row)

        ber_str = f"  DCT: extracted={dct_extracted}, BER={row['dct_ber']}" if has_dct_roi else ""
        neural_str = f"  Neural: BER={row['neural_ber']}" if neural_wm and neural_bits else ""
        print(f"{vp.name}")
        print(f"  visible: {r['hit']}/{r['total']} ({r['rate']:.2f}%) | NCC mean={r['mean']:.4f}{ber_str}{neural_str}")

    # CSV
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
