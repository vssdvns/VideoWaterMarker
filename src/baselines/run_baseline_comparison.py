"""
Baseline comparison scaffold (proposal Section 3.1).

Compares our hybrid watermarking system against VideoSeal and ItoV
on the same video set and attack suite.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Compare watermarking methods (ours vs VideoSeal vs ItoV)"
    )
    parser.add_argument("--clean", required=True, help="Original clean video")
    parser.add_argument("--ours", required=True, help="Our watermarked video")
    parser.add_argument("--videoseal", default="", help="VideoSeal output (if available)")
    parser.add_argument("--itov", default="", help="ItoV output (if available)")
    parser.add_argument("--attack_dir", default="", help="Directory of attacked videos")
    parser.add_argument("--out_csv", default="baseline_comparison.csv")
    args = parser.parse_args()

    import sys
    import torch
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.evaluate_quality import evaluate_pair

    methods = [("ours", args.ours)]
    if args.videoseal and os.path.exists(args.videoseal):
        methods.append(("videoseal", args.videoseal))
    if args.itov and os.path.exists(args.itov):
        methods.append(("itov", args.itov))

    rows = []
    for name, vid_path in methods:
        try:
            _, summary = evaluate_pair(
                clean_video=args.clean,
                test_video=vid_path,
                lpips_net="alex",
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_frames=100,
                sample_every=5,
                force_resize=False,
                quiet=True,
            )
            rows.append({
                "method": name,
                "video": vid_path,
                "psnr_mean": summary["psnr_mean"],
                "ssim_mean": summary["ssim_mean"],
                "lpips_mean": summary["lpips_mean"],
            })
        except Exception as e:
            rows.append({"method": name, "video": vid_path, "error": str(e)})

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "video", "psnr_mean", "ssim_mean", "lpips_mean", "error"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote: {out_path}")
    for r in rows:
        if "error" in r:
            print(f"  {r['method']}: ERROR - {r['error']}")
        else:
            print(f"  {r['method']}: PSNR={r['psnr_mean']:.2f} SSIM={r['ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
