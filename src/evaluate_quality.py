import os
import csv
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# SSIM
try:
    from skimage.metrics import structural_similarity as sk_ssim
except ImportError as e:
    raise ImportError("Missing dependency: scikit-image. Install: pip install scikit-image") from e

# LPIPS
try:
    import torch
    import lpips
except ImportError as e:
    raise ImportError("Missing dependency: lpips/torch. Install: pip install lpips torch torchvision") from e


@dataclass
class FrameMetrics:
    psnr: float
    ssim: float
    lpips: float


def compute_psnr_uint8(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """PSNR on uint8 images in [0..255]."""
    diff = (bgr1.astype(np.float32) - bgr2.astype(np.float32))
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim_uint8(bgr1: np.ndarray, bgr2: np.ndarray) -> float:
    """SSIM on uint8 images in [0..255], computed in RGB space."""
    rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
    rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
    score = sk_ssim(rgb1, rgb2, channel_axis=2, data_range=255)
    return float(score)


class LPIPSComputer:
    """
    LPIPS expects tensors in [-1, 1], shape [N,3,H,W] in RGB.
    """
    def __init__(self, net: str = "alex", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def compute(self, bgr1: np.ndarray, bgr2: np.ndarray) -> float:
        rgb1 = cv2.cvtColor(bgr1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)

        t1 = torch.from_numpy(rgb1).float().permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W in [0..255]
        t2 = torch.from_numpy(rgb2).float().permute(2, 0, 1).unsqueeze(0)

        # Normalize to [-1, 1]
        t1 = (t1 / 127.5) - 1.0
        t2 = (t2 / 127.5) - 1.0

        t1 = t1.to(self.device)
        t2 = t2.to(self.device)

        val = self.model(t1, t2)
        return float(val.item())


def open_video(path: str) -> cv2.VideoCapture:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def read_frame(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def resize_to_match(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    h, w = frame.shape[:2]
    if (h, w) == (th, tw):
        return frame
    return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)


def evaluate_pair(
    clean_video: str,
    test_video: str,
    lpips_net: str,
    device: str,
    max_frames: int,
    sample_every: int,
    force_resize: bool,
    quiet: bool,
) -> Tuple[List[FrameMetrics], Dict[str, float]]:
    cap_clean = open_video(clean_video)
    cap_test = open_video(test_video)

    lp = LPIPSComputer(net=lpips_net, device=device)

    metrics: List[FrameMetrics] = []
    i = 0
    kept = 0

    # Use first clean frame to determine size if resize enabled
    first_clean = read_frame(cap_clean)
    first_test = read_frame(cap_test)
    if first_clean is None or first_test is None:
        cap_clean.release()
        cap_test.release()
        raise RuntimeError("One of the videos has no frames.")

    target_hw = first_clean.shape[:2]  # (H,W)

    # rewind (we consumed one frame)
    cap_clean.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_test.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        frame_clean = read_frame(cap_clean)
        frame_test = read_frame(cap_test)
        if frame_clean is None or frame_test is None:
            break

        if (i % sample_every) != 0:
            i += 1
            continue

        if force_resize:
            frame_test = resize_to_match(frame_test, target_hw)

        # If still mismatch, stop (safer than wrong alignment)
        if frame_clean.shape[:2] != frame_test.shape[:2]:
            cap_clean.release()
            cap_test.release()
            raise RuntimeError(
                f"Frame size mismatch at frame {i}: clean {frame_clean.shape[:2]} vs test {frame_test.shape[:2]}. "
                f"Use --force_resize to auto-resize test to clean."
            )

        psnr = compute_psnr_uint8(frame_clean, frame_test)
        ssim = compute_ssim_uint8(frame_clean, frame_test)
        lpv = lp.compute(frame_clean, frame_test)

        metrics.append(FrameMetrics(psnr=psnr, ssim=ssim, lpips=lpv))

        kept += 1
        if not quiet and kept % 50 == 0:
            print(f"  processed {kept} sampled frames...")

        if max_frames > 0 and kept >= max_frames:
            break

        i += 1

    cap_clean.release()
    cap_test.release()

    arr_psnr = np.array([m.psnr for m in metrics], dtype=np.float32)
    arr_ssim = np.array([m.ssim for m in metrics], dtype=np.float32)
    arr_lpips = np.array([m.lpips for m in metrics], dtype=np.float32)

    summary = {
        "frames": float(len(metrics)),
        "psnr_mean": float(arr_psnr.mean()) if len(arr_psnr) else float("nan"),
        "psnr_std": float(arr_psnr.std(ddof=0)) if len(arr_psnr) else float("nan"),
        "ssim_mean": float(arr_ssim.mean()) if len(arr_ssim) else float("nan"),
        "ssim_std": float(arr_ssim.std(ddof=0)) if len(arr_ssim) else float("nan"),
        "lpips_mean": float(arr_lpips.mean()) if len(arr_lpips) else float("nan"),
        "lpips_std": float(arr_lpips.std(ddof=0)) if len(arr_lpips) else float("nan"),
    }
    return metrics, summary


def parse_methods(method_args: List[str]) -> Dict[str, str]:
    """
    Expects repeated: --method name=path.mp4
    Example: --method baseline=out/baseline.mp4 --method heuristic=out/heuristic.mp4
    """
    out: Dict[str, str] = {}
    for m in method_args:
        if "=" not in m:
            raise ValueError(f"Invalid --method '{m}'. Use name=path")
        name, path = m.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Invalid --method '{m}': empty name")
        out[name] = path
    if not out:
        raise ValueError("No methods provided. Use --method name=path at least once.")
    return out


def write_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "method",
        "video",
        "frames",
        "psnr_mean",
        "psnr_std",
        "ssim_mean",
        "ssim_std",
        "lpips_mean",
        "lpips_std",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_per_frame_csv(path: str, method_name: str, video_path: str, metrics: List[FrameMetrics]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = ["method", "video", "frame_idx_sampled", "psnr", "ssim", "lpips"]
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            w.writeheader()
        for idx, m in enumerate(metrics):
            w.writerow(
                {
                    "method": method_name,
                    "video": video_path,
                    "frame_idx_sampled": idx,
                    "psnr": m.psnr,
                    "ssim": m.ssim,
                    "lpips": m.lpips,
                }
            )


def main():
    ap = argparse.ArgumentParser(
        description="Compute PSNR / SSIM / LPIPS between clean video and watermarked outputs (baseline/heuristic/DL/hybrid)."
    )
    ap.add_argument("--clean", required=True, help="Path to original clean video")
    ap.add_argument(
        "--method",
        action="append",
        default=[],
        help="Repeat: name=path_to_method_video. Example: --method baseline=out/base.mp4",
    )
    ap.add_argument("--out_dir", default="eval_out", help="Output directory for CSVs")
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    ap.add_argument("--max_frames", type=int, default=0, help="Max sampled frames (0 = all)")
    ap.add_argument("--sample_every", type=int, default=1, help="Evaluate every Nth frame (1 = all frames)")
    ap.add_argument("--force_resize", action="store_true", help="Resize test frames to match clean video resolution")
    ap.add_argument("--save_per_frame", action="store_true", help="Write per-frame metrics CSV (can be large)")
    ap.add_argument("--quiet", action="store_true", help="Less logging")
    args = ap.parse_args()

    methods = parse_methods(args.method)

    summary_rows: List[Dict[str, object]] = []
    per_frame_csv = os.path.join(args.out_dir, "metrics_per_frame.csv")

    # If per-frame enabled, clear file first to avoid appending old runs
    if args.save_per_frame:
        os.makedirs(args.out_dir, exist_ok=True)
        if os.path.exists(per_frame_csv):
            os.remove(per_frame_csv)

    for name, vid in methods.items():
        if not args.quiet:
            print(f"\nEvaluating method: {name}")
            print(f"  clean: {args.clean}")
            print(f"  test : {vid}")

        metrics, summary = evaluate_pair(
            clean_video=args.clean,
            test_video=vid,
            lpips_net=args.lpips_net,
            device=args.device,
            max_frames=args.max_frames,
            sample_every=max(1, args.sample_every),
            force_resize=args.force_resize,
            quiet=args.quiet,
        )

        row = {
            "method": name,
            "video": vid,
            **summary,
        }
        summary_rows.append(row)

        if args.save_per_frame:
            write_per_frame_csv(per_frame_csv, name, vid, metrics)

        if not args.quiet:
            print(
                f"  frames={int(summary['frames'])} "
                f"PSNR={summary['psnr_mean']:.3f}±{summary['psnr_std']:.3f} "
                f"SSIM={summary['ssim_mean']:.4f}±{summary['ssim_std']:.4f} "
                f"LPIPS={summary['lpips_mean']:.4f}±{summary['lpips_std']:.4f}"
            )

    summary_csv = os.path.join(args.out_dir, "metrics_summary.csv")
    write_summary_csv(summary_csv, summary_rows)

    if not args.quiet:
        print(f"\nWrote summary: {summary_csv}")
        if args.save_per_frame:
            print(f"Wrote per-frame: {per_frame_csv}")


if __name__ == "__main__":
    main()