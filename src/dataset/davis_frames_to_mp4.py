from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def detect_frame_pattern(seq_dir: Path) -> Optional[tuple[str, int]]:
    """
    DAVIS frames are usually like 00000.jpg, 00001.jpg...
    We detect the extension and assume %05d numbering.
    Returns (pattern, start_number).
    """
    files = sorted([p for p in seq_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not files:
        return None

    # Find first numeric stem (e.g., "00000")
    for p in files[:10]:
        if p.stem.isdigit():
            start = int(p.stem)
            # Use the first file's extension
            pattern = str(seq_dir / f"%05d{p.suffix.lower()}")
            return pattern, start

    return None


def convert_one(
    seq_dir: str,
    out_dir: str,
    fps: int,
    crf: int,
    preset: str,
) -> dict:
    seq_path = Path(seq_dir)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / f"{seq_path.name}.mp4"
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"sequence": seq_path.name, "ok": True, "out": str(out_path), "skipped": True, "err": ""}

    info = detect_frame_pattern(seq_path)
    if info is None:
        return {"sequence": seq_path.name, "ok": False, "out": "", "skipped": False, "err": "No frames found"}

    pattern, start_number = info

    # -start_number is important if sequence starts at 00000 vs 00001 etc.
    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-framerate", str(fps),
        "-start_number", str(start_number),
        "-i", pattern,

        # Make sure dimensions are even (required by libx264/yuv420p)
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",

        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]

    p = run(cmd)
    if p.returncode != 0:
        return {"sequence": seq_path.name, "ok": False, "out": "", "skipped": False, "err": p.stderr.strip()}

    return {"sequence": seq_path.name, "ok": True, "out": str(out_path), "skipped": False, "err": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--davis_root", type=str, default="data/raw/DAVIS")
    ap.add_argument("--images_subdir", type=str, default="JPEGImages/480p")
    ap.add_argument("--out_dir", type=str, default="data/raw/DAVIS_VIDEOS")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--crf", type=int, default=18)          # good quality for source videos
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    images_root = Path(args.davis_root) / args.images_subdir
    if not images_root.exists():
        raise FileNotFoundError(f"Not found: {images_root}\n"
                                f"Expected DAVIS frames at: {args.davis_root}/{args.images_subdir}")

    seq_dirs = sorted([p for p in images_root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise RuntimeError(f"No sequence folders found under {images_root}")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(convert_one, str(sd), args.out_dir, args.fps, args.crf, args.preset)
            for sd in seq_dirs
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = sum(1 for r in results if r["ok"])
    bad = len(results) - ok

    # Print failures clearly
    print(f"Converted sequences: ok={ok}, bad={bad}")
    for r in sorted(results, key=lambda x: x["sequence"]):
        if not r["ok"]:
            print(f"FAIL {r['sequence']}: {r['err']}")

    print(f"Output videos in: {args.out_dir}")


if __name__ == "__main__":
    main()