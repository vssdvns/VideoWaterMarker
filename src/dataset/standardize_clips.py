# src/dataset/standardize_clips.py
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess command and capture stdout/stderr."""
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def standardize_one(
    clip_id: str,
    dataset: str,
    src_path: str,
    out_dir: str,
    height: int,
    fps: int,
    seconds: int,
    crf: int,
    preset: str,
    keep_audio: bool,
) -> dict:
    """
    Standardize one clip to:
      - duration: seconds (cuts early if source longer; ends early if source shorter)
      - fps
      - height (width auto, aspect preserved)
      - H.264, yuv420p
      - optional audio (default: drop audio for speed/consistency)
    """
    src = Path(src_path)
    out_root = Path(out_dir) / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    out_path = out_root / f"{clip_id}.mp4"

    # Skip if already exists and non-empty
    if out_path.exists() and out_path.stat().st_size > 0:
        return {"id": clip_id, "dataset": dataset, "ok": True, "out_path": str(out_path), "skipped": True, "err": ""}

    # Ensure even dimensions for libx264/yuv420p:
    # scale keeps AR, then trunc to even dims.
    vf = f"fps={fps},scale=-2:{height}:flags=bicubic,scale=trunc(iw/2)*2:trunc(ih/2)*2"

    cmd = [
        "ffmpeg", "-y",
        "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-t", str(seconds),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
    ]

    if not keep_audio:
        cmd += ["-an"]  # drop audio

    cmd += [str(out_path)]

    p = run(cmd)
    if p.returncode != 0:
        return {"id": clip_id, "dataset": dataset, "ok": False, "out_path": "", "skipped": False, "err": p.stderr.strip()}

    return {"id": clip_id, "dataset": dataset, "ok": True, "out_path": str(out_path), "skipped": False, "err": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", type=str, default="data/manifests/clips_150.csv")
    ap.add_argument("--out_dir", type=str, default="data/clips_150")
    ap.add_argument("--height", type=int, default=408)  # iteration: 408p
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=10)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", type=str, default="veryfast")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--keep_audio", action="store_true", help="Keep audio (default drops audio).")
    ap.add_argument("--report_csv", type=str, default="data/manifests/standardize_report.csv")
    args = ap.parse_args()

    manifest = Path(args.manifest_csv)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows: list[dict] = []
    with manifest.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # Required columns: id,dataset,label,src_path
            rows.append(row)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    futures = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for row in rows:
            futures.append(
                ex.submit(
                    standardize_one,
                    row["id"],
                    row["dataset"],
                    row["src_path"],
                    args.out_dir,
                    args.height,
                    args.fps,
                    args.seconds,
                    args.crf,
                    args.preset,
                    args.keep_audio,
                )
            )

        for fut in as_completed(futures):
            results.append(fut.result())

    # Write report
    with Path(args.report_csv).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "dataset", "ok", "out_path", "skipped", "err"])
        w.writeheader()
        for x in sorted(results, key=lambda d: (d["dataset"], d["id"])):
            w.writerow(x)

    ok = sum(1 for x in results if x["ok"])
    bad = len(results) - ok
    print(f"Standardized clips: ok={ok} bad={bad}")
    print(f"Outputs -> {args.out_dir}")
    print(f"Report  -> {args.report_csv}")


if __name__ == "__main__":
    main()