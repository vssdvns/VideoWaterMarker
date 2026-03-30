"""
HLS/DASH segment watermarking for OTT pipeline integration (proposal).

Processes HLS (.m3u8) segments: watermarks video in each segment,
outputs new manifest + segments for just-in-time delivery.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np


def parse_m3u8(manifest_path: str) -> list[str]:
    """
    Parse HLS manifest, return list of segment URIs/paths.
    Supports local paths and relative URLs.
    """
    base = Path(manifest_path).parent
    segments = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if not line.startswith(("http://", "https://")):
                seg = base / line
            else:
                seg = line
            segments.append(str(seg))
    return segments


def read_ts_frames(ts_path: str) -> list[np.ndarray]:
    """Read video frames from .ts file using OpenCV."""
    cap = cv2.VideoCapture(ts_path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
    cap.release()
    return frames


def write_ts_frames(frames: list[np.ndarray], out_path: str, fps: float = 30) -> None:
    """Write frames to .ts/.mp4 using ffmpeg."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tf.name, fourcc, fps, (w, h))
        for fr in frames:
            out.write(fr)
        out.release()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tf.name,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "copy", "-movflags", "+faststart",
                out_path,
            ],
            check=True,
            capture_output=True,
        )
    Path(tf.name).unlink(missing_ok=True)


def watermark_hls_segments(
    manifest_path: str,
    output_dir: str | Path,
    watermark_fn: Callable[[np.ndarray], np.ndarray],
    *,
    segment_ext: str = ".mp4",
) -> str:
    """
    Watermark all segments from HLS manifest.

    Args:
        manifest_path: Path to .m3u8 file
        output_dir: Directory for watermarked segments
        watermark_fn: Function frame -> watermarked_frame
        segment_ext: Output segment extension (.mp4 or .ts)

    Returns:
        Path to new manifest
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = parse_m3u8(manifest_path)
    base = Path(manifest_path).parent
    new_segments = []

    for i, seg_uri in enumerate(segments):
        seg_path = base / seg_uri if not seg_uri.startswith("http") else None
        if seg_path and not seg_path.exists():
            continue
        if seg_path is None:
            continue  # Skip remote URLs without download for now

        frames = read_ts_frames(str(seg_path))
        if not frames:
            continue

        watermarked = [watermark_fn(fr) for fr in frames]
        out_name = f"seg_{i:05d}{segment_ext}"
        out_path = output_dir / out_name
        write_ts_frames(watermarked, str(out_path))
        new_segments.append(out_name)

    # Write new manifest
    new_manifest = output_dir / "playlist.m3u8"
    with open(new_manifest, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:10\n#EXT-X-MEDIA-SEQUENCE:0\n")
        for seg in new_segments:
            f.write(f"#EXTINF:10.0,\n{seg}\n")
        f.write("#EXT-X-ENDLIST\n")
    return str(new_manifest)


def default_watermark_fn(text: str = "OTT", alpha: float = 0.4) -> Callable[[np.ndarray], np.ndarray]:
    """Create a simple text watermark function."""

    def _watermark(frame: np.ndarray) -> np.ndarray:
        overlay = frame.copy()
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = w / 1280.0
        cv2.putText(overlay, text, (w - 200, h - 30), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0).astype(np.uint8)

    return _watermark


def run_hls_watermark_cli(
    manifest: str,
    output_dir: str,
    text: str = "OTT",
    use_hybrid: bool = True,
) -> None:
    """
    CLI entry: watermarks HLS segments using our hybrid pipeline.
    """
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.video_watermark_demo import (
        compute_complexity_map,
        combine_maps,
        choose_low_complexity_region,
    )
    from src.models.saliency_deeplab import DeepLabSaliency

    deeplab = DeepLabSaliency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    def wm_fn(frame: np.ndarray) -> np.ndarray:
        lap = compute_complexity_map(frame)
        dl = deeplab.get_saliency_map(frame)
        guide = combine_maps(lap, dl, 0.6)
        h, w = frame.shape[:2]
        scale = w / 1280.0
        tw, th = cv2.getTextSize(text, font, scale, 2)[0]
        box_w, box_h = tw + 20, th + 20
        x, y = choose_low_complexity_region(guide, box_w, box_h, prefer_edges=True)
        overlay = frame.copy()
        cv2.putText(overlay, text, (x + 10, y + box_h - 10), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.4, frame, 0.6, 0).astype(np.uint8)

    output_manifest = watermark_hls_segments(manifest, output_dir, wm_fn)
    print(f"Watermarked manifest: {output_manifest}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text", default="OTT")
    args = ap.parse_args()
    run_hls_watermark_cli(args.manifest, args.output, args.text)
