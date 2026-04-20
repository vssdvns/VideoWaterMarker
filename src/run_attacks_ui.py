"""Run selected attacks for UI. Returns (name, output_path) for each."""

from __future__ import annotations

import subprocess
from pathlib import Path

# The UI uses the same ffmpeg defaults every time so attack runs stay quiet,
# overwrite old files, and behave consistently during interactive testing.
FFMPEG = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]

# This dictionary is the menu the Streamlit app exposes in the attack tab.
# The key is the user-facing attack id, and the value is the ffmpeg snippet
# needed to generate that distorted version of the video.
ATTACKS: dict[str, list[str]] = {
    "reencode_crf28": ["-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "reencode_crf35": ["-c:v", "libx264", "-crf", "35", "-preset", "veryfast"],
    "down_up": ["-vf", "scale=854:480,scale=1280:720", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "blur_sigma2": ["-vf", "gblur=sigma=2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "blur_sigma4": ["-vf", "gblur=sigma=4", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "noise20": ["-vf", "noise=alls=20:allf=t+u", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "crop40": ["-vf", "crop=iw-80:ih-80:40:40", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "crop80": ["-vf", "crop=iw-160:ih-160:80:80", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "fps15": ["-vf", "fps=15", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "grayscale": ["-vf", "hue=s=0", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
    "rotate2deg": ["-vf", "rotate=2*PI/180:fillcolor=black,crop=iw:ih", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"],
}


def run_attacks(input_path: Path, out_dir: Path, attack_names: list[str]) -> list[tuple[str, Path]]:
    """Run selected attacks. Returns [(attack_name, output_path), ...]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in attack_names:
        # Ignore unknown attack names so the UI can stay resilient even if
        # a stale selection or typo makes it here.
        if name not in ATTACKS:
            continue
        out_path = out_dir / f"{name}.mp4"
        # Build one ffmpeg command for the selected attack and store the file
        # under the attack name so later detection results stay easy to read.
        cmd = FFMPEG + ["-i", str(input_path)] + ATTACKS[name] + [str(out_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            results.append((name, out_path))
        except subprocess.CalledProcessError:
            # Skip failed attacks instead of crashing the whole UI run.
            pass
    return results
