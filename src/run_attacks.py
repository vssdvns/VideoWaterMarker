from __future__ import annotations

from pathlib import Path
import argparse
import subprocess


# Shared ffmpeg flags used by every attack command so each run is quiet,
# non-interactive, and safe to overwrite during repeated benchmarks.
FFMPEG_BASE = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats"]


def run(cmd: list[str]) -> None:
    # Run one fully-built ffmpeg command and let failures raise immediately
    # so the benchmark stops on broken attack settings instead of hiding errors.
    subprocess.run(cmd, check=True)


def out(out_dir: Path, name: str) -> str:
    # Keep output path building in one helper so the attack list stays readable.
    return str(out_dir / name)


def main():
    # Read the source clip and the destination folder where all attacked
    # versions will be written.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each tuple is one named attack plus the ffmpeg arguments that create it.
    # The rest of the script simply loops over this list and executes them.
    attacks: list[tuple[str, list[str]]] = []

    # A) Re-encode
    attacks.append(("reencode_crf28.mp4", ["-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))
    attacks.append(("reencode_crf35.mp4", ["-c:v", "libx264", "-crf", "35", "-preset", "veryfast"]))

    # B) Down+up scale
    attacks.append(("down_up.mp4", [
        "-vf", "scale=854:480,scale=1280:720",
        "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"
    ]))

    # C) Blur
    attacks.append(("blur_sigma2.mp4", ["-vf", "gblur=sigma=2", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))
    attacks.append(("blur_sigma4.mp4", ["-vf", "gblur=sigma=4", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # D) Noise
    attacks.append(("noise20.mp4", ["-vf", "noise=alls=20:allf=t+u", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))
    attacks.append(("noise40.mp4", ["-vf", "noise=alls=40:allf=t+u", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # E) Crop
    attacks.append(("crop40.mp4", ["-vf", "crop=iw-80:ih-80:40:40", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))
    attacks.append(("crop80.mp4", ["-vf", "crop=iw-160:ih-160:80:80", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # F) FPS
    attacks.append(("fps15.mp4", ["-vf", "fps=15", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # G) EQ
    attacks.append(("eq_gamma12_contrast11.mp4", ["-vf", "eq=gamma=1.20:contrast=1.10:brightness=0.03", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))
    attacks.append(("eq_gamma085_contrast09.mp4", ["-vf", "eq=gamma=0.85:contrast=0.90:brightness=-0.03", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # H) Saturation / grayscale
    attacks.append(("saturation15.mp4", ["-vf", "eq=saturation=1.5", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))
    attacks.append(("grayscale.mp4", ["-vf", "hue=s=0", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # I) Rotate
    attacks.append(("rotate2deg.mp4", ["-vf", "rotate=2*PI/180:fillcolor=black,crop=iw:ih", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # J) Sharpen
    attacks.append(("unsharp.mp4", ["-vf", "unsharp=5:5:1.0:5:5:0.0", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    # K) Denoise
    attacks.append(("denoise_hqdn3d.mp4", ["-vf", "hqdn3d=1.5:1.5:6:6", "-c:v", "libx264", "-crf", "28", "-preset", "veryfast"]))

    print(f"[ATTACK] running {len(attacks)} attacks -> {out_dir}")
    for name, params in attacks:
        # Build one ffmpeg command per attack and write the result to a
        # predictable filename so detection scripts can consume them later.
        cmd = FFMPEG_BASE + ["-i", str(inp)] + params + [out(out_dir, name)]
        run(cmd)

    print("[ATTACK] done")


if __name__ == "__main__":
    main()