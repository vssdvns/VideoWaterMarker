from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

DERF_URL = "https://media.xiph.org/video/derf/"


def sh(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def sh_out(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.stdout.strip()


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_webm_links(html: str, base_url: str) -> list[str]:
    # href="something.webm"
    links = re.findall(r'href="([^"]+\.webm)"', html, flags=re.IGNORECASE)
    seen = set()
    out: list[str] = []
    for href in links:
        absu = urljoin(base_url, href)
        if absu not in seen:
            seen.add(absu)
            out.append(absu)
    return out


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as r, open(dst, "wb") as f:
        f.write(r.read())


def ffprobe_ok_video(path: Path) -> bool:
    """
    True if ffprobe can read duration and at least one video stream.
    """
    try:
        out = sh_out([
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration,size",
            "-of", "default=nk=1:nw=1",
            str(path)
        ])
        # output should include duration and size lines
        parts = [p.strip() for p in out.splitlines() if p.strip()]
        if len(parts) < 2:
            return False
        duration = float(parts[0])
        size = int(parts[1])
        if duration <= 0.1:
            return False
        if size < 50_000:  # <50KB probably junk
            return False
        return True
    except Exception:
        return False


def ffprobe_wh(path: Path) -> tuple[int, int]:
    out = sh_out([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(path)
    ])
    if "x" not in out:
        return (0, 0)
    w, h = out.split("x")
    return (int(w), int(h))


def make_mp4_clip(
    src: Path,
    out_mp4: Path,
    clip_seconds: float,
    target: str,  # "720" or "1080"
    crf: int = 18,
) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # we force output canvas = 1280x720 or 1920x1080, preserving aspect ratio
    if target == "720":
        W, H = 1280, 720
    else:
        W, H = 1920, 1080

    vf = (
        f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,"
        f"fps=30"
    )

    # Important:
    # -t clip_seconds trims to fixed duration
    # -an drops audio
    sh([
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(src),
        "-t", f"{clip_seconds:.3f}",
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",
        str(out_mp4)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/clips_xiph_hd")
    ap.add_argument("--tmp_dir", type=str, default="data/_tmp_xiph")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--clip_seconds", type=float, default=10.0)
    ap.add_argument("--mix_1080_ratio", type=float, default=0.5)
    ap.add_argument("--max_sources", type=int, default=600)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--min_src_kb", type=int, default=200, help="Skip downloaded sources smaller than this (KB)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_html(DERF_URL)
    links = parse_webm_links(html, DERF_URL)[: args.max_sources]
    if not links:
        print("[XIPH] No .webm links found on:", DERF_URL)
        sys.exit(1)

    want_1080 = int(round(args.count * args.mix_1080_ratio))
    want_720 = args.count - want_1080

    made_total = 0
    made_1080 = 0
    made_720 = 0

    print(f"[XIPH] found webm links: {len(links)}")
    print(f"[XIPH] target outputs: {args.count}  (1080p={want_1080}, 720p={want_720})")
    print(f"[XIPH] writing to: {out_dir}")
    print(f"[XIPH] tmp_dir: {tmp_dir}")

    for i, url in enumerate(links):
        if made_total >= args.count:
            break

        name = url.split("/")[-1]
        src_path = tmp_dir / name

        # 1) download
        try:
            download(url, src_path)
        except Exception as e:
            print(f"[XIPH] skip download failed: {url} ({e})")
            continue

        # quick size gate (avoid 261 bytes garbage)
        if src_path.stat().st_size < args.min_src_kb * 1024:
            continue

        # 2) validate decodability
        if not ffprobe_ok_video(src_path):
            continue

        # 3) check source resolution capability
        w, h = ffprobe_wh(src_path)
        if w == 0 or h == 0:
            continue

        can_1080 = (w >= 1920 or h >= 1080)
        can_720 = (w >= 1280 or h >= 720)

        if not can_720 and not can_1080:
            continue

        # decide target based on remaining quota
        target = None
        if made_1080 < want_1080 and can_1080:
            target = "1080"
        elif made_720 < want_720 and can_720:
            target = "720"
        elif made_1080 < want_1080 and can_720:
            # do NOT upscale a 720 source to 1080; keep honest
            target = "720"
        else:
            continue

        out_name = f"xiph_{made_total:03d}_{target}p.mp4"
        out_path = out_dir / out_name

        # 4) convert to mp4 clip
        try:
            make_mp4_clip(
                src=src_path,
                out_mp4=out_path,
                clip_seconds=args.clip_seconds,
                target=target,
                crf=args.crf,
            )
        except subprocess.CalledProcessError:
            continue

        # 5) validate output (avoid 1KB mp4s)
        if not ffprobe_ok_video(out_path):
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass
            continue

        made_total += 1
        if target == "1080":
            made_1080 += 1
        else:
            made_720 += 1

        print(f"[XIPH] made {made_total}/{args.count}: {out_name}")

    print(f"\n[XIPH] DONE. made_total={made_total}  1080p={made_1080}  720p={made_720}")
    if made_total < args.count:
        print("[XIPH] NOTE: Could not reach requested count. Increase --max_sources or lower --min_src_kb.")


if __name__ == "__main__":
    main()