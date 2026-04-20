from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    # This lightweight record lets worker processes report success or failure
    # back to the parent benchmark process in a structured way.
    ok: bool
    clip_path: str
    clip_id: str
    run_dir: str
    error: str = ""


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    # Print each command before running it so long benchmark jobs leave behind
    # a readable execution trail in the terminal logs.
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _clip_id_from_path(clips_dir: Path, clip_path: Path) -> str:
    """
    Example:
      clips_dir = data/clips_150
      clip_path = data/clips_150/davis/davis_davis_blackswan.mp4
      -> davis__davis_davis_blackswan
    """
    # Turn a nested clip path into a stable id that can be reused in folder
    # names and combined CSV rows without carrying the full path around.
    rel = clip_path.relative_to(clips_dir).with_suffix("")
    parts = list(rel.parts)
    if len(parts) >= 2:
        return f"{parts[0]}__{parts[-1]}"
    return parts[-1]


def process_one(
    python_exe: str,
    root: Path,
    clips_dir: Path,
    out_root: Path,
    clip_path: Path,
    frame_step: int,
    search_pad: int,
    thr: float,
) -> RunResult:
    # Each clip gets its own run directory so watermarking, attacks, and
    # detection outputs stay grouped together and can be inspected later.
    clip_id = _clip_id_from_path(clips_dir, clip_path)
    run_dir = out_root / "runs" / clip_id
    run_dir.mkdir(parents=True, exist_ok=True)

    watermarked = run_dir / "watermarked.mp4"
    positions = run_dir / "positions.json"
    attacks_dir = run_dir / "attacks"
    detect_csv = run_dir / "robustness_results.csv"

    try:
        # Step 1: create the watermarked clip and save its positions metadata.
        _run(
            [
                python_exe,
                str(root / "src" / "watermark_video.py"),
                "--input",
                str(clip_path),
                "--output",
                str(watermarked),
                "--positions_out",
                str(positions),
            ]
        )

        # Step 2: generate the attacked copies that will be used for robustness testing.
        _run(
            [
                python_exe,
                str(root / "src" / "run_attacks.py"),
                "--input",
                str(watermarked),
                "--out_dir",
                str(attacks_dir),
            ]
        )

        # Step 3: run detection across the attacked outputs and save one CSV per clip.
        _run(
            [
                python_exe,
                str(root / "src" / "detect_watermark.py"),
                "--attacks_dir",
                str(attacks_dir),
                "--pos_json",
                str(positions),
                "--csv_out",
                str(detect_csv),
                "--frame_step",
                str(frame_step),
                "--search_pad",
                str(search_pad),
                "--thr",
                str(thr),
                "--global_fallback",
            ]
        )

        # Returning a structured success result makes it easy for the parent
        # process to summarize many worker jobs at the end.
        return RunResult(True, str(clip_path), clip_id, str(run_dir))

    except subprocess.CalledProcessError as e:
        return RunResult(False, str(clip_path), clip_id, str(run_dir), error=f"CalledProcessError: {e}")
    except Exception as e:
        return RunResult(False, str(clip_path), clip_id, str(run_dir), error=str(e))


def combine_results(out_root: Path, combined_csv: Path) -> None:
    # After all per-clip jobs finish, merge their CSV files into one table
    # that is easier to analyze in spreadsheets or plotting scripts.
    rows: list[dict] = []
    runs_dir = out_root / "runs"
    for run_dir in sorted(runs_dir.glob("*")):
        clip_id = run_dir.name
        csv_path = run_dir / "robustness_results.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["clip_id"] = clip_id
                rows.append(r)

    combined_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        combined_csv.write_text("", encoding="utf-8")
        print("[BENCH] combined empty ->", combined_csv)
        return

    # Make a stable column order
    fieldnames = ["clip_id"] + [k for k in rows[0].keys() if k != "clip_id"]
    with open(combined_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print("[BENCH] combined ->", combined_csv)


def main():
    # This script is the outer benchmark driver: it discovers clips, fans the
    # work out across processes, and then combines all results into one CSV.
    root = Path(__file__).resolve().parents[2]  # .../src/runner/run_benchmark.py -> project root
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips_dir", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--search_pad", type=int, default=60)
    parser.add_argument("--thr", type=float, default=0.35)
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    # Discover all clips first so the user can see the workload size before
    # the expensive watermark/attack/detect pipeline begins.
    clips = sorted(clips_dir.rglob("*.mp4"))
    if args.limit and args.limit > 0:
        clips = clips[: args.limit]

    print("[BENCH] clips_dir:", clips_dir)
    print("[BENCH] out_root :", out_root)
    print("[BENCH] clips    :", len(clips))
    print("[BENCH] workers  :", args.workers)
    print("[BENCH] python   :", python_exe)

    ok = 0
    bad = 0

    results_dir = out_root / "results"
    combined_csv = results_dir / "benchmark_results.csv"

    # Process each clip independently in parallel. This keeps the benchmark
    # simple because each worker owns its own output directory.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                process_one,
                python_exe,
                root,
                clips_dir,
                out_root,
                cp,
                args.frame_step,
                args.search_pad,
                args.thr,
            )
            for cp in clips
        ]

        for fut in as_completed(futs):
            r = fut.result()
            if r.ok:
                ok += 1
                print(f"[BENCH] OK  {r.clip_path}")
            else:
                bad += 1
                print(f"[BENCH] BAD {r.clip_path}\n  -> {r.error}")

    print(f"[BENCH] done. ok={ok} bad={bad}")
    # Build the final all-clips summary only after every worker has finished.
    combine_results(out_root, combined_csv)


if __name__ == "__main__":
    main()