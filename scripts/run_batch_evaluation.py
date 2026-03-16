"""
Batch evaluation: watermark all (or a subset of) input videos, run attacks, detect.

Produces aggregate metrics to present workflow efficiency:
- Visible detection rate (avg across attacks & videos)
- DCT extraction rate (% attacks where payload recovered)
- DCT BER (when extracted)
- Neural BER (when --with_neural)
- Overall "traceability score"
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "input"
ATTACKS_BASE = ROOT / "data" / "attacks_batch"


def find_videos(limit: int | None, max_mb: float | None) -> list[Path]:
    """Find MP4/MOV files, optionally limited and by size."""
    videos = []
    for ext in ("*.mp4", "*.mov"):
        videos.extend(INPUT_DIR.glob(ext))
    videos = sorted(videos, key=lambda p: p.stat().st_size)
    if max_mb:
        max_bytes = int(max_mb * 1024 * 1024)
        videos = [v for v in videos if v.stat().st_size <= max_bytes]
    if limit:
        videos = videos[:limit]
    return videos


def run_cmd(cmd: list[str]) -> bool:
    r = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"[ERR] {' '.join(cmd)}: {r.stderr}")
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5, help="Max videos to test (default 5)")
    parser.add_argument("--max_mb", type=float, default=20, help="Skip videos larger than this MB")
    parser.add_argument("--no_watermark", action="store_true", help="Skip watermarking (use existing)")
    parser.add_argument("--with_neural", action="store_true", help="Include neural watermark (visible+DCT+neural)")
    args = parser.parse_args()

    OUTPUT_DIR = ROOT / "data" / "output" / ("batch_eval_neural" if args.with_neural else "batch_eval")
    videos = find_videos(args.limit, args.max_mb)
    if not videos:
        print("No videos found in", INPUT_DIR)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    if not python.exists():
        python = ROOT / "venv_gpu" / "Scripts" / "python.exe"
    py = str(python)

    print(f"[BATCH] Testing {len(videos)} videos: {[v.name for v in videos]}")
    print(f"[BATCH] Mode: {'visible + DCT + NEURAL' if args.with_neural else 'visible + DCT'}")
    print()

    all_results = []

    for i, video in enumerate(videos):
        name = video.stem
        out_vid = OUTPUT_DIR / f"{name}_watermarked.mp4"
        out_pos = OUTPUT_DIR / f"{name}_watermarked.positions.json"
        attacks_dir = ATTACKS_BASE / name

        print(f"[{i+1}/{len(videos)}] {video.name}")

        if not args.no_watermark:
            wm_script = "scripts/watermark_all_three.py" if args.with_neural else "scripts/watermark_with_dct.py"
            ok = run_cmd([
                py, wm_script,
                "--input", str(video.resolve()),
                "--output", str(out_vid.resolve()),
                "--positions", str(out_pos.resolve()),
            ])
            if not ok:
                print("  -> Watermark failed, skipping")
                continue

        attacks_dir.mkdir(parents=True, exist_ok=True)
        ok = run_cmd([
            py, "src/run_attacks.py",
            "--input", str(out_vid),
            "--out_dir", str(attacks_dir),
        ])
        if not ok:
            print("  -> Attacks failed, skipping")
            continue

        eval_cmd = [
            py, "scripts/evaluate_attacks.py",
            "--attacks_dir", str(attacks_dir),
            "--pos_json", str(out_pos),
            "--csv_out", str(OUTPUT_DIR / f"{name}_results.csv"),
            "--global_fallback",
        ]
        if args.with_neural:
            eval_cmd.append("--with_neural")
        ok = run_cmd(eval_cmd)
        if not ok:
            print("  -> Evaluation failed, skipping")
            continue

        csv_path = OUTPUT_DIR / f"{name}_results.csv"
        if csv_path.exists():
            rows = list(csv.DictReader(csv_path.open()))
            vis_rates = [float(r.get("detected_rate_percent", 0)) for r in rows]
            dct_yes = sum(1 for r in rows if r.get("dct_extracted") == "yes")
            dct_bers = []
            for r in rows:
                b = r.get("dct_ber", "")
                if b and b not in ("N/A", "N/A (no extract)", "N/A (no DCT)"):
                    try:
                        dct_bers.append(float(b))
                    except ValueError:
                        pass
            neural_bers = []
            neural_yes = 0
            if "neural_ber" in (rows[0].keys() if rows else []):
                neural_yes = sum(1 for r in rows if r.get("neural_extracted") == "yes")
                for r in rows:
                    b = r.get("neural_ber", "")
                    if b and b not in ("N/A",):
                        try:
                            neural_bers.append(float(b))
                        except ValueError:
                            pass
            rec = {
                "video": name,
                "visible_avg": sum(vis_rates) / len(vis_rates) if vis_rates else 0,
                "dct_extraction_rate": 100 * dct_yes / len(rows) if rows else 0,
                "dct_ber_avg": sum(dct_bers) / len(dct_bers) if dct_bers else None,
            }
            if args.with_neural:
                rec["neural_extraction_rate"] = 100 * neural_yes / len(rows) if rows else 0
                rec["neural_ber_avg"] = sum(neural_bers) / len(neural_bers) if neural_bers else None
            all_results.append(rec)

    if not all_results:
        print("\nNo results to aggregate.")
        return 1

    # Aggregate
    vis_avg = sum(r["visible_avg"] for r in all_results) / len(all_results)
    dct_ext_avg = sum(r["dct_extraction_rate"] for r in all_results) / len(all_results)
    dct_bers = [r["dct_ber_avg"] for r in all_results if r.get("dct_ber_avg") is not None]
    dct_ber_avg = sum(dct_bers) / len(dct_bers) if dct_bers else None

    neural_ext_avg = None
    neural_ber_avg = None
    if args.with_neural and all_results and "neural_extraction_rate" in all_results[0]:
        neural_ext_avg = sum(r["neural_extraction_rate"] for r in all_results) / len(all_results)
        neural_bers = [r["neural_ber_avg"] for r in all_results if r.get("neural_ber_avg") is not None]
        neural_ber_avg = sum(neural_bers) / len(neural_bers) if neural_bers else None

    # Traceability: vis 50%, DCT 30%, neural 20% when all three; else vis 60% + DCT 40%
    trace_vis = vis_avg / 100
    trace_dct = (dct_ext_avg / 100) * (1 - (dct_ber_avg or 0.5))
    if neural_ext_avg is not None and neural_ber_avg is not None:
        trace_neural = (neural_ext_avg / 100) * (1 - neural_ber_avg)
        traceability_score = 100 * (0.5 * trace_vis + 0.3 * trace_dct + 0.2 * trace_neural)
    else:
        traceability_score = 100 * (0.6 * trace_vis + 0.4 * trace_dct)

    print("\n" + "=" * 60)
    print("WORKFLOW EFFICIENCY REPORT" + (" (visible + DCT + NEURAL)" if args.with_neural else " (visible + DCT)"))
    print("=" * 60)
    print(f"Videos tested: {len(all_results)}")
    print(f"Visible detection (avg across attacks): {vis_avg:.2f}%")
    print(f"DCT extraction rate (avg): {dct_ext_avg:.2f}%")
    print(f"DCT BER (avg when extracted): {dct_ber_avg:.2%}" if dct_ber_avg is not None else "DCT BER: N/A")
    if neural_ext_avg is not None:
        print(f"Neural extraction rate (avg): {neural_ext_avg:.2f}%")
        print(f"Neural BER (avg when extracted): {neural_ber_avg:.2%}" if neural_ber_avg is not None else "Neural BER: N/A")
    print(f"\nTraceability score: {traceability_score:.1f}/100")
    if args.with_neural and neural_ext_avg is not None:
        print("  (50% visible + 30% DCT + 20% neural, BER-penalized)")
    else:
        print("  (60% visible + 40% DCT recovery, BER-penalized)")
    print("=" * 60)

    fieldnames = ["video", "visible_avg", "dct_extraction_rate", "dct_ber_avg"]
    if args.with_neural and all_results and "neural_extraction_rate" in all_results[0]:
        fieldnames.extend(["neural_extraction_rate", "neural_ber_avg"])
    summary_path = OUTPUT_DIR / "workflow_efficiency_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
        w.writerow({})
        agg = {"video": "AGGREGATE", "visible_avg": f"{vis_avg:.2f}", "dct_extraction_rate": f"{dct_ext_avg:.2f}", "dct_ber_avg": f"{dct_ber_avg:.2%}" if dct_ber_avg is not None else "N/A"}
        if neural_ext_avg is not None:
            agg["neural_extraction_rate"] = f"{neural_ext_avg:.2f}"
            agg["neural_ber_avg"] = f"{neural_ber_avg:.2%}" if neural_ber_avg is not None else "N/A"
        w.writerow(agg)
        w.writerow({"video": "traceability_score", "visible_avg": f"{traceability_score:.1f}", "dct_extraction_rate": "", "dct_ber_avg": ""})
    print(f"\nSummary saved: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
