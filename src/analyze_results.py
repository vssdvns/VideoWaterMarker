from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="data/bench_hd/results/benchmark_results.csv",
        help="Path to benchmark_results.csv",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    # sanity: show a quick hint which run this is
    print(f"\n[ANALYZE] CSV -> {csv_path.resolve()}")
    print(f"[ANALYZE] rows={len(df)}  clips={df['clip_id'].nunique()}  attacks={df['attack'].nunique()}")

    print("\n=== Mean detection per attack ===")
    print(df.groupby("attack")["detected_rate_percent"].mean().sort_values(ascending=False))

    print("\n=== Overall mean detection ===")
    print(df["detected_rate_percent"].mean())


if __name__ == "__main__":
    main()