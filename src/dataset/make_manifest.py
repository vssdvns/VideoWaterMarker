from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class ClipRow:
    id: str
    dataset: str
    label: str
    src_path: str


def iter_videos(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def infer_label(dataset: str, path: Path, root: Path) -> str:
    """
    UCF/HMDB: label = first directory under root (class folder) if present.
    DAVIS_VIDEOS: label = 'davis' (or you can use sequence name).
    """
    ds = dataset.lower()
    if ds in {"ucf101", "hmdb51"}:
        try:
            rel = path.relative_to(root)
            parts = rel.parts
            # expected: <class>/<file>
            return parts[0] if len(parts) >= 2 else "unknown"
        except Exception:
            return "unknown"
    return "davis"


def sample_paths(paths: list[Path], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    paths = list(paths)
    if len(paths) <= n:
        return paths
    return rng.sample(paths, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf_root", type=str, required=True)
    ap.add_argument("--hmdb_root", type=str, required=True)
    ap.add_argument("--davis_root", type=str, required=True)
    ap.add_argument("--n_each", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_csv", type=str, default="data/manifests/clips_150.csv")
    args = ap.parse_args()

    roots = [
        ("ucf101", Path(args.ucf_root)),
        ("hmdb51", Path(args.hmdb_root)),
        ("davis", Path(args.davis_root)),
    ]

    rows: list[ClipRow] = []

    for dataset, root in roots:
        if not root.exists():
            raise FileNotFoundError(f"{dataset} root not found: {root}")

        vids = sorted(iter_videos(root))
        if not vids:
            raise RuntimeError(f"No videos found under {dataset} root: {root}")

        picked = sample_paths(vids, args.n_each, args.seed)

        for p in picked:
            label = infer_label(dataset, p, root)
            clip_id = f"{dataset}_{label}_{p.stem}"
            rows.append(
                ClipRow(
                    id=clip_id,
                    dataset=dataset,
                    label=label,
                    src_path=str(p.resolve()),
                )
            )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "dataset", "label", "src_path"])
        for r in rows:
            w.writerow([r.id, r.dataset, r.label, r.src_path])

    print(f"Wrote {len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()