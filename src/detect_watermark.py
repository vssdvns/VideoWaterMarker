from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from .dct_watermark import extract_dct_watermark
except ImportError:
    extract_dct_watermark = None


def build_time_index(positions):
    # positions sorted by time
    times = np.array([p.get("t_ms", 0.0) for p in positions], dtype=np.float32)
    return times


def find_nearest_pos_idx(times: np.ndarray, t_ms: float) -> int:
    i = int(np.searchsorted(times, t_ms))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    if abs(t_ms - float(times[i - 1])) <= abs(float(times[i]) - t_ms):
        return i - 1
    return i


def build_text_template(text: str, box_w: int, box_h: int, font_scale: float, thickness: int) -> np.ndarray:
    tpl = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(tpl, text, (10, box_h - 10), font, float(font_scale), (255, 255, 255), int(thickness), cv2.LINE_AA)
    return cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def ncc_max_multiscale(
    search_g: np.ndarray,
    template_g: np.ndarray,
    scales=(0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40),
) -> float:
    """
    Best NCC over multiple template scales.
    Returns -1 if no valid scale fits in search region.
    """
    best = -1.0
    th0, tw0 = template_g.shape[:2]
    sh, sw = search_g.shape[:2]

    for s in scales:
        tw = max(5, int(round(tw0 * s)))
        th = max(5, int(round(th0 * s)))

        if tw > sw or th > sh:
            continue

        tpl = cv2.resize(template_g, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search_g, tpl, cv2.TM_CCOEFF_NORMED)
        m = float(res.max())
        if m > best:
            best = m

    return best


def global_fallback_score(
    frame_g: np.ndarray,
    template_g: np.ndarray,
    downscale: float = 0.5,
    scales=(0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40),
) -> float:
    """
    Global search across whole frame (optionally downscaled for speed).
    IMPORTANT: if we downscale the frame, we must downscale the template too.
    """
    ds = float(downscale) if downscale is not None else 1.0
    if not (0.0 < ds <= 1.0):
        ds = 1.0

    if ds < 1.0:
        h, w = frame_g.shape[:2]
        nw = max(64, int(round(w * ds)))
        nh = max(64, int(round(h * ds)))
        frame_g_small = cv2.resize(frame_g, (nw, nh), interpolation=cv2.INTER_AREA)

        th, tw = template_g.shape[:2]
        ntw = max(10, int(round(tw * ds)))
        nth = max(10, int(round(th * ds)))
        template_small = cv2.resize(template_g, (ntw, nth), interpolation=cv2.INTER_AREA)

        return ncc_max_multiscale(frame_g_small, template_small, scales=scales)

    return ncc_max_multiscale(frame_g, template_g, scales=scales)

def detect_video(
    video_path: Path,
    positions,
    template_g,
    frame_step=5,
    search_pad=60,
    thr=0.35,
    enable_global_fallback=True,
    global_downscale=0.5,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    box_w = int(positions[0]["box_w"])
    box_h = int(positions[0]["box_h"])
    times = build_time_index(positions)

    scores = []
    total = 0
    hit = 0
    dct_payload: Optional[bytes] = None
    has_dct_roi = bool(positions and positions[0].get("dct_roi"))
    dct_strength = 5.0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % frame_step != 0:
            idx += 1
            continue

        h, w = frame.shape[:2]
        t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        pos_idx = find_nearest_pos_idx(times, t_ms)
        pos = positions[pos_idx]
        if bool(pos.get("skip_visible", False) or not pos.get("visible_watermark", True)):
            idx += 1
            continue

        x = int(pos["x"])
        y = int(pos["y"])

        # ---- Stage A: local ROI ----
        x0 = clamp(x - search_pad, 0, w - 1)
        y0 = clamp(y - search_pad, 0, h - 1)
        x1 = clamp(x + box_w + search_pad, 0, w)
        y1 = clamp(y + box_h + search_pad, 0, h)

        roi = frame[y0:y1, x0:x1]
        local_score = -1.0
        if roi.size > 0:
            roi_g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            local_score = ncc_max_multiscale(roi_g, template_g)

        best_score = local_score

        # ---- Stage B: global fallback if local fails ----
        if enable_global_fallback and (best_score < thr):
            frame_g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gscore = global_fallback_score(
                frame_g, template_g, downscale=global_downscale
            )
            if gscore > best_score:
                best_score = gscore
        if best_score < 0:
            best_score = 0.0
            
        scores.append(best_score)
        total += 1
        if best_score >= thr:
            hit += 1

        # DCT payload extraction (invisible watermark)
        if extract_dct_watermark is not None and has_dct_roi and dct_payload is None:
            roi_info = pos.get("dct_roi", {})
            if roi_info:
                rx = int(roi_info.get("x", 0))
                ry = int(roi_info.get("y", 0))
                rw = int(roi_info.get("w", 64))
                rh = int(roi_info.get("h", 64))
                roi = (rx, ry, rw, rh)
                extracted = extract_dct_watermark(frame, roi, strength=dct_strength, use_ecc=False)
                if extracted and len(extracted) > 0:
                    dct_payload = extracted

        idx += 1

    cap.release()

    if total == 0:
        return {"total": 0, "hit": 0, "rate": 0.0, "mean": 0.0, "p50": 0.0, "p10": 0.0, "p90": 0.0, "dct_payload": None}

    scores_np = np.array(scores, dtype=np.float32)
    result = {
        "total": total,
        "hit": hit,
        "rate": 100.0 * hit / total,
        "mean": float(scores_np.mean()),
        "p10": float(np.percentile(scores_np, 10)),
        "p50": float(np.percentile(scores_np, 50)),
        "p90": float(np.percentile(scores_np, 90)),
    }
    if dct_payload is not None:
        result["dct_payload"] = dct_payload
    else:
        result["dct_payload"] = None
    return result


def main():
    root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks_dir", type=str, default=str(root / "data" / "attacks"))
    parser.add_argument("--pos_json", type=str, default=str(root / "data" / "output" / "final_watermarked.positions.json"))
    parser.add_argument("--csv_out", type=str, default=str(root / "data" / "output" / "robustness_results.csv"))

    parser.add_argument("--frame_step", type=int, default=5)
    parser.add_argument("--search_pad", type=int, default=60)
    parser.add_argument("--thr", type=float, default=0.35)

    # NEW:
    parser.add_argument("--global_fallback", action="store_true", help="Enable global search if local ROI fails")
    parser.add_argument("--global_downscale", type=float, default=0.5, help="Downscale factor for global search (0.5 = half-res)")

    args = parser.parse_args()

    attacks_dir = Path(args.attacks_dir)
    pos_json = Path(args.pos_json)

    positions = json.loads(pos_json.read_text(encoding="utf-8"))

    text = positions[0].get("text", "VideoWaterMarker")
    box_w = int(positions[0]["box_w"])
    box_h = int(positions[0]["box_h"])

    # ✅ NEW: read actual embedding parameters
    font_scale = float(positions[0].get("font_scale", 1.0))
    thickness  = int(positions[0].get("thickness", 2))

    template_g = build_text_template(
        text,
        box_w,
        box_h,
        font_scale,
        thickness,
    )

    vids = sorted(attacks_dir.glob("*.mp4"))
    if not vids:
        print("No attacked videos found in:", attacks_dir)
        return

    print(f"[DETECT] Using threshold={args.thr}, frame_step={args.frame_step}, search_pad={args.search_pad}")
    print(f"[DETECT] Positions: {pos_json.name} ({len(positions)} frames)")
    print(f"[DETECT] Global fallback: {bool(args.global_fallback)} (downscale={args.global_downscale})")

    results = []
    for vp in vids:
        r = detect_video(
            vp,
            positions,
            template_g,
            frame_step=args.frame_step,
            search_pad=args.search_pad,
            thr=args.thr,
            enable_global_fallback=bool(args.global_fallback),
            global_downscale=args.global_downscale,
        )

        print(f"\n{vp.name}")
        print(f"  detected: {r['hit']}/{r['total']}  ({r['rate']:.2f}%)")
        print(f"  NCC mean: {r['mean']:.4f} | p10/p50/p90: {r['p10']:.4f}/{r['p50']:.4f}/{r['p90']:.4f}")

        results.append({
            "attack": vp.name,
            "detected_rate_percent": round(r["rate"], 2),
            "mean_ncc": round(r["mean"], 4),
            "p10_ncc": round(r["p10"], 4),
            "p50_ncc": round(r["p50"], 4),
            "p90_ncc": round(r["p90"], 4),
        })

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved CSV to:", csv_path)


if __name__ == "__main__":
    main()