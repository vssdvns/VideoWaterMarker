from pathlib import Path
import cv2
import numpy as np

from models.saliency_deeplab import DeepLabSaliency
from video_watermark_demo import compute_complexity_map, choose_low_complexity_region

def compute_flow_farneback(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    return flow


def warp_map_with_flow(map2d: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = map2d.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    warped = cv2.remap(
        map2d.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def get_box_size(text: str = "VideoWaterMarker") -> tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    return text_w + 20, text_h + 20


def clamp_xy(x: int, y: int, width: int, height: int, box_w: int, box_h: int) -> tuple[int, int]:
    x = max(0, min(x, width - box_w - 1))
    y = max(0, min(y, height - box_h - 1))
    return x, y


def ema_smooth_positions(positions: list[tuple[int, int]], alpha: float = 0.8, max_jump: int = 150) -> list[tuple[int, int]]:
    """Apply the same EMA smoothing logic used in the watermarking code."""
    if not positions:
        return []

    max_jump_sq = max_jump * max_jump
    out = [positions[0]]
    prev_x, prev_y = positions[0]

    for (cx, cy) in positions[1:]:
        dx = cx - prev_x
        dy = cy - prev_y
        dist_sq = dx * dx + dy * dy

        if dist_sq <= max_jump_sq:
            x = int(alpha * prev_x + (1 - alpha) * cx)
            y = int(alpha * prev_y + (1 - alpha) * cy)
        else:
            x, y = cx, cy

        out.append((x, y))
        prev_x, prev_y = x, y

    return out


def jitter_stats(positions: list[tuple[int, int]], big_jump_thresh: int = 50) -> dict:
    if len(positions) < 2:
        return {"avg_move": 0.0, "max_move": 0.0, "pct_big_jumps": 0.0}

    moves = []
    big = 0
    for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        moves.append(d)
        if d >= big_jump_thresh:
            big += 1

    moves = np.array(moves, dtype="float32")
    return {
        "avg_move": float(moves.mean()),
        "max_move": float(moves.max()),
        "pct_big_jumps": float(100.0 * big / len(moves)),
    }


def mean_eval_saliency(eval_maps: list[np.ndarray], positions: list[tuple[int, int]], box_w: int, box_h: int) -> float:
    vals = []
    for m, (x, y) in zip(eval_maps, positions):
        region = m[y : y + box_h, x : x + box_w]
        vals.append(float(region.mean()))
    return float(np.mean(vals))


def main():
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "input" / "sample.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    box_w, box_h = get_box_size("VideoWaterMarker")

    # We'll evaluate placement quality using BOTH:
    # - Laplacian complexity maps
    # - DeepLab saliency maps (slower)
    deeplab = DeepLabSaliency()

    frames = []
    lap_maps = []
    dl_maps = []

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1

        frames.append(frame)
        lap_maps.append(compute_complexity_map(frame))
        dl_maps.append(deeplab.get_saliency_map(frame))

        if idx % 20 == 0:
            print(f"[STATS] Loaded {idx} frames...")

    cap.release()

    if not frames:
        raise RuntimeError("No frames read from video.")

    h, w = frames[0].shape[:2]

    # --- Build placements (unsmoothed) ---
    # Simple fixed
    simple_pos = []
    for _ in frames:
        x = w - box_w - 20
        y = h - box_h - 20
        simple_pos.append(clamp_xy(x, y, w, h, box_w, box_h))

    # Heuristic (Laplacian-based guide)
    heur_pos = []
    for m in lap_maps:
        x, y = choose_low_complexity_region(m, box_w, box_h, stride=32)
        heur_pos.append(clamp_xy(x, y, w, h, box_w, box_h))

    # DeepLab (DeepLab guide)
    dl_pos = []
    for m in dl_maps:
        x, y = choose_low_complexity_region(m, box_w, box_h, stride=32)
        dl_pos.append(clamp_xy(x, y, w, h, box_w, box_h))

    # --- Smoothed versions ---
    heur_pos_s = ema_smooth_positions(heur_pos, alpha=0.8, max_jump=150)
    dl_pos_s = ema_smooth_positions(dl_pos, alpha=0.8, max_jump=150)
    
    
    # DeepLab + Optical Flow fused guide map
    flow_beta = 0.7  # try 0.6/0.8 later if needed
    dl_flow_pos = []

    prev_frame = None
    prev_guide = None

    for frame, curr_dl in zip(frames, dl_maps):
        guide = curr_dl

        if prev_frame is not None and prev_guide is not None:
            flow = compute_flow_farneback(prev_frame, frame)
            prev_warped = warp_map_with_flow(prev_guide, flow)

            b = float(np.clip(flow_beta, 0.0, 1.0))
            guide = (b * guide + (1.0 - b) * prev_warped).astype(np.float32)
            guide = cv2.normalize(guide, None, 0.0, 1.0, cv2.NORM_MINMAX)

        x, y = choose_low_complexity_region(guide, box_w, box_h, stride=32)
        dl_flow_pos.append(clamp_xy(x, y, w, h, box_w, box_h))

        prev_frame = frame
        prev_guide = guide

    dl_flow_pos_s = ema_smooth_positions(dl_flow_pos, alpha=0.8, max_jump=150)
    
    # HYBRID(wD=0.50) + Optical Flow fused guide map
    wD_hybrid = 0.50
    flow_beta_h = 0.7

    hyb_flow_pos = []
    prev_frame = None
    prev_guide = None

    for frame, lap_m, dl_m in zip(frames, lap_maps, dl_maps):
        guide = (1.0 - wD_hybrid) * lap_m + wD_hybrid * dl_m
        guide = cv2.normalize(guide, None, 0.0, 1.0, cv2.NORM_MINMAX)

        if prev_frame is not None and prev_guide is not None:
            flow = compute_flow_farneback(prev_frame, frame)
            prev_warped = warp_map_with_flow(prev_guide, flow)

            b = float(np.clip(flow_beta_h, 0.0, 1.0))
            guide = (b * guide + (1.0 - b) * prev_warped).astype(np.float32)
            guide = cv2.normalize(guide, None, 0.0, 1.0, cv2.NORM_MINMAX)

        x, y = choose_low_complexity_region(guide, box_w, box_h, stride=32)
        hyb_flow_pos.append(clamp_xy(x, y, w, h, box_w, box_h))

        prev_frame = frame
        prev_guide = guide

    hyb_flow_pos_s = ema_smooth_positions(hyb_flow_pos, alpha=0.8, max_jump=150)

    
    # HYBRID placements (unsmoothed + smoothed) - using combined map
    # --- Compute metrics ---
    def report(name: str, pos: list[tuple[int, int]]):
        j = jitter_stats(pos, big_jump_thresh=50)
        lap_mean = mean_eval_saliency(lap_maps, pos, box_w, box_h)
        dl_mean = mean_eval_saliency(dl_maps, pos, box_w, box_h)
        print(f"\n{name}")
        print(f"  avg_move(px):   {j['avg_move']:.3f}")
        print(f"  max_move(px):   {j['max_move']:.3f}")
        print(f"  big_jumps(%):   {j['pct_big_jumps']:.2f}")
        print(f"  Laplacian mean: {lap_mean:.6f}  (lower=better)")
        print(f"  DeepLab mean:   {dl_mean:.6f}  (lower=better)")

    print("\n=== Temporal Stability + Placement Quality ===")
    report("SIMPLE (fixed)", simple_pos)
    report("HEURISTIC (unsmoothed)", heur_pos)
    report("HEURISTIC (smoothed)", heur_pos_s)
    report("DEEPLAB (unsmoothed)", dl_pos)
    report("DEEPLAB (smoothed)", dl_pos_s)
    report("DEEPLAB+FLOW (unsmoothed)", dl_flow_pos)
    report("DEEPLAB+FLOW (smoothed)", dl_flow_pos_s)


    # HYBRID placements (unsmoothed + smoothed) - using combined map
    for wD in [0.25, 0.50, 0.75]:
        hyb_pos = []
        for lap_m, dl_m in zip(lap_maps, dl_maps):
            combined = (1.0 - wD) * lap_m + wD * dl_m
            combined = cv2.normalize(combined, None, 0.0, 1.0, cv2.NORM_MINMAX)

            x, y = choose_low_complexity_region(combined, box_w, box_h, stride=32)
            hyb_pos.append(clamp_xy(x, y, w, h, box_w, box_h))

        hyb_pos_s = ema_smooth_positions(hyb_pos, alpha=0.8, max_jump=150)

        report(f"HYBRID(wD={wD:.2f}) (unsmoothed)", hyb_pos)
        report(f"HYBRID(wD={wD:.2f}) (smoothed)", hyb_pos_s)
        report("HYBRID(wD=0.50)+FLOW (unsmoothed)", hyb_flow_pos)
        report("HYBRID(wD=0.50)+FLOW (smoothed)", hyb_flow_pos_s)


if __name__ == "__main__":
    main()
