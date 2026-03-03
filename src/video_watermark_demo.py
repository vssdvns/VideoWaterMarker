from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from models.saliency_deeplab import DeepLabSaliency


def compute_flow_farneback(prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
    """Dense optical flow from prev -> curr using Farneback. Returns flow (H,W,2)."""
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
    """
    Warp a 2D map from prev frame into current frame using flow(prev->curr).
    map2d: (H,W) float32
    flow:  (H,W,2) float32 where flow[x,y] = (dx,dy)
    """
    h, w = map2d.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    # For remap: source coords in prev that map to current pixel.
    # current(x,y) came from prev(x - dx, y - dy)
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        map2d.astype(np.float32),
        map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def combine_maps(lap_map: np.ndarray, dl_map: np.ndarray, w_deeplab: float) -> np.ndarray:
    """
    Weighted fusion of Laplacian complexity and DeepLab saliency maps.
    Both maps should be normalized in [0,1].
    Returns combined map in [0,1] (approximately).
    """
    w_deeplab = float(np.clip(w_deeplab, 0.0, 1.0))
    w_lap = 1.0 - w_deeplab
    combined = (w_lap * lap_map) + (w_deeplab * dl_map)
    combined = cv2.normalize(combined, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return combined.astype("float32")



def compute_complexity_map(frame: np.ndarray) -> np.ndarray:
    """Return a normalized 'complexity' map based on edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    comp = np.abs(lap)
    comp = cv2.normalize(comp, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return comp.astype("float32")


def _in_edge_zone(x: int, y: int, box_w: int, box_h: int, w: int, h: int, edge_margin: float) -> bool:
    """True if window (x,y) is in the edge band (outer margin from each side)."""
    pad_w = max(box_w, int(w * edge_margin))
    pad_h = max(box_h, int(h * edge_margin))
    # Center of window
    cx = x + box_w / 2
    cy = y + box_h / 2
    # In edge zone if near left, right, top, or bottom
    return (
        cx <= pad_w or cx >= w - pad_w or
        cy <= pad_h or cy >= h - pad_h
    )


def choose_low_complexity_region(
    map2d: np.ndarray,
    box_w: int,
    box_h: int,
    stride: int = 32,
    prefer_edges: bool = False,
    edge_margin: float = 0.12,
) -> tuple[int, int]:
    """
    Slide a window over the given 2D map and return (x, y) for the
    top-left corner of the lowest-average-value window.
    (Low value = good place for watermark.)

    If prefer_edges=True, only consider windows in the outer edge band
    (within edge_margin of frame borders). This keeps the watermark near
    corners/edges for less intrusion and better crop resilience.
    """
    h, w = map2d.shape
    best_score = float("inf")
    fallback = (max(0, w - box_w - 20), max(0, h - box_h - 20))
    best_xy = fallback

    for y in range(0, max(h - box_h, 1), stride):
        for x in range(0, max(w - box_w, 1), stride):
            if prefer_edges and not _in_edge_zone(x, y, box_w, box_h, w, h, edge_margin):
                continue
            region = map2d[y : y + box_h, x : x + box_w]
            if region.shape[0] < box_h or region.shape[1] < box_w:
                continue
            score = float(region.mean())
            if score < best_score:
                best_score = score
                best_xy = (x, y)

    return best_xy


def add_text_watermark_fixed(
    input_path: Path,
    output_path: Path,
    text: str = "VideoWaterMarker",
    alpha: float = 0.5,
    use_saliency_model: bool = False,
    use_hybrid: bool = False,
    w_deeplab: float = 0.5,
    temporal_smoothing: bool = True,
    temporal_alpha: float = 0.8,
    max_jump: int = 150,
    save_positions: bool = False,
    positions_path: Path | None = None,
) -> None:
    """Simple baseline: fixed bottom-right watermark for all frames."""
    print(f"[FIXED] Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[FIXED] FPS: {fps}, Res: {width}x{height}, Frames: {frame_count}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    s = width / 1280.0
    font_scale = float(np.clip(1.0 * s, 0.8, 2.0))
    thickness = max(2, int(round(2 * s)))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Fixed bottom-right placement (same box size as others)
    box_w = text_w + 20
    box_h = text_h + 20
    x = width - box_w - 20
    y = height - box_h - 20
    text_x = x + 10
    text_y = y + box_h - 10

    positions = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)

        if save_positions and positions_path is not None:
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            positions.append({
                "frame": frame_idx, "t_ms": t_ms, "x": x, "y": y,
                "box_w": box_w, "box_h": box_h, "text": text,
                "font_scale": font_scale, "thickness": thickness, "alpha": alpha,
            })
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[FIXED] Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    out.release()
    if save_positions and positions_path and positions:
        import json
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(positions_path, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2)
    print(f"[FIXED] Saved watermarked video to: {output_path}")


def add_text_watermark_to_video(
    input_path: Path,
    output_path: Path,
    text: str = "VideoWaterMarker",
    alpha: float = 0.5,
    use_saliency_model: bool = False,

    use_hybrid: bool = False,
    w_deeplab: float = 0.5,

    temporal_smoothing: bool = True,
    temporal_alpha: float = 0.8,
    max_jump: int = 150,
    use_optical_flow: bool = False,
    flow_beta: float = 0.7,
    save_positions: bool = False,
    positions_path: Path | None = None,
    prefer_edges: bool = True,
    edge_margin: float = 0.12,

) -> None:
    """
    Add a semi-transparent text watermark per frame.

    If use_saliency_model=True:
        - Use DeepLab-based saliency map to avoid foreground.
    Else:
        - Use Laplacian complexity map.

    If temporal_smoothing=True:
        - Smooth watermark position over time to avoid jitter.
    """

    if use_hybrid:
        mode = f"HYBRID(wD={w_deeplab:.2f})"
    elif use_saliency_model:
        mode = "DEEPLAB"
    else:
        mode = "HEURISTIC"
    if use_optical_flow:
        mode += f"+FLOW(b={flow_beta:.2f})"

    print(f"[{mode}] Opening video: {input_path}")
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[{mode}] FPS: {fps}, Res: {width}x{height}, Frames: {frame_count}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # ✅ scale watermark with resolution (relative to 1280 baseline)
    s = width / 1280.0
    font_scale = float(np.clip(1.0 * s, 0.8, 2.0))
    thickness = max(2, int(round(2 * s)))

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 20
    box_h = text_h + 20

    saliency_model = None
    if use_saliency_model:
        saliency_model = DeepLabSaliency()

    prev_x, prev_y = None, None
    max_jump_sq = max_jump * max_jump

    frame_idx = 0
    prev_frame = None
    prev_guide_map = None
    positions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        lap_map = compute_complexity_map(frame)

        if use_hybrid:
            # Ensure DeepLab model exists
            if saliency_model is None:
                saliency_model = DeepLabSaliency()
            dl_map = saliency_model.get_saliency_map(frame)
            guide_map = combine_maps(lap_map, dl_map, w_deeplab=w_deeplab)

        elif use_saliency_model:
            if saliency_model is None:
                saliency_model = DeepLabSaliency()
            guide_map = saliency_model.get_saliency_map(frame)

        else:
            guide_map = lap_map
            
        # --- Optical-flow temporal consistency ---
        if use_optical_flow and prev_frame is not None and prev_guide_map is not None:
            flow = compute_flow_farneback(prev_frame, frame)
            prev_warped = warp_map_with_flow(prev_guide_map, flow)

            b = float(np.clip(flow_beta, 0.0, 1.0))
            guide_map = (b * guide_map + (1.0 - b) * prev_warped).astype(np.float32)
            guide_map = cv2.normalize(guide_map, None, 0.0, 1.0, cv2.NORM_MINMAX)





        # Instantaneous best candidate from the guide map
        cand_x, cand_y = choose_low_complexity_region(
            guide_map, box_w, box_h, stride=32,
            prefer_edges=prefer_edges, edge_margin=edge_margin,
        )

        # Temporal smoothing of position
        if temporal_smoothing and prev_x is not None and prev_y is not None:
            dx = cand_x - prev_x
            dy = cand_y - prev_y
            dist_sq = dx * dx + dy * dy

            if dist_sq <= max_jump_sq:
                # Smooth: EMA between previous and candidate
                x = int(temporal_alpha * prev_x + (1.0 - temporal_alpha) * cand_x)
                y = int(temporal_alpha * prev_y + (1.0 - temporal_alpha) * cand_y)
            else:
                # Likely a scene change; allow jump
                x, y = cand_x, cand_y
        else:
            x, y = cand_x, cand_y

        # Clamp so the box stays fully inside frame
        x = max(0, min(x, width - box_w - 1))
        y = max(0, min(y, height - box_h - 1))

        prev_x, prev_y = x, y
        
        if save_positions:
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            positions.append({
                "frame": int(frame_idx),
                "t_ms": t_ms,                      # ✅ ADD THIS
                "x": int(x),
                "y": int(y),
                "box_w": int(box_w),
                "box_h": int(box_h),
                "text": text,
                "mode": mode,
                "w_deeplab": float(w_deeplab),
                "flow_beta": float(flow_beta),
                "temporal_alpha": float(temporal_alpha),
                "max_jump": int(max_jump),
                "alpha": float(alpha),
                "font_scale": float(font_scale),
                "thickness": int(thickness),
            })

        overlay = frame.copy()
        text_x = x + 10
        text_y = y + box_h - 10

        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)
        prev_frame = frame.copy()
        prev_guide_map = guide_map.copy()

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"[{mode}] Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"[{mode}] Saved watermarked video to: {output_path}")
    if save_positions:
        import json
        if positions_path is None:
            positions_path = output_path.with_suffix(".positions.json")
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(positions_path, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2)
        print(f"[{mode}] Saved positions to: {positions_path}")



def add_text_watermark_from_positions(
    input_path: Path,
    output_path: Path,
    positions_path: Path,
    text: str = "VideoWaterMarker",
    alpha: float = 0.5,
) -> None:
    """
    Apply watermark at positions from a JSON file (for manual-override / re-export).
    positions_path: path to positions.json (from previous watermark run or UI editor).
    """
    import json

    with open(positions_path, "r", encoding="utf-8") as f:
        positions = json.load(f)
    if not positions:
        raise ValueError("positions.json is empty")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_idx = 0
    pos_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pos_idx = min(frame_idx, len(positions) - 1)
        p = positions[pos_idx]
        x = int(p["x"])
        y = int(p["y"])
        box_w = int(p["box_w"])
        box_h = int(p["box_h"])
        font_scale = float(p.get("font_scale", 1.0))
        thickness = int(p.get("thickness", 2))

        x = max(0, min(x, width - box_w - 1))
        y = max(0, min(y, height - box_h - 1))
        text_x = x + 10
        text_y = y + box_h - 10

        overlay = frame.copy()
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()


def add_text_watermark_fixed_at(
    input_path: Path,
    output_path: Path,
    x: int,
    y: int,
    text: str = "VideoWaterMarker",
    alpha: float = 0.5,
    positions_path: Path | None = None,
) -> None:
    """
    Fixed watermark at user-specified (x, y) for all frames.
    Optionally saves positions.json for detection.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    font = cv2.FONT_HERSHEY_SIMPLEX
    s = width / 1280.0
    font_scale = float(np.clip(1.0 * s, 0.8, 2.0))
    thickness = max(2, int(round(2 * s)))
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 20
    box_h = text_h + 20

    x = max(0, min(x, width - box_w - 1))
    y = max(0, min(y, height - box_h - 1))
    text_x = x + 10
    text_y = y + box_h - 10

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    positions = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        out.write(frame)
        if positions_path is not None:
            t_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
            positions.append({
                "frame": frame_idx, "t_ms": t_ms, "x": x, "y": y,
                "box_w": box_w, "box_h": box_h, "text": text,
                "font_scale": font_scale, "thickness": thickness, "alpha": alpha,
            })
        frame_idx += 1

    cap.release()
    out.release()
    if positions_path and positions:
        import json
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(positions_path, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2)


def get_frame_heatmap(
    video_path: Path,
    frame_idx: int,
    method: str = "hybrid",
    w_deeplab: float = 0.6,
    saliency_model: DeepLabSaliency | None = None,
    prefer_edges: bool = True,
    edge_margin: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, float, int]:
    """
    Get frame at index, compute heatmap, and suggested watermark position.
    Returns: (frame_bgr, heatmap_overlay_rgb, x, y, box_w, box_h, font_scale, thickness)
    method: "fixed" | "heuristic" | "deeplab" | "hybrid"
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read frame {frame_idx}")

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    s = w / 1280.0
    font_scale = float(np.clip(1.0 * s, 0.8, 2.0))
    thickness = max(2, int(round(2 * s)))
    (text_w, text_h), _ = cv2.getTextSize("VideoWaterMarker", font, font_scale, thickness)
    box_w = text_w + 20
    box_h = text_h + 20

    if method == "fixed":
        x = w - box_w - 20
        y = h - box_h - 20
        guide_map = compute_complexity_map(frame)
    else:
        lap_map = compute_complexity_map(frame)
        if method == "heuristic":
            guide_map = lap_map
        else:
            if saliency_model is None:
                saliency_model = DeepLabSaliency()
            dl_map = saliency_model.get_saliency_map(frame)
            if method == "deeplab":
                guide_map = dl_map
            else:
                guide_map = combine_maps(lap_map, dl_map, w_deeplab)
        x, y = choose_low_complexity_region(
            guide_map, box_w, box_h, stride=32,
            prefer_edges=prefer_edges, edge_margin=edge_margin,
        )

    x = max(0, min(x, w - box_w - 1))
    y = max(0, min(y, h - box_h - 1))

    guide_norm = cv2.normalize(guide_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(guide_norm, cv2.COLORMAP_JET)
    heatmap_overlay = cv2.addWeighted(frame, 0.6, heatmap_rgb, 0.4, 0)

    return frame, heatmap_overlay, x, y, box_w, box_h, font_scale, thickness


def interpolate_keyframes_to_positions(
    keyframes: list[tuple[int, int, int]],
    frame_count: int,
    base_pos: dict,
    fps: float = 30.0,
) -> list[dict]:
    """
    Interpolate (x, y) from keyframes to all frames.
    keyframes: [(frame_idx, x, y), ...] sorted by frame_idx.
    base_pos: template dict with box_w, box_h, text, font_scale, thickness, alpha.
    """
    if not keyframes:
        return []
    keyframes = sorted(keyframes, key=lambda k: k[0])

    def interp(frame_idx: int) -> tuple[int, int]:
        if frame_idx <= keyframes[0][0]:
            return keyframes[0][1], keyframes[0][2]
        if frame_idx >= keyframes[-1][0]:
            return keyframes[-1][1], keyframes[-1][2]
        for i in range(len(keyframes) - 1):
            f0, x0, y0 = keyframes[i]
            f1, x1, y1 = keyframes[i + 1]
            if f0 <= frame_idx <= f1:
                t = (frame_idx - f0) / max(1, f1 - f0)
                x = int(x0 + t * (x1 - x0))
                y = int(y0 + t * (y1 - y0))
                return x, y
        return keyframes[-1][1], keyframes[-1][2]

    positions = []
    for fidx in range(frame_count):
        x, y = interp(fidx)
        t_ms = 1000.0 * fidx / max(1.0, fps)
        p = dict(base_pos)
        p["frame"] = fidx
        p["t_ms"] = t_ms
        p["x"] = x
        p["y"] = y
        positions.append(p)
    return positions


def add_text_watermark_from_keyframes(
    input_path: Path,
    output_path: Path,
    keyframes: list[tuple[int, int, int]],
    text: str = "VideoWaterMarker",
    alpha: float = 0.5,
    positions_path: Path | None = None,
) -> None:
    """
    Apply watermark using keyframe-interpolated positions.
    keyframes: [(frame_idx, x, y), ...]. Positions for other frames are interpolated.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    font = cv2.FONT_HERSHEY_SIMPLEX
    s = width / 1280.0
    font_scale = float(np.clip(1.0 * s, 0.8, 2.0))
    thickness = max(2, int(round(2 * s)))
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 20
    box_h = text_h + 20

    base_pos = {
        "box_w": box_w, "box_h": box_h, "text": text,
        "font_scale": font_scale, "thickness": thickness, "alpha": alpha,
    }
    positions = interpolate_keyframes_to_positions(keyframes, frame_count, base_pos, fps)

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json
        json.dump(positions, f, indent=2)
        tmp_path = Path(f.name)
    try:
        add_text_watermark_from_positions(
            input_path, output_path, tmp_path,
            text=text, alpha=alpha,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if positions_path and positions:
        import json
        positions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(positions_path, "w", encoding="utf-8") as f:
            json.dump(positions, f, indent=2)


if __name__ == "__main__":
    from pathlib import Path

    # Choose what to run
    MODE = "ablation"   # "final" or "ablation"

    project_root = Path(__file__).resolve().parents[1]
    in_path = project_root / "data" / "input" / "sample.mp4"

    # ---------- FINAL: one output only ----------
    out_final = project_root / "data" / "output" / "final_watermarked.mp4"
    save_positions=True,
    positions_path=out_final.with_suffix(".positions.json"),

    # ---------- ABLATION outputs ----------
    
    out_deeplab_flow = project_root / "data" / "output" / "watermarked_deeplab_flow" / "sample.mp4"
    out_hybrid_50_flow = project_root / "data" / "output" / "hybrid_wd0.50_flow" / "sample.mp4"
    
    
    out_simple = project_root / "data" / "output" / "simple" / "sample.mp4"
    out_heuristic = project_root / "data" / "output" / "heuristic" / "sample.mp4"
    out_deeplab = project_root / "data" / "output" / "deeplab" / "sample.mp4"
    out_hybrid_25 = project_root / "data" / "output" / "hybrid_wd0.25" / "sample.mp4"
    out_hybrid_50 = project_root / "data" / "output" / "hybrid_wd0.50" / "sample.mp4"
    out_hybrid_75 = project_root / "data" / "output" / "hybrid_wd0.75" / "sample.mp4"

    print(f"[INFO] Input video: {in_path}")
    print(f"[INFO] MODE: {MODE}")

    if MODE == "final":
        print(f"[INFO] Final output: {out_final}")

        # FINAL method: HYBRID(wD=0.50) + FLOW(beta=0.7) + EMA smoothing
        add_text_watermark_to_video(
            input_path=in_path,
            output_path=out_final,
            text="VideoWaterMarker",
            alpha=0.5,
            use_hybrid=True,
            w_deeplab=0.50,
            use_optical_flow=True,
            flow_beta=0.7,
            temporal_smoothing=True,
            temporal_alpha=0.8,
            max_jump=150,
            
            # next phase (detection): enable these when ready
            save_positions=True,
            positions_path=out_final.with_suffix(".positions.json"),
        )

    elif MODE == "ablation":
        print("[INFO] Running ablation suite...")

        # 1) Simple fixed watermark
        add_text_watermark_fixed(
            input_path=in_path,
            output_path=out_simple,
            text="VideoWaterMarker",
            alpha=0.5,
        )

        # 2) Heuristic Laplacian-based placement + EMA
        add_text_watermark_to_video(
            input_path=in_path,
            output_path=out_heuristic,
            text="VideoWaterMarker",
            alpha=0.5,
            use_saliency_model=False,
            temporal_smoothing=True,
            temporal_alpha=0.8,
            max_jump=150,
        )

        # 3) DeepLab placement + EMA
        add_text_watermark_to_video(
            input_path=in_path,
            output_path=out_deeplab,
            text="VideoWaterMarker",
            alpha=0.5,
            use_saliency_model=True,
            temporal_smoothing=True,
            temporal_alpha=0.8,
            max_jump=150,
        )

        # 4) Hybrid sweeps + EMA
        for wD, out_path in [
            (0.25, out_hybrid_25),
            (0.50, out_hybrid_50),
            (0.75, out_hybrid_75),
        ]:
            add_text_watermark_to_video(
                input_path=in_path,
                output_path=out_path,
                text="VideoWaterMarker",
                alpha=0.5,
                use_hybrid=True,
                w_deeplab=wD,
                temporal_smoothing=True,
                temporal_alpha=0.8,
                max_jump=150,
            )

        # 5) DeepLab + FLOW + EMA
        add_text_watermark_to_video(
            input_path=in_path,
            output_path=out_deeplab_flow,
            text="VideoWaterMarker",
            alpha=0.5,
            use_saliency_model=True,
            use_optical_flow=True,
            flow_beta=0.7,
            temporal_smoothing=True,
            temporal_alpha=0.8,
            max_jump=150,
        )

        # 6) Hybrid(wD=0.50) + FLOW + EMA
        add_text_watermark_to_video(
            input_path=in_path,
            output_path=out_hybrid_50_flow,
            text="VideoWaterMarker",
            alpha=0.5,
            use_hybrid=True,
            w_deeplab=0.50,
            use_optical_flow=True,
            flow_beta=0.7,
            temporal_smoothing=True,
            temporal_alpha=0.8,
            max_jump=150,
        )

    else:
        raise ValueError("MODE must be 'final' or 'ablation'")

