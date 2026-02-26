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


def choose_low_complexity_region(
    map2d: np.ndarray,
    box_w: int,
    box_h: int,
    stride: int = 32,
) -> tuple[int, int]:
    """
    Slide a window over the given 2D map and return (x, y) for the
    top-left corner of the lowest-average-value window.
    (Low value = good place for watermark.)
    """
    h, w = map2d.shape
    best_score = float("inf")
    best_xy = (w - box_w - 20, h - box_h - 20)  # fallback: bottom-right-ish

    for y in range(0, max(h - box_h, 1), stride):
        for x in range(0, max(w - box_w, 1), stride):
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

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"[FIXED] Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    out.release()
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
        cand_x, cand_y = choose_low_complexity_region(guide_map, box_w, box_h, stride=32)

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



if __name__ == "__main__":
    from pathlib import Path

    # Choose what to run
    MODE = "final"   # "final" or "ablation"

    project_root = Path(__file__).resolve().parents[1]
    in_path = project_root / "data" / "input" / "sample.mp4"

    # ---------- FINAL: one output only ----------
    out_final = project_root / "data" / "output" / "final_watermarked.mp4"
    save_positions=True,
    positions_path=out_final.with_suffix(".positions.json"),

    # ---------- ABLATION outputs ----------
    out_simple = project_root / "data" / "output" / "sample_watermarked_simple.mp4"
    out_heuristic = project_root / "data" / "output" / "sample_watermarked_heuristic_smooth.mp4"
    out_deeplab = project_root / "data" / "output" / "sample_watermarked_deeplab_smooth.mp4"
    out_hybrid_25 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.25.mp4"
    out_hybrid_50 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.50.mp4"
    out_hybrid_75 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.75.mp4"
    out_deeplab_flow = project_root / "data" / "output" / "sample_watermarked_deeplab_flow.mp4"
    out_hybrid_50_flow = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.50_flow.mp4"

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

