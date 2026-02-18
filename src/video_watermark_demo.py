from pathlib import Path
import cv2
import numpy as np

from models.saliency_deeplab import DeepLabSaliency

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
    font_scale = 1.0
    thickness = 2
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
    font_scale = 1.0
    thickness = 2

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 20
    box_h = text_h + 20

    saliency_model = None
    if use_saliency_model:
        saliency_model = DeepLabSaliency()

    prev_x, prev_y = None, None
    max_jump_sq = max_jump * max_jump

    frame_idx = 0
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

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"[{mode}] Processed {frame_idx}/{frame_count} frames...")

    cap.release()
    out.release()
    print(f"[{mode}] Saved watermarked video to: {output_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    in_path = project_root / "data" / "input" / "sample.mp4"

    out_simple = project_root / "data" / "output" / "sample_watermarked_simple.mp4"
    out_heuristic = project_root / "data" / "output" / "sample_watermarked_heuristic_smooth.mp4"
    out_deeplab = project_root / "data" / "output" / "sample_watermarked_deeplab_smooth.mp4"
    
    out_hybrid_50 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.50.mp4"
    out_hybrid_70 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.70.mp4"
    out_hybrid_25 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.25.mp4"
    out_hybrid_75 = project_root / "data" / "output" / "sample_watermarked_hybrid_wd0.75.mp4"



    print(f"[INFO] Input video:      {in_path}")
    print(f"[INFO] Simple output:    {out_simple}")
    print(f"[INFO] Heuristic output: {out_heuristic}")
    print(f"[INFO] DeepLab output:   {out_deeplab}")

    # 1) Simple fixed watermark (no smoothing needed)
    add_text_watermark_fixed(
        input_path=in_path,
        output_path=out_simple,
        text="VideoWaterMarker",
        alpha=0.5,
    )

    # 2) Heuristic Laplacian-based placement with temporal smoothing
    add_text_watermark_to_video(
        input_path=in_path,
        output_path=out_heuristic,
        text="VideoWaterMarker",
        alpha=0.5,
        use_saliency_model=False,
        temporal_smoothing=True,
    )

    # 3) DeepLab-based saliency placement with temporal smoothing
    add_text_watermark_to_video(
        input_path=in_path,
        output_path=out_deeplab,
        text="VideoWaterMarker",
        alpha=0.5,
        use_saliency_model=True,
        temporal_smoothing=True,
    )
    # 4) Hybrid (balanced)
    add_text_watermark_to_video(
        input_path=in_path,
        output_path=out_hybrid_50,
        text="VideoWaterMarker",
        alpha=0.5,
        use_hybrid=True,
        w_deeplab=0.50,
        temporal_smoothing=True,
    )

    # 5) Hybrid (more semantic avoidance)
    add_text_watermark_to_video(
        input_path=in_path,
        output_path=out_hybrid_70,
        text="VideoWaterMarker",
        alpha=0.5,
        use_hybrid=True,
        w_deeplab=0.70,
        temporal_smoothing=True,
    )
    # 6) Hybrid (more heuristic / edge-smoothness)
    add_text_watermark_to_video(
        input_path=in_path,
        output_path=out_hybrid_25,
        text="VideoWaterMarker",
        alpha=0.5,
        use_hybrid=True,
        w_deeplab=0.25,
        temporal_smoothing=True,
    )

    # 7) Hybrid (more semantic avoidance)
    add_text_watermark_to_video(
        input_path=in_path,
        output_path=out_hybrid_75,
        text="VideoWaterMarker",
        alpha=0.5,
        use_hybrid=True,
        w_deeplab=0.75,
        temporal_smoothing=True,
    )
