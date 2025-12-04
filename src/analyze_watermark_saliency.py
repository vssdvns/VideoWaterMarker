from pathlib import Path
import cv2
import numpy as np

from models.saliency_deeplab import DeepLabSaliency
from video_watermark_demo import compute_complexity_map, choose_low_complexity_region


def compute_watermark_box_positions(
    frame: np.ndarray,
    method: str,
    text: str = "VideoWaterMarker",
) -> tuple[int, int, int, int]:
    """
    Given a frame and a method name:
      - 'simple'    -> fixed bottom-right
      - 'heuristic' -> Laplacian-based placement
      - 'deeplab'   -> DeepLab-based placement
    Return (x, y, box_w, box_h).
    """
    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_w + 20
    box_h = text_h + 20

    if method == "simple":
        x = w - box_w - 20
        y = h - box_h - 20
        return x, y, box_w, box_h

    if method == "heuristic":
        guide_map = compute_complexity_map(frame)
        x, y = choose_low_complexity_region(guide_map, box_w, box_h, stride=32)
        return x, y, box_w, box_h

    if method == "deeplab":
        # For metrics we can share a global DeepLabSaliency instance
        raise RuntimeError("deeplab method requires saliency_model; see main()")

    raise ValueError(f"Unknown method: {method}")


def analyze_methods_on_video(
    video_path: Path,
    use_deeplab_for_eval: bool = False,
    max_frames: int | None = None,
) -> None:
    """
    Compare average saliency under the watermark region for:
      - simple
      - heuristic
      - deeplab

    Saliency for evaluation is:
      - Laplacian complexity (default)
      - or DeepLabSaliency if use_deeplab_for_eval=True
    """

    print(f"[METRIC] Opening video for analysis: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    saliency_model = None
    placement_deeplab_model = DeepLabSaliency()
    if use_deeplab_for_eval:
        saliency_model = placement_deeplab_model  # reuse same model

    sums = {"simple": 0.0, "heuristic": 0.0, "deeplab": 0.0}
    counts = {"simple": 0, "heuristic": 0, "deeplab": 0}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames is not None and frame_idx > max_frames:
            break

        h, w = frame.shape[:2]

        # Evaluation saliency map
        if use_deeplab_for_eval and saliency_model is not None:
            eval_map = saliency_model.get_saliency_map(frame)
        else:
            eval_map = compute_complexity_map(frame)

        # SIMPLE: fixed bottom-right
        x_s, y_s, bw, bh = compute_watermark_box_positions(frame, "simple")
        region_s = eval_map[y_s : y_s + bh, x_s : x_s + bw]
        sums["simple"] += float(region_s.mean())
        counts["simple"] += 1

        # HEURISTIC: Laplacian-based placement
        x_h, y_h, _, _ = compute_watermark_box_positions(frame, "heuristic")
        region_h = eval_map[y_h : y_h + bh, x_h : x_h + bw]
        sums["heuristic"] += float(region_h.mean())
        counts["heuristic"] += 1

        # DEEPLAB: DeepLab-based placement
        dl_saliency_map = placement_deeplab_model.get_saliency_map(frame)
        x_d, y_d = choose_low_complexity_region(dl_saliency_map, bw, bh, stride=32)
        region_d = eval_map[y_d : y_d + bh, x_d : x_d + bw]
        sums["deeplab"] += float(region_d.mean())
        counts["deeplab"] += 1

        if frame_idx % 20 == 0:
            print(f"[METRIC] Processed {frame_idx} frames...")

    cap.release()

    print("\n[METRIC] Average saliency under watermark region:")
    for method in ["simple", "heuristic", "deeplab"]:
        if counts[method] > 0:
            avg = sums[method] / counts[method]
        else:
            avg = float("nan")
        print(f"  {method:9s}: {avg:.6f}  (n={counts[method]})")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "input" / "sample.mp4"

    print("[METRIC] Using Laplacian complexity as evaluation saliency:")
    analyze_methods_on_video(
        video_path=video_path,
        use_deeplab_for_eval=False,
        max_frames=None,  # or set e.g. 100 if it's slow
    )

    print("\n[METRIC] Using DeepLab-based saliency as evaluation (slower):")
    analyze_methods_on_video(
        video_path=video_path,
        use_deeplab_for_eval=True,
        max_frames=100,  # limit frames for speed if you want
    )
