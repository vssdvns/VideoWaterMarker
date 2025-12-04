from pathlib import Path
import cv2
import numpy as np


def compute_complexity_map(frame: np.ndarray) -> np.ndarray:
    """Return a normalized 'complexity' map based on edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    comp = np.abs(lap)
    comp = cv2.normalize(comp, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return comp


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    video_path = project_root / "data" / "input" / "sample.mp4"
    out_dir = project_root / "data" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Grab the first frame (you can change to any index later)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to read a frame from the video.")

    # Compute complexity map
    comp = compute_complexity_map(frame)

    # Convert to 0–255 uint8 for saving/colormap
    comp_norm = cv2.normalize(comp, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Color heatmap (blue→red style)
    comp_color = cv2.applyColorMap(comp_norm, cv2.COLORMAP_JET)

    # Overlay heatmap on original frame
    overlay = cv2.addWeighted(comp_color, 0.6, frame, 0.4, 0)

    # Save images
    cv2.imwrite(str(out_dir / "frame_original.jpg"), frame)
    cv2.imwrite(str(out_dir / "complexity_gray.jpg"), comp_norm)
    cv2.imwrite(str(out_dir / "complexity_heatmap.jpg"), comp_color)
    cv2.imwrite(str(out_dir / "overlay_heatmap_on_frame.jpg"), overlay)

    print(f"[INFO] Saved debug images to: {out_dir}")
