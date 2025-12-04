from pathlib import Path
import cv2


def add_text_watermark(
    input_path: Path,
    output_path: Path,
    text: str = "VideoWaterMarker",
    alpha: float = 0.5,
) -> None:
    """Add a semi-transparent text watermark to the bottom-right of an image."""

    print(f"[INFO] Reading image: {input_path}")
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    overlay = img.copy()
    h, w = img.shape[:2]
    print(f"[INFO] Image size: {w} x {h}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2

    # Measure text size to place it nicely
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = w - text_w - 20
    y = h - 20

    # Draw on overlay, then blend with original for transparency
    cv2.putText(
        overlay,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Blend overlay (with text) and original image
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # ✅ Save the *image* as second argument
    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), img)
    if not saved:
        raise RuntimeError(f"Failed to save output image to: {output_path}")
    print(f"[INFO] Saved watermarked image to: {output_path}")


if __name__ == "__main__":
    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parents[1] 
    in_path = project_root / "data" / "input" / "sample.jpg"
    out_path = project_root / "data" / "output" / "sample_watermarked.jpg"

    print(f"[INFO] Input path:  {in_path}")
    print(f"[INFO] Output path: {out_path}")

    add_text_watermark(in_path, out_path)
