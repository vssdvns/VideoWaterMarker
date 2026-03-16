"""
FastAPI microservice for OTT pipeline integration (proposal-aligned).

Exposes the watermarking pipeline as a REST API for just-in-time embedding.
Designed to plug into HLS/DASH adaptive streaming workflows.
"""

from __future__ import annotations

import io
import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np

# Project imports (lazy to support optional deps)
ROOT = Path(__file__).resolve().parents[1]


def _ensure_api_dirs():
    base = ROOT / "data" / "api"
    (base / "input").mkdir(parents=True, exist_ok=True)
    (base / "output").mkdir(parents=True, exist_ok=True)
    return base


router = APIRouter(prefix="/api/v1", tags=["watermark"])


@router.get("/health")
def health():
    """Health check for load balancers and orchestration."""
    return {"status": "ok", "service": "VideoWaterMarker"}


@router.post("/watermark")
async def watermark_video(
    video: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None),
    method: str = Form("Hybrid"),
    alpha: float = Form(0.5),
    use_dct: bool = Form(True),
):
    """
    Watermark an uploaded video with user/session fingerprint.

    Returns the watermarked video file and positions JSON for detection.
    """
    base = _ensure_api_dirs()
    job_id = str(uuid.uuid4())[:8]

    content = await video.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty video file")

    ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    inp_path = base / "input" / f"{job_id}{ext}"
    inp_path.write_bytes(content)

    out_path = base / "output" / f"{job_id}_watermarked.mp4"
    pos_path = base / "output" / f"{job_id}_positions.json"

    try:
        from src.video_watermark_demo import add_text_watermark_fixed, add_text_watermark_to_video
        from src.fingerprint import encode_fingerprint
        from src.crypto_payload import encode_payload

        text = encode_fingerprint(user_id or "anonymous", None, None)
        dct_payload = None
        if use_dct and user_id:
            payload_bytes, _ = encode_payload(
                user_id, session_id, None, None,
                secret_key=None, use_aes=False, use_ecc=True,
            )
            dct_payload = payload_bytes

        if method == "Fixed":
            add_text_watermark_fixed(
                input_path=inp_path,
                output_path=out_path,
                text=text,
                alpha=alpha,
                save_positions=True,
                positions_path=pos_path,
            )
        else:
            add_text_watermark_to_video(
                input_path=inp_path,
                output_path=out_path,
                text=text,
                alpha=alpha,
                use_saliency_model=(method in ("DeepLab", "Hybrid")),
                use_hybrid=(method == "Hybrid"),
                w_deeplab=0.6,
                save_positions=True,
                positions_path=pos_path,
                embed_dct_payload=dct_payload,
            )

        if not out_path.exists():
            raise HTTPException(status_code=500, detail="Watermarking failed")

        return FileResponse(
            out_path,
            media_type="video/mp4",
            filename=f"watermarked_{job_id}.mp4",
            headers={"X-Positions-Path": str(pos_path)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if inp_path.exists():
            inp_path.unlink(missing_ok=True)


@router.post("/watermark/positions")
async def get_positions(job_id: str = Form(...)):
    """Retrieve positions JSON for a watermark job (for detection)."""
    base = _ensure_api_dirs()
    pos_path = base / "output" / f"{job_id}_positions.json"
    if not pos_path.exists():
        raise HTTPException(status_code=404, detail="Positions not found")
    with open(pos_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)


@router.post("/detect")
async def detect_watermark(
    video: UploadFile = File(...),
    positions: UploadFile = File(...),
    threshold: float = Form(0.40),
):
    """
    Detect watermark and extract fingerprint from video.

    Requires the positions JSON from the watermarking step.
    """
    base = _ensure_api_dirs()
    job_id = str(uuid.uuid4())[:8]

    vid_content = await video.read()
    pos_content = await positions.read()
    if not vid_content:
        raise HTTPException(status_code=400, detail="Empty video")

    ext = Path(video.filename or "video.mp4").suffix or ".mp4"
    vid_path = base / "input" / f"{job_id}_detect{ext}"
    vid_path.write_bytes(vid_content)

    pos_data = json.loads(pos_content)
    template_text = pos_data[0].get("text", "VideoWaterMarker") if pos_data else "VideoWaterMarker"
    box_w = pos_data[0].get("box_w", 200) if pos_data else 200
    box_h = pos_data[0].get("box_h", 50) if pos_data else 50

    try:
        from src.detect_watermark import detect_video, build_text_template

        template_g = build_text_template(template_text, box_w, box_h, 1.0, 2)
        result = detect_video(
            vid_path,
            pos_data,
            template_g,
            thr=threshold,
            enable_global_fallback=True,
        )

        # Decode DCT payload if present
        dct_decoded = None
        if result.get("dct_payload"):
            try:
                from src.crypto_payload import decode_payload
                dct_decoded = decode_payload(
                    result["dct_payload"],
                    use_aes=False, use_ecc=True,
                )
            except Exception:
                pass

        return {
            "detection_rate": result["rate"],
            "frames_checked": result["total"],
            "fingerprint_text": template_text,
            "dct_payload_decoded": dct_decoded,
            "raw": {k: v for k, v in result.items() if k != "dct_payload"},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if vid_path.exists():
            vid_path.unlink(missing_ok=True)


def create_app():
    """Create FastAPI app with router (for uvicorn --factory)."""
    try:
        from fastapi import FastAPI
        _app = FastAPI(
            title="VideoWaterMarker API",
            description="OTT pipeline watermarking microservice",
            version="1.0",
        )
        _app.include_router(router)
        return _app
    except ImportError:
        return None


app = create_app()


if __name__ == "__main__":
    import uvicorn
    if app:
        uvicorn.run(app, host="0.0.0.0", port=8000)
