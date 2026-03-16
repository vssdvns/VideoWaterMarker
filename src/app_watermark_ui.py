"""
Phase 1 + 2: Video Watermarking UI with User-Specific Fingerprint

Streamlit app for:
- Upload video
- Choose watermarking method (Fixed, Heuristic, DeepLab, Hybrid)
- Traceability mode: embed user ID / location / device in watermark
- Generate watermarked video
- Preview and manually adjust position via heatmap
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import cv2
import numpy as np

import json

from src.detect_watermark import detect_video, build_text_template
from src.run_attacks_ui import ATTACKS, run_attacks as run_attacks_ui
from src.fingerprint import encode_fingerprint
from src.video_watermark_demo import (
    add_text_watermark_fixed,
    add_text_watermark_to_video,
    add_text_watermark_fixed_at,
    add_text_watermark_from_positions,
    add_text_watermark_from_keyframes,
    get_frame_heatmap,
)

# Session state keys
INPUT_PATH = "input_path"
OUTPUT_PATH = "output_path"
POSITIONS_PATH = "positions_path"
FRAME_COUNT = "frame_count"
VIDEO_READY = "video_ready"
SALIENCY_MODEL = "saliency_model"
KEYFRAMES = "keyframes"

APP_DIR = ROOT / "data" / "app"
APP_INPUT = APP_DIR / "input"
APP_OUTPUT = APP_DIR / "output"


def ensure_dirs():
    APP_INPUT.mkdir(parents=True, exist_ok=True)
    APP_OUTPUT.mkdir(parents=True, exist_ok=True)


def init_session():
    if INPUT_PATH not in st.session_state:
        st.session_state[INPUT_PATH] = None
    if OUTPUT_PATH not in st.session_state:
        st.session_state[OUTPUT_PATH] = None
    if POSITIONS_PATH not in st.session_state:
        st.session_state[POSITIONS_PATH] = None
    if FRAME_COUNT not in st.session_state:
        st.session_state[FRAME_COUNT] = 0
    if VIDEO_READY not in st.session_state:
        st.session_state[VIDEO_READY] = False
    if SALIENCY_MODEL not in st.session_state:
        st.session_state[SALIENCY_MODEL] = None
    if KEYFRAMES not in st.session_state:
        st.session_state[KEYFRAMES] = []


@st.cache_resource
def get_saliency_model():
    from src.models.saliency_deeplab import DeepLabSaliency
    return DeepLabSaliency()


def convert_to_h264_for_preview(src: Path, dst: Path) -> bool:
    """Convert mp4v (OpenCV) to H.264 for browser playback. Returns True if successful."""
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
        return dst.exists()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _apply_neural_on_video(src: Path, dst: Path, payload_bits: list[int], model_dir: Path) -> bool:
    """Apply neural watermark on existing video. Returns True if successful."""
    enc_p = model_dir / "encoder.pt"
    dec_p = model_dir / "decoder.pt"
    if not enc_p.exists() or not dec_p.exists():
        return False
    try:
        from src.neural_watermark.embed import NeuralWatermarker
        import torch
        wm = NeuralWatermarker(encoder_path=enc_p, decoder_path=dec_p,
            device="cuda" if torch.cuda.is_available() else "cpu")
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
        n = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = wm.embed(frame, payload_bits)
            out.write(frame)
            n += 1
        cap.release()
        out.release()
        return n > 0
    except Exception:
        return False


def run_watermark(method: str, input_path: Path, output_path: Path, positions_path: Path,
                  text: str, alpha: float, w_deeplab: float,
                  prefer_edges: bool = True, edge_margin: float = 0.12,
                  embed_dct_payload: Optional[bytes] = None,
                  use_neural: bool = False,
                  neural_payload_bits: Optional[list[int]] = None,
                  saliency_type: str = "deeplab",
                  use_optical_flow: bool = False, use_raft_flow: bool = False) -> None:
    ensure_dirs()
    model_dir = ROOT / "data" / "models" / "neural_wm"

    if method == "Fixed" and not embed_dct_payload and not use_neural:
        add_text_watermark_fixed(
            input_path=input_path,
            output_path=output_path,
            text=text,
            alpha=alpha,
            save_positions=True,
            positions_path=positions_path,
        )
    else:
        use_hybrid = method == "Hybrid"
        use_saliency = method in ("DeepLab", "Hybrid")
        add_text_watermark_to_video(
            input_path=input_path,
            output_path=output_path,
            text=text,
            alpha=alpha,
            use_saliency_model=use_saliency,
            use_hybrid=use_hybrid,
            w_deeplab=w_deeplab,
            temporal_smoothing=True,
            temporal_alpha=0.8,
            max_jump=150,
            use_optical_flow=use_optical_flow,
            use_raft_flow=use_raft_flow,
            save_positions=True,
            positions_path=positions_path,
            prefer_edges=prefer_edges,
            edge_margin=edge_margin,
            embed_dct_payload=embed_dct_payload,
            saliency_type=saliency_type,
        )

    # Apply neural on top if requested
    if use_neural and neural_payload_bits and model_dir.exists():
        tmp = output_path.parent / (output_path.stem + "_tmp.mp4")
        output_path.rename(tmp)
        if _apply_neural_on_video(tmp, output_path, neural_payload_bits, model_dir):
            tmp.unlink(missing_ok=True)
            # Save neural payload in positions for detection
            if positions_path.exists():
                data = json.loads(positions_path.read_text(encoding="utf-8"))
                if data:
                    data[0]["neural_payload_bits"] = neural_payload_bits
                    positions_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:
            tmp.rename(output_path)


def main():
    st.set_page_config(page_title="Video Watermarker", page_icon="🎬", layout="wide")
    init_session()
    ensure_dirs()

    st.title("🎬 Hybrid Video Watermarker")
    st.markdown("Upload a video, choose a watermarking method, and optionally adjust the placement manually.")

    # --- Sidebar: Upload & settings ---
    with st.sidebar:
        st.header("Settings")
        uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded:
            input_path = APP_INPUT / "uploaded.mp4"
            with open(input_path, "wb") as f:
                f.write(uploaded.read())
            st.session_state[INPUT_PATH] = str(input_path)
            cap = cv2.VideoCapture(str(input_path))
            st.session_state[FRAME_COUNT] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            st.success(f"Loaded: {uploaded.name}")

        method = st.selectbox(
            "Watermark method",
            ["Fixed", "Heuristic", "DeepLab", "Hybrid"],
            index=3,
            help="Fixed: bottom-right. Heuristic: Laplacian. DeepLab: semantic. Hybrid: Laplacian+DeepLab.",
        )
        saliency_type = st.selectbox(
            "Saliency model (Hybrid/DeepLab)",
            ["deeplab", "u2net"],
            index=0,
            help="U²-Net: proposal-aligned salient object detection. DeepLab: semantic segmentation.",
        )
        use_optical_flow = st.checkbox(
            "Use optical flow (temporal consistency)",
            value=False,
            help="Farneback by default. Enable RAFT below for proposal-aligned flow.",
        )
        use_raft_flow = False
        if use_optical_flow:
            use_raft_flow = st.checkbox(
                "Use RAFT optical flow (proposal)",
                value=False,
                help="RAFT instead of Farneback. Requires more GPU memory.",
            )

        st.subheader("Watermark types")
        use_visible = st.checkbox("Visible (text overlay)", value=True,
            help="Semi-transparent text on video. Always recommended for traceability.")
        use_dct = st.checkbox("DCT (invisible, in ROI)", value=False,
            help="Embed payload in DCT coefficients. Robust to compression. Requires traceability or custom ID.")
        use_neural = st.checkbox("Neural (invisible, full-frame)", value=False,
            help="Learning-based invisible watermark. Requires trained encoder/decoder in data/models/neural_wm/.")

        st.subheader("Traceability (Phase 2)")
        use_traceability = st.checkbox(
            "Use user-specific fingerprint",
            value=False,
            help="Embed user ID in watermark for piracy traceability. Each user gets a unique visible code.",
        )
        if use_traceability:
            user_id = st.text_input("User ID", value="user_001", help="Account or session identifier")
            location = st.text_input("Location (optional)", value="", placeholder="e.g. US-CA, EU")
            device = st.text_input("Device (optional)", value="", placeholder="e.g. web, ios, android")
            use_hash = st.checkbox("Use hashed fingerprint (privacy)", value=False,
                help="Short hash instead of raw ID. Requires lookup table to trace back.")
            text = encode_fingerprint(user_id, location or None, device or None, use_hash=use_hash)
            st.caption(f"Watermark text: **{text}**")
            if use_dct and not user_id:
                st.caption(":warning: DCT requires User ID.")
        else:
            text = st.text_input("Watermark text", value="VideoWaterMarker")
            user_id = ""
            location = ""
            device = ""

        alpha = st.slider("Opacity", 0.2, 1.0, 0.5)
        w_deeplab = 0.6
        if method == "Hybrid":
            w_deeplab = st.slider("DeepLab weight (hybrid)", 0.2, 0.9, 0.6)

        st.subheader("Placement")
        prefer_edges = st.checkbox("Prefer edges/corners", value=True,
            help="Restrict watermark to outer 12% of frame. Less intrusive, better for crop resilience.")
        edge_margin = 0.12
        if prefer_edges:
            edge_margin = st.slider("Edge margin", 0.08, 0.25, 0.12, 0.01,
                help="Fraction of frame width/height for edge zone (larger = stricter).")

    # Build DCT payload (needs user_id or text for encoding)
    embed_dct_payload = None
    if use_dct and (user_id or text):
        try:
            from src.crypto_payload import encode_payload
            payload_bytes, _ = encode_payload(
                user_id or text, None, location or None, device or None,
                use_aes=False, use_ecc=True,
            )
            embed_dct_payload = payload_bytes
        except ImportError:
            pass

    # Neural payload: 8 bits from hash of user_id or text
    neural_payload_bits = None
    if use_neural:
        h = hash((user_id or text) or "default") & 0xFF
        neural_payload_bits = [(h >> i) & 1 for i in range(8)]

    # --- Main: Tabs ---
    input_path_raw = st.session_state.get(INPUT_PATH)
    input_path = Path(input_path_raw) if input_path_raw and Path(input_path_raw).exists() else None
    output_path = APP_OUTPUT / "watermarked.mp4"
    positions_path = APP_OUTPUT / "positions.json"

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Generate", "Preview", "Manual adjust", "Detect (no generate)", "Attack test", "Traceability info"]
    )

    with tab1:
        if not input_path:
            st.info("👆 Upload a video in the sidebar to start.")
        else:
            if st.button("🚀 Generate watermarked video"):
                with st.spinner("Generating watermarked video... (this may take a minute)"):
                    try:
                        run_watermark(
                            method=method,
                            input_path=input_path,
                            output_path=output_path,
                            positions_path=positions_path,
                            text=text,
                            alpha=alpha,
                            w_deeplab=w_deeplab,
                            prefer_edges=prefer_edges,
                            edge_margin=edge_margin,
                            embed_dct_payload=embed_dct_payload,
                            use_neural=use_neural,
                            neural_payload_bits=neural_payload_bits,
                            saliency_type=saliency_type,
                            use_optical_flow=use_optical_flow,
                            use_raft_flow=use_raft_flow,
                        )
                        st.session_state[OUTPUT_PATH] = str(output_path)
                        st.session_state[POSITIONS_PATH] = str(positions_path)
                        st.session_state[VIDEO_READY] = True
                        st.success("Done! Check the Preview tab.")
                    except Exception as e:
                        st.error(str(e))

    with tab2:
        if not input_path:
            st.info("👆 Upload a video and generate first to preview.")
        elif not output_path.exists():
            st.info("Generate a watermarked video first.")
        else:
            # OpenCV writes mp4v which browsers often can't play. Convert to H.264 for preview.
            preview_path = APP_OUTPUT / "watermarked_preview.mp4"
            manual_preview = APP_OUTPUT / "watermarked_manual_preview.mp4"
            active_output = st.session_state.get(OUTPUT_PATH) or str(output_path)
            is_manual = active_output and "manual" in Path(active_output).name
            src = Path(active_output)
            dst = manual_preview if is_manual else preview_path

            if not src.exists():
                st.warning("Video file not found. Generate or re-export first.")
            else:
                needs_convert = not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime
                if needs_convert:
                    convert_to_h264_for_preview(src, dst)
                if dst.exists():
                    st.video(str(dst))
                else:
                    st.video(str(src))
                    st.caption(
                        "If the video doesn't play: install [ffmpeg](https://ffmpeg.org/) for browser-compatible preview, "
                        "or use the download button below."
                    )
                with open(src, "rb") as f:
                    st.download_button("Download watermarked video", f, file_name=src.name, mime="video/mp4")
                pos_data = {}
                for pp in [positions_path, APP_OUTPUT / "positions_manual.json", APP_OUTPUT / "positions.json"]:
                    if pp.exists():
                        try:
                            pos_data = json.loads(pp.read_text(encoding="utf-8"))
                            break
                        except Exception:
                            pass
                has_dct = bool(pos_data and pos_data[0].get("dct_roi"))
                has_neural = bool(pos_data and pos_data[0].get("neural_payload_bits"))
                wm_list = ["Visible"]
                if has_dct:
                    wm_list.append("DCT")
                if has_neural:
                    wm_list.append("Neural")
                summary = f"""Video Watermarking Summary
=========================
Method: {method}
Watermark text: {text}
Opacity: {alpha}
Output: {src.name}
Watermarks applied: {", ".join(wm_list)}
Prefer edges: {prefer_edges}
Edge margin: {edge_margin}
"""
                st.download_button("📄 Export summary (txt)", data=summary, file_name="watermark_summary.txt", mime="text/plain", key="dl_summary")

                # Original vs watermarked comparison
                st.markdown("---")
                st.subheader("Compare: Original vs Watermarked")
                fc = st.session_state.get(FRAME_COUNT, 100)
                cmp_frame = st.slider("Frame", 0, max(0, fc - 1), fc // 2 if fc else 0, key="cmp_frame")
                cap_orig = cv2.VideoCapture(str(input_path))
                cap_wm = cv2.VideoCapture(str(src))
                if cap_orig.isOpened() and cap_wm.isOpened():
                    cap_orig.set(cv2.CAP_PROP_POS_FRAMES, cmp_frame)
                    cap_wm.set(cv2.CAP_PROP_POS_FRAMES, cmp_frame)
                    _, orig_frame = cap_orig.read()
                    _, wm_frame = cap_wm.read()
                    cap_orig.release()
                    cap_wm.release()
                    if orig_frame is not None and wm_frame is not None:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                        with c2:
                            st.image(cv2.cvtColor(wm_frame, cv2.COLOR_BGR2RGB), caption="Watermarked", use_container_width=True)

    with tab3:
        if not input_path:
            st.info("👆 Upload a video and generate first to adjust.")
        elif not output_path.exists():
            st.info("Generate a watermarked video first to enable manual adjustment.")
        else:
            st.markdown("""
            Adjust the watermark position. **Blue = low saliency (better)**.  
            **Tip:** Add keyframes at different frames and re-export — positions are interpolated between keyframes.
            """)

            frame_count = st.session_state.get(FRAME_COUNT, 100)
            frame_idx = st.slider("Frame", 0, max(0, frame_count - 1), 0, help="Frame to preview on map")

            heatmap_method = method.lower() if method != "Fixed" else "heuristic"
            if heatmap_method == "fixed":
                heatmap_method = "heuristic"

            try:
                frame, heatmap_overlay, x, y, box_w, box_h, _, _ = get_frame_heatmap(
                    input_path,
                    frame_idx,
                    method=heatmap_method,
                    w_deeplab=w_deeplab,
                    saliency_model=get_saliency_model() if method in ("DeepLab", "Hybrid") else None,
                    prefer_edges=prefer_edges,
                    edge_margin=edge_margin,
                )
            except Exception as e:
                st.error(f"Could not compute heatmap: {e}")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.caption("Saliency / complexity map (blue = less intrusive)")
                    st.image(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

                with col2:
                    # Preset position buttons
                    w_f, h_f = frame.shape[1], frame.shape[0]
                    pad = 20
                    presets = [
                        ("↖ Top-left", pad, pad),
                        ("↗ Top-right", w_f - box_w - pad, pad),
                        ("↙ Bottom-left", pad, h_f - box_h - pad),
                        ("↘ Bottom-right", w_f - box_w - pad, h_f - box_h - pad),
                        ("⬇ Bottom-center", (w_f - box_w) // 2, h_f - box_h - pad),
                    ]
                    for label, px, py in presets:
                        if st.button(label, key=f"preset_{label}"):
                            st.session_state["manual_x"] = max(0, min(px, w_f - box_w))
                            st.session_state["manual_y"] = max(0, min(py, h_f - box_h))
                    new_x = st.slider("X position", 0, max(1, frame.shape[1] - box_w),
                                      st.session_state.get("manual_x", x), key="manual_x")
                    new_y = st.slider("Y position", 0, max(1, frame.shape[0] - box_h),
                                      st.session_state.get("manual_y", y), key="manual_y")

                    preview = frame.copy()
                    cv2.rectangle(preview, (new_x, new_y), (new_x + box_w, new_y + box_h), (0, 255, 0), 2)
                    st.caption("Preview with watermark box")
                    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), use_container_width=True)

                    keyframes: list = st.session_state.get(KEYFRAMES, [])
                    if st.button("Add keyframe at current frame"):
                        keyframes.append((frame_idx, new_x, new_y))
                        keyframes.sort(key=lambda k: k[0])
                        st.session_state[KEYFRAMES] = keyframes
                        st.success(f"Keyframe at frame {frame_idx}")
                    if keyframes:
                        st.caption(f"Keyframes: {len(keyframes)} — {', '.join(f'f{k[0]}' for k in keyframes)}")
                        if st.button("Clear keyframes"):
                            st.session_state[KEYFRAMES] = []

                use_keyframes = len(keyframes) >= 1
                btn_label = "Re-export (interpolate keyframes)" if use_keyframes else "Re-export with new position"
                if st.button(btn_label):
                    with st.spinner("Re-exporting..."):
                        out_manual = APP_OUTPUT / "watermarked_manual.mp4"
                        pos_manual = APP_OUTPUT / "positions_manual.json"
                        try:
                            if use_keyframes:
                                add_text_watermark_from_keyframes(
                                    input_path=input_path,
                                    output_path=out_manual,
                                    keyframes=keyframes,
                                    text=text,
                                    alpha=alpha,
                                    positions_path=pos_manual,
                                )
                            else:
                                add_text_watermark_fixed_at(
                                    input_path=input_path,
                                    output_path=out_manual,
                                    x=new_x,
                                    y=new_y,
                                    text=text,
                                    alpha=alpha,
                                    positions_path=pos_manual,
                                )
                            st.session_state[OUTPUT_PATH] = str(out_manual)
                            st.success(f"Saved to {out_manual.name}. Check Preview tab.")
                        except Exception as e:
                            st.error(str(e))

    with tab4:
        st.markdown("""
        ## Detect & Trace (No Generate Required)

        Upload a **watermarked video** and its **positions.json** to detect the embedded fingerprint.  
        Works with **previous files** — no need to generate in this session.

        **Required:** The `positions.json` saved when that video was originally watermarked.
        """)
        use_last_video = False
        if output_path.exists():
            use_last_video = st.checkbox("Use last generated watermarked video", value=False, key="use_last")
        detect_video_upload = None
        if not use_last_video:
            detect_video_upload = st.file_uploader(
                "Upload watermarked video to analyze",
                type=["mp4", "avi", "mov", "mkv"],
                key="detect_video",
            )
        detect_pos_upload = st.file_uploader(
            "Upload positions.json (from when this video was watermarked)",
            type=["json"],
            key="detect_pos",
        )
        pos_from_app = None
        if APP_OUTPUT.exists():
            pos_files = list(APP_OUTPUT.glob("*.json"))
            if pos_files:
                pos_from_app = st.selectbox(
                    "Or select from recently generated",
                    options=[""] + [p.name for p in pos_files],
                    format_func=lambda x: x or "(choose one)",
                    key="detect_pos_select",
                )

        thr = st.slider("Detection threshold (NCC)", 0.25, 0.55, 0.40, 0.01,
            help="Higher = stricter. 0.40 is a good balance.")

        if st.button("Detect watermark", key="detect_btn"):
            video_path = None
            positions = None

            if use_last_video and output_path.exists():
                video_path = output_path
            elif detect_video_upload:
                vpath = APP_INPUT / "detect_uploaded.mp4"
                vpath.parent.mkdir(parents=True, exist_ok=True)
                with open(vpath, "wb") as f:
                    f.write(detect_video_upload.read())
                video_path = vpath
            if detect_pos_upload:
                positions = json.loads(detect_pos_upload.read().decode("utf-8"))
            elif pos_from_app:
                pos_path = APP_OUTPUT / pos_from_app
                if pos_path.exists():
                    with open(pos_path, "r", encoding="utf-8") as f:
                        positions = json.load(f)

            if not video_path or not positions:
                st.error("Upload both a watermarked video and a positions.json file.")
            else:
                try:
                    text = positions[0].get("text", "VideoWaterMarker")
                    box_w = int(positions[0]["box_w"])
                    box_h = int(positions[0]["box_h"])
                    font_scale = float(positions[0].get("font_scale", 1.0))
                    thickness = int(positions[0].get("thickness", 2))
                    template_g = build_text_template(text, box_w, box_h, font_scale, thickness)

                    with st.spinner("Running detection..."):
                        result = detect_video(
                            video_path,
                            positions,
                            template_g,
                            frame_step=5,
                            search_pad=60,
                            thr=thr,
                            enable_global_fallback=True,
                            global_downscale=0.5,
                        )

                    st.metric("Detection rate", f"{result['rate']:.1f}%")
                    st.metric("Embedded fingerprint", text)
                    if result.get("dct_payload"):
                        try:
                            from src.crypto_payload import decode_payload
                            decoded = decode_payload(result["dct_payload"], use_aes=False, use_ecc=True)
                            if decoded:
                                st.metric("DCT payload (decoded)", decoded)
                        except Exception:
                            st.caption("DCT payload extracted (decode failed)")
                    st.caption("The fingerprint text identifies which user/session the video originated from.")
                    with st.expander("Details"):
                        det = {
                            "frames_sampled": result["total"],
                            "frames_detected": result["hit"],
                            "mean_ncc": round(result["mean"], 4),
                            "threshold": thr,
                        }
                        if result.get("dct_payload"):
                            det["dct_extracted"] = True
                        st.json(det)
                except Exception as e:
                    st.error(str(e))

    with tab5:
        st.markdown("""
        ## Attack Testing (Robustness)

        Apply distortion attacks to a watermarked video and run detection to test robustness.
        Requires ffmpeg.
        """)
        at_use_last = output_path.exists() and st.checkbox("Use last generated video", value=False, key="at_use_last")
        at_video_upload = None
        if not at_use_last:
            at_video_upload = st.file_uploader("Or upload watermarked video", type=["mp4", "avi", "mov"], key="at_video")
        at_pos_upload = st.file_uploader("Upload positions.json", type=["json"], key="at_pos")
        at_pos_select = None
        if APP_OUTPUT.exists():
            pos_files = list(APP_OUTPUT.glob("*.json"))
            if pos_files:
                at_pos_select = st.selectbox("Or select positions", [""] + [p.name for p in pos_files], key="at_pos_sel")
        at_attacks = st.multiselect("Select attacks", list(ATTACKS.keys()), default=["blur_sigma2", "reencode_crf28", "grayscale"])
        at_thr = st.slider("Detection threshold", 0.25, 0.55, 0.40, 0.01, key="at_thr")

        if st.button("Run attacks & detect", key="at_run"):
            at_video_path = None
            at_positions = None
            if at_use_last and output_path.exists():
                at_video_path = output_path
            elif at_video_upload:
                ap = APP_INPUT / "attack_input.mp4"
                ap.parent.mkdir(parents=True, exist_ok=True)
                with open(ap, "wb") as f:
                    f.write(at_video_upload.read())
                at_video_path = ap
            if at_pos_upload:
                at_positions = json.loads(at_pos_upload.read().decode("utf-8"))
            elif at_pos_select:
                pp = APP_OUTPUT / at_pos_select
                if pp.exists():
                    with open(pp, "r") as f:
                        at_positions = json.load(f)

            if not at_video_path or not at_positions or not at_attacks:
                st.error("Provide video, positions, and select at least one attack.")
            else:
                at_out = APP_OUTPUT / "attacks_ui"
                at_out.mkdir(parents=True, exist_ok=True)
                with st.spinner("Running attacks..."):
                    results = run_attacks_ui(at_video_path, at_out, at_attacks)
                if not results:
                    st.error("Attack run failed (ffmpeg required).")
                else:
                    text = at_positions[0].get("text", "VideoWaterMarker")
                    box_w = int(at_positions[0]["box_w"])
                    box_h = int(at_positions[0]["box_h"])
                    font_scale = float(at_positions[0].get("font_scale", 1.0))
                    thickness = int(at_positions[0].get("thickness", 2))
                    template_g = build_text_template(text, box_w, box_h, font_scale, thickness)

                    st.subheader("Results")
                    rows = []
                    for name, out_path in results:
                        r = detect_video(out_path, at_positions, template_g, frame_step=5, search_pad=60,
                                        thr=at_thr, enable_global_fallback=True, global_downscale=0.5)
                        rows.append({"Attack": name, "Detection %": round(r["rate"], 1), "Mean NCC": round(r["mean"], 4)})
                    st.dataframe(rows, use_container_width=True)
                    avg = sum(row["Detection %"] for row in rows) / len(rows) if rows else 0
                    st.metric("Average detection across attacks", f"{avg:.1f}%")

    with tab6:
        st.markdown("""
        ## User-Specific Fingerprint (Traceability)

        When **traceability mode** is enabled, each user receives a video with a **unique watermark**.
        If the video is leaked or pirated, the watermark can be traced back to the source account.

        ### Watermark types
        - **Visible**: Semi-transparent text overlay. Robust, easy to detect.
        - **DCT**: Invisible payload in DCT coefficients (Reed-Solomon). Robust to compression.
        - **Neural**: Learning-based invisible watermark. Requires trained models in `data/models/neural_wm/`.

        ### Flow
        1. **OTT/Streaming platform** assigns a user ID (and optionally location, device) to each session.
        2. **Watermarking** embeds this as visible text and/or invisible DCT/neural payloads.
        3. **Piracy detection**: Found copy → detect/extract watermarks → lookup in registry → identify user.

        ### Modes
        - **Direct ID**: Short user IDs shown as-is (`ID:alice`, `ID:user_001`). Best for demos.
        - **Hashed**: Longer IDs hashed to 8 chars (`ID:a1b2c3d4`). Requires a lookup table mapping hash → user.

        ### Example fingerprint formats
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.code("ID:user_001\nID:alice\nID:a1b2c3d4", language="text")
        with col2:
            st.code("ID:a1b2c3d4.US-CA.ios\nID:x9f2e1. web", language="text")
        st.markdown("""
        With **location** and **device**, the format is `ID:code.loc.device` for extra context.

        ### Detect & Trace
        Use the **Detect (no generate)** tab to upload a watermarked video and trace back the embedded fingerprint.

        ### Attack Test
        Use the **Attack test** tab to apply distortions (blur, crop, etc.) and measure detection robustness.
        """)


if __name__ == "__main__":
    main()
