"""
Neural watermark embedding and extraction (inference).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

try:
    from .models import Encoder, Decoder
except ImportError:
    from models import Encoder, Decoder


def _load_payload_bits(encoder_path: Path | str, decoder_path: Path | str) -> int:
    """Load payload_bits from config.json if present, else default 48."""
    # Keep model metadata next to the checkpoints so inference can recover the
    # right payload size without hardcoding it in multiple places.
    enc_p = Path(encoder_path)
    config_path = enc_p.parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("payload_bits", 48)
    return 48


class NeuralWatermarker:
    def __init__(
        self,
        encoder_path: Optional[str | Path] = None,
        decoder_path: Optional[str | Path] = None,
        payload_bits: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        delta_scale: float = 0.06,
    ) -> None:
        # Build the encoder/decoder pair, infer payload size from config when
        # possible, and load any saved checkpoints requested by the caller.
        enc_p = Path(encoder_path) if encoder_path else None
        dec_p = Path(decoder_path) if decoder_path else None
        self.payload_bits = payload_bits if payload_bits is not None else (
            _load_payload_bits(enc_p, dec_p) if enc_p and dec_p and enc_p.exists() else 48
        )
        self.encoder = Encoder(payload_bits=self.payload_bits)
        self.decoder = Decoder(payload_bits=self.payload_bits)
        self.encoder.set_delta_scale(delta_scale)
        self.device = device
        if encoder_path and Path(encoder_path).exists():
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        if decoder_path and Path(decoder_path).exists():
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        self.encoder.eval().to(device)
        self.decoder.eval().to(device)

    def embed(
        self,
        frame: np.ndarray,
        payload_bits: list[int] | np.ndarray,
        delta_scale: float | None = None,
        preserve_quality: bool = True,
        color_mode: str = "bias_corrected",
        delta_smooth_sigma: float = 0.0,
    ) -> np.ndarray:
        """
        Embed payload in frame. Returns watermarked BGR frame.

        preserve_quality=True (default): Compute delta at 256×256, upsample it, add to
        full-res frame. Keeps original detail; only adds subtle perturbation.

        color_mode:
          - "rgb": Add delta to all RGB channels (may cause color shift)
          - "bias_corrected" (default): Zero-mean delta per channel before adding; reduces color cast
          - "luma_only": Add delta only to luminance (Y) in YCbCr; colors preserved exactly

        delta_smooth_sigma: If > 0, apply Gaussian blur to delta to reduce haloing around
          high-contrast edges (e.g. tree branches). 1.0–1.5 typically helps. 0 = no smoothing.
        """
        # Normalize the payload length first so the network always sees the
        # exact bit count it was trained to embed.
        payload_bits = list(payload_bits) if not isinstance(payload_bits, list) else payload_bits
        if len(payload_bits) < self.payload_bits:
            payload_bits = list(payload_bits) + [0] * (self.payload_bits - len(payload_bits))
        payload_bits = payload_bits[:self.payload_bits]
        # Resize to the model's working resolution, run the encoder there, and
        # later bring only the learned delta back to the original frame size.
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cover_small = cv2.resize(rgb, (256, 256))

        t = torch.from_numpy(cover_small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        p = torch.tensor([payload_bits], dtype=torch.float32, device=self.device)
        t = t.to(self.device)

        # Allow UI or scripts to override strength at inference time without
        # rebuilding the model.
        if delta_scale is not None:
            self.encoder.set_delta_scale(delta_scale)

        with torch.no_grad():
            out_small = self.encoder(t, p)

        if preserve_quality:
            # Only add the DELTA to full-res frame; preserve original detail
            cover_t = t
            delta_small = (out_small - cover_t).squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Upsample delta to full resolution (bilinear for smooth perturbation)
            delta_full = cv2.resize(delta_small, (w, h), interpolation=cv2.INTER_LINEAR)

            # Edge feathering: taper delta toward zero at frame margins to avoid visible bands
            # (Neural encoder can produce different output at 256x256 edges; upsampling creates margin artifacts)
            margin_pct = 0.05
            y_coord = np.linspace(0, 1, h)
            x_coord = np.linspace(0, 1, w)
            fade_y = np.where(y_coord < margin_pct, y_coord / margin_pct,
                              np.where(y_coord > 1 - margin_pct, (1 - y_coord) / margin_pct, 1.0))
            fade_x = np.where(x_coord < margin_pct, x_coord / margin_pct,
                              np.where(x_coord > 1 - margin_pct, (1 - x_coord) / margin_pct, 1.0))
            mask = (fade_y[:, np.newaxis] * fade_x[np.newaxis, :]).astype(np.float32)
            mask = np.stack([mask] * 3, axis=-1)
            delta_full = delta_full * mask

            # Color handling decides how much the residual is allowed to change
            # perceived color versus only brightness.
            if color_mode == "bias_corrected":
                # Remove per-channel DC offset (eliminates uniform color tint)
                for c in range(3):
                    delta_full[:, :, c] -= delta_full[:, :, c].mean()
            elif color_mode == "luma_only":
                # Add delta only to luminance; chroma (Cb, Cr) unchanged → colors preserved
                # Luma weight: 0.299R + 0.587G + 0.114B
                delta_luma = (
                    0.299 * delta_full[:, :, 0]
                    + 0.587 * delta_full[:, :, 1]
                    + 0.114 * delta_full[:, :, 2]
                )
                delta_full = np.stack([delta_luma] * 3, axis=-1)  # same luma delta to R,G,B proportionally

            # Soften delta to reduce haloing around high-contrast edges (e.g. tree branches)
            if delta_smooth_sigma > 0:
                k = max(3, int(6 * delta_smooth_sigma) | 1)  # odd kernel
                delta_full = cv2.GaussianBlur(delta_full, (k, k), delta_smooth_sigma)

            # Merge the residual back into the original full-resolution frame.
            frame_f = rgb.astype(np.float32) / 255.0
            watermarked = np.clip(frame_f + delta_full, 0, 1).astype(np.float32)
            out_np = (watermarked * 255).clip(0, 255).astype(np.uint8)
        else:
            # Legacy: replace with upscaled output (low quality)
            out_np = (out_small.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            out_np = cv2.resize(out_np, (w, h))

        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract payload bits from frame (hard 0/1 via threshold)."""
        # This is the simple inference path used when only a binary bit vector
        # is needed and per-bit confidence does not matter.
        probs = self.extract_probs(frame)
        return (probs > 0.5).astype(np.float32)

    def extract_probs(self, frame: np.ndarray) -> np.ndarray:
        """Extract payload probabilities from frame. Use for confidence-weighted voting."""
        # The decoder runs on the same normalized 256x256 view used during
        # training, then returns sigmoid probabilities for each payload bit.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (256, 256))
        t = torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(self.device)
        with torch.no_grad():
            logits = self.decoder(t)
        return torch.sigmoid(logits).float().cpu().numpy().flatten()


def _test():
    """Quick test: embed and extract, report bit error rate."""
    # This self-check gives a quick sanity test after training without needing
    # the full video UI pipeline.
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    enc_path = root / "data" / "models" / "neural_wm" / "encoder.pt"
    dec_path = root / "data" / "models" / "neural_wm" / "decoder.pt"
    if not enc_path.exists():
        print("Run training first: py -m src.neural_watermark.train --overnight --epochs 50")
        return
    wm = NeuralWatermarker(encoder_path=enc_path, decoder_path=dec_path, device="cpu")
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    payload = np.random.randint(0, 2, wm.payload_bits).tolist()
    out = wm.embed(img, payload)
    ext = wm.extract(out)
    err = sum(1 for a, b in zip(payload, ext[:wm.payload_bits]) if a != int(b)) / wm.payload_bits
    print(f"Test: BER = {err*100:.1f}% ({'OK' if err < 0.1 else 'needs more training'})")


if __name__ == "__main__":
    _test()
