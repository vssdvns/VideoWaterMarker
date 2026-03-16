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
    ) -> None:
        enc_p = Path(encoder_path) if encoder_path else None
        dec_p = Path(decoder_path) if decoder_path else None
        self.payload_bits = payload_bits if payload_bits is not None else (
            _load_payload_bits(enc_p, dec_p) if enc_p and dec_p and enc_p.exists() else 48
        )
        self.encoder = Encoder(payload_bits=self.payload_bits)
        self.decoder = Decoder(payload_bits=self.payload_bits)
        self.device = device
        if encoder_path and Path(encoder_path).exists():
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        if decoder_path and Path(decoder_path).exists():
            self.decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
        self.encoder.eval().to(device)
        self.decoder.eval().to(device)

    def embed(self, frame: np.ndarray, payload_bits: list[int] | np.ndarray) -> np.ndarray:
        """Embed payload in frame. Returns watermarked BGR frame."""
        payload_bits = list(payload_bits) if not isinstance(payload_bits, list) else payload_bits
        if len(payload_bits) < self.payload_bits:
            payload_bits = list(payload_bits) + [0] * (self.payload_bits - len(payload_bits))
        payload_bits = payload_bits[:self.payload_bits]
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to 256 for model
        small = cv2.resize(rgb, (256, 256))
        t = torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        p = torch.tensor([payload_bits], dtype=torch.float32, device=self.device)
        t = t.to(self.device)
        with torch.no_grad():
            out = self.encoder(t, p)
        out_np = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        out_np = cv2.resize(out_np, (w, h))
        return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)

    def extract(self, frame: np.ndarray) -> np.ndarray:
        """Extract payload bits from frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (256, 256))
        t = torch.from_numpy(small).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        t = t.to(self.device)
        with torch.no_grad():
            logits = self.decoder(t)
        pred = (torch.sigmoid(logits) > 0.5).float().cpu().numpy().flatten()
        return pred


def _test():
    """Quick test: embed and extract, report bit error rate."""
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
