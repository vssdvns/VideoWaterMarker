"""
U²-Net saliency detection wrapper (proposal: segmentation and saliency).

Low-attention regions = low saliency = 1 - saliency_map (background).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

try:
    from .u2net import U2NETP
except ImportError:
    from models.u2net import U2NETP

U2NETP_WEIGHTS_URL = (
    "https://huggingface.co/netradrishti/u2net-saliency/resolve/"
    "c723f1a701ee5d3240a28bb5a1d57b9e713698c5/models/u2netp.pth"
)


class U2NetSaliency:
    """
    U²-Net for salient object detection. Returns map where low = background (good for watermark).
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: Literal["cpu", "cuda"] | None = None,
        long_side: int = 320,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.long_side = long_side

        print("[U2NET] Loading U2NETP...")
        self.model = U2NETP(in_ch=3, out_ch=1).to(device)
        self.model.eval()

        if weights_path:
            path = Path(weights_path)
        else:
            cache_dir = Path(__file__).resolve().parents[2] / "data" / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            path = cache_dir / "u2netp.pth"
            if not path.exists():
                self._download_weights(path)

        if path.exists():
            state = torch.load(path, map_location=device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
            print(f"[U2NET] Loaded weights from {path}")
        else:
            print("[U2NET] No weights found; using random init (saliency may be poor)")

    def _download_weights(self, dest: Path) -> None:
        try:
            import urllib.request
            print(f"[U2NET] Downloading weights from {U2NETP_WEIGHTS_URL}...")
            urllib.request.urlretrieve(U2NETP_WEIGHTS_URL, dest)
        except Exception as e:
            print(f"[U2NET] Download failed: {e}. Place u2netp.pth manually in {dest.parent}")

    @torch.no_grad()
    def get_saliency_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns float32 map [0,1], same HxW as input.
        High = salient (foreground); low = background.
        For watermark placement we want low = 1 - saliency or invert.
        """
        h, w = frame_bgr.shape[:2]
        scale = self.long_side / max(h, w)
        if scale < 1.0:
            nw = int(w * scale)
            nh = int(h * scale)
            img = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (nw, nh), cv2.INTER_AREA)
        else:
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ImageNet normalization
        img_t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_t = (img_t - mean) / std
        img_t = img_t.to(self.device)

        d0, *_ = self.model(img_t)
        sal = d0.squeeze(0).squeeze(0).cpu().numpy()
        sal = cv2.resize(sal, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(sal, 0.0, 1.0).astype(np.float32)
