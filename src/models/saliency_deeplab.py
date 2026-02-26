from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import torch
from torchvision import models, transforms
torch.backends.cudnn.benchmark = True



@dataclass
class DeepLabSaliencyConfig:
    long_side: int = 512  # resize longest side to this for speed
    device: Literal["cpu", "cuda"] = "cpu"


class DeepLabSaliency:
    """
    Wrapper around torchvision's DeepLabV3-ResNet50 segmentation model
    to produce a 'saliency-like' map.

    Idea:
      - Use pretrained DeepLabV3 on COCO.
      - Get per-pixel class probabilities.
      - Treat background (class 0) as "non-salient".
      - Saliency = 1 - P(background).
    """

    def __init__(self, config: DeepLabSaliencyConfig | None = None) -> None:
        if config is None:
            # Auto-pick CUDA if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            config = DeepLabSaliencyConfig(device=device)  # type: ignore[arg-type]
        self.config = config

        print(f"[SAL] Loading DeepLabV3-ResNet50 on {self.config.device}...")
        weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = models.segmentation.deeplabv3_resnet50(weights=weights)
        self.model.eval().to(self.config.device)

        # --- NORMALIZATION FIX ---
        # Newer torchvision may not expose 'mean'/'std' in weights.meta.
        # Fall back to standard ImageNet stats when missing.
        meta = getattr(weights, "meta", {}) or {}
        if isinstance(meta, dict):
            mean = meta.get("mean", (0.485, 0.456, 0.406))
            std = meta.get("std", (0.229, 0.224, 0.225))
        else:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),  # HWC uint8 -> CHW float32 [0,1]
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _prepare_image(self, frame_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
        """
        Convert BGR frame to normalized tensor and resize long side to config.long_side.
        Returns:
          - tensor: (1,3,H',W')
          - original_size: (H, W)
          - resized_size: (H', W')
        """
        h, w = frame_bgr.shape[:2]
        orig_size = (h, w)

        # BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize so that long side = long_side
        scale = self.config.long_side / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w
        resized_size = (new_h, new_w)

        # To tensor & normalize
        img_t = self.preprocess(frame_rgb)  # (3,H',W')
        img_t = img_t.unsqueeze(0).to(self.config.device)  # (1,3,H',W')
        return img_t, orig_size, resized_size

    @torch.no_grad()
    def get_saliency_map(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Returns a float32 saliency map in [0,1], same HxW as input frame.
        Higher values mean more 'foreground / salient'.
        """
        img_t, orig_size, resized_size = self._prepare_image(frame_bgr)

        # Forward pass
        if self.config.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(img_t)
        else:
            outputs = self.model(img_t)
            
        logits = outputs["out"]  # (1, C, H', W')
        probs = torch.softmax(logits, dim=1)  # per-class probabilities

        # Background is class 0 for COCO segmentation models
        background_prob = probs[:, 0, :, :]  # (1, H', W')
        saliency = 1.0 - background_prob  # foreground = 1 - P(background)

        # Resize back to original frame size
        saliency_np = saliency.squeeze(0).cpu().numpy()  # (H', W')
        saliency_np = cv2.resize(
            saliency_np,
            (orig_size[1], orig_size[0]),  # (W, H)
            interpolation=cv2.INTER_LINEAR,
        )

        # Normalize to [0,1]
        saliency_np = np.clip(saliency_np, 0.0, 1.0)
        return saliency_np.astype("float32")
