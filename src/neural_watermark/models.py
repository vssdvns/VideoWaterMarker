"""
Neural encoder-decoder for steganographic watermarking (proposal).

U-Net/ResNet style: embed payload bits in image pixels, recover under attack.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """
    Encoder: (cover, payload_bits) -> watermarked_image.
    Payload is a bit tensor; output delta is added to cover.
    """

    def __init__(
        self,
        payload_bits: int = 48,
        img_channels: int = 3,
        base_ch: int = 64,
        delta_scale: float = 0.12,
    ) -> None:
        super().__init__()
        self.payload_bits = payload_bits
        self.delta_scale = delta_scale
        self.down1 = ConvBlock(img_channels + payload_bits, base_ch)      # concat payload to spatial
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.conv3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.conv2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.conv1 = ConvBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, img_channels, 1)
        self.tanh = nn.Tanh()

    def set_delta_scale(self, scale: float) -> None:
        """Set perturbation scale (e.g. 0.12 phase1, 0.08 phase2)."""
        self.delta_scale = scale

    def _expand_payload(self, payload: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Expand [B, C] payload to [B, C, H, W] spatial."""
        b, c = payload.shape
        return payload.view(b, c, 1, 1).expand(b, c, h, w)

    def forward(
        self, cover: torch.Tensor, payload: torch.Tensor
    ) -> torch.Tensor:
        """
        cover: (B, 3, H, W) in [0,1]
        payload: (B, payload_bits) in {0,1}
        Returns: watermarked (B, 3, H, W), delta clipped for imperceptibility.
        """
        h, w = cover.shape[2], cover.shape[3]
        payload_spatial = self._expand_payload(payload, h, w)
        x = torch.cat([cover, payload_spatial], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))

        u3 = self.up3(b)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        delta = self.tanh(self.out(u1)) * self.delta_scale
        return (cover + delta).clamp(0, 1)


class Decoder(nn.Module):
    """Decoder: watermarked_image -> recovered payload bits."""

    def __init__(
        self,
        payload_bits: int = 48,
        img_channels: int = 3,
        base_ch: int = 96,
    ) -> None:
        super().__init__()
        self.payload_bits = payload_bits
        self.encoder = nn.Sequential(
            ConvBlock(img_channels, base_ch),
            nn.MaxPool2d(2),
            ConvBlock(base_ch, base_ch * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_ch * 2, base_ch * 4),
            nn.MaxPool2d(2),
            ConvBlock(base_ch * 4, base_ch * 8),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, payload_bits),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Returns (B, payload_bits) logits."""
        feat = self.encoder(img)
        return self.fc(feat)


class AttackSimulator(nn.Module):
    """
    Differentiable attack simulation for training.
    Applies: JPEG-like, blur, noise, resize.
    """

    def __init__(self, jpeg_quality: float = 0.5, blur_sigma: float = 0.5, noise_std: float = 0.02) -> None:
        super().__init__()
        self.jpeg_quality = jpeg_quality
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std

    def forward(
        self, x: torch.Tensor, training: bool = True
    ) -> torch.Tensor:
        if not training:
            return x
        # Blur (approx with conv)
        if self.blur_sigma > 0:
            k = 5
            sigma = self.blur_sigma
            kh = torch.exp(-((torch.arange(k, device=x.device).float() - k // 2) ** 2) / (2 * sigma ** 2))
            k2d = (kh.unsqueeze(0) * kh.unsqueeze(1)).unsqueeze(0).unsqueeze(0)
            k2d = k2d.expand(x.shape[1], 1, k, k) / k2d.sum()
            x = torch.nn.functional.conv2d(x, k2d, padding=k // 2, groups=x.shape[1])
        # Noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x, device=x.device) * self.noise_std
        return x.clamp(0, 1)


class DecoderDelta(nn.Module):
    """
    Decoder that reads from the residual (watermarked - cover).
    Used only during training to provide direct gradient signal to the encoder:
    "put decodable information in your delta". Improves encoder learning.
    At inference we use only Decoder(watermarked), not DecoderDelta.
    """

    def __init__(self, payload_bits: int = 48, base_ch: int = 64) -> None:
        super().__init__()
        self.payload_bits = payload_bits
        self.encoder = nn.Sequential(
            ConvBlock(3, base_ch),
            nn.MaxPool2d(2),
            ConvBlock(base_ch, base_ch * 2),
            nn.MaxPool2d(2),
            ConvBlock(base_ch * 2, base_ch * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_ch * 4, 128),
            nn.ReLU(),
            nn.Linear(128, payload_bits),
        )

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: (B, 3, H, W), typically in [-0.15, 0.15]. Normalized for decoder."""
        x = (delta + 0.15) / 0.3  # map to [0, 1]
        x = x.clamp(0, 1)
        feat = self.encoder(x)
        return self.fc(feat)
