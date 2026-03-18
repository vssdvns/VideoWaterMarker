"""
Training script for neural watermark encoder-decoder (proposal).

Trains with attack simulation: re-encoding, blur, noise, resize.

Overnight / low-resource usage:
  py -m src.neural_watermark.train --overnight --epochs 100

Standard:
  py -m src.neural_watermark.train --data_dir data/input --epochs 50
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cv2

try:
    from .models import Encoder, Decoder, DecoderDelta, AttackSimulator
except ImportError:
    from models import Encoder, Decoder, DecoderDelta, AttackSimulator


class FramePayloadDataset(Dataset):
    """Dataset of frames with random payload bits."""

    def __init__(
        self,
        data_dir: str | Path,
        max_frames: int = 10000,
        size: int = 256,
        synthetic: bool = False,
        max_videos: int = 50,
        max_frames_per_video: int = 50,
        payload_bits: int = 48,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.size = size
        self.synthetic = synthetic
        self.payload_bits = payload_bits
        self.frames: list[Path] = []

        if not synthetic:
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                self.frames.extend(self.data_dir.rglob(ext))
            videos = []
            for ext in ("*.mp4", "*.avi", "*.mov"):
                videos.extend(self.data_dir.rglob(ext))
            videos = list(dict.fromkeys(videos))[:max_videos]
            for vi, v in enumerate(videos):
                cap = cv2.VideoCapture(str(v))
                n = 0
                while n < max_frames_per_video:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    fp = self.data_dir / "cache" / f"{v.stem}_{n:04d}.png"
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    if not fp.exists():
                        cv2.imwrite(str(fp), fr)
                    self.frames.append(fp)
                    n += 1
                cap.release()
                if (vi + 1) % 5 == 0:
                    gc.collect()
            self.frames = list(dict.fromkeys(self.frames))[:max_frames]

        if not self.frames and not synthetic:
            raise FileNotFoundError(
                f"No images/videos in {data_dir}. "
                "Add .mp4/.jpg files to data/input, or use --synthetic to train on random images."
            )

        self._length = len(self.frames) if self.frames else max_frames

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        payload = torch.randint(0, 2, (self.payload_bits,), dtype=torch.float32)
        if self.synthetic or not self.frames:
            img = np.random.randint(0, 256, (self.size, self.size, 3), dtype=np.uint8)
        else:
            idx = i % len(self.frames)
            img = cv2.imread(str(self.frames[idx]))
            if img is None:
                img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return img_t, payload


def train_one_epoch(
    enc: Encoder,
    dec: Decoder,
    dec_delta: DecoderDelta | None,
    dec_delta_opt: torch.optim.Optimizer | None,
    attack: AttackSimulator,
    loader: torch.utils.data.DataLoader,
    enc_opt: torch.optim.Optimizer,
    dec_opt: torch.optim.Optimizer,
    device: torch.device,
    gc_every_n_batches: int = 0,
    attack_weight: float = 0.5,
    mse_weight: float = 2.0,
    delta_weight: float = 0.5,
) -> tuple[float, float, float]:
    enc.train()
    dec.train()
    if dec_delta is not None:
        dec_delta.train()
    total_loss = 0.0
    total_bce = 0.0
    total_mse = 0.0
    n = 0
    for bi, (cover, payload) in enumerate(loader):
        if gc_every_n_batches and bi > 0 and bi % gc_every_n_batches == 0:
            gc.collect()
        cover = cover.to(device)
        payload = payload.to(device)
        watermarked = enc(cover, payload)
        delta = watermarked - cover
        pred_clean = dec(watermarked)
        pred_attacked = dec(attack(watermarked))
        bce_clean = F.binary_cross_entropy_with_logits(pred_clean, payload)
        bce_attacked = F.binary_cross_entropy_with_logits(pred_attacked, payload)
        bce = (1 - attack_weight) * bce_clean + attack_weight * bce_attacked
        mse = F.mse_loss(watermarked, cover)
        loss = bce + mse_weight * mse
        if dec_delta is not None and delta_weight > 0:
            pred_delta = dec_delta(delta)
            bce_delta = F.binary_cross_entropy_with_logits(pred_delta, payload)
            loss = loss + delta_weight * bce_delta
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        if dec_delta_opt is not None:
            dec_delta_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        if dec_delta is not None:
            torch.nn.utils.clip_grad_norm_(dec_delta.parameters(), 1.0)
        enc_opt.step()
        dec_opt.step()
        if dec_delta_opt is not None:
            dec_delta_opt.step()
        total_loss += loss.item()
        total_bce += bce.item()
        total_mse += mse.item()
        n += 1
    n = max(1, n)
    return total_loss / n, total_bce / n, total_mse / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/input", help="Frames or video directory")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8, help="Smaller = less memory (use 2 for slow PC)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out_dir", default="data/models/neural_wm")
    ap.add_argument("--device", default=None, help="cpu or cuda (auto if not set)")
    ap.add_argument("--synthetic", action="store_true",
        help="Use random synthetic images (no real data needed)")
    ap.add_argument("--overnight", action="store_true",
        help="Low-resource mode: batch=2, size=128, cpu, fewer frames. Train overnight.")
    ap.add_argument("--size", type=int, default=256, help="Image size (128 = less memory)")
    ap.add_argument("--max_videos", type=int, default=50, help="Max videos to load")
    ap.add_argument("--max_frames_per_video", type=int, default=50,
        help="Frames per video (fewer = faster cache)")
    ap.add_argument("--mse_weight", type=float, default=2.0,
        help="MSE weight (lower = encoder can embed more; was 10)")
    ap.add_argument("--attack_weight", type=float, default=0.5,
        help="Weight for attacked-path loss (0.2=mostly clean early)")
    ap.add_argument("--phase1_epochs", type=int, default=40,
        help="Phase 1: clean-only training (no attack). Phase 2: add attacks.")
    ap.add_argument("--payload_bits", type=int, default=16,
        help="Payload bits (fewer = easier; 8/16 recommended)")
    ap.add_argument("--phase0_epochs", type=int, default=25,
        help="Phase 0: BCE-only (mse=0). Longer = more time to learn to embed.")
    ap.add_argument("--delta_weight", type=float, default=0.8,
        help="Weight for delta-decoder loss (direct encoder signal; 0.8 recommended)")
    ap.add_argument("--lr_phase0", type=float, default=3e-3,
        help="Learning rate for phase 0 (higher = faster initial learning)")
    ap.add_argument("--blur_sigma", type=float, default=0.35,
        help="Attack blur strength (0.35 gentler than 0.5)")
    ap.add_argument("--noise_std", type=float, default=0.015,
        help="Attack noise std (0.015 gentler than 0.02)")
    ap.add_argument("--resize_scale", type=float, default=0.0,
        help="Resize-down-up attack scale (0.5 = 50%% then back; 0 = off)")
    ap.add_argument("--fast", action="store_true",
        help="Fast preset: 8 bits, 100 frames, size 64, no attacks, phase0=40, ~20–30min for 100 epochs")
    ap.add_argument("--improved", action="store_true",
        help="Improved preset: 8 bits, 180 ep, phase0=40, delta_w=1.0, gentle attacks, low MSE early")
    ap.add_argument("--max_frames", type=int, default=2000, help="Max frames in dataset")
    args = ap.parse_args()

    # Overnight mode: gentle on PC
    if args.overnight:
        args.batch = 2
        args.size = 128
        args.device = "cpu"
        args.max_videos = 20
        args.max_frames_per_video = 25
        print("[TRAIN] Overnight mode: batch=2, size=128, CPU, reduced frames")

    # Fast preset: 8 bits, few frames, small size, no attacks, optimized for ~20–30min
    if args.fast:
        args.payload_bits = 8
        args.size = 64
        args.batch = 16
        args.phase0_epochs = 40
        args.phase1_epochs = 9999
        args.max_videos = 5
        args.max_frames_per_video = 20
        args.max_frames = 100
        args.delta_weight = 1.0
        print("[TRAIN] Fast mode: 8 bits, 100 frames, size 64, no attacks, phase0=40, delta_w=1.0")

    # Improved preset: 8 bits, 180–200 ep, gentle attack curriculum, low MSE early, resize attack
    if args.improved:
        args.payload_bits = 8
        args.epochs = max(args.epochs, 180)
        args.phase0_epochs = 40
        args.phase1_epochs = 60
        args.delta_weight = 1.0
        args.size = min(args.size, 128)
        args.blur_sigma = 0.25
        args.noise_std = 0.01
        args.resize_scale = 0.5
        print("[TRAIN] Improved: 8 bits, 180 ep, phase0=40, phase1=60, phase2=80, gentle attacks, resize 0.5")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fallback: PyTorch may be CPU-only, or GPU incompatible (e.g. RTX 50 sm_120 needs cu128)
    cuda_ok = False
    if args.device == "cuda":
        try:
            t = torch.zeros(1, device="cuda")
            torch.cuda.synchronize()  # Force kernel execution (catches sm_120 incompatibility)
            cuda_ok = True
        except Exception as e:
            print(f"[TRAIN] CUDA device unavailable ({e}). Using CPU.")
            print("         For RTX 50-series, run: powershell -File scripts/setup_gpu_blackwell.ps1")
            args.device = "cpu"
    else:
        cuda_ok = args.device == "cuda"

    if cuda_ok:
        torch.backends.cudnn.benchmark = args.fast

    root = Path(__file__).resolve().parents[2]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_bits = args.payload_bits
    print(f"[TRAIN] data_dir={data_dir}, batch={args.batch}, size={args.size}, device={args.device}, payload_bits={payload_bits}")
    ds = FramePayloadDataset(
        data_dir,
        max_frames=args.max_frames,
        size=args.size,
        synthetic=args.synthetic,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
        payload_bits=payload_bits,
    )
    print(f"[TRAIN] dataset size={len(ds)} frames")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=False
    )

    device = torch.device(args.device)
    enc = Encoder(payload_bits=payload_bits).to(device)
    dec = Decoder(payload_bits=payload_bits).to(device)
    dec_delta = DecoderDelta(payload_bits=payload_bits).to(device)
    attack = AttackSimulator(blur_sigma=args.blur_sigma, noise_std=args.noise_std, resize_scale=args.resize_scale).to(device)

    lr0 = args.lr_phase0
    enc_opt = torch.optim.Adam(enc.parameters(), lr=lr0)
    dec_opt = torch.optim.Adam(dec.parameters(), lr=lr0)
    dec_delta_opt = torch.optim.Adam(dec_delta.parameters(), lr=lr0)

    phase0_epochs = min(args.phase0_epochs, args.epochs)
    phase1_epochs = min(phase0_epochs + args.phase1_epochs, args.epochs)
    phase2_epochs = args.epochs - phase1_epochs
    print(f"[TRAIN] Phase 0: {phase0_epochs} epochs (BCE-only, mse=0, delta_w={args.delta_weight}, lr={lr0})")
    print(f"[TRAIN] Phase 1: {phase1_epochs - phase0_epochs} epochs (clean, delta=0.12, mse=0.2)")
    print(f"[TRAIN] Phase 2: {phase2_epochs} epochs (attacks, delta=0.08, mse=0.5->1.0)")
    print(f"[TRAIN] Target: BCE < 0.1 (good), BCE < 0.05 (very good). Random = 0.693")

    enc.set_delta_scale(0.12)
    gc_interval = 20 if args.overnight else 0

    for ep in range(args.epochs):
        if ep == phase0_epochs:
            mse_weight = 0.3
            print(f"[TRAIN] Phase 1 started (mse={mse_weight})")
        if ep == phase1_epochs:
            enc.set_delta_scale(0.08)
            enc_opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
            dec_opt = torch.optim.Adam(dec.parameters(), lr=1e-3)
            print(f"[TRAIN] Phase 2 started (attack enabled, delta=0.08)")
        if ep < phase0_epochs:
            attack_weight = 0.0
            mse_weight = 0.0
            delta_weight = args.delta_weight
        elif ep < phase1_epochs:
            attack_weight = 0.0
            mse_weight = 0.05 if args.improved else (0.1 if args.fast else 0.2)
            delta_weight = args.delta_weight * 0.6
        else:
            frac = (ep - phase1_epochs) / max(1, phase2_epochs)
            attack_weight = (0.1 + 0.4 * frac) if args.improved else (0.2 + 0.6 * frac)
            mse_weight = (0.2 + 0.5 * frac) if args.improved else (0.5 + 0.5 * frac)
            delta_weight = 0.3
        loss, bce, mse = train_one_epoch(
            enc, dec, dec_delta, dec_delta_opt, attack, loader,
            enc_opt, dec_opt, device,
            gc_every_n_batches=gc_interval,
            attack_weight=attack_weight,
            mse_weight=mse_weight,
            delta_weight=delta_weight,
        )
        print(f"Epoch {ep+1}/{args.epochs} loss={loss:.4f} BCE={bce:.4f} MSE={mse:.6f}")
        if (ep + 1) % 5 == 0:
            torch.save(enc.state_dict(), out_dir / "encoder.pt")
            torch.save(dec.state_dict(), out_dir / "decoder.pt")
            print(f"  -> checkpoint saved")
        gc.collect()

    torch.save(enc.state_dict(), out_dir / "encoder.pt")
    torch.save(dec.state_dict(), out_dir / "decoder.pt")
    import json
    with open(out_dir / "config.json", "w") as f:
        json.dump({"payload_bits": payload_bits}, f)
    print(f"\n[TRAIN] Done. Saved to {out_dir} (payload_bits={payload_bits})")
    print("""
=== What to do after training ===
1. Use the trained model in Python:
   from src.neural_watermark.embed import NeuralWatermarker
   wm = NeuralWatermarker(encoder_path='data/models/neural_wm/encoder.pt',
                         decoder_path='data/models/neural_wm/decoder.pt')
   watermarked = wm.embed(frame_bgr, payload_bits=[0,1,0,...])  # 48 bits
   extracted = wm.extract(watermarked_frame)

2. Or run a quick test:
   python -m src.neural_watermark.embed --test  # if embed has --test
""")


if __name__ == "__main__":
    main()
