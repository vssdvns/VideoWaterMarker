# Neural Watermark Training — Optimized Parameters

## Current Values (Tuned for Learning)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **payload_bits** | 16 | Fewer bits = easier to learn (try 8 for very fast convergence) |
| **phase0_epochs** | 25 | BCE-only phase; longer lets model learn to embed before MSE |
| **delta_weight** | 0.8 | Delta-decoder loss; direct gradient to encoder |
| **lr_phase0** | 3e-3 | Higher LR in phase 0 for faster initial learning |
| **mse_weight (phase1)** | 0.2 | Lower than 0.3 so encoder can embed more |
| **mse_weight (phase2)** | 0.5 → 1.0 | Curriculum: start gentle, increase over epochs |
| **blur_sigma** | 0.35 | Gentler attack (was 0.5) |
| **noise_std** | 0.015 | Gentler noise (was 0.02) |
| **Decoder dropout** | 0.05 | Reduced from 0.1 to avoid underfitting |
| **Gradient clip** | 1.0 | Prevents instability |

## Target Metrics

| Metric | Random | Good | Very Good |
|--------|--------|------|-----------|
| BCE | 0.693 | < 0.2 | < 0.05 |
| MSE | 0 | 0.001–0.01 | < 0.005 |

## Recommended Commands

**Improved preset** (8 bits, 180 ep, gentle attacks, low MSE):
```cmd
python -m src.neural_watermark.train --improved --device cuda
```

**Custom:**
```cmd
python -m src.neural_watermark.train --data_dir data --epochs 150 --batch 4 --size 128 --device cuda --payload_bits 16 --phase0_epochs 25 --delta_weight 0.8 --lr_phase0 3e-3
```

## Tuning Guide

| If BCE stays high | Try |
|-------------------|-----|
| Stuck ~0.69 | `--phase0_epochs 40`, `--payload_bits 8` |
| Drops then plateaus | `--lr_phase0 5e-3`, `--delta_weight 1.0` |
| Unstable / NaN | `--lr_phase0 1e-3`, smaller batch |
| MSE too high (visible) | Increase `mse_weight` in code for phase1 |
