# Baseline Comparison (Proposal Section: Establish Baselines)

The project proposal specifies benchmarking against two state-of-the-art methods:

1. **VideoSeal (2024)** — Meta's open neural video watermarking  
   - Paper: [arXiv:2402.11668](https://arxiv.org/abs/2402.11668)  
   - Open and light; temporal propagation; robust to common transforms

2. **ItoV (2023)** — Adapting deep image watermarking to video  
   - Paper: [arXiv:2309.01390](https://arxiv.org/abs/2309.01390)  
   - Highlights temporal consistency challenges

## Running Baselines

### VideoSeal

```bash
# Clone and setup (when available)
# git clone https://github.com/facebookresearch/VideoSeal
# pip install -r requirements.txt

# Run on test video
# python videoseal_embed.py --input video.mp4 --output watermarked.mp4
# python videoseal_extract.py --input watermarked.mp4
```

### ItoV

```bash
# Clone and setup (when available)
# git clone https://github.com/...
# Follow paper's implementation instructions
```

## Comparison Script

Use `run_baseline_comparison.py` to evaluate our method vs. baselines on the same attack suite:

```bash
python -m src.baselines.run_baseline_comparison \
  --clean data/input/test.mp4 \
  --ours data/output/watermarked.mp4 \
  --videoseal data/baselines/videoseal_out.mp4 \
  --attack_dir data/output/attacks_ui
```

Output: CSV with PSNR, SSIM, LPIPS, detection rate, BER per method and per attack.
