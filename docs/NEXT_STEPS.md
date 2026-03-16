# Next Steps — Neural Watermark on Video

Focus: **embed the neural watermark on your video** (no OTT optimization for now).

---

## 1. Embed Watermark on a Video (Ready Now)

If you have trained models (`encoder.pt`, `decoder.pt` in `data/models/neural_wm/`):

```cmd
python scripts/embed_neural_video.py input.mp4 output_watermarked.mp4
```

**Custom payload** (e.g. user ID as bits):

```cmd
python scripts/embed_neural_video.py input.mp4 output.mp4 --payload 0,1,0,1,1,0,1,0,0,1
```

**With GPU:**

```cmd
python scripts/embed_neural_video.py input.mp4 output.mp4 --device cuda
```

---

## 2. If Training Hasn’t Finished or Models Are Missing

```cmd
venv_gpu\Scripts\activate
python -m src.neural_watermark.train --data_dir data --epochs 100 --batch 4 --size 128 --device cuda --payload_bits 16
```

Models are saved to `data/models/neural_wm/` when training finishes.

---

## 3. Quick Test on Single Image

```cmd
python -m src.neural_watermark.embed --test
```

Reports Bit Error Rate (BER). Low BER = model is working; high BER = needs more training.

---

## 4. Current vs Future Steps

| Now (Simple) | Later (Optional) |
|--------------|------------------|
| Embed on full video | Per-user/session payloads |
| Fixed or custom payload | HLS/DASH segment watermarking |
| One output file | OTT CDN integration |
| Manual script | UI / API integration |

---

## Summary

1. **Embed video:** `python scripts/embed_neural_video.py input.mp4 output.mp4`
2. **Check models:** Ensure `data/models/neural_wm/encoder.pt` and `decoder.pt` exist
3. **Improve training:** If BER is high, keep training or try `--payload_bits 16`
