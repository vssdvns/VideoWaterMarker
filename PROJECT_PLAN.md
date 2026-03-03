# Video Watermarking Project – Master's Project Plan

**Institution:** CSU Sacramento  
**Type:** Master's Project (capstone / coursework project)  
**Scope:** Applied research with deliverable system + short report

---

## Project Summary

Hybrid video watermarking system that places semi-transparent text watermarks in **less intrusive** regions using:
- Laplacian structural complexity + DeepLab semantic saliency (hybrid fusion)
- Temporal smoothing (EMA) and optional optical flow
- Multi-scale NCC detection with global fallback

---

## Phases

### Phase 1: UI Application ✅

**Goal:** End-to-end application for non-technical use.

- [x] Upload video
- [x] Choose watermarking method (Fixed, Heuristic, DeepLab, Hybrid)
- [x] Generate watermarked video with progress feedback
- [x] Preview watermarked output
- [x] Manual position editor: view saliency/complexity map, adjust watermark position, re-export

**Deliverable:** Working Streamlit app in `src/app_watermark_ui.py`

**Run:** `streamlit run src/app_watermark_ui.py` (from project root)

---

### Phase 2: User-Specific Fingerprint ✅

**Goal:** Traceability for piracy detection.

- [x] Per-user watermark text (user ID or hash)
- [x] Encode in visible text for OTT/streaming scenario
- [x] Optional: location/device-based variations
- [x] UI: traceability mode, user ID / location / device inputs
- [x] CLI demo: `python -m src.demo_fingerprint --input video.mp4 --users user_001 user_002`

**Deliverables:** `src/fingerprint.py`, extended UI, `src/demo_fingerprint.py`

---

### Phase 3: Documentation & Submission ✅

**Goal:** Master's project submission package.

- [x] Short report: `docs/REPORT.md` (~15 pages)
- [x] Code repository with README
- [ ] Demo video / screenshots (optional)

---

## Technical Stack

- **Watermarking:** OpenCV, PyTorch (DeepLabV3-ResNet50)
- **UI:** Streamlit
- **Detection:** OpenCV NCC, multi-scale, global fallback

---

## Success Criteria (Master's Project)

1. Functional UI for upload → watermark → preview → manual adjust
2. All 4 methods (fixed, heuristic, DeepLab, hybrid) available
3. Benchmark results reproducible (≈94.6% detection at thr=0.40)
4. Clear documentation and run instructions

---

## Timeline Suggestion

| Phase   | Duration | Status      |
|---------|----------|-------------|
| Phase 1 | 1–2 weeks| Done        |
| Phase 2 | 1–2 weeks| Done        |
| Phase 3 | 1 week   | Pending     |
