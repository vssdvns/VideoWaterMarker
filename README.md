\# VideoWaterMarker



Model-guided adaptive video watermarking project for my master's work at CSU Sacramento.



The goal is to place visible watermarks in \*\*less intrusive regions\*\* of each video frame by using

both simple image statistics and deep learning–based saliency, as a first step toward

user-specific, piracy-resistant watermarking for OTT platforms.



---



\## Features (Phase 1)



This repo currently implements \*\*three watermarking strategies\*\* and a simple evaluation pipeline:



\### 1. Simple fixed watermark (baseline)



\- Constant text watermark (e.g. `VideoWaterMarker`) placed at the \*\*bottom-right\*\* corner

&nbsp; of every frame.

\- Implemented in: `src/video\_watermark\_demo.py` → `add\_text\_watermark\_fixed(...)`.



\### 2. Heuristic adaptive watermark (Laplacian complexity)



\- Computes a \*\*Laplacian-based "complexity map"\*\* per frame using OpenCV.

\- Treats low-complexity (smooth) regions as better locations for the watermark.

\- Slides a window over the map and selects the \*\*lowest-average-complexity\*\* window.

\- Implemented in:  

&nbsp; - `compute\_complexity\_map(...)`  

&nbsp; - `choose\_low\_complexity\_region(...)`  

&nbsp; - used by `add\_text\_watermark\_to\_video(..., use\_saliency\_model=False)`.



\### 3. DeepLab-based adaptive watermark (semantic saliency)



\- Uses a pretrained \*\*DeepLabV3–ResNet50\*\* segmentation model (from `torchvision`)

&nbsp; to obtain a per-pixel saliency / foreground map.

\- Interprets \*\*background / low-saliency\*\* pixels as better locations for the watermark.

\- Again uses the sliding window to place the watermark in the \*\*least-salient region\*\*.

\- Implemented in:

&nbsp; - `src/models/saliency\_deeplab.py` (`DeepLabSaliency`),

&nbsp; - `add\_text\_watermark\_to\_video(..., use\_saliency\_model=True)`.



---



\## Evaluation



The script `src/analyze\_watermark\_saliency.py` compares the three strategies:



\- For each frame in the \*\*original\*\* `sample.mp4`, it computes:

&nbsp; - a Laplacian complexity map, and

&nbsp; - (optionally) a DeepLab-based saliency map.

\- For each method:

&nbsp; - it computes where the watermark \*\*would\*\* be placed,

&nbsp; - extracts that region from the evaluation map,

&nbsp; - and averages the saliency values.



This gives \*\*average saliency under the watermark region\*\*:



\- Lower value ⇒ watermark sits in a "less important" region (better).



Early results on one test video show:



\- The \*\*heuristic method\*\* minimizes edge-based complexity (very smooth regions),

\- The \*\*DeepLab-based method\*\* minimizes semantic saliency (avoids foreground objects),

\- The \*\*simple baseline\*\* is worst on both metrics.



---



\## Repository Structure



```text

VideoWaterMarker/

&nbsp; .gitignore

&nbsp; README.md

&nbsp; test\_setup.py           # quick check for torch, cv2, numpy, matplotlib versions



&nbsp; data/

&nbsp;   input/                # place your input images/videos here (ignored by git)

&nbsp;   output/               # watermarked outputs (ignored by git)

&nbsp;   debug/                # heatmaps / debug images (ignored by git)



&nbsp; src/

&nbsp;   video\_watermark\_demo.py       # main script: generates 3 watermarked videos

&nbsp;   watermark\_demo.py             # basic image watermark example

&nbsp;   debug\_complexity\_heatmap.py   # saves visualizations of the complexity heatmap

&nbsp;   analyze\_watermark\_saliency.py # evaluation of placement strategies



&nbsp;   models/

&nbsp;     saliency\_deeplab.py         # DeepLabV3-ResNet50 wrapper for saliency maps



