# AirPLAi Track B: Player Detection & Tracking

Detects and tracks all players across a basketball clip using RF-DETR + ByteTrack, producing MOT-format output and an annotated video.

**Pipeline:** RF-DETR detection → ByteTrack tracking → MOT output + annotated video

## Requirements

- Python 3.11
- CUDA 12.x GPU recommended (tested on Colab T4)

```bash
pip install supervision==0.27.0 inference roboflow python-dotenv opencv-python numpy tqdm pycuda onnxruntime-gpu==1.19.2
```

> **GPU note:** `onnxruntime-gpu==1.19.2` is pinned specifically for CUDA 12.x (tested on Colab T4). Newer versions (1.20+) require CUDA 13 and will fail with a `libcudart` error. If you upgrade and hit that error, downgrade back to 1.19.2.

> **CPU-only fallback:** replace `pycuda onnxruntime-gpu==1.19.2` with `onnxruntime`. Significantly slower.

Set your Roboflow API key:
```bash
echo "ROBOFLOW_API_KEY=your_key_here" > code/.env
```

## Reproduce Results

```bash
python code/run_pipeline.py --video data/sample.mp4 --output results/
```

Outputs:
- `results/annotated.mp4` — video with bounding boxes and track ID labels
- `results/mot.txt` — MOT-format output: `frame,track_id,x,y,w,h,conf,-1,-1,-1`

## Running in Google Colab (T4 GPU)

**Recommended.** The pipeline requires a CUDA-capable GPU. If you don't have one locally, run it in Google Colab with a T4 GPU runtime (Runtime → Change runtime type → T4 GPU).

**Option A — Use the included notebook:** Open `code/Track_B_pipeline.ipynb` directly in Colab. It contains all cells pre-configured and ready to run. The notebook clones this repo and executes `run_pipeline.py` — no manual copy-pasting needed.

**Option B — Open a new notebook** and run each cell in order:

**Cell 1 — Mount Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — Clone repo**
```python
!git clone https://github.com/Ericwentz5/airplai-track-a.git
%cd airplai-track-a
```

**Cell 3 — Install dependencies**
```python
!pip install -q supervision==0.27.0 inference roboflow python-dotenv opencv-python numpy tqdm pycuda onnxruntime-gpu==1.19.2
```

**Cell 4 — Set API key** (add `ROBOFLOW_API_KEY` to Colab Secrets first)
```python
from google.colab import userdata
import os
os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')
```

**Cell 5 — Run pipeline**
```python
!python code/run_pipeline.py --video data/sample.mp4 --output results/
```

**Cell 6 — Save results to Google Drive**
```python
!cp -r results/ '/content/drive/MyDrive/AirPlai/results/'
```

## Data

A 10-second sample clip (`data/sample.mp4`) is included for sanity-checking. It shows a Macalester College basketball game filmed from a fixed sideline camera (orange jerseys vs. black jerseys, 1080p, no broadcast graphics).

The full evaluation clip is not included due to file size. Contact the repository owner to obtain it.

## Models

All models are downloaded automatically via the Roboflow API — no local weights needed.

| Model | ID | Purpose |
|-------|----|---------|
| RF-DETR | `basketball-player-detection-3-ycjdo/4` | Player detection |
| ByteTrack | supervision built-in | Multi-object tracking |

## File Structure

```
airplai-track-a/
├── design.pdf                          # Part 1 system design document
├── report.pdf                          # Part 2 methodology and results
├── README.md
├── code/
│   ├── src/
│   │   └── config.py                   # model IDs, class IDs, paths
│   ├── run_pipeline.py                 # end-to-end inference script
│   ├── Track_B_pipeline.ipynb          # Colab notebook
│   └── requirements.txt
├── data/
│   ├── sample.mp4                      # 10-second sample clip
│   └── README.md
└── results/
    ├── macalester_short_annotated.mp4  # annotated output video
    ├── macalester_mot.txt              # MOT-format tracking output
    ├── metrics/
    │   └── metrics.json                # MOTA evaluation results
    ├── visualizations/                 # screenshots and clips of tracking working
    └── failure_cases/                  # screenshots and clips of failure modes
```

## Troubleshooting

**`pycuda` build fails**
pycuda requires CUDA toolkit headers to compile. If it fails, use the CPU fallback instead:
```bash
pip install onnxruntime
```
Then set `CUDA_VISIBLE_DEVICES=''` before running. Slower but works.

**`libcudart` error with onnxruntime-gpu**
Newer versions of `onnxruntime-gpu` (1.20+) require CUDA 13. Pin to 1.19.2:
```bash
pip install onnxruntime-gpu==1.19.2
```

**`Could not open video` error**
Check the video path is correct and the file exists. Use absolute paths in Colab (e.g. `/content/airplai-track-a/data/sample.mp4`).

**`ROBOFLOW_API_KEY` not found**
Make sure you've added it to Colab Secrets (key icon in left sidebar) or created a `code/.env` file locally.

## AI Assistance

This project was developed with Claude Code (Anthropic) for code structure, implementation, and debugging. All model choices, evaluation design, and analysis are the author's own.
