# AirPLAi Track B: Player Detection & Tracking

Detects and tracks all players across a basketball clip using RF-DETR + ByteTrack, producing MOT-format output and an annotated video.

**Pipeline:** RF-DETR detection → ByteTrack tracking → MOT output + annotated video

## Requirements

- Python 3.11
- CUDA 12.x GPU recommended (tested on Colab T4)

```bash
pip install supervision==0.27.0 inference roboflow python-dotenv opencv-python numpy tqdm pycuda onnxruntime-gpu==1.19.2
```

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

Open a new Colab notebook and run each cell in order:

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
├── code/
│   ├── src/
│   │   └── config.py       # model IDs, class IDs, paths
│   ├── run_pipeline.py     # end-to-end inference script
│   └── requirements.txt
├── data/
│   ├── sample.mp4          # 10-second sample clip
│   └── README.md
└── results/                # output video and mot.txt written here
```

## AI Assistance

This project was developed with Claude Code (Anthropic) for code structure, implementation, and debugging. All model choices, evaluation design, and analysis are the author's own.
