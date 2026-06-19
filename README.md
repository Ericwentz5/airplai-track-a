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
python code/run_pipeline.py --video path/to/clip.mov --output results/
```

Outputs:
- `results/annotated.mp4` — video with bounding boxes and track ID labels
- `results/mot.txt` — MOT-format output: `frame,track_id,x,y,w,h,conf,-1,-1,-1`

## Running in Google Colab (T4 GPU)

```python
from google.colab import drive, userdata
import os
drive.mount('/content/drive')

!git clone https://github.com/Ericwentz5/airplai-track-a.git
%cd airplai-track-a
!pip install -q supervision==0.27.0 inference roboflow python-dotenv opencv-python numpy tqdm pycuda onnxruntime-gpu==1.19.2

os.environ['ROBOFLOW_API_KEY'] = userdata.get('ROBOFLOW_API_KEY')

VIDEO_PATH = '/content/drive/MyDrive/AirPlai/test_mac_1.mov'
!python code/run_pipeline.py --video "{VIDEO_PATH}" --output results/
```

## Data

The evaluation video (`test_mac_1.mov`) is a ~16 second clip of a Macalester College basketball game filmed from a fixed sideline camera. It is not included in the repository due to size. A short sample clip is included in `data/` for sanity-checking.

A 10-second sample (`data/sample.mp4`) is included for sanity-checking. To obtain the full evaluation clip, contact the repository owner.

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
│   └── README.md           # data instructions
└── results/                # output video, mot.txt
```

## AI Assistance

This project was developed with Claude Code (Anthropic) for code structure, implementation, and debugging. All model choices, evaluation design, and analysis are the author's own.
