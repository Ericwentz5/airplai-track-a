# Track A: Team & Jersey ID

Predicts team assignment and jersey number for each detected player in a basketball video clip.

**Pipeline:** RF-DETR detection → SigLIP embeddings → UMAP → KMeans (teams) + SmolVLM2 OCR (jersey numbers)

## Setup

Python 3.11 required.

```bash
conda create -n airplai python=3.11
conda activate airplai
conda install -c conda-forge llvmlite numba umap-learn
pip install -r requirements.txt
```

Create a `.env` file in the `code/` directory:
```
ROBOFLOW_API_KEY=your_key_here
```

## Reproduce Results

**Step 1 — Run the full pipeline on the evaluation video:**
```bash
python run_pipeline.py --video path/to/test_mac_1.mov --output ../results/
```

**Step 2 — Build the labeled eval set** (requires manual labeling, ~15 min):
```bash
python build_eval_set.py --video path/to/test_mac_1.mov
```

**Step 3 — Run evaluation:**
```bash
python src/evaluate.py
```
Outputs metrics to stdout and saves `results/metrics/metrics.json`.

## Models

All models are loaded via the Roboflow hosted API — no local weights needed.

| Model | Roboflow ID | Purpose |
|-------|-------------|---------|
| RF-DETR | `basketball-player-detection-3-ycjdo/4` | Player + number detection |
| SmolVLM2 | `basketball-jersey-numbers-ocr/3` | Jersey number OCR |
| SigLIP | `google/siglip-base-patch16-224` (HuggingFace) | Player embeddings |

## File Structure

```
code/
├── src/
│   ├── config.py          # all settings
│   ├── extract_crops.py   # detection + crop extraction
│   ├── team_classifier.py # SigLIP → UMAP → KMeans
│   ├── jersey_ocr.py      # SmolVLM2 OCR + validation
│   └── evaluate.py        # accuracy metrics
├── run_pipeline.py        # end-to-end inference
├── build_eval_set.py      # manual labeling tool
└── requirements.txt
```

## AI Assistance

This project was developed with Claude Code (Anthropic) used for code structure,
implementation scaffolding, and debugging. All model choices, evaluation design,
and analysis are the author's own.
