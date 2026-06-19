# Data

## Sample Clip

`sample.mp4` is a 10-second clip of a Macalester College basketball game filmed from a fixed sideline camera. Included for sanity-checking the pipeline.

- **Teams:** Macalester College (orange jerseys) vs. opponent (black jerseys)
- **Camera:** Fixed, single angle, no broadcast graphics
- **Resolution:** 1080p

## Full Evaluation Clip

The full evaluation clip (`test_mac_1.mov`, ~60 seconds) is not included due to file size. Contact the repository owner to obtain it.

## Running on the Sample

```bash
python code/run_pipeline.py --video data/sample.mp4 --output results/
```
