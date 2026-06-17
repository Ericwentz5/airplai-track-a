# Data

## Video Footage

The evaluation video (`test_mac_1.mov`) is a ~60 second clip of a Macalester College
basketball game filmed from a fixed sideline camera.

- **Teams:** Macalester College (orange jerseys) vs. opponent (black jerseys)
- **Camera:** Fixed, single angle, no broadcast graphics
- **Resolution:** 1080p

This clip is not included in the repository due to size. A short 10-second sample
(`sample.mov`) is included for sanity-checking the pipeline.

To reproduce results on the full clip, contact the repository owner.

## Eval Set

`eval_set.json` contains manually labeled player and number crops extracted from
`test_mac_1.mov`. Each entry has:

- `path` — relative path to the crop image
- `team_label` — 0 (Macalester/orange) or 1 (opponent/black)
- `number_label` — jersey number string, e.g. "23"

To rebuild the eval set from scratch:
```bash
python code/build_eval_set.py --video path/to/test_mac_1.mov
```

## Limitations

- Single game clip, single camera angle
- ~100 labeled crops — sufficient for evaluation but not statistical significance
- Jersey numbers only labeled when clearly visible (facing camera, not blurred)
