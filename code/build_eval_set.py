"""
Builds the ground truth evaluation set.

Step 1: Extracts player and number crops from the video.
Step 2: Saves them to data/crops/.
Step 3: Prompts you in the terminal to label each crop.
Step 4: Saves labeled data to data/eval_set.json.

Run this once before running evaluate.py.

Usage:
    python build_eval_set.py --video path/to/clip.mov
"""

import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from extract_crops import extract_crops
from config import VIDEO_PATH, EVAL_SET_PATH, CROPS_DIR


def label_player_crops(meta: list) -> list:
    """Show each player crop and ask for team label (0 or 1)."""
    print("\n--- Labeling player crops ---")
    print("Enter 0 = Macalester (orange), 1 = Opponent (black), s = skip, q = quit\n")

    labeled = []
    for entry in meta:
        img = cv2.imread(entry["path"])
        if img is None:
            continue

        cv2.imshow("Player Crop - Label Team (0/1/s/q)", img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            continue
        elif key in (ord("0"), ord("1")):
            entry["team_label"] = int(chr(key))
            labeled.append(entry)

    cv2.destroyAllWindows()
    return labeled


def label_number_crops(meta: list) -> list:
    """Show each number crop and ask for jersey number."""
    print("\n--- Labeling number crops ---")
    print("Type the jersey number and press Enter. Leave blank to skip, 'q' to quit.\n")

    labeled = []
    for entry in meta:
        img = cv2.imread(entry["path"])
        if img is None:
            continue

        cv2.imshow("Number Crop - Type Jersey Number", img)
        cv2.waitKey(100)

        number = input(f"  [{Path(entry['path']).name}] Jersey number: ").strip()
        if number == "q":
            break
        elif number == "":
            continue
        else:
            entry["number_label"] = number
            labeled.append(entry)

    cv2.destroyAllWindows()
    return labeled


def main(video_path: str):
    print(f"Extracting crops from {video_path}...")
    meta = extract_crops(video_path, output_dir=CROPS_DIR, stride=60)

    player_meta = label_player_crops(meta["player_crops"])
    number_meta = label_number_crops(meta["number_crops"])

    eval_set = {
        "video": video_path,
        "player_crops": player_meta,
        "number_crops": number_meta,
        "notes": "Labeled manually from test_mac_1.mov. Macalester=0 (orange), Opponent=1 (black).",
    }

    Path(EVAL_SET_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SET_PATH, "w") as f:
        json.dump(eval_set, f, indent=2)

    print(f"\nEval set saved to {EVAL_SET_PATH}")
    print(f"  {len(player_meta)} labeled player crops")
    print(f"  {len(number_meta)} labeled number crops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=VIDEO_PATH)
    args = parser.parse_args()
    main(args.video)
