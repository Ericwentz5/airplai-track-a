"""
Evaluation metrics for Track A: Team & Jersey ID.

Loads ground truth from the eval set (built by build_eval_set.py)
and computes:
  - Team classification accuracy
  - Jersey OCR exact match rate
  - Per-team breakdown
  - Confusion matrix
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix

from config import EVAL_SET_PATH


def load_eval_set(path: str = EVAL_SET_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def evaluate_team_classification(eval_set: dict) -> dict:
    """
    eval_set["player_crops"] must have:
      - "team_label":    int (0 or 1, ground truth)
      - "team_pred":     int (0 or 1, model prediction)
    """
    entries = [e for e in eval_set["player_crops"] if e.get("team_label") is not None and e.get("team_pred") is not None]

    if not entries:
        return {"error": "No labeled player crops found"}

    gt = [e["team_label"] for e in entries]
    pred = [e["team_pred"] for e in entries]

    acc = accuracy_score(gt, pred)
    cm = confusion_matrix(gt, pred).tolist()

    # per-team accuracy
    per_team = {}
    for team_id in [0, 1]:
        mask = [i for i, g in enumerate(gt) if g == team_id]
        if mask:
            team_gt   = [gt[i] for i in mask]
            team_pred = [pred[i] for i in mask]
            per_team[team_id] = round(accuracy_score(team_gt, team_pred), 4)

    return {
        "n_samples": len(entries),
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "per_team_accuracy": per_team,
    }


def evaluate_jersey_ocr(eval_set: dict) -> dict:
    """
    eval_set["number_crops"] must have:
      - "number_label": str (ground truth jersey number)
      - "number_pred":  str (model prediction)
    """
    entries = [e for e in eval_set["number_crops"] if e.get("number_label") is not None and e.get("number_pred") is not None]

    if not entries:
        return {"error": "No labeled number crops found"}

    exact_matches = sum(1 for e in entries if e["number_pred"] == e["number_label"])
    exact_match_rate = exact_matches / len(entries)

    # breakdown by failure type
    no_read   = [e for e in entries if e["number_pred"] == ""]
    wrong_num = [e for e in entries if e["number_pred"] != e["number_label"] and e["number_pred"] != ""]

    return {
        "n_samples": len(entries),
        "exact_match_rate": round(exact_match_rate, 4),
        "no_read_rate": round(len(no_read) / len(entries), 4),
        "wrong_number_rate": round(len(wrong_num) / len(entries), 4),
        "failure_examples": [
            {"label": e["number_label"], "pred": e["number_pred"], "path": e["path"]}
            for e in wrong_num[:10]
        ],
    }


def run_evaluation(eval_set_path: str = EVAL_SET_PATH) -> dict:
    eval_set = load_eval_set(eval_set_path)
    return {
        "team_classification": evaluate_team_classification(eval_set),
        "jersey_ocr": evaluate_jersey_ocr(eval_set),
    }


if __name__ == "__main__":
    import json
    results = run_evaluation()
    print(json.dumps(results, indent=2))
