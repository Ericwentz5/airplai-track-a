"""
Jersey number OCR using Roboflow's SmolVLM2 model.

Reads jersey numbers from padded number crops and validates
them using a consecutive-agreement tracker.
"""

import re
from collections import defaultdict

from inference import get_model

from config import ROBOFLOW_API_KEY, JERSEY_OCR_MODEL_ID, JERSEY_OCR_PROMPT, OCR_CONSECUTIVE_FRAMES


def load_ocr_model():
    return get_model(model_id=JERSEY_OCR_MODEL_ID, api_key=ROBOFLOW_API_KEY)


def read_number(model, crop) -> str:
    """Run OCR on a single crop. Returns digit string or empty string."""
    raw = model.predict(crop, JERSEY_OCR_PROMPT)[0]
    digits = re.sub(r"[^0-9]", "", str(raw))
    return digits if 1 <= len(digits) <= 2 else ""


class ConsecutiveNumberTracker:
    """
    Locks a jersey number to a track ID only after it appears
    N consecutive times consistently.
    """

    def __init__(self, n_consecutive: int = OCR_CONSECUTIVE_FRAMES):
        self.n = n_consecutive
        self._history: dict[int, list[str]] = defaultdict(list)
        self._locked: dict[int, str] = {}

    def update(self, track_id: int, number: str) -> None:
        if track_id in self._locked:
            return
        history = self._history[track_id]
        if history and history[-1] != number:
            self._history[track_id] = [number]
        else:
            history.append(number)
            if len(history) >= self.n and number != "":
                self._locked[track_id] = number

    def get(self, track_id: int) -> str:
        return self._locked.get(track_id, "?")

    def get_all(self) -> dict[int, str]:
        return dict(self._locked)

    def reset(self) -> None:
        self._history.clear()
        self._locked.clear()
