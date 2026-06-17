"""
Runs RF-DETR on a video and extracts:
  - player crops (for team classification)
  - number crops (for jersey OCR)

Saves crops to disk and returns metadata for downstream use.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import supervision as sv
from inference import get_model

from config import (
    ROBOFLOW_API_KEY,
    PLAYER_DETECTION_MODEL_ID,
    PLAYER_DETECTION_MODEL_CONFIDENCE,
    PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    NUMBER_CLASS_ID,
    PLAYER_CLASS_IDS,
    STRIDE,
    CROPS_DIR,
)


def extract_crops(video_path: str, output_dir: str = CROPS_DIR, stride: int = STRIDE):
    """
    Extract player and number crops from a video.
    Returns a dict with lists of crop paths and metadata.
    """
    player_dir = Path(output_dir) / "players"
    number_dir = Path(output_dir) / "numbers"
    player_dir.mkdir(parents=True, exist_ok=True)
    number_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)

    frame_generator = sv.get_video_frames_generator(video_path, stride=stride)
    video_info = sv.VideoInfo.from_video_path(video_path)
    total = video_info.total_frames // stride

    player_crops_meta = []
    number_crops_meta = []

    for frame_idx, frame in tqdm(enumerate(frame_generator), total=total, desc="Extracting crops"):
        frame_h, frame_w = frame.shape[:2]

        result = model.infer(
            frame,
            confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
            iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
        )[0]
        detections = sv.Detections.from_inference(result)

        # --- Player crops (scaled down to jersey region) ---
        player_dets = detections[np.isin(detections.class_id, PLAYER_CLASS_IDS)]
        boxes = sv.scale_boxes(xyxy=player_dets.xyxy, factor=0.4)
        for i, (box, class_id) in enumerate(zip(boxes, player_dets.class_id)):
            crop = sv.crop_image(frame, box)
            if crop.size == 0:
                continue
            fname = f"frame{frame_idx:05d}_player{i:02d}.jpg"
            path = str(player_dir / fname)
            cv2.imwrite(path, crop)
            player_crops_meta.append({
                "path": path,
                "frame": frame_idx * stride,
                "class_id": int(class_id),
                "team_label": None,  # filled in by build_eval_set.py
            })

        # --- Number crops (padded) ---
        number_dets = detections[detections.class_id == NUMBER_CLASS_ID]
        padded = sv.clip_boxes(
            sv.pad_boxes(xyxy=number_dets.xyxy, px=10, py=10),
            (frame_w, frame_h),
        )
        for i, box in enumerate(padded):
            crop = sv.crop_image(frame, box)
            crop = sv.resize_image(crop, resolution_wh=(224, 224))
            if crop.size == 0:
                continue
            fname = f"frame{frame_idx:05d}_number{i:02d}.jpg"
            path = str(number_dir / fname)
            cv2.imwrite(path, crop)
            number_crops_meta.append({
                "path": path,
                "frame": frame_idx * stride,
                "number_label": None,  # filled in by build_eval_set.py
            })

    print(f"Saved {len(player_crops_meta)} player crops → {player_dir}")
    print(f"Saved {len(number_crops_meta)} number crops  → {number_dir}")

    return {
        "player_crops": player_crops_meta,
        "number_crops": number_crops_meta,
    }


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else CROPS_DIR
    from config import VIDEO_PATH
    extract_crops(VIDEO_PATH)
