"""
End-to-end Track A pipeline: Team Classification + Jersey OCR.

Usage:
    python run_pipeline.py --video path/to/clip.mov --output results/

What it does:
    1. Runs RF-DETR to detect players and jersey number regions
    2. Fits team classifier (SigLIP → UMAP → KMeans) on player crops
    3. Runs SmolVLM2 OCR on number crops, validates across frames
    4. Writes an annotated output video
    5. Saves predictions to results/metrics/predictions.json
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

import supervision as sv
from inference import get_model

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    ROBOFLOW_API_KEY,
    PLAYER_DETECTION_MODEL_ID,
    PLAYER_DETECTION_MODEL_CONFIDENCE,
    PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
    NUMBER_CLASS_ID,
    PLAYER_CLASS_IDS,
    STRIDE,
    TEAM_NAMES,
    TEAM_COLORS,
)
from team_classifier import TeamClassifier
from jersey_ocr import load_ocr_model, read_number, ConsecutiveNumberTracker


def collect_training_crops(video_path: str, model, stride: int = STRIDE) -> list:
    crops = []
    for frame in tqdm(sv.get_video_frames_generator(video_path, stride=stride), desc="Collecting crops"):
        result = model.infer(frame, confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD)[0]
        dets = sv.Detections.from_inference(result)
        dets = dets[np.isin(dets.class_id, PLAYER_CLASS_IDS)]
        boxes = sv.scale_boxes(xyxy=dets.xyxy, factor=0.4)
        crops += [sv.crop_image(frame, box) for box in boxes]
    return crops


def run(video_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    ocr_model = load_ocr_model()

    # --- Fit team classifier ---
    print("Fitting team classifier...")
    crops = collect_training_crops(video_path, detection_model)
    classifier = TeamClassifier()
    classifier.fit(crops)

    # --- Annotators ---
    team_colors = sv.ColorPalette.from_hex([TEAM_COLORS[TEAM_NAMES[0]], TEAM_COLORS[TEAM_NAMES[1]]])
    box_annotator = sv.BoxAnnotator(color=team_colors, thickness=2, color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color=team_colors, text_color=sv.Color.WHITE, color_lookup=sv.ColorLookup.INDEX)

    tracker = sv.ByteTrack()
    number_tracker = ConsecutiveNumberTracker()

    video_info = sv.VideoInfo.from_video_path(video_path)
    output_video = str(output_dir / "annotated.mp4")
    predictions = []

    with sv.VideoSink(output_video, video_info) as sink:
        for frame_idx, frame in tqdm(enumerate(sv.get_video_frames_generator(video_path)), total=video_info.total_frames, desc="Processing video"):
            frame_h, frame_w = frame.shape[:2]

            result = detection_model.infer(frame, confidence=PLAYER_DETECTION_MODEL_CONFIDENCE, iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD)[0]
            dets = sv.Detections.from_inference(result)

            # --- Players ---
            player_dets = dets[np.isin(dets.class_id, PLAYER_CLASS_IDS)]
            player_dets = tracker.update_with_detections(player_dets)

            boxes = sv.scale_boxes(xyxy=player_dets.xyxy, factor=0.4)
            player_crops = [sv.crop_image(frame, box) for box in boxes]

            if player_crops:
                teams = classifier.predict(player_crops)
            else:
                teams = np.array([])

            # --- Jersey OCR (every 30 frames) ---
            if frame_idx % 30 == 0:
                number_dets = dets[dets.class_id == NUMBER_CLASS_ID]
                padded = sv.clip_boxes(sv.pad_boxes(xyxy=number_dets.xyxy, px=10, py=10), (frame_w, frame_h))

                number_crops = [
                    sv.resize_image(sv.crop_image(frame, box), resolution_wh=(224, 224))
                    for box in padded
                ]

                print(f"[frame {frame_idx}] number_dets={len(number_dets)}")
                # match number crops to player tracks via IoU
                if len(number_dets) > 0 and len(player_dets) > 0:
                    iou = sv.box_iou_batch(player_dets.xyxy, number_dets.xyxy)
                    for num_idx, num_crop in enumerate(number_crops):
                        player_idx = np.argmax(iou[:, num_idx])
                        if iou[player_idx, num_idx] > 0.1 and player_dets.tracker_id is not None:
                            tid = int(player_dets.tracker_id[player_idx])
                            number = read_number(ocr_model, num_crop)
                            print(f"  track {tid} → OCR: {repr(number)}")
                            number_tracker.update(tid, number)

            # --- Build labels ---
            labels = []
            for i, tid in enumerate(player_dets.tracker_id if player_dets.tracker_id is not None else []):
                team = int(teams[i]) if i < len(teams) else 0
                number = number_tracker.get(int(tid))
                labels.append(f"#{number} {TEAM_NAMES[team]}")
                predictions.append({
                    "frame": frame_idx,
                    "track_id": int(tid),
                    "team_pred": team,
                    "number_pred": number,
                })

            # --- Annotate ---
            annotated = frame.copy()
            team_ids = teams.astype(int) if len(teams) > 0 else np.array([], dtype=int)
            annotated = box_annotator.annotate(annotated, player_dets, custom_color_lookup=team_ids)
            annotated = label_annotator.annotate(annotated, player_dets, labels=labels, custom_color_lookup=team_ids)
            sink.write_frame(annotated)

    # --- Save predictions ---
    pred_path = output_dir / "predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)

    print(f"\nAnnotated video: {output_video}")
    print(f"Predictions:     {pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True, help="Path to input video")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()
    run(args.video, args.output)
