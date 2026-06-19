"""
Track B pipeline: Player Detection + Tracking.

Usage:
    python run_pipeline.py --video path/to/clip.mov --output results/

What it does:
    1. Runs RF-DETR to detect players each frame
    2. Runs ByteTrack to assign consistent track IDs across frames
    3. Writes an annotated output video with track ID labels
    4. Saves MOT-format output to results/mot.txt
"""

import os
import sys
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
    PLAYER_CLASS_IDS,
)


def run(video_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_model = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    tracker = sv.ByteTrack()

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.WHITE)

    video_info = sv.VideoInfo.from_video_path(video_path)
    output_video = str(output_dir / "annotated.mp4")
    mot_rows = []

    with sv.VideoSink(output_video, video_info) as sink:
        for frame_idx, frame in tqdm(
            enumerate(sv.get_video_frames_generator(video_path)),
            total=video_info.total_frames,
            desc="Processing",
        ):
            result = detection_model.infer(
                frame,
                confidence=PLAYER_DETECTION_MODEL_CONFIDENCE,
                iou_threshold=PLAYER_DETECTION_MODEL_IOU_THRESHOLD,
            )[0]
            dets = sv.Detections.from_inference(result)
            player_dets = dets[np.isin(dets.class_id, PLAYER_CLASS_IDS)]
            player_dets = tracker.update_with_detections(player_dets)

            if player_dets.tracker_id is not None:
                for i, tid in enumerate(player_dets.tracker_id):
                    x1, y1, x2, y2 = player_dets.xyxy[i].astype(int)
                    w, h = x2 - x1, y2 - y1
                    conf = float(player_dets.confidence[i]) if player_dets.confidence is not None else 1.0
                    mot_rows.append(f"{frame_idx + 1},{tid},{x1},{y1},{w},{h},{conf:.4f},-1,-1,-1")

            labels = [f"#{tid}" for tid in (player_dets.tracker_id if player_dets.tracker_id is not None else [])]
            annotated = box_annotator.annotate(frame.copy(), player_dets)
            annotated = label_annotator.annotate(annotated, player_dets, labels=labels)
            sink.write_frame(annotated)

    mot_path = output_dir / "mot.txt"
    with open(mot_path, "w") as f:
        f.write("\n".join(mot_rows) + "\n")

    print(f"\nAnnotated video: {output_video}")
    print(f"MOT output:      {mot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",  required=True, help="Path to input video")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()
    run(args.video, args.output)
