import os
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]

# --- Models ---
# RF-DETR: detects ball, number, player, player-in-possession,
#          player-jump-shot, player-layup-dunk, player-shot-block, referee, rim
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9

# SmolVLM2 jersey number OCR
JERSEY_OCR_MODEL_ID = "basketball-jersey-numbers-ocr/3"
JERSEY_OCR_PROMPT = "Read the number."

# SigLIP for team classification embeddings
SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"

# --- Class IDs for basketball-player-detection-3-ycjdo/4 ---
BALL_CLASS_ID = 0
BALL_IN_BASKET_CLASS_ID = 1
NUMBER_CLASS_ID = 2
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player + action variants
REFEREE_CLASS_ID = 8

# --- Team info (Macalester footage) ---
TEAM_NAMES = {
    0: "Macalester",
    1: "Opponent",
}

TEAM_COLORS = {
    "Macalester": "#FF6B00",  # orange
    "Opponent":   "#1a1a1a",  # black
}

# --- Crop sampling ---
STRIDE = 30       # sample every Nth frame for team classifier training
BATCH_SIZE = 32   # SigLIP embedding batch size

# --- Jersey OCR validation ---
OCR_CONSECUTIVE_FRAMES = 1  # number must appear N times to be accepted

# --- Paths ---
VIDEO_PATH = os.environ.get("VIDEO_PATH", "data/test_mac_1.mov")
CROPS_DIR = "data/crops"
EVAL_SET_PATH = "data/eval_set.json"
