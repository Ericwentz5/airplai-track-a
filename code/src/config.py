import os
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.environ["ROBOFLOW_API_KEY"]

# --- Models ---
PLAYER_DETECTION_MODEL_ID = "basketball-player-detection-3-ycjdo/4"
PLAYER_DETECTION_MODEL_CONFIDENCE = 0.4
PLAYER_DETECTION_MODEL_IOU_THRESHOLD = 0.9

# --- Class IDs for basketball-player-detection-3-ycjdo/4 ---
BALL_CLASS_ID = 0
BALL_IN_BASKET_CLASS_ID = 1
NUMBER_CLASS_ID = 2
PLAYER_CLASS_IDS = [3, 4, 5, 6, 7]  # player + action variants
REFEREE_CLASS_ID = 8

# --- Paths ---
VIDEO_PATH = os.environ.get("VIDEO_PATH", "data/sample.mp4")
