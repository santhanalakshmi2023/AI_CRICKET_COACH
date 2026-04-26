"""
config.py — Global constants and configuration
"""

# ── Model paths ────────────────────────────────────────────────────────────────
POSE_MODEL_PATH  = "yolov8m-pose.pt"   # YOLOv8 pose model
OBJ_MODEL_PATH   = "yolov8n.pt"        # YOLOv8 object-detection model

# GGUF model file for Gemma 3n 270M (llama-cpp-python)
# Download from HuggingFace: google/gemma-3n-E2B-it-GGUF  (270M variant)
GEMMA_MODEL_PATH = "models/gemma-3n-270m.gguf"

# ── Video output ───────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_PATH = "output.mp4"

# ── Detection thresholds ───────────────────────────────────────────────────────
BALL_IMPACT_SPEED_THRESHOLD = 5        # pixels/frame
BAT_BBOX_PADDING            = 30       # pixels around bat bbox for impact check
WRIST_TRAIL_LENGTH          = 30       # frames to keep wrist trail
ANGLE_BUFFER_SIZE           = 5        # rolling-average window

# YOLO class IDs (COCO)
COCO_BALL_CLASS  = 32
COCO_BAT_CLASS   = 39   # sports-ball → repurposed; swap to custom model ID as needed

# ── Scoring weights ────────────────────────────────────────────────────────────
BALANCE_WEIGHT = 0.4
POWER_WEIGHT   = 0.3
TIMING_WEIGHT  = 0.3

# ── Gemma inference ────────────────────────────────────────────────────────────
GEMMA_MAX_TOKENS = 300
GEMMA_TEMPERATURE = 0.7

# ── Database ───────────────────────────────────────────────────────────────────
DB_PATH = "cricket_ai.db"
