"""
Central configuration for the OpenBot PC Controller.
All hardware IPs, model parameters, and paths in one place.
"""
import os

# ─── Hardware Endpoints ────────────────────────────────────────
ESP8266_WS_URL = "ws://192.168.1.12:81"
ESP32_CAM_STREAM_URL = "http://192.168.1.13:81/stream"

# ─── Camera ────────────────────────────────────────────────────
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600

# ─── Model ─────────────────────────────────────────────────────
# NVIDIA PilotNet standard input size
MODEL_INPUT_WIDTH = 200
MODEL_INPUT_HEIGHT = 66

# ─── Motor ─────────────────────────────────────────────────────
MOTOR_SPEED_MIN = -100
MOTOR_SPEED_MAX = 100
DEFAULT_DRIVE_SPEED = 70  # Default speed for keyboard driving

# ─── Servo ─────────────────────────────────────────────────────
SERVO_MIN = 0
SERVO_MAX = 180
SERVO_CENTER = 90

# ─── Paths ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Data Collection ──────────────────────────────────────────
RECORDING_FPS = 10  # Frames per second to save during recording

# ─── Training Defaults ────────────────────────────────────────
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 200
DEFAULT_LEARNING_RATE = 0.001
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation

# ─── Autopilot ────────────────────────────────────────────────
AUTOPILOT_FPS = 10  # Inference rate
