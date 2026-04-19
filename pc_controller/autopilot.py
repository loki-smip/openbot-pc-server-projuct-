"""
Autopilot v4 — Dual-head classification inference.
Predicts discrete motor action + servo tilt position from camera frame.
"""
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from config import (
    MODELS_DIR, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT,
    AUTOPILOT_FPS, DEFAULT_DRIVE_SPEED, SERVO_POSITIONS,
)
from model import (
    ActionNet, action_to_command, class_to_servo,
    NUM_MOTOR_ACTIONS, NUM_SERVO_POSITIONS,
)
from trainer import crop_and_resize


# Motor action names for display
MOTOR_ACTION_NAMES = [
    "STOP", "FWD", "BWD", "LEFT", "RIGHT",
    "FWD+L", "FWD+R", "BWD+L", "BWD+R"
]

# Backward compat alias
ACTION_NAMES = MOTOR_ACTION_NAMES


class Autopilot:
    def __init__(self, camera_stream, car_connection):
        self.camera = camera_stream
        self.car = car_connection
        self._running = False
        self._thread = None
        self._model = None
        self._device = None
        self._model_name = ""
        self._inference_fps = 0.0
        self._last_prediction = [0.0, 0.0]
        self._last_servo = 90
        self._last_action = "—"
        self._speed = DEFAULT_DRIVE_SPEED
        self._to_tensor = transforms.ToTensor()

    @property
    def is_running(self):
        return self._running

    @property
    def model_name(self):
        return self._model_name

    @property
    def inference_fps(self):
        return self._inference_fps

    @property
    def last_prediction(self):
        return self._last_prediction.copy()

    @property
    def last_servo(self):
        return self._last_servo

    @property
    def last_action(self):
        return self._last_action

    def start(self, model_name: str, speed: int = None):
        if self._running:
            return {"error": "Autopilot already running"}

        if speed:
            self._speed = max(20, min(100, speed))

        import os
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")
        if not os.path.isfile(model_path):
            return {"error": f"Model not found: {model_path}"}

        try:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = ActionNet().to(self._device)

            checkpoint = torch.load(model_path, map_location=self._device, weights_only=False)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()

            self._model_name = model_name
            val_acc = checkpoint.get('val_acc', '?')
            motor_acc = checkpoint.get('val_motor_acc', '?')
            servo_acc = checkpoint.get('val_servo_acc', '?')
            acc_str = f"{100*val_acc:.1f}%" if isinstance(val_acc, float) else val_acc
            print(f"[Autopilot] Model loaded: {model_name} (combined: {acc_str})")
            if isinstance(motor_acc, float) and isinstance(servo_acc, float):
                print(f"[Autopilot]   Motor acc: {100*motor_acc:.1f}%, Servo acc: {100*servo_acc:.1f}%")

        except Exception as e:
            return {"error": f"Failed to load model: {e}"}

        self._running = True
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()
        return {"status": "running", "model": model_name}

    def stop(self):
        if not self._running:
            return {"error": "Autopilot not running"}

        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

        self.car.send_command(0, 0, 90)  # Stop motors, center servo
        self._last_prediction = [0.0, 0.0]
        self._last_servo = 90
        self._last_action = "—"

        result = {"status": "stopped", "model": self._model_name}
        self._model_name = ""
        return result

    def _inference_loop(self):
        interval = 1.0 / AUTOPILOT_FPS
        fps_timer = time.time()
        fps_count = 0

        # Smoothing: majority vote over last N frames
        motor_votes = []
        servo_votes = []
        vote_window = 3

        print(f"[Autopilot] Autonomous mode ACTIVE | Speed: {self._speed} | {AUTOPILOT_FPS} FPS")

        while self._running:
            start = time.time()

            frame = self.camera.get_frame_cv2()
            if frame is None:
                time.sleep(0.05)
                continue

            # Same preprocessing as training
            preprocessed = crop_and_resize(frame)
            img_tensor = self._to_tensor(preprocessed).unsqueeze(0).to(self._device)

            # Inference — dual head
            with torch.no_grad():
                motor_logits, servo_logits = self._model(img_tensor)
                motor_probs = F.softmax(motor_logits, dim=1)
                servo_probs = F.softmax(servo_logits, dim=1)
                predicted_motor = torch.argmax(motor_probs, dim=1).item()
                predicted_servo = torch.argmax(servo_probs, dim=1).item()
                motor_confidence = motor_probs[0, predicted_motor].item()

            # Majority vote smoothing (prevents flickering)
            motor_votes.append(predicted_motor)
            servo_votes.append(predicted_servo)
            if len(motor_votes) > vote_window:
                motor_votes.pop(0)
            if len(servo_votes) > vote_window:
                servo_votes.pop(0)

            from collections import Counter
            smoothed_motor = Counter(motor_votes).most_common(1)[0][0]
            smoothed_servo = Counter(servo_votes).most_common(1)[0][0]

            # Convert to commands
            left, right = action_to_command(smoothed_motor, self._speed)
            servo_angle = class_to_servo(smoothed_servo)

            # Update state
            action_name = MOTOR_ACTION_NAMES[smoothed_motor] if smoothed_motor < len(MOTOR_ACTION_NAMES) else f"A{smoothed_motor}"
            self._last_action = f"{action_name} ({100*motor_confidence:.0f}%)"
            self._last_prediction = [left / 100.0, right / 100.0]
            self._last_servo = servo_angle

            # Send to car — motor + servo
            self.car.send_command(left, right, servo_angle)

            # FPS tracking
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                self._inference_fps = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()

            elapsed = time.time() - start
            if interval - elapsed > 0:
                time.sleep(interval - elapsed)

        print("[Autopilot] Stopped")
        self.car.send_command(0, 0, 90)
