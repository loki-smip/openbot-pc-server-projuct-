"""
Autopilot v3 — Classification-based inference.
Predicts discrete action (stop/forward/left/right/etc) and maps to motor command.
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
    AUTOPILOT_FPS, DEFAULT_DRIVE_SPEED
)
from model import ActionNet, action_to_command, NUM_ACTIONS, ACTION_TABLE
from trainer import crop_and_resize


# Action names for display
ACTION_NAMES = [
    "STOP", "FWD", "BWD", "LEFT", "RIGHT",
    "FWD+L", "FWD+R", "BWD+L", "BWD+R"
]


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
            acc_str = f"{100*val_acc:.1f}%" if isinstance(val_acc, float) else val_acc
            print(f"[Autopilot] Model loaded: {model_name} (accuracy: {acc_str})")

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

        self.car.send_command(0, 0)
        self._last_prediction = [0.0, 0.0]
        self._last_action = "—"

        result = {"status": "stopped", "model": self._model_name}
        self._model_name = ""
        return result

    def _inference_loop(self):
        interval = 1.0 / AUTOPILOT_FPS
        fps_timer = time.time()
        fps_count = 0

        # Smoothing: require N consecutive same predictions to change action
        prev_action = 0
        action_votes = []
        vote_window = 3  # Majority vote over last 3 frames

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

            # Inference
            with torch.no_grad():
                logits = self._model(img_tensor)
                probs = F.softmax(logits, dim=1)
                predicted_action = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_action].item()

            # Majority vote smoothing (prevents flickering)
            action_votes.append(predicted_action)
            if len(action_votes) > vote_window:
                action_votes.pop(0)

            # Use most common action in the vote window
            from collections import Counter
            vote_counts = Counter(action_votes)
            smoothed_action = vote_counts.most_common(1)[0][0]

            # Convert to motor command at current speed
            left, right = action_to_command(smoothed_action, self._speed)

            # Update state
            action_name = ACTION_NAMES[smoothed_action] if smoothed_action < len(ACTION_NAMES) else f"A{smoothed_action}"
            self._last_action = f"{action_name} ({100*confidence:.0f}%)"
            self._last_prediction = [left / 100.0, right / 100.0]

            # Send to car
            self.car.send_command(left, right)

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
        self.car.send_command(0, 0)
