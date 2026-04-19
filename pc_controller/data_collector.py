"""
Synchronized data collector — pairs camera frames with motor commands.
Saves images + CSV for training.
"""
import csv
import os
import threading
import time
from datetime import datetime

from config import DATASETS_DIR, RECORDING_FPS


class DataCollector:
    def __init__(self, camera_stream, car_connection):
        self.camera = camera_stream
        self.car = car_connection
        self._recording = False
        self._thread = None
        self._session_dir = None
        self._images_dir = None
        self._csv_path = None
        self._csv_file = None
        self._csv_writer = None
        self._frame_count = 0
        self._session_name = ""
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────
    @property
    def is_recording(self):
        return self._recording

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def session_name(self):
        return self._session_name

    def start_recording(self, session_name: str = None):
        """Start a new recording session."""
        if self._recording:
            return {"error": "Already recording"}

        # Create session directory
        if not session_name:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self._session_name = session_name

        self._session_dir = os.path.join(DATASETS_DIR, session_name)
        self._images_dir = os.path.join(self._session_dir, "images")
        os.makedirs(self._images_dir, exist_ok=True)

        # Open CSV file
        self._csv_path = os.path.join(self._session_dir, "data.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(["timestamp", "image_path", "left", "right", "servo"])

        self._frame_count = 0
        self._recording = True

        # Start capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        print(f"[Recorder] ✓ Recording started: {session_name}")
        return {"status": "recording", "session": session_name}

    def stop_recording(self):
        """Stop and finalize the current recording session."""
        if not self._recording:
            return {"error": "Not recording"}

        self._recording = False
        if self._thread:
            self._thread.join(timeout=2)

        # Close CSV
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

        result = {
            "status": "stopped",
            "session": self._session_name,
            "frames": self._frame_count,
            "path": self._session_dir,
        }
        print(f"[Recorder] ■ Recording stopped: {self._frame_count} frames saved")
        return result

    def list_datasets(self):
        """List all recorded datasets with metadata."""
        datasets = []
        if not os.path.exists(DATASETS_DIR):
            return datasets

        for name in sorted(os.listdir(DATASETS_DIR)):
            session_path = os.path.join(DATASETS_DIR, name)
            if not os.path.isdir(session_path):
                continue

            csv_path = os.path.join(session_path, "data.csv")
            images_dir = os.path.join(session_path, "images")

            frame_count = 0
            if os.path.isfile(csv_path):
                with open(csv_path, "r") as f:
                    frame_count = max(0, sum(1 for _ in f) - 1)  # Subtract header

            img_count = 0
            if os.path.isdir(images_dir):
                img_count = len([f for f in os.listdir(images_dir) if f.endswith(".jpg")])

            datasets.append({
                "name": name,
                "frames": frame_count,
                "images": img_count,
                "path": session_path,
            })

        return datasets

    # ── Internal ────────────────────────────────────────────
    def _capture_loop(self):
        """Background loop capturing frames at fixed rate."""
        interval = 1.0 / RECORDING_FPS

        while self._recording:
            start = time.time()

            frame_jpeg = self.camera.get_frame_jpeg()
            if frame_jpeg is None:
                time.sleep(0.05)
                continue

            cmd = self.car.last_command
            timestamp = time.time()

            # Save image
            img_filename = f"{self._frame_count:06d}.jpg"
            img_path = os.path.join(self._images_dir, img_filename)
            with open(img_path, "wb") as f:
                f.write(frame_jpeg)

            # Write CSV row
            with self._lock:
                if self._csv_writer:
                    self._csv_writer.writerow([
                        f"{timestamp:.6f}",
                        f"images/{img_filename}",
                        cmd["left"],
                        cmd["right"],
                        cmd.get("servo", 90),
                    ])
                    self._csv_file.flush()

            self._frame_count += 1

            # Maintain target FPS
            elapsed = time.time() - start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
