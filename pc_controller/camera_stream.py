"""
MJPEG stream reader for ESP32-CAM.
Runs in background thread, provides latest frame on demand.
"""
import threading
import time
import cv2
import numpy as np
import requests

from config import ESP32_CAM_STREAM_URL


class CameraStream:
    def __init__(self, url=ESP32_CAM_STREAM_URL):
        self.url = url
        self._frame_jpeg = None  # Latest frame as JPEG bytes
        self._frame_cv2 = None   # Latest frame as numpy (BGR)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False
        self._fps = 0.0
        self._frame_count = 0

    # ── Public API ──────────────────────────────────────────
    @property
    def connected(self):
        return self._connected

    @property
    def fps(self):
        return self._fps

    @property
    def frame_count(self):
        return self._frame_count

    def get_frame_jpeg(self) -> bytes:
        """Get latest frame as JPEG bytes (for streaming to browser)."""
        with self._lock:
            return self._frame_jpeg

    def get_frame_cv2(self) -> np.ndarray:
        """Get latest frame as OpenCV BGR numpy array (for AI processing)."""
        with self._lock:
            return self._frame_cv2.copy() if self._frame_cv2 is not None else None

    def start(self):
        """Begin reading stream in background."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop reading stream."""
        self._running = False
        self._connected = False

    # ── Internal ────────────────────────────────────────────
    def _read_loop(self):
        """Continuously read MJPEG stream with auto-reconnect."""
        backoff = 1.0
        while self._running:
            try:
                print(f"[Camera] Connecting to {self.url}...")
                response = requests.get(self.url, stream=True, timeout=10)
                response.raise_for_status()
                print(f"[Camera] ✓ Stream connected!")
                self._connected = True
                backoff = 1.0

                # Read MJPEG stream
                buffer = b""
                fps_timer = time.time()
                fps_count = 0

                for chunk in response.iter_content(chunk_size=4096):
                    if not self._running:
                        break
                    buffer += chunk

                    # Look for JPEG frame boundaries
                    while True:
                        start = buffer.find(b"\xff\xd8")  # JPEG SOI
                        end = buffer.find(b"\xff\xd9")    # JPEG EOI

                        if start == -1 or end == -1 or end <= start:
                            break

                        # Extract complete JPEG frame
                        jpeg_data = buffer[start:end + 2]
                        buffer = buffer[end + 2:]

                        # Decode to OpenCV
                        np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            with self._lock:
                                self._frame_jpeg = jpeg_data
                                self._frame_cv2 = frame
                                self._frame_count += 1

                            fps_count += 1
                            elapsed = time.time() - fps_timer
                            if elapsed >= 1.0:
                                self._fps = fps_count / elapsed
                                fps_count = 0
                                fps_timer = time.time()

            except Exception as e:
                print(f"[Camera] Stream error: {e}")
                self._connected = False

            if self._running:
                print(f"[Camera] Reconnecting in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
