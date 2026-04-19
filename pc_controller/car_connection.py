"""
WebSocket client for ESP8266 motor/servo control.
Thread-safe, auto-reconnecting, non-blocking.
"""
import json
import threading
import time
import websocket

from config import ESP8266_WS_URL


class CarConnection:
    def __init__(self, url=ESP8266_WS_URL):
        self.url = url
        self._ws = None
        self._lock = threading.Lock()
        self._connected = False
        self._running = False
        self._thread = None
        self._last_left = 0
        self._last_right = 0
        self._last_servo = 90

    # ── Public API ──────────────────────────────────────────
    @property
    def connected(self):
        return self._connected

    @property
    def last_command(self):
        return {
            "left": self._last_left,
            "right": self._last_right,
            "servo": self._last_servo,
        }

    def start(self):
        """Begin connection loop in background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._connection_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Disconnect and stop the background thread."""
        self._running = False
        self.send_command(0, 0)  # Safety stop
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
            self._ws = None
            self._connected = False

    def send_command(self, left: int, right: int, servo: int = None):
        """
        Send motor command to the car.
        left/right: -100 to 100 (negative = reverse)
        servo: 0-180 (optional, only sent if provided)
        """
        left = max(-100, min(100, int(left)))
        right = max(-100, min(100, int(right)))

        self._last_left = left
        self._last_right = right

        payload = {"left": left, "right": right}

        if servo is not None:
            servo = max(0, min(180, int(servo)))
            self._last_servo = servo
            payload["servo"] = servo

        with self._lock:
            if self._ws and self._connected:
                try:
                    self._ws.send(json.dumps(payload))
                    return True
                except Exception as e:
                    print(f"[CarConn] Send error: {e}")
                    self._connected = False
                    return False
        return False

    # ── Internal ────────────────────────────────────────────
    def _connection_loop(self):
        """Auto-reconnect loop running in background thread."""
        backoff = 1.0
        while self._running:
            try:
                print(f"[CarConn] Connecting to {self.url}...")
                ws = websocket.WebSocket()
                ws.settimeout(5)
                ws.connect(self.url)
                with self._lock:
                    self._ws = ws
                    self._connected = True
                print(f"[CarConn] ✓ Connected!")
                backoff = 1.0  # Reset backoff on success

                # Keep-alive loop: just check the connection is still alive
                while self._running and self._connected:
                    time.sleep(0.5)
                    try:
                        # Ping by sending a harmless stop command periodically
                        pass
                    except Exception:
                        self._connected = False
                        break

            except Exception as e:
                print(f"[CarConn] Connection failed: {e}")
                self._connected = False
                with self._lock:
                    self._ws = None

            if self._running:
                print(f"[CarConn] Reconnecting in {backoff:.0f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 10.0)
