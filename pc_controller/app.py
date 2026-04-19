"""
OpenBot PC Controller — Flask Server
Main entry point. Serves dashboard and API endpoints.
"""
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, Response

from config import DEFAULT_DRIVE_SPEED
from car_connection import CarConnection
from camera_stream import CameraStream
from data_collector import DataCollector
from trainer import Trainer
from autopilot import Autopilot

# ─── Initialize Components ────────────────────────────────────
app = Flask(__name__)

car = CarConnection()
camera = CameraStream()
collector = DataCollector(camera, car)
trainer = Trainer()
pilot = Autopilot(camera, car)

# Start hardware connections
car.start()
camera.start()


# ─── Dashboard Page ───────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ─── Video Feed ───────────────────────────────────────────────
@app.route("/api/video_feed")
def video_feed():
    """Proxy MJPEG stream to the browser."""
    def generate():
        while True:
            frame = camera.get_frame_jpeg()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.033)  # ~30 FPS max

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ─── Manual Control ──────────────────────────────────────────
@app.route("/api/control", methods=["POST"])
def control():
    """Send manual motor/servo command."""
    data = request.get_json()
    left = data.get("left", 0)
    right = data.get("right", 0)
    servo = data.get("servo", None)

    success = car.send_command(left, right, servo)
    return jsonify({"ok": success, "left": left, "right": right})


# ─── Recording ────────────────────────────────────────────────
@app.route("/api/recording/start", methods=["POST"])
def recording_start():
    data = request.get_json() or {}
    session_name = data.get("session_name", None)
    result = collector.start_recording(session_name)
    return jsonify(result)


@app.route("/api/recording/stop", methods=["POST"])
def recording_stop():
    result = collector.stop_recording()
    return jsonify(result)


@app.route("/api/recording/status")
def recording_status():
    return jsonify({
        "recording": collector.is_recording,
        "session": collector.session_name,
        "frames": collector.frame_count,
    })


# ─── Datasets ─────────────────────────────────────────────────
@app.route("/api/datasets")
def datasets_list():
    return jsonify(collector.list_datasets())


# ─── Training ─────────────────────────────────────────────────
@app.route("/api/train", methods=["POST"])
def train_start():
    data = request.get_json()
    sessions = data.get("sessions", [])
    epochs = data.get("epochs", 50)
    batch_size = data.get("batch_size", 32)
    lr = data.get("learning_rate", 0.0001)
    model_name = data.get("model_name", "autopilot")

    if not sessions:
        return jsonify({"error": "No sessions selected"}), 400

    result = trainer.start_training(sessions, epochs, batch_size, lr, model_name)
    return jsonify(result)


@app.route("/api/train/status")
def train_status():
    """Server-Sent Events stream for training progress."""
    def generate():
        while True:
            progress = trainer.progress
            yield f"data: {json.dumps(progress)}\n\n"
            if progress["status"] in ("completed", "error", "idle"):
                if progress["status"] != "idle":
                    time.sleep(1)
                break
            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/train/progress")
def train_progress():
    """Simple GET endpoint for training progress."""
    return jsonify(trainer.progress)


# ─── Models ────────────────────────────────────────────────────
@app.route("/api/models")
def models_list():
    return jsonify(trainer.list_models())


# ─── Autopilot ─────────────────────────────────────────────────
@app.route("/api/autopilot/start", methods=["POST"])
def autopilot_start():
    data = request.get_json()
    model_name = data.get("model_name", "autopilot")
    result = pilot.start(model_name)
    if "error" in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/autopilot/stop", methods=["POST"])
def autopilot_stop():
    result = pilot.stop()
    return jsonify(result)


@app.route("/api/autopilot/status")
def autopilot_status():
    return jsonify({
        "running": pilot.is_running,
        "model": pilot.model_name,
        "fps": round(pilot.inference_fps, 1),
        "prediction": pilot.last_prediction,
    })


# ─── System Status ─────────────────────────────────────────────
@app.route("/api/status")
def system_status():
    return jsonify({
        "car_connected": car.connected,
        "camera_connected": camera.connected,
        "camera_fps": round(camera.fps, 1),
        "recording": collector.is_recording,
        "recording_frames": collector.frame_count,
        "training": trainer.is_training,
        "autopilot": pilot.is_running,
        "autopilot_model": pilot.model_name,
        "autopilot_fps": round(pilot.inference_fps, 1),
    })


# ─── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  OpenBot PC Controller")
    print("  Dashboard: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
