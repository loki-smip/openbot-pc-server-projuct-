# OpenBot PC Server Project

An end-to-end autonomous driving system for small RC cars. A PC runs the brain — collecting driving data, training a neural network, and running real-time inference — while an ESP8266 handles motor control and an ESP32-CAM streams live video over Wi-Fi.

Everything runs on your local network. No cloud. No subscriptions. Just hardware, Python, and a browser.

---

## Table of Contents

- [What This Project Does](#what-this-project-does)
- [System Architecture](#system-architecture)
- [Hardware You Need](#hardware-you-need)
- [Wiring Diagram](#wiring-diagram)
- [Uploading Firmware to the ESP8266](#uploading-firmware-to-the-esp8266)
- [Setting Up the ESP32-CAM](#setting-up-the-esp32-cam)
- [Installing the PC Server](#installing-the-pc-server)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [Step 1: Power On the Hardware](#step-1-power-on-the-hardware)
  - [Step 2: Start the Server](#step-2-start-the-server)
  - [Step 3: Collect Training Data](#step-3-collect-training-data)
  - [Step 4: Diagnose Your Data](#step-4-diagnose-your-data)
  - [Step 5: Train the Model](#step-5-train-the-model)
  - [Step 6: Run Autonomous Mode](#step-6-run-autonomous-mode)
- [Model Architecture](#model-architecture)
- [Project File Structure](#project-file-structure)
- [API Reference](#api-reference)
- [Data Format](#data-format)
- [Tips for Good Training Data](#tips-for-good-training-data)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What This Project Does

You drive the car manually through a web dashboard while the system records every camera frame paired with your motor commands. That recorded dataset trains a convolutional neural network to map raw pixels to driving actions. Once trained, the model takes over — reading the camera feed in real time and steering the car on its own.

The full pipeline:

```
Drive manually  →  Record frames + commands  →  Train CNN  →  Car drives itself
```

This is imitation learning (behavioral cloning). The car learns to copy your driving behavior.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Your PC                              │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │  Flask Server (app.py) — http://localhost:5000       │  │
│   │                                                      │  │
│   │  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │  │
│   │  │ Camera   │  │  Data    │  │     Trainer       │  │  │
│   │  │ Stream   │  │Collector │  │  (PyTorch CNN)    │  │  │
│   │  │ Reader   │  │          │  │                   │  │  │
│   │  └────┬─────┘  └──────────┘  └───────────────────┘  │  │
│   │       │                                              │  │
│   │  ┌────┴─────┐              ┌───────────────────┐    │  │
│   │  │Autopilot │──────────────│  Car Connection   │    │  │
│   │  │(Inference)│             │  (WebSocket)      │    │  │
│   │  └──────────┘              └─────────┬─────────┘    │  │
│   └──────────────────────────────────────┼──────────────┘  │
└──────────────────────────────────────────┼──────────────────┘
                                           │
                         Wi-Fi (LAN)       │
                    ┌──────────────┬────────┘
                    │              │
           ┌────────┴───┐  ┌──────┴──────┐
           │ ESP32-CAM  │  │   ESP8266   │
           │  (Camera)  │  │  (Motors)   │
           │            │  │  + Servo    │
           │ MJPEG @    │  │ WebSocket  │
           │ 800x600    │  │  Port 81   │
           └────────────┘  └─────────────┘
```

**Two microcontrollers, two jobs:**
- ESP32-CAM streams 800×600 MJPEG video over HTTP
- ESP8266 receives motor commands over WebSocket and drives the L298N motor driver

The PC server bridges them — it reads the camera, serves a web dashboard, handles data recording, runs training, and performs real-time inference.

---

## Hardware You Need

| Component | Purpose | Approx. Cost |
|---|---|---|
| ESP8266 (NodeMCU or Wemos D1 Mini) | Motor controller + servo | ~$3 |
| ESP32-CAM (with OV2640) | Camera streaming | ~$5 |
| L298N Motor Driver | Dual H-bridge for 2 DC motors | ~$2 |
| 2× DC Gear Motors | Left and right wheels | ~$3 |
| SG90 Servo (optional) | Camera pan | ~$1 |
| 2S LiPo Battery (7.4V) or 4×AA pack | Power supply | ~$5 |
| Robot car chassis | Frame, wheels, caster | ~$5 |
| Jumper wires | Connections | ~$2 |
| USB cable (Micro-USB) | Flashing firmware | — |

**Total: roughly $25–30 USD**

---

## Wiring Diagram

### ESP8266 to L298N Motor Driver

```
ESP8266 Pin   →   L298N Pin      →   Purpose
─────────────────────────────────────────────
D5 (GPIO14)   →   IN1              Left motor forward
D6 (GPIO12)   →   IN2              Left motor reverse
D7 (GPIO13)   →   IN3              Right motor forward
D8 (GPIO15)   →   IN4              Right motor reverse
GND           →   GND              Common ground
```

### ESP8266 Servo Connection

```
ESP8266 Pin   →   Servo Wire     →   Purpose
─────────────────────────────────────────────
D1 (GPIO5)    →   Signal (orange)    PWM control
GND           →   Ground (brown)     Common ground
VIN (5V)      →   Power (red)        Power supply
```

### Power

```
Battery (+)   →   L298N 12V input    Motor power
Battery (-)   →   L298N GND          Common ground
L298N 5V out  →   ESP8266 VIN        Logic power (regulated)
```

**The ESP32-CAM runs independently** — power it via its own USB cable or a separate 5V regulator. It connects to the same Wi-Fi network but has no physical wires to the ESP8266.

---

## Uploading Firmware to the ESP8266

### Requirements

- [Arduino IDE](https://www.arduino.cc/en/software) (version 2.x recommended)
- ESP8266 board package installed

### Step-by-step

1. **Install the ESP8266 board package:**
   - Open Arduino IDE → File → Preferences
   - In "Additional Board Manager URLs," add:
     ```
     http://arduino.esp8266.com/stable/package_esp8266com_index.json
     ```
   - Go to Tools → Board → Board Manager → search "ESP8266" → Install

2. **Install required libraries:**
   - Go to Sketch → Include Library → Manage Libraries
   - Search and install each of these:
     - `WebSockets` by Markus Sattler (Links2004)
     - `ArduinoJson` by Benoit Blanchon (version 6.x)

3. **Configure your Wi-Fi:**
   - Open `firmware_esp8266/esp8266_car/esp8266_car.ino`
   - Find lines 24-25 and fill in your Wi-Fi name and password:
     ```cpp
     const char* ssid = "YOUR_WIFI_NAME";
     const char* password = "YOUR_WIFI_PASSWORD";
     ```

4. **Select the board:**
   - Tools → Board → ESP8266 Boards → "NodeMCU 1.0 (ESP-12E Module)"
   - Tools → Port → select the COM port your ESP8266 is on

5. **Upload:**
   - Click the Upload button (→ arrow)
   - Wait for "Done uploading"
   - Open Serial Monitor (115200 baud) to see the assigned IP address

6. **Note the IP address** — you'll see something like `ESP8266 IP address: 192.168.1.12` in the serial monitor. You need this for the PC server config.

---

## Setting Up the ESP32-CAM

The ESP32-CAM runs the standard **CameraWebServer** example sketch that ships with the ESP32 Arduino core.

1. **Install ESP32 board package** in Arduino IDE:
   - Add this URL to Board Manager URLs:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Board Manager → search "ESP32" → Install

2. **Open the example sketch:**
   - File → Examples → ESP32 → Camera → CameraWebServer

3. **Configure the sketch:**
   - Select your camera model (usually `#define CAMERA_MODEL_AI_THINKER`)
   - Set your Wi-Fi credentials
   - Set resolution to SVGA (800×600) in the code if not default

4. **Upload** using an FTDI adapter (the ESP32-CAM has no built-in USB-to-serial):
   - Connect GPIO0 to GND for flash mode
   - Upload the sketch
   - Disconnect GPIO0 from GND and reset

5. **Verify:**
   - Open Serial Monitor to find the IP address
   - Navigate to `http://<ESP32_IP>:81/stream` — you should see live video

---

## Installing the PC Server

### Prerequisites

- Python 3.10 or newer
- pip (comes with Python)
- A computer on the same Wi-Fi network as both microcontrollers

### Step-by-step

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/openbot-pc-server-project.git
   cd openbot-pc-server-project
   ```

2. **Install Python dependencies:**
   ```bash
   cd pc_controller
   pip install -r requirements.txt
   ```

   This installs:
   - `flask` — web server
   - `torch` + `torchvision` — neural network training and inference
   - `opencv-python` — image processing
   - `websocket-client` — communication with the ESP8266
   - `requests` — reading the camera stream
   - `numpy` — numerical operations
   - `Pillow` — image utilities

3. **Set your device IP addresses:**
   - Open `pc_controller/config.py`
   - Update these two lines with the IPs from your Serial Monitors:
     ```python
     ESP8266_WS_URL = "ws://192.168.1.12:81"          # Your ESP8266 IP
     ESP32_CAM_STREAM_URL = "http://192.168.1.13:81/stream"  # Your ESP32-CAM IP
     ```

4. **Verify everything works:**
   ```bash
   python app.py
   ```
   Open `http://localhost:5000` in your browser. You should see the dashboard with a live camera feed and motor controls.

---

## Step-by-Step Usage Guide

### Step 1: Power On the Hardware

1. Power up the ESP8266 car — wait 5 seconds for Wi-Fi connection
2. Power up the ESP32-CAM — wait 5 seconds for Wi-Fi connection
3. Verify both devices have joined your Wi-Fi network (check your router or Serial Monitor)

### Step 2: Start the Server

```bash
cd pc_controller
python app.py
```

Open `http://localhost:5000` in your browser. The dashboard shows:

- **Live camera feed** from the ESP32-CAM
- **Manual controls** — WASD keys or on-screen D-pad
- **Recording panel** — start/stop data collection
- **Training panel** — configure and launch training
- **Autopilot panel** — load a model and go autonomous

Check the status bar at the bottom — both "Car" and "Camera" indicators should turn green.

### Step 3: Collect Training Data

This is the most important step. The quality of your autonomous driving depends entirely on the quality of your training data.

1. Click **"Start Recording"** in the Recording panel (or give the session a name first)
2. Drive the car using **WASD keys**:
   - `W` — Forward
   - `S` — Backward
   - `A` — Turn left
   - `D` — Turn right
   - `W+A` — Forward-left arc
   - `W+D` — Forward-right arc
   - Release all — Stop
3. Drive around the space you want the car to navigate autonomously
4. Click **"Stop Recording"** when done

**How much data?** Aim for at least 3,000–5,000 frames. At 10 FPS, that is roughly 5–8 minutes of active driving. Record multiple sessions with different paths.

### Step 4: Diagnose Your Data

Before training, check that your data is usable:

```bash
python diagnose_data.py
```

This prints a breakdown of your command distribution, value ranges, and flags problems like:
- Too many idle frames (car sitting still)
- Not enough turns (model won't learn to steer)
- Low command variety

It also saves a `diagnostic_grid.jpg` showing what the model actually "sees" after cropping — open this image to visually confirm the camera perspective makes sense.

### Step 5: Train the Model

**Option A — From the web dashboard:**
1. Go to the Training panel
2. Select which recorded sessions to train on
3. Set epochs (start with 50–100), batch size (32), and learning rate (0.001)
4. Click "Start Training"
5. Watch the live progress bar — training stops early if accuracy stops improving

**Option B — From the command line:**
```bash
# Training happens through the web API, but you can also monitor in the terminal
# The server prints training progress to stdout
```

The best model is saved automatically to `pc_controller/trained_models/autopilot.pth`.

**What to expect:**
- Training loss should decrease over the first 20–50 epochs
- Validation accuracy should reach 70–90% depending on data quality
- Early stopping kicks in after 30 epochs without improvement
- A well-trained model file is around 1–2 MB

### Step 6: Run Autonomous Mode

1. Go to the Autopilot panel in the dashboard
2. Select the trained model
3. Click **"Start Autopilot"**
4. The car starts driving itself using the camera feed

The autopilot runs at 10 FPS, uses majority-vote smoothing over 3 frames to prevent flickering, and maps predicted actions to motor commands at the configured speed.

To stop: click **"Stop Autopilot"** or close the browser tab (the car will stop on disconnect).

---

## Model Architecture

The model is called **ActionNet** — a classification CNN that predicts one of 9 discrete driving actions from a single camera frame.

### Why Classification, Not Regression?

Keyboard-driven data only produces a handful of unique motor command combinations (forward, left, right, etc.). Trying to regress continuous steering values from this kind of data produces washed-out averages near zero. Classification with cross-entropy loss works much better for discrete control.

### The 9 Actions

| Index | Action | Left Motor | Right Motor |
|---|---|---|---|
| 0 | STOP | 0 | 0 |
| 1 | FORWARD | +70 | +70 |
| 2 | BACKWARD | -70 | -70 |
| 3 | TURN LEFT | -49 | +49 |
| 4 | TURN RIGHT | +49 | -49 |
| 5 | FORWARD + LEFT | +21 | +70 |
| 6 | FORWARD + RIGHT | +70 | +21 |
| 7 | BACKWARD + LEFT | -21 | -70 |
| 8 | BACKWARD + RIGHT | -70 | -21 |

Motor values scale proportionally with the speed setting (these are shown at speed=70).

### Network Layers

```
Input: 66 × 200 × 3 RGB image (cropped from 800×600, top 40% removed)

Convolutional Feature Extractor:
├── Conv2d(3→24, 5×5, stride=2)  + BatchNorm + ELU
├── Conv2d(24→36, 5×5, stride=2) + BatchNorm + ELU
├── Conv2d(36→48, 5×5, stride=2) + BatchNorm + ELU
├── Conv2d(48→64, 3×3, stride=1) + BatchNorm + ELU
└── Conv2d(64→64, 3×3, stride=1) + BatchNorm + ELU

Spatial Dropout (15%)

Classification Head:
├── Flatten
├── Dropout (35%)
├── Linear(64×1×18 → 64) + ELU
├── Dropout (35%)
└── Linear(64 → 9)    ← raw logits, one per action

Output: 9-class probability distribution (via softmax at inference)
```

**Parameter count:** ~145,000 trainable parameters

The architecture is based on NVIDIA's PilotNet with modifications:
- BatchNorm after every convolution for training stability
- ELU activations instead of ReLU
- Classification head instead of regression
- Kaiming initialization for all weights
- Aggressive dropout and spatial dropout to fight overfitting

### Preprocessing Pipeline

```
Raw 800×600 BGR frame from camera
        │
        ▼
Crop top 40% (remove ceiling/sky)
        │
        ▼
Convert BGR → RGB
        │
        ▼
Resize to 200×66 (INTER_AREA)
        │
        ▼
ToTensor() → float [0, 1]
        │
        ▼
Shape: [batch, 3, 66, 200]
```

The same `crop_and_resize()` function is used in both training and inference to guarantee consistency.

### Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW (weight_decay=5e-3) |
| Scheduler | OneCycleLR (cosine anneal) |
| Loss | CrossEntropyLoss (label_smoothing=0.2) |
| Batch Size | 32 |
| Learning Rate | 0.001 (peak) |
| Early Stopping | 30 epochs patience (by val accuracy) |
| Gradient Clipping | max_norm=1.0 |
| Class Balancing | WeightedRandomSampler (inverse frequency) |

### Data Augmentation

Applied during training to improve generalization:

- **Horizontal flip** — with mirrored action labels (left↔right)
- **Random shadow** — simulates lighting variation
- **Random brightness** — HSV channel scaling
- **Gaussian blur** — simulates motion blur and camera noise
- **Random translation** — shifts the image 0–10% in any direction
- **Random erasing (cutout)** — masks a random rectangle to force the model to use the full image

---

## Project File Structure

```
openbot-pc-server-project/
│
├── README.md                  ← You are here
├── huggingface.md             ← Hugging Face model card
│
├── firmware_esp8266/
│   └── esp8266_car/
│       └── esp8266_car.ino    ← Arduino firmware for motor control
│
└── pc_controller/
    ├── app.py                 ← Flask server — main entry point
    ├── config.py              ← All settings (IPs, model params, paths)
    ├── model.py               ← ActionNet CNN architecture + action tables
    ├── trainer.py             ← Training pipeline (PyTorch)
    ├── autopilot.py           ← Real-time autonomous inference loop
    ├── camera_stream.py       ← MJPEG stream reader (background thread)
    ├── car_connection.py      ← WebSocket client for ESP8266
    ├── data_collector.py      ← Synchronized frame + command recorder
    ├── diagnose_data.py       ← Dataset analysis and diagnostic tool
    ├── requirements.txt       ← Python dependencies
    │
    ├── templates/
    │   └── index.html         ← Web dashboard (single-page app)
    │
    ├── static/
    │   ├── css/style.css      ← Dashboard styling
    │   └── js/main.js         ← Dashboard logic (controls, API calls)
    │
    ├── datasets/              ← Recorded driving sessions (created at runtime)
    │   └── session_YYYYMMDD_HHMMSS/
    │       ├── images/
    │       │   ├── 000000.jpg
    │       │   ├── 000001.jpg
    │       │   └── ...
    │       └── data.csv
    │
    └── trained_models/        ← Saved model weights (created at runtime)
        └── autopilot.pth
```

---

## API Reference

All endpoints are served at `http://localhost:5000`.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web dashboard |
| GET | `/api/video_feed` | MJPEG video stream (proxy from ESP32-CAM) |
| POST | `/api/control` | Send motor command `{"left": int, "right": int, "servo": int}` |
| POST | `/api/recording/start` | Begin recording `{"session_name": "optional_name"}` |
| POST | `/api/recording/stop` | Stop recording |
| GET | `/api/recording/status` | Current recording state |
| GET | `/api/datasets` | List all recorded datasets |
| POST | `/api/train` | Start training `{"sessions": [...], "epochs": 50, ...}` |
| GET | `/api/train/status` | SSE stream of training progress |
| GET | `/api/train/progress` | Polling endpoint for training status |
| GET | `/api/models` | List trained models |
| POST | `/api/autopilot/start` | Start autopilot `{"model_name": "autopilot"}` |
| POST | `/api/autopilot/stop` | Stop autopilot |
| GET | `/api/autopilot/status` | Autopilot state + FPS + last prediction |
| GET | `/api/status` | Full system status (car, camera, recording, training, autopilot) |

### WebSocket Protocol (ESP8266)

The ESP8266 listens on port 81 for JSON messages:

```json
{"left": 70, "right": 70}
{"left": -50, "right": 50, "servo": 90}
```

- `left` / `right`: motor speed from -100 (full reverse) to +100 (full forward)
- `servo`: camera pan angle, 0–180 degrees (optional)

---

## Data Format

Each recording session creates a folder under `pc_controller/datasets/` containing:

### data.csv

```csv
timestamp,image_path,left,right
1713500000.123456,images/000000.jpg,70,70
1713500000.223456,images/000001.jpg,70,21
1713500000.323456,images/000002.jpg,0,0
```

| Column | Type | Description |
|---|---|---|
| timestamp | float | Unix timestamp (seconds) |
| image_path | string | Relative path to JPEG frame |
| left | int | Left motor command (-100 to 100) |
| right | int | Right motor command (-100 to 100) |

### Images

- Format: JPEG
- Resolution: 800×600 (raw from ESP32-CAM)
- Naming: zero-padded sequential (`000000.jpg`, `000001.jpg`, ...)
- Rate: 10 frames per second

---

## Tips for Good Training Data

1. **Drive actively.** Keep the car moving. Long idle periods dilute the dataset with useless "stop" frames.

2. **Include lots of turns.** Drive figure-8s, weave between obstacles, make sharp corners. Forward-only data produces a model that only goes straight.

3. **Vary your speed.** Mix fast straight sections with slow careful turns.

4. **Cover the full area.** Drive every path the car might take during autonomous mode. The model can only reproduce what it has seen.

5. **Record multiple sessions.** Three 3-minute sessions from different starting positions are better than one 9-minute session.

6. **Check your data.** Run `python diagnose_data.py` after every recording session. Fix problems before training.

7. **Consistent lighting.** The camera sees color and brightness. If you train in daylight and run at night, expect poor results.

---

## Troubleshooting

### "Car connected" stays red

- Check that the ESP8266 is powered on and connected to Wi-Fi
- Verify the IP address in `config.py` matches the Serial Monitor output
- Make sure your PC is on the same Wi-Fi network
- Try pinging the ESP8266: `ping 192.168.1.12`

### "Camera connected" stays red

- Check that the ESP32-CAM is powered on and connected to Wi-Fi
- Test the stream URL directly in your browser: `http://192.168.1.13:81/stream`
- Verify the IP in `config.py`
- The ESP32-CAM sometimes needs a reset after power-on

### Training accuracy stays below 50%

- Run `python diagnose_data.py` — look for class imbalance
- You likely have too many STOP frames. Drive more actively during recording.
- Collect more data. 500 frames is not enough; aim for 3,000+.
- Reduce the learning rate to 0.0005 or 0.0001

### Car just goes straight in autopilot

- Your training data probably lacks turns. Record a new session with more turning.
- Check the diagnostic grid image — if the cropped view doesn't show the ground/obstacles, the model has nothing useful to learn from.

### WebSocket disconnect errors

- The ESP8266 drops the connection if it doesn't receive data for ~30 seconds. The server sends keep-alive commands automatically, but if your PC goes to sleep the connection will die.
- The server auto-reconnects. Wait a few seconds.

### "No module named torch"

```bash
pip install torch torchvision
```

On Windows without a GPU, use:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## License

MIT License — use this project however you want. Build cool robots.
