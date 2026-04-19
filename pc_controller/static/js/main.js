/* ═══════════════════════════════════════════════════════════
   OpenBot PC Controller — Frontend Logic
   Keyboard control, API calls, real-time status updates
   ═══════════════════════════════════════════════════════════ */

// ─── State ────────────────────────────────────────────────
const state = {
    keys: { w: false, a: false, s: false, d: false },
    speed: 70,
    currentLeft: 0,
    currentRight: 0,
    currentServo: 90,
    servoStep: 22,  // Match the 9-position bucket spacing (~22° steps)
    controlInterval: null,
    statusInterval: null,
    recordingPollInterval: null,
    trainingPollInterval: null,
    autopilotPollInterval: null,
};

// ─── Keyboard Control ─────────────────────────────────────
document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();
    if (['w', 'a', 's', 'd', ' ', 'q', 'e'].includes(key)) {
        e.preventDefault();
        if (key === ' ') {
            // Emergency stop
            state.keys.w = false;
            state.keys.a = false;
            state.keys.s = false;
            state.keys.d = false;
            // Reset servo to center
            state.currentServo = 90;
            updateServoDisplay();
            sendServo(90);
        } else if (key === 'q') {
            // Tilt up (decrease angle — 0° is fully up)
            state.currentServo = Math.max(0, state.currentServo - state.servoStep);
            updateServoDisplay();
            sendServo(state.currentServo);
        } else if (key === 'e') {
            // Tilt down (increase angle — 180° is fully down)
            state.currentServo = Math.min(180, state.currentServo + state.servoStep);
            updateServoDisplay();
            sendServo(state.currentServo);
        } else {
            state.keys[key] = true;
        }
        highlightKey(key, true);
        updateMotors();
    }
});

document.addEventListener('keyup', (e) => {
    const key = e.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key)) {
        e.preventDefault();
        state.keys[key] = false;
        highlightKey(key, false);
        updateMotors();
    }
    if (['q', 'e', ' '].includes(key)) {
        highlightKey(key, false);
    }
});

function highlightKey(key, active) {
    const kbds = document.querySelectorAll('kbd');
    const keyMap = { 'w': 'W', 'a': 'A', 's': 'S', 'd': 'D', ' ': 'Space', 'q': 'Q', 'e': 'E' };
    const label = keyMap[key];
    kbds.forEach(kbd => {
        if (kbd.textContent.trim() === label) {
            kbd.classList.toggle('active', active);
        }
    });
}

function updateServoDisplay() {
    const slider = document.getElementById('servo-slider');
    const value = document.getElementById('servo-value');
    if (slider && value) {
        slider.value = state.currentServo;
        value.textContent = state.currentServo + '°';
    }
}

function updateMotors() {
    const speed = state.speed;
    let left = 0;
    let right = 0;

    // Forward / Backward
    if (state.keys.w) {
        left += speed;
        right += speed;
    }
    if (state.keys.s) {
        left -= speed;
        right -= speed;
    }

    // Turning (differential drive)
    if (state.keys.a) {
        left -= speed * 0.7;
        right += speed * 0.7;
    }
    if (state.keys.d) {
        left += speed * 0.7;
        right -= speed * 0.7;
    }

    // Clamp
    left = Math.max(-100, Math.min(100, Math.round(left)));
    right = Math.max(-100, Math.min(100, Math.round(right)));

    // Only send if changed
    if (left !== state.currentLeft || right !== state.currentRight) {
        state.currentLeft = left;
        state.currentRight = right;
        sendControl(left, right);
    }

    // Update overlay bars
    updateBars(left, right);
}

function updateBars(left, right) {
    const barLeft = document.getElementById('bar-left');
    const barRight = document.getElementById('bar-right');
    const labelLeft = document.getElementById('label-left');
    const labelRight = document.getElementById('label-right');

    const leftPct = Math.abs(left);
    const rightPct = Math.abs(right);

    barLeft.style.height = leftPct + '%';
    barRight.style.height = rightPct + '%';

    barLeft.style.background = left >= 0
        ? 'linear-gradient(to top, #6366f1, #818cf8)'
        : 'linear-gradient(to top, #ef4444, #f87171)';
    barRight.style.background = right >= 0
        ? 'linear-gradient(to top, #6366f1, #818cf8)'
        : 'linear-gradient(to top, #ef4444, #f87171)';

    labelLeft.textContent = `L: ${left}`;
    labelRight.textContent = `R: ${right}`;
}

// ─── Speed Slider ─────────────────────────────────────────
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');

speedSlider.addEventListener('input', () => {
    state.speed = parseInt(speedSlider.value);
    speedValue.textContent = state.speed;
});

// ─── Servo Slider ─────────────────────────────────────────
const servoSlider = document.getElementById('servo-slider');
const servoValue = document.getElementById('servo-value');

servoSlider.addEventListener('input', () => {
    const angle = parseInt(servoSlider.value);
    state.currentServo = angle;
    servoValue.textContent = angle + '°';
    sendServo(angle);
});

// ─── API Calls ────────────────────────────────────────────
async function sendControl(left, right) {
    try {
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ left, right }),
        });
    } catch (err) {
        console.error('Control send failed:', err);
    }
}

async function sendServo(angle) {
    try {
        await fetch('/api/control', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ left: state.currentLeft, right: state.currentRight, servo: angle }),
        });
    } catch (err) {
        console.error('Servo send failed:', err);
    }
}

// ─── Recording ────────────────────────────────────────────
async function startRecording() {
    const nameInput = document.getElementById('session-name');
    const body = {};
    if (nameInput.value.trim()) {
        body.session_name = nameInput.value.trim();
    }

    try {
        const res = await fetch('/api/recording/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('btn-rec-start').disabled = true;
        document.getElementById('btn-rec-stop').disabled = false;
        document.getElementById('rec-status').textContent = '🔴 REC';
        document.getElementById('rec-status').style.color = '#ef4444';

        // Update mode indicator
        setMode('recording');

        // Poll recording status
        state.recordingPollInterval = setInterval(pollRecordingStatus, 500);
    } catch (err) {
        alert('Failed to start recording: ' + err.message);
    }
}

async function stopRecording() {
    try {
        const res = await fetch('/api/recording/stop', { method: 'POST' });
        const data = await res.json();

        document.getElementById('btn-rec-start').disabled = false;
        document.getElementById('btn-rec-stop').disabled = true;
        document.getElementById('rec-status').textContent = 'Idle';
        document.getElementById('rec-status').style.color = '';

        setMode('manual');

        if (state.recordingPollInterval) {
            clearInterval(state.recordingPollInterval);
            state.recordingPollInterval = null;
        }

        // Refresh datasets list
        loadDatasets();

        if (data.frames) {
            document.getElementById('rec-frames').textContent = data.frames;
        }
    } catch (err) {
        alert('Failed to stop recording: ' + err.message);
    }
}

async function pollRecordingStatus() {
    try {
        const res = await fetch('/api/recording/status');
        const data = await res.json();
        document.getElementById('rec-frames').textContent = data.frames;
    } catch (err) { /* ignore */ }
}

// ─── Training ─────────────────────────────────────────────
async function loadDatasets() {
    try {
        const res = await fetch('/api/datasets');
        const datasets = await res.json();
        const select = document.getElementById('train-datasets');
        select.innerHTML = '';

        datasets.forEach(ds => {
            const opt = document.createElement('option');
            opt.value = ds.name;
            opt.textContent = `${ds.name} (${ds.frames} frames)`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load datasets:', err);
    }
}

async function startTraining() {
    const select = document.getElementById('train-datasets');
    const selected = Array.from(select.selectedOptions).map(o => o.value);

    if (selected.length === 0) {
        alert('Please select at least one dataset!');
        return;
    }

    const epochs = parseInt(document.getElementById('train-epochs').value);
    const batchSize = parseInt(document.getElementById('train-batch').value);
    const lr = parseFloat(document.getElementById('train-lr').value);
    const modelName = document.getElementById('train-model-name').value.trim() || 'autopilot';

    try {
        const res = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sessions: selected,
                epochs,
                batch_size: batchSize,
                learning_rate: lr,
                model_name: modelName,
            }),
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Show progress section
        document.getElementById('train-progress-section').classList.remove('hidden');
        document.getElementById('btn-train').disabled = true;

        // Poll training progress
        state.trainingPollInterval = setInterval(pollTrainingProgress, 800);
    } catch (err) {
        alert('Failed to start training: ' + err.message);
    }
}

async function pollTrainingProgress() {
    try {
        const res = await fetch('/api/train/progress');
        const p = await res.json();

        const bar = document.getElementById('train-progress-bar');
        const text = document.getElementById('train-progress-text');
        const epochInfo = document.getElementById('train-epoch-info');
        const lossInfo = document.getElementById('train-loss-info');
        const message = document.getElementById('train-message');

        bar.style.width = p.progress_pct + '%';
        text.textContent = p.progress_pct + '%';
        epochInfo.textContent = `Epoch ${p.epoch}/${p.total_epochs}`;
        lossInfo.textContent = `Train: ${p.train_loss} | Val: ${p.val_loss}`;
        message.textContent = p.message;

        if (p.status === 'completed' || p.status === 'error') {
            clearInterval(state.trainingPollInterval);
            state.trainingPollInterval = null;
            document.getElementById('btn-train').disabled = false;

            // Refresh models list
            loadModels();

            if (p.status === 'completed') {
                bar.style.background = 'linear-gradient(90deg, #22c55e, #4ade80)';
            } else {
                bar.style.background = 'linear-gradient(90deg, #ef4444, #f87171)';
            }
        }
    } catch (err) { /* ignore */ }
}

// ─── Autopilot ────────────────────────────────────────────
async function loadModels() {
    try {
        const res = await fetch('/api/models');
        const models = await res.json();
        const select = document.getElementById('autopilot-model');
        select.innerHTML = '';

        models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name;
            opt.textContent = `${m.name} (${m.size_mb} MB)`;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load models:', err);
    }
}

async function startAutopilot() {
    const select = document.getElementById('autopilot-model');
    const modelName = select.value;

    if (!modelName) {
        alert('Please select a trained model first!');
        return;
    }

    try {
        const res = await fetch('/api/autopilot/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName }),
        });
        const data = await res.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        document.getElementById('btn-ap-start').disabled = true;
        document.getElementById('btn-ap-stop').disabled = false;
        setMode('autopilot');

        // Poll autopilot status
        state.autopilotPollInterval = setInterval(pollAutopilotStatus, 300);
    } catch (err) {
        alert('Failed to start autopilot: ' + err.message);
    }
}

async function stopAutopilot() {
    try {
        await fetch('/api/autopilot/stop', { method: 'POST' });

        document.getElementById('btn-ap-start').disabled = false;
        document.getElementById('btn-ap-stop').disabled = true;
        setMode('manual');

        if (state.autopilotPollInterval) {
            clearInterval(state.autopilotPollInterval);
            state.autopilotPollInterval = null;
        }

        document.getElementById('ap-fps').textContent = '0';
        document.getElementById('ap-prediction').textContent = '—';
        document.getElementById('ap-servo').textContent = '90°';
        updateBars(0, 0);
    } catch (err) {
        alert('Failed to stop autopilot: ' + err.message);
    }
}

async function pollAutopilotStatus() {
    try {
        const res = await fetch('/api/autopilot/status');
        const data = await res.json();

        document.getElementById('ap-fps').textContent = data.fps;

        // Show action name if available, otherwise motor values
        if (data.action) {
            document.getElementById('ap-prediction').textContent = data.action;
        } else {
            document.getElementById('ap-prediction').textContent =
                `L:${data.prediction[0]} R:${data.prediction[1]}`;
        }

        // Show servo tilt angle
        if (data.servo !== undefined) {
            document.getElementById('ap-servo').textContent = data.servo + '°';
            // Sync the servo slider with autopilot output
            const slider = document.getElementById('servo-slider');
            const valEl = document.getElementById('servo-value');
            if (slider && valEl) {
                slider.value = data.servo;
                valEl.textContent = data.servo + '°';
            }
        }

        // Update overlay bars with autopilot prediction
        const left = Math.round(data.prediction[0] * 100);
        const right = Math.round(data.prediction[1] * 100);
        updateBars(left, right);

        if (!data.running) {
            clearInterval(state.autopilotPollInterval);
            state.autopilotPollInterval = null;
            document.getElementById('btn-ap-start').disabled = false;
            document.getElementById('btn-ap-stop').disabled = true;
            setMode('manual');
        }
    } catch (err) { /* ignore */ }
}

// ─── Mode Indicator ───────────────────────────────────────
function setMode(mode) {
    const badge = document.getElementById('mode-indicator');
    badge.className = 'mode-badge ' + mode;

    const labels = {
        manual:    'MANUAL',
        recording: '⏺ RECORDING',
        autopilot: '🤖 AUTOPILOT',
    };
    badge.textContent = labels[mode] || mode.toUpperCase();
}

// ─── System Status Polling ────────────────────────────────
async function pollSystemStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();

        // Car connection pill
        const carPill = document.getElementById('pill-car');
        carPill.className = 'pill ' + (data.car_connected ? 'connected' : 'disconnected');

        // Camera connection pill
        const camPill = document.getElementById('pill-camera');
        camPill.className = 'pill ' + (data.camera_connected ? 'connected' : 'disconnected');

        // FPS
        document.getElementById('cam-fps').textContent = data.camera_fps;

        // Camera feed visibility
        const feed = document.getElementById('camera-feed');
        const noMsg = document.getElementById('no-camera-msg');
        if (data.camera_connected) {
            feed.style.display = '';
            noMsg.classList.add('hidden');
        } else {
            noMsg.classList.remove('hidden');
        }
    } catch (err) {
        // Server probably not reachable
        document.getElementById('pill-car').className = 'pill disconnected';
        document.getElementById('pill-camera').className = 'pill disconnected';
    }
}

// ─── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    // Load initial data
    loadDatasets();
    loadModels();

    // Start status polling
    state.statusInterval = setInterval(pollSystemStatus, 1000);
    pollSystemStatus();

    console.log('🤖 OpenBot PC Controller initialized (with servo tilt)');
});
