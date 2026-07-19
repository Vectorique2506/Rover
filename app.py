"""
Hazard Inspector - Web Dashboard (single file)
================================================
Runs your existing FSM (YOLO detection -> guidance -> Moondream VLM analysis ->
Arduino alarm) but serves it as a live web dashboard instead of a cv2.imshow
window. Open http://localhost:5000 in a browser (or from your phone on the
same network using your PC's LAN IP).

HOW IT WORKS
------------
- A dedicated CameraReader thread continuously drains the camera stream and
  always keeps only the LATEST frame. This is what fixes growing video
  latency: OpenCV/HTTP-MJPEG streams buffer frames in the OS network socket,
  and if nothing reads fast enough the delay keeps growing. By reading in a
  tight loop with nothing blocking it, the buffer never backs up.
- The FSM thread (YOLO detection -> guidance -> Moondream VLM analysis ->
  Arduino alarm) always grabs whatever the CURRENT latest frame is, runs
  detection/analysis on it, and pushes the annotated result to the browser.
  Even if the VLM call takes a few seconds, video resumes at full speed the
  moment it's done -- no backlog accumulates because the CameraReader kept
  draining the stream the whole time.
- Flask streams the annotated JPEG as an MJPEG feed at /video_feed (this is
  what your <img> tag on the dashboard points at).
- The dashboard page also polls /api/status every second to update the state
  badge, alarm indicator, camera-connection banner, and hazard log.

SETUP (step by step)
---------------------
1. Put this file in the SAME folder as yolov8n.pt (e.g. next to your `core_ai`
   folder, or copy yolov8n.pt beside this file).
2. Install dependencies:
       pip install flask opencv-python ultralytics moondream pyserial
3. Edit the CONFIG block below:
       - PHONE_CAMERA_URL -> your IP webcam stream URL
       - ARDUINO_PORT     -> your Arduino's COM port (or set to None to skip it)
4. Run it:
       python app.py
5. Open the dashboard:
       http://localhost:5000
   (or http://<your-pc-ip>:5000 from another device on the same WiFi)
6. Press the "Stop" button on the page (or Ctrl+C in the terminal) to shut
   down cleanly.
"""

import cv2
import time
import threading
import serial
import moondream as md
from PIL import Image
from ultralytics import YOLO
from flask import Flask, Response, jsonify, render_template_string

# --- CONFIGURATION ---
PHONE_CAMERA_URL = "http://10.92.204.152:8080/video"
ARDUINO_PORT = 'COM3'   # set to None if you don't have the Arduino connected
BAUD_RATE = 9600
YOLO_WEIGHTS = 'yolov8n.pt'
YOLO_CLASS_IDS = [28]      # class(es) treated as "hazard" (28 = suitcase in COCO -- change if your object is different)
YOLO_CONF_THRESHOLD = 0.35 # lower = more sensitive/more false positives, higher = fewer/more confident detections
DEBUG_LOG_DETECTIONS = True  # prints EVERY object YOLO sees (name + confidence) so you can find the right class ID
STREAM_MAX_WIDTH = 640  # frames are downscaled to this width before detection/encoding (lower = less latency)
FRAME_ROTATE = None     # None, 90, 180, or 270 -- set this if your video looks sideways/upside down
JPEG_QUALITY = 70       # 0-100; lower = smaller/faster frames over the network, slight quality loss

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Shared state between the background inference thread and the Flask routes
# ---------------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg_bytes = None
        self.state = "INITIALIZING"
        self.alarm = False
        self.log = []
        self.running = True
        self.camera_ok = False
        self.camera_message = "Connecting to camera..."

    def set_camera_status(self, ok, message=""):
        with self.lock:
            self.camera_ok = ok
            self.camera_message = message

    def update_frame(self, frame):
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            with self.lock:
                self.jpeg_bytes = buf.tobytes()

    def get_frame(self):
        with self.lock:
            return self.jpeg_bytes

    def set_state(self, state):
        with self.lock:
            self.state = state

    def set_alarm(self, active):
        with self.lock:
            self.alarm = active

    def add_log(self, entry):
        with self.lock:
            self.log.append(entry)
            self.log = self.log[-50:]  # keep last 50 entries

    def snapshot(self):
        with self.lock:
            return {
                "state": self.state,
                "alarm": self.alarm,
                "log": list(self.log[-15:]),
                "camera_ok": self.camera_ok,
                "camera_message": self.camera_message,
            }


shared = SharedState()


# ---------------------------------------------------------------------------
# Dedicated camera-reading thread.
#
# Why this exists: cv2.VideoCapture reading an HTTP/MJPEG stream buffers
# frames in the OS network socket. If the consumer (our FSM loop) is busy
# doing something slow (YOLO, and especially the multi-second VLM call),
# frames pile up in that buffer. The next read() then returns the OLDEST
# buffered frame, not the newest -- so the visible delay keeps growing the
# longer analysis takes. This class continuously drains the stream in its
# own thread and always exposes only the LATEST frame, so nothing backs up.
# ---------------------------------------------------------------------------
class CameraReader:
    def __init__(self, url):
        self.url = url
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.running = True
        self.cap = cv2.VideoCapture(url)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # helps on some backends, harmless if ignored
        except Exception:
            pass
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _reader_loop(self):
        consecutive_failures = 0
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                consecutive_failures += 1
                with self.lock:
                    self.ok = False
                if consecutive_failures == 1 or consecutive_failures % 20 == 0:
                    msg = f"No frames received from {self.url} ({consecutive_failures} failed reads)"
                    print(f"WARNING: {msg}")
                    shared.set_camera_status(False, msg)
                time.sleep(0.3)
                continue

            if consecutive_failures > 0:
                print("Camera stream recovered.")
            consecutive_failures = 0
            shared.set_camera_status(True, "Live")

            with self.lock:
                self.frame = frame
                self.ok = True
            # no sleep here -- we WANT this loop running as fast as the
            # stream allows, to keep draining the socket buffer

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def stop(self):
        self.running = False
        self.cap.release()


# ---------------------------------------------------------------------------
# The FSM itself (your original logic, adapted to write into `shared`
# instead of calling cv2.imshow)
# ---------------------------------------------------------------------------
class HazardInspectorFSM:
    def __init__(self):
        self.state = "SCANNING"
        shared.set_state(self.state)

        print("Loading YOLO...")
        self.yolo_model = YOLO(YOLO_WEIGHTS)

        print("Loading VLM (Moondream)...")
        self.vlm = md.vl(model='moondream')

        self.arduino = None
        if ARDUINO_PORT:
            try:
                self.arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
                time.sleep(2)
                print(f"Connected to Arduino on {ARDUINO_PORT}")
            except Exception as e:
                print(f"CRITICAL: Could not connect to Arduino: {e}")
                self.arduino = None

    def trigger_alarm(self, active):
        shared.set_alarm(active)
        if self.arduino:
            self.arduino.write(b"ALARM_ON\n" if active else b"ALARM_OFF\n")

    def get_vlm_analysis(self, frame):
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)  # moondream requires a PIL.Image, not a numpy array
            answer = self.vlm.query(pil_img, "Is this a safety hazard? If so, describe it in 5 words.")
            return str(answer.answer)
        except Exception as e:
            print(f"VLM ERROR: {e}")
            return "vlm Busy/Failed"

    def set_state(self, new_state):
        self.state = new_state
        shared.set_state(new_state)

    def run(self, camera: CameraReader):
        print("Starting Hazard Inspector System...")
        frame_count = 0

        while shared.running:
            ret, frame = camera.get_latest()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            # Fix sideways/upside-down video if needed (set FRAME_ROTATE above)
            if FRAME_ROTATE == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif FRAME_ROTATE == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif FRAME_ROTATE == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Downscale before doing anything else -- smaller frame = faster
            # YOLO inference AND a much smaller JPEG to push over the network
            h, w = frame.shape[:2]
            if w > STREAM_MAX_WIDTH:
                scale = STREAM_MAX_WIDTH / w
                frame = cv2.resize(frame, (STREAM_MAX_WIDTH, int(h * scale)))

            # --- 1. AI INFERENCE (YOLO) ---
            detections = []
            if frame_count % 3 == 0:
                # Run on ALL classes (not just YOLO_CLASS_IDS) so we can log
                # everything YOLO actually sees -- this is the fastest way to
                # find the correct class ID if detection "isn't working".
                results = self.yolo_model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)
                all_boxes = results[0].boxes

                if DEBUG_LOG_DETECTIONS and len(all_boxes) > 0:
                    names = self.yolo_model.names
                    seen = ", ".join(
                        f"{names[int(b.cls[0])]}({float(b.conf[0]):.2f})" for b in all_boxes
                    )
                    print(f"[YOLO sees]: {seen}")

                # Filter down to only the class(es) we treat as a hazard
                detections = [b for b in all_boxes if int(b.cls[0]) in YOLO_CLASS_IDS]

                if len(detections) > 0 and self.state == "SCANNING":
                    print(">> Hazard spotted!")
                    self.set_state("TARGET_LOCKED")

            # --- 2. FSM LOGIC ---
            if self.state == "TARGET_LOCKED":
                if len(detections) > 0:
                    self.trigger_alarm(True)

                    x1, y1, x2, y2 = detections[0].xyxy[0]
                    box_center = (x1 + x2) / 2

                    if box_center < 250:
                        cv2.putText(frame, "MOVE LEFT", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif box_center > 390:
                        cv2.putText(frame, "MOVE RIGHT", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, "STEADY - ANALYZING...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        self.set_state("ANALYZING")
                else:
                    self.trigger_alarm(False)
                    self.set_state("SCANNING")

            elif self.state == "ANALYZING":
                print(">> Running Deep Analysis...")
                report = self.get_vlm_analysis(frame)
                timestamp = time.strftime("%H:%M:%S")

                if "red" in report.lower() and "box" in report.lower():
                    entry = f"[{timestamp}] RED BOX CONFIRMED: {report}"
                else:
                    entry = f"[{timestamp}] {report}"

                shared.add_log(entry)
                print(f"REPORT: {report}")

                self.trigger_alarm(False)
                self.set_state("SCANNING")
            else:
                self.trigger_alarm(False)

            # --- 3. OVERLAY (burned into the frame, then streamed to the browser) ---
            for box in detections:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

            cv2.putText(frame, f"STATE: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            shared.update_frame(frame)

        if self.arduino:
            self.arduino.close()
        print("Inspector loop stopped.")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hazard Inspector Dashboard</title>
<style>
  :root { --bg:#0e0f11; --panel:#17181c; --line:#2a2c31; --text:#e8e9ec; --muted:#8b8f98; --accent:#3ecf8e; --danger:#ff4d4f; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:16px 24px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size:18px; margin:0; font-weight:600; letter-spacing:.3px; }
  .layout { display:grid; grid-template-columns:1.4fr 1fr; gap:16px; padding:16px; max-width:1200px; margin:0 auto; }
  @media (max-width:860px){ .layout{ grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
  .video-wrap { position:relative; border-radius:8px; overflow:hidden; background:#000; }
  .video-wrap img { width:100%; display:block; }
  .badges { display:flex; gap:10px; margin-top:12px; flex-wrap:wrap; }
  .badge { padding:6px 12px; border-radius:20px; font-size:13px; font-weight:600; border:1px solid var(--line); }
  .badge.state { background:#1c2b22; color:var(--accent); }
  .badge.alarm-off { background:#20242b; color:var(--muted); }
  .badge.alarm-on { background:#3a1414; color:var(--danger); animation:pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:.5;} }
  h2 { font-size:14px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 10px; }
  .log { max-height:520px; overflow-y:auto; display:flex; flex-direction:column-reverse; gap:6px; }
  .log-entry { font-size:12.5px; padding:8px 10px; background:#1c1d21; border-radius:6px; border-left:3px solid var(--accent); }
  .cam-banner { display:none; background:#3a1414; color:#ff9d9e; border:1px solid #5a1c1d; border-radius:8px; padding:10px 14px; font-size:13px; margin-bottom:12px; }
  .cam-banner.show { display:block; }
</style>
</head>
<body>
<header>
  <h1>🛠 Hazard Inspector</h1>
  <span id="conn" style="color:var(--muted); font-size:13px;">connecting…</span>
</header>
<div class="layout">
  <div class="panel">
    <div class="cam-banner" id="camBanner"></div>
    <div class="video-wrap">
      <img src="/video_feed" alt="live feed">
    </div>
    <div class="badges">
      <span class="badge state" id="stateBadge">STATE: --</span>
      <span class="badge alarm-off" id="alarmBadge">ALARM: OFF</span>
    </div>
  </div>
  <div class="panel">
    <h2>Hazard Log</h2>
    <div class="log" id="log"></div>
  </div>
</div>
<script>
async function poll() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('conn').textContent = 'live';
    document.getElementById('stateBadge').textContent = 'STATE: ' + data.state;
    const camBanner = document.getElementById('camBanner');
    if (!data.camera_ok) {
      camBanner.textContent = '⚠ ' + data.camera_message;
      camBanner.className = 'cam-banner show';
    } else {
      camBanner.className = 'cam-banner';
    }
    const alarmEl = document.getElementById('alarmBadge');
    if (data.alarm) {
      alarmEl.textContent = 'ALARM: ON';
      alarmEl.className = 'badge alarm-on';
    } else {
      alarmEl.textContent = 'ALARM: OFF';
      alarmEl.className = 'badge alarm-off';
    }
    const logEl = document.getElementById('log');
    logEl.innerHTML = data.log.map(e => `<div class="log-entry">${e}</div>`).join('');
  } catch (e) {
    document.getElementById('conn').textContent = 'disconnected';
  }
}
setInterval(poll, 1000);
poll();
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)


def mjpeg_generator():
    while True:
        frame = shared.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)


@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    return jsonify(shared.snapshot())


if __name__ == '__main__':
    print(f"Connecting to camera stream: {PHONE_CAMERA_URL}")
    camera = CameraReader(PHONE_CAMERA_URL).start()

    inspector = HazardInspectorFSM()
    worker = threading.Thread(target=inspector.run, args=(camera,), daemon=True)
    worker.start()

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        shared.running = False
        camera.stop()