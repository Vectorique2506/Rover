# Project H.A.R.D. (Hazard Assessment & Reporting Device)

##Team
##Andrew Prakash  andrewb2506#gmail.com
##Piyush Tandon   ppiyush_be24@thapar.edu


> **"Detect. Analyze. Alert. Protect."**  
> An advanced, edge-native distributed AI ecosystem engineered for the **Qualcomm Snapdragon Multiverse Hackathon 2026**.

---

 Executive Summary

In high-risk industrial environments, safety isn't just a protocol—it’s a lifeline. Yet, traditional safety inspections remain slow, reactive, and dangerously prone to human error.

**H.A.R.D.** transforms standard hardware into an intelligent, zero-latency safety guardian. By seamlessly bridging an Android smartphone, a Snapdragon Copilot+ PC, and an Arduino UNO Q microcontroller, H.A.R.D. creates a localized, distributed AI inspection network. It scans environments, pinpoints physical hazards via hardware-accelerated computer vision, provides real-time haptic/visual alerts, and deploys local Vision-Language Models (VLMs) to generate deep, contextual incident reports—**100% on the edge, entirely offline. **

---

##  The Core Problem

Industrial spaces (factories, construction sites, deep-logistics warehouses) are inherently dynamic. Current inspection frameworks fail because they are:

*   **Reactive, Not Proactive:** Incidents are documented *after* they happen, rather than intercepted in real time.
*   **Cognitive Overload:** Inspectors navigating dangerous zones must split their focus between spotting hazards and writing manual logs.
*   **Cloud-Dependent Failure Points:** Industrial environments often suffer from dead zones. Cloud-reliant AI models fail precisely when an inspector needs them most.
*   **The Reporting Bottleneck:** Standard safety reporting requires manual entry, creating hours of delayed administrative friction.

---

##  The Distributed Solution

H.A.R.D. avoids the pitfalls of monolithic systems by distributing the computational load across specialized hardware nodes, maximizing local Qualcomm architecture.

| Device Node | Operational Layer | Core Responsibility |
| :--- | :--- | :-- |
|  Android Phone| Edge Capture & UI | Live video ingestion, spatial anchoring, and user-facing augmented dashboard. |
|  Snapdragon Copilot+ PC | Neural Processing Hub | Real-time Object Detection (**Hexagon NPU**) + Complex Vision-Language Reasoning (**Adreno GPU**). |
|  Arduino UNO Q | Physical Actuation Layer | Time-of-Flight (ToF) spatial telemetry, instantaneous audio/visual haptic alarms. |





![alt text](https://github.com/Vectorique2506/Rover/blob/main/WhatsApp%20Image%202026-07-19%20at%2012.13.27%20PM.jpeg)

---

##  Key Capabilities & Innovations

*   **Hardware-Accelerated Edge Inference:** Real-time object detection powered by custom-quantized models running locally on the Qualcomm Hexagon NPU.
*   **Local Vision-Language Reasoning:** Deep contextual safety analysis powered by `Qwen2.5-VL`, running entirely on-device without cloud data leakage.
*   **Distributed Physical Telemetry:** Seamless device-to-device synchronization via micro-latency local communication protocols.
*   **Zero-Trust Offline Resilience:** Built natively for environments with zero internet connectivity, ensuring data privacy and continuous operational uptime.
*   **Autonomous Documentation:** Eradicates paperwork by automatically compiling timestamps, confidence matrices, hazard imagery, and mitigation steps instantly.

---



##  System Architecture & Data Flow













##System Infrastrucutre

#  Setup & Installation

Follow the steps below to set up and run **Project H.A.R.D.** on your local machine.

---

## Prerequisites

### Hardware

- Snapdragon Copilot+ PC (or Windows PC)
- Android Smartphone
- Arduino UNO Q
- VL53L0X Time-of-Flight (ToF) Sensor
- LED
- Buzzer

### Software
- Python 3.11 or later
- Arduino IDE
- IP Webcam (Android App)

---

# 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/HARD.git
cd HARD
```

---

# 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3. Install Python Dependencies

Upgrade pip

```bash
python -m pip install --upgrade pip
```

Install all required packages

```bash
pip install -r requirements.txt
```

If `requirements.txt` is unavailable, install manually:

```bash
pip install ultralytics
pip install opencv-python
pip install pyserial
pip install numpy
pip install pillow
pip install torch torchvision
pip install transformers
pip install accelerate
pip install sentencepiece
pip install requests
```

---

# 4. Python Requirements

Create a file named **requirements.txt** containing:

```text
ultralytics>=8.3.0
opencv-python>=4.10.0
numpy>=1.26.0
pyserial>=3.5
torch>=2.3.0
torchvision>=0.18.0
transformers>=4.53.0
accelerate>=1.0.0
sentencepiece>=0.2.0
Pillow>=10.0.0
requests>=2.32.0
```

---

# 5. Install Arduino Libraries

Open **Arduino IDE** and install the following libraries from **Library Manager**:

- Adafruit VL53L0X
- Wire

Upload the Arduino firmware located in:

```text
arduino/hard_controller.ino
```

Select:

- **Board:** Arduino UNO Q
- **Port:** COM3 (or your detected COM Port)

---

# 6. Android Camera Setup

1. Install **IP Webcam** from the Play Store.
2. Open the application.
3. Start the camera server.
4. Note the streaming URL displayed by the app.

Example:

```text
http://192.168.1.100:8080/video
```

Update the following line in the Python code:

```python
PHONE_CAMERA_URL = "http://192.168.1.100:8080/video"
```

Ensure both the smartphone and PC are connected to the same Wi-Fi network.

---

# 7. Configure Arduino Port

Update the serial port in the Python script:

```python
ARDUINO_PORT = "COM3"
```

Replace `COM3` with the port assigned to your Arduino.

---

# 8. Download YOLO Model

The project uses **YOLOv8 Nano**.

It will download automatically on the first run:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

Alternatively, manually place `yolov8n.pt` inside the `models/` directory.

---

# 9. (Optional) Local Vision-Language Model

To enable offline hazard reasoning, install the Hugging Face dependencies:

```bash
pip install transformers accelerate
```

The first execution will automatically download the **Qwen2.5-VL** model and cache it locally for future offline use.

---

# 10. Run the Project

Start the application:

```bash
python hazard_detection.py
```

Expected console output:

```text
Connecting to Arduino...
Loading YOLO model...
Connecting to Android camera...
System Ready
Inspecting Environment...
```

---

# Project Structure

```text
HARD/
│
├── arduino/
│   └── hard_controller.ino
│
├── models/
│   └── yolov8n.pt
│
├── src/
│   ├── hazard_detection.py
│   ├── report_generator.py
│   ├── vlm_reasoning.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Running the Complete System

1. Upload the Arduino firmware.
2. Connect the ToF sensor, LED, and buzzer.
3. Start the IP Webcam server on the Android phone.
4. Connect the phone and PC to the same Wi-Fi network.
5. Update the camera URL and Arduino COM port.
6. Run the Python application.
7. Point the camera toward the inspection area.

The system will:

- Detect hazards using YOLOv8.
- Measure object proximity using the ToF sensor.
- Trigger LED and buzzer alerts for nearby hazards.
- Perform local Vision-Language reasoning using Qwen2.5-VL.
- Generate an automated incident report containing hazard descriptions, timestamps, confidence scores, and suggested mitigation actions.

---







![alt text](https://github.com/Vectorique2506/Rover/blob/main/WhatsApp%20Image%202026-07-19%20at%2010.52.18%20AM.jpeg)




