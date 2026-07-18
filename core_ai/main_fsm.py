import cv2
import time
import json
import base64
import requests
import numpy as np
import paho.mqtt.client as mqtt
from npu_yolo_loader import load_npu_model, preprocess_frame

# --- CONFIGURATION ---
PHONE_CAMERA_URL = "http://192.168.x.x:8080/video" 
MQTT_BROKER_IP = "localhost"
VLM_API_URL = "http://localhost:8000/v1/chat/completions"

DISTANCE_THRESHOLD_M = 1.0
CONFIDENCE_THRESHOLD = 0.5

class ArduinoMQTTClient:
    def __init__(self, broker_ip):
        # Version 2.0+ fix for Paho MQTT
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "AI_PC_Brain")
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.current_distance = 2.0
        try:
            self.client.connect(broker_ip, 1883, 60)
            self.client.loop_start()
        except:
            print("MQTT Broker not found. Continuing in Offline Mode.")

    def on_connect(self, client, userdata, flags, rc):
        self.client.subscribe("sensor/tof/distance")
        
    def on_message(self, client, userdata, msg):
        if msg.topic == "sensor/tof/distance":
            try:
                payload = json.loads(msg.payload.decode())
                self.current_distance = float(payload.get("distance", 2.0))
            except: pass

    def send_buzzer_cmd(self, state):
        self.client.publish("arduino/command", json.dumps({"buzzer": state}))
        
    def update_dashboard(self, state, hazard_log=""):
        self.client.publish("dashboard/status", json.dumps({"state": state, "log": hazard_log}))

class HazardInspectorFSM:
    def __init__(self):
        self.state = "SCANNING"
        self.yolo_model = load_npu_model() # Using ultralytics model
        self.comms = ArduinoMQTTClient(MQTT_BROKER_IP)
        
    def run(self):
        print("Starting H.A.R.D. System...")
        url="http://10.92.204.152:8080/video"
        cap = cv2.VideoCapture(url)

        frame_count=0
        
        if not cap.isOpened():
            print("CRITICAL:stream failed to open")
            return
        
        
        
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream lost,retrying...")
                time.sleep(0.5)
                continue

            frame_count += 1


            if frame_count % 3 == 0:
                results = self.yolo_model(frame, verbose=False)
                # Keep your existing logic for results here:
                if len(results[0].boxes) > 0:
                    print(">> Hazard spotted!")
                    self.state = "TARGET_LOCKED"
            
            # --- FSM LOGIC (Outside the %3 block so it runs every frame) ---
            if self.state == "TARGET_LOCKED":
                if self.comms.current_distance <= DISTANCE_THRESHOLD_M:
                    self.comms.send_buzzer_cmd("FAST_BEEP")
                    self.state = "ANALYZING"
                # ... rest of your FSM ...

            # HUD overlay (Runs every frame)
            cv2.putText(frame, f"STATE: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Inspector HUD", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

            










            if self.state == "SCANNING":
                # Ultralytics inference
                results = self.yolo_model(frame, verbose=False)
                if len(results[0].boxes) > 0:
                    print(">> Hazard spotted!")
                    self.state = "TARGET_LOCKED"
                    
            elif self.state == "TARGET_LOCKED":
                if self.comms.current_distance <= DISTANCE_THRESHOLD_M:
                    self.comms.send_buzzer_cmd("FAST_BEEP")
                    self.state = "ANALYZING"
                elif self.comms.current_distance > 3.0:
                    self.state = "SCANNING"
                    
            elif self.state == "ANALYZING":
                print(">> Analyzing hazard...")
                self.comms.send_buzzer_cmd("STOP")
                time.sleep(2) # Simulate Cloud AI reasoning
                self.comms.update_dashboard("REPORT_READY", "Hazard identified: Cardboard box.")
                time.sleep(2)
                self.state = "SCANNING"
            
            # HUD overlay
            cv2.putText(frame, f"STATE: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Inspector HUD", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    inspector = HazardInspectorFSM()
    inspector.run()


    


    