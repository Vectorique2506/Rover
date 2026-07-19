import cv2
import time
import serial
import moondream as md
from ultralytics import YOLO

# --- CONFIGURATION ---
PHONE_CAMERA_URL = "http://10.92.204.152:8080/video"
ARDUINO_PORT = 'COM3' # CHANGE THIS TO YOUR ACTUAL PORT
BAUD_RATE = 9600

class HazardInspectorFSM:
    def __init__(self):
        self.state = "SCANNING"
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Load Moondream VLM
        print("Loading VLM (Moondream)...")
        self.vlm = md.vl(model='moondream')
        
        # Initialize Serial for Arduino Buzzer/LED
        try:
            self.arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
            time.sleep(2) 
            print(f"Connected to Arduino on {ARDUINO_PORT}")
        except Exception as e:
            print(f"CRITICAL: Could not connect to Arduino: {e}")
            self.arduino = None
        
        self.hazard_log = []

    def trigger_alarm(self, active):
        if self.arduino:
            if active:
                self.arduino.write(b"ALARM_ON\n")
            else:
                self.arduino.write(b"ALARM_OFF\n")

    def get_vlm_analysis(self, frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            answer = self.vlm.query(img, "Is this a safety hazard? If so, describe it in 5 words.")
            return str(answer.answer)
        except Exception as e:
            print(f"VLMERROR: {e}")
            return "vlm Busy/Failed"
        

    def run(self):
        print("Starting Hazard Inspector System...")
        cap = cv2.VideoCapture(PHONE_CAMERA_URL)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: 
                time.sleep(0.5)
                continue
            
            frame_count += 1
            
            # --- 1. AI INFERENCE (YOLO) ---
            detections = []
            if frame_count % 3 == 0:
                results = self.yolo_model(frame, verbose=False,classes=[28])
                detections = results[0].boxes
                
                if len(detections) > 0 and self.state == "SCANNING":
                    print(">> Hazard spotted!")
                    self.state = "TARGET_LOCKED"
            
            # --- 2. FSM LOGIC ---
            if self.state == "TARGET_LOCKED":
                if len(detections) > 0:
                    self.trigger_alarm(True) # Alarm ON
                    
                    x1, y1, x2, y2 = detections[0].xyxy[0]
                    box_center = (x1 + x2) / 2
                    
                    # Guidance overlay
                    if box_center < 250:
                        cv2.putText(frame, "MOVE LEFT", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    elif box_center > 390:
                        cv2.putText(frame, "MOVE RIGHT", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    else:
                        cv2.putText(frame, "STEADY - ANALYZING...", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        self.state = "ANALYZING"
                else:
                    self.trigger_alarm(False)
                    self.state = "SCANNING"
                    
            elif self.state == "ANALYZING":
                print(">> Running Deep Analysis...")
                report = self.get_vlm_analysis(frame)

                if("red" in report.lower() and "box" in report.lower()):
                    self.hazard_log.append(f"[{timestamp}] RED BOX CONFIRMED: {report}")
                else:
                    self.hazard_log.append(f"[{timestamp}] False Alarm.")

                self.state="SCANNING"

                



                   



                timestamp = time.strftime("%H:%M:%S")
                self.hazard_log.append(f"[{timestamp}] {report}")
                print(f"REPORT: {report}")
                
                time.sleep(2) 
                self.trigger_alarm(False)
                self.state = "SCANNING"
            else:
                self.trigger_alarm(False)

            # --- 3. HUD OVERLAY ---
            for box in detections:
                x1, y1, x2, y2 = box.xyxy[0]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
            
            # Draw Log Panel
            cv2.rectangle(frame, (400, 0), (640, 480), (30, 30, 30), -1)
            cv2.putText(frame, "LOG:", (410, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos = 60
            for entry in self.hazard_log[-10:]:
                cv2.putText(frame, entry[:30], (410, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_pos += 20

            cv2.putText(frame, f"STATE: {self.state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Inspector HUD", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()
        if self.arduino: self.arduino.close()

if __name__ == "__main__":
    inspector = HazardInspectorFSM()
    inspector.run()








