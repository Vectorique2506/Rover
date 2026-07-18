import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

from ultralytics import YOLO
import cv2
import numpy as np

def load_npu_model(model_path="yolov8n.pt"):
    """
    Downloads and loads the standard YOLOv8n model.
    """
    print("Loading YOLOv8n (Standard)...")
    # This automatically downloads the weights if they aren't there
    model = YOLO('yolov8n.pt') 
    return model

def preprocess_frame(frame, imgsz=640):
    # YOLO ultralytics handles resizing and normalization internally
    return frame

# Update the main_fSM.py logic below to match this












# --- HACKATHON TEST EXECUTION ---
if __name__ == "__main__":
    # 1. Load the model
    # Note: Make sure you pip install onnxruntime-qnn (not the standard onnxruntime)
    yolo_session = load_npu_model("yolov8n_hazard_qnn.onnx")
    
    if yolo_session:
        # 2. Get input details
        input_name = yolo_session.get_inputs()[0].name
        
        # 3. Grab a dummy frame (replace this with your IP webcam frame later)
        dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # 4. Preprocess
        input_tensor = preprocess_frame(dummy_frame)
        
        # 5. RUN INFERENCE!
        print("Running inference on NPU...")
        outputs = yolo_session.run(None, {input_name: input_tensor})
        
        print(f"Inference complete! Output tensor shape: {outputs[0].shape}")