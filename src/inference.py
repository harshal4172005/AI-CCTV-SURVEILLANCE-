import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from av.video.frame import VideoFrame
from src.violation_logger import ViolationLogger
import uuid
import os
from datetime import datetime
import torch

# Class names for the PPE dataset
CLASS_NAMES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Set the device for inference to GPU if available, otherwise use CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_path):
    """
    Load the YOLOv8 model from given path.
    """
    # Print the device being used for troubleshooting
    st.info(f"✅ Attempting to load model on device: {DEVICE}")

    try:
        # Check if the specified model path exists
        if os.path.exists(model_path):
            # Load the custom trained model
            model = YOLO(model_path)
            st.success(f"✅ Model loaded successfully from {model_path} on device: {DEVICE}")
        else:
            # Fallback to default model if custom model doesn't exist
            st.warning(f"⚠️ Custom model not found at {model_path}, using default YOLOv8 model")
            model = YOLO('yolov8n.pt')
            st.info("ℹ️ Note: Default model may not detect PPE violations properly")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None
    
def log_violations(results, img_array):
    logger = ViolationLogger()
    violation_classes = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]

    if results and len(results) > 0 and hasattr(results[0], 'boxes'):
        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        names = results[0].names if hasattr(results[0], 'names') else {}

        for i, cls in enumerate(clss):
            label = names[cls] if cls < len(names) else str(cls)
            if label in violation_classes:
                x1, y1, x2, y2 = map(int, boxes[i])
                cropped = img_array[y1:y2, x1:x2]
                
                # Create a subfolder for each violation type
                violation_folder = os.path.join("app", "assets", label)
                os.makedirs(violation_folder, exist_ok=True)
                
                filename = f"{uuid.uuid4()}.jpg"
                save_path = os.path.join(violation_folder, filename)

                cv2.imwrite(save_path, cropped)
                logger.add_violation(save_path, label)


def predict_image(model, image):
    """
    Predict and return image with bounding boxes for uploaded image.
    """
    try:
        img_array = np.array(image.convert("RGB"))
        
        # Use lower confidence threshold for better detection
        results = model(img_array, device=DEVICE, conf=0.25, iou=0.45)
        log_violations(results, img_array)

        # Draw bounding boxes and labels manually
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Debug: Print all detections
            print(f"🔍 Found {len(boxes)} detections:")
            for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                label = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else str(cls)
                print(f"   {i+1}: {label} (confidence: {conf:.3f})")
            
            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else str(cls)
                color = (0, 255, 0) if 'NO-' not in label else (0, 0, 255)
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_array, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img_array
        else:
            st.warning("⚠️ No detections found in the image. Try lowering the confidence threshold.")
            return img_array
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return np.array(image.convert("RGB"))

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None
        self.frame_skip = 2
        self.frame_count = 0
        self.last_results = None
        self.processed_frames = 0
        self.start_time = time.time()
        self.fps = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.processed_frames += 1

        self.frame_count += 1
        if self.frame_count % self.frame_skip == 0 and self.model is not None:
            results = self.model(img, device=DEVICE, verbose=False)
            self.last_results = results
            log_violations(results, img)
            
            # Calculate FPS
            end_time = time.time()
            if (end_time - self.start_time) > 1:
                self.fps = self.processed_frames / (end_time - self.start_time)
                self.processed_frames = 0
                self.start_time = end_time

        if self.last_results is not None:
            results = self.last_results
            if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.model.names[cls] if hasattr(self.model, 'names') and cls < len(self.model.names) else str(cls)
                    color = (0, 255, 0) if 'NO-' not in label else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'FPS: {self.fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return VideoFrame.from_ndarray(img, format="bgr24")


def get_or_create_transformer(model):
    if "yolo_transformer" not in st.session_state or st.session_state["yolo_transformer"] is None:
        st.session_state["yolo_transformer"] = YOLOVideoTransformer()
    st.session_state["yolo_transformer"].model = model
    return st.session_state["yolo_transformer"]

def predict_webcam(model):
    st.title("Real-time Webcam Detection")

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=lambda: get_or_create_transformer(model),
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
        async_transform=True,
    )

def get_detection_summary(results):
    """
    Get a summary of detections for display.
    """
    if not results or len(results) == 0:
        return "No detections found"
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return "No detections found"
    
    detections = {}
    for i, box in enumerate(result.boxes.xyxy):
        if hasattr(result.boxes, 'cls') and len(result.boxes.cls) > i:
            cls_id = int(result.boxes.cls[i])
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            detections[class_name] = detections.get(class_name, 0) + 1
    
    summary = []
    for class_name, count in detections.items():
        summary.append(f"{class_name}: {count}")
    
    return ", ".join(summary) if summary else "No detections found"