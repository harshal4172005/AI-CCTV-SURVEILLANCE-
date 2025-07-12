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

# Class names for the PPE dataset
CLASS_NAMES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def load_model(model_path):
    """
    Load the YOLOv8 model from given path.
    """
    try:
        model = YOLO(model_path)
        st.success(f"✅ Model loaded successfully from {model_path}")
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
                filename = f"{uuid.uuid4()}.jpg"
                save_path = os.path.join("app/assets", filename)
                os.makedirs("app/assets", exist_ok=True)
                cv2.imwrite(save_path, cropped)
                logger.add_violation(save_path, label)


def predict_image(model, image):
    """
    Predict and return image with bounding boxes for uploaded image.
    """
    try:
        img_array = np.array(image.convert("RGB"))
        results = model(img_array)
        log_violations(results, img_array)


        # Draw bounding boxes and labels manually
        if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else str(cls)
                color = (0, 255, 0) if 'NO-' not in label else (0, 0, 255)
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img_array, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return img_array
        else:
            st.warning("⚠️ No detections found in the image.")
            return img_array
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return np.array(image.convert("RGB"))

class YOLOVideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if self.model is not None:
            results = self.model(img)

            # ✅ Log violations for webcam too
            log_violations(results, img)

            if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls in zip(boxes, confs, clss):
                    x1, y1, x2, y2 = map(int, box)
                    label = self.model.names[cls] if hasattr(self.model, 'names') and cls < len(self.model.names) else str(cls)
                    color = (0, 255, 0) if 'NO-' not in label else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return VideoFrame.from_ndarray(img, format="bgr24")

def get_or_create_transformer(model):
    # Always create or update the transformer in session state
    if "yolo_transformer" not in st.session_state or st.session_state["yolo_transformer"] is None:
        st.session_state["yolo_transformer"] = YOLOVideoTransformer()
    st.session_state["yolo_transformer"].model = model
    return st.session_state["yolo_transformer"]

def predict_webcam(model):
    st.title("Real-time Webcam Detection")

    webrtc_streamer(
        key="yolo-webcam",
        video_transformer_factory=lambda: get_or_create_transformer(model),
        media_stream_constraints={"video": True, "audio": False},
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
            class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class_{cls_id}"
            detections[class_name] = detections.get(class_name, 0) + 1
    
    summary = []
    for class_name, count in detections.items():
        summary.append(f"{class_name}: {count}")
    
    return ", ".join(summary) if summary else "No detections found"
