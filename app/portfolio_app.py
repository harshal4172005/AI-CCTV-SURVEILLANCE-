import streamlit as st
import os
import sys
import numpy as np
import plotly.express as px
import cv2
from src.violation_logger import ViolationLogger
from src.report_generator import PDFReport, CSVReport
from PIL import Image

# ✅ Add parent directory to Python path BEFORE importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import load_model, predict_image, predict_webcam, get_detection_summary, YOLOVideoTransformer, DEVICE

# 🎨 Professional Page Configuration
st.set_page_config(
    page_title="AI CCTV Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 📊 Model Status with Animation
@st.cache_resource
def load_cached_model():
    model_path = "app/models/best.pt"
    if not os.path.exists(model_path):
        return None
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
model = load_cached_model()

# 🎛️ Enhanced Sidebar with Professional Design
with st.sidebar:
    st.markdown("### 🎛️ Detection Options")
    
    option = st.radio("Select input source:", 
                     ["📊 Dashboard", "📷 Single Image", "📁 Batch Processing", "📹 Real-time Webcam", "📑 Violations Report"],
                     index=0)
    
    st.markdown("---")
    
    # --- LIVE STATISTICS ---
    st.markdown("### 📈 Live Statistics")
    
    if "images_processed_count" not in st.session_state:
        st.session_state["images_processed_count"] = 0
    if "logger" not in st.session_state:
        st.session_state["logger"] = ViolationLogger()
    if "yolo_transformer" not in st.session_state:
        st.session_state["yolo_transformer"] = None

    if st.session_state["yolo_transformer"] and hasattr(st.session_state["yolo_transformer"], 'fps'):
        images_processed = st.session_state["yolo_transformer"].processed_frames
    else:
        images_processed = st.session_state["images_processed_count"]

    total_violations = len(st.session_state["logger"].get_violations())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Images Processed", f"{images_processed}")
    with col2:
        st.metric("Violations Logged", f"{total_violations}")
    
    st.markdown("---")
    
    # 🔍 Detection Classes Info
    st.markdown("### 🔍 Detection Classes")
    
    classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
    
    for class_name in classes:
        if "NO-" in class_name:
            st.markdown(f"🔴 {class_name} (Violation)")
        else:
            st.markdown(f"🟢 {class_name} (Compliant)")
            
    # 🚨 Live Violation Log Section
    st.markdown("---")
    st.markdown("### 🚨 Live Violation Log")
    
    violations = st.session_state["logger"].get_violations()
    total_violations = len(violations)
    
    st.markdown(f"**Total Violations:** `{total_violations}`")
    
    violation_counts = {}
    for v in violations:
        violation_type = v['violation_type']
        violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

    for v_type, count in violation_counts.items():
        st.markdown(f"- **{v_type}**: `{count}`")

# 📊 Dashboard View
if option == "📊 Dashboard":
    st.markdown("<h1>AI CCTV Surveillance System</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Advanced PPE Detection for Construction Site Safety</h4>", unsafe_allow_html=True)

    violations = st.session_state["logger"].get_violations()
    
    if not violations:
        st.info("No violations logged yet. Start a session to see real-time data.")
    else:
        violation_counts = {}
        for v in violations:
            violation_type = v['violation_type']
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

        chart_data = {
            'Violation Type': list(violation_counts.keys()),
            'Count': list(violation_counts.values())
        }

        fig = px.bar(chart_data, x='Violation Type', y='Count',
                     title="Real-time Violation Analysis",
                     color='Count',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

# 📷 Single Image Upload with Enhanced UI
elif option == "📷 Single Image":
    st.markdown("<h2>📷 Single Image Detection</h2>", unsafe_allow_html=True)
    
    image_file = st.file_uploader("Upload an image for analysis", 
                                 type=["jpg", "jpeg", "png"], 
                                 help="Upload a single image to detect PPE and safety violations")
    
    if image_file:
        col1, col2 = st.columns(2)
        image = Image.open(image_file)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            if model is None:
                st.error("Model not loaded. Please ensure app/models/best.pt exists and is valid.")
                st.image(image, caption="Original Image (No Detection Available)", use_container_width=True)
            else:
                try:
                    img_array = np.array(image.convert("RGB"))
                    results = model(img_array, device=DEVICE)
                    result_img = img_array.copy()
                    if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        for box, conf, cls in zip(boxes, confs, clss):
                            x1, y1, x2, y2 = map(int, box)
                            label = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else str(cls)
                            color = (0, 255, 0) if 'NO-' not in label else (0, 0, 255)
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(result_img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    st.image(result_img, caption="Detected Objects", use_container_width=True)
                    st.session_state["images_processed_count"] += 1
                except Exception as e:
                    st.warning(f"Detection failed: {e}")
                    st.image(image, caption="Detected Objects (No Detection Available)", use_container_width=True)

# 🗂️ Multiple Image Upload with Enhanced UI
elif option == "📁 Batch Processing":
    st.markdown("<h2>📁 Batch Image Processing</h2>", unsafe_allow_html=True)
    
    image_files = st.file_uploader("Upload multiple images for batch analysis", 
                                  type=["jpg", "jpeg", "png"], 
                                  accept_multiple_files=True,
                                  help="Upload multiple images to batch process")
    
    if image_files:
        st.info(f"📁 Processing {len(image_files)} images...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image_file in enumerate(image_files):
            status_text.text(f"Processing {image_file.name}... ({i+1}/{len(image_files)})")
            progress_bar.progress((i + 1) / len(image_files))
            
            st.markdown(f"### 📸 Image {i+1}: {image_file.name}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(image_file)
                st.image(image, caption=f"Original - {image_file.name}", use_container_width=True)
            
            with col2:
                if model is None:
                    st.error("Model not loaded. Please ensure app/models/best.pt exists and is valid.")
                    st.image(image, caption=f"Original - {image_file.name} (No Detection Available)", use_container_width=True)
                else:
                    with st.spinner(f"🔍 Processing {image_file.name}..."):
                        result_img = predict_image(model, image)
                        
                        img_array = np.array(image.convert("RGB"))
                        results = model(img_array, device=DEVICE)
                        summary = get_detection_summary(results)
                        
                        st.image(result_img, caption=f"Detected - {image_file.name}", use_container_width=True)
                        st.markdown(f"**📊 Summary:** {summary}")
            
            st.markdown("---")
            st.session_state["images_processed_count"] += 1
        
        status_text.text("✅ Batch processing completed!")
        st.success(f"Successfully processed {len(image_files)} images!")

# 🖥️ Webcam Mode with Enhanced UI
elif option == "📹 Real-time Webcam":
    st.markdown("<h2>📹 Real-time Webcam Detection</h2>", unsafe_allow_html=True)
    st.info("🎥 Click 'Start Webcam Detection' to begin. Click 'Stop' to end.")

    if "webcam_active" not in st.session_state:
        st.session_state["webcam_active"] = False

    if not st.session_state["webcam_active"]:
        if st.button("🎥 Start Webcam Detection", key="start_webcam_portfolio"):
            st.session_state["webcam_active"] = True

    if st.session_state["webcam_active"]:
        if model is None:
            st.error("Model not loaded. Please ensure app/models/best.pt exists and is valid.")
            st.session_state["webcam_active"] = False
        else:
            predict_webcam(model)
            if st.button("🛑 Stop Webcam Detection", key="stop_webcam_portfolio"):
                st.session_state["webcam_active"] = False
                st.rerun()

# 📑 Violations Report Section
elif option == "📑 Violations Report":
    st.markdown("<h2>📑 Session Violations Report</h2>", unsafe_allow_html=True)
    st.markdown("<p>View all PPE violations detected during this session and export them to PDF or CSV.</p>", unsafe_allow_html=True)

    violations = st.session_state["logger"].get_violations()
    
    if not violations:
        st.info("✅ No violations recorded yet.")
    else:
        violation_counts = {}
        for v in violations:
            violation_type = v['violation_type']
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

        st.subheader("📊 Violation Summary")
        st.markdown(f"**Total Violations:** `{len(violations)}`")
        
        for v_type, count in violation_counts.items():
            st.markdown(f"- **{v_type}**: `{count}` violations")
        
        st.subheader("📸 Violation Logs")

        for i, v in enumerate(violations):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(v["image_path"], width=120, caption=f"{v['violation_type']}")
            with col2:
                st.markdown(f"**Type:** {v['violation_type']}  \n**Time:** {v['timestamp']}")
            st.markdown("---")

        col_gen_pdf, col_gen_csv, col_clear = st.columns([1, 1, 1])
        with col_gen_pdf:
            if st.button("📄 Generate PDF Report"):
                pdf = PDFReport()
                pdf_path = pdf.generate(violations)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name="ppe_violations_report.pdf")

        with col_gen_csv:
            if st.button("📊 Generate CSV Report"):
                csv = CSVReport()
                csv_path = csv.generate(violations)
                with open(csv_path, "rb") as f:
                    st.download_button("⬇️ Download CSV", f, file_name="ppe_violations_report.csv")

        with col_clear:
            if st.button("🗑️ Clear Violations"):
                st.session_state["logger"].clear()
                st.success("Violations cleared.")
                st.rerun()

# 📊 Footer with Portfolio Information
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🛡️ AI CCTV Surveillance System | Advanced PPE Detection for Construction Site Safety</p>
    <p>Built with YOLOv8, Streamlit, and Modern Web Technologies</p>
    <p>Portfolio Showcase Project | Professional AI/ML Implementation</p>
</div>
""", unsafe_allow_html=True)