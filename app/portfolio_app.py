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

# 🎨 Premium Page Configuration
st.set_page_config(
    page_title="AI CCTV Surveillance System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 Premium CSS with Animations
st.markdown("""
<style>
:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --accent: #4facfe;
    --dark-bg: #0a0e1a;
    --light-bg: #f5f6fa;
    --card-bg-dark: rgba(255,255,255,0.03);
    --card-bg-light: rgba(0,0,0,0.03);
    --glass-bg-dark: rgba(255,255,255,0.07);
    --glass-bg-light: rgba(0,0,0,0.07);
    --text-primary-dark: #fff;
    --text-primary-light: #23272F;
    --text-secondary-dark: #b8c5d1;
    --text-secondary-light: #23272F;
    --border-color-dark: rgba(255,255,255,0.1);
    --border-color-light: rgba(0,0,0,0.1);
    --shadow-lg: 0 25px 50px -12px rgba(0,0,0,0.25);
    --shadow-xl: 0 35px 60px -15px rgba(0,0,0,0.3);
}
body, .main, .stApp {
    background: var(--dark-bg) !important;
    color: var(--text-primary-dark) !important;
    font-family: 'Inter', 'JetBrains Mono', sans-serif;
    transition: background 0.5s, color 0.5s;
}
[data-theme="light"] body, [data-theme="light"] .main, [data-theme="light"] .stApp {
    background: var(--light-bg) !important;
    color: var(--text-primary-light) !important;
}
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 10px; }
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: radial-gradient(circle at 20% 20%, var(--primary) 0%, transparent 60%),
                radial-gradient(circle at 80% 80%, var(--secondary) 0%, transparent 60%),
                radial-gradient(circle at 40% 60%, var(--accent) 0%, transparent 60%);
    opacity: 0.08;
    z-index: -1;
    pointer-events: none;
}
.hero-section { text-align: center; padding: 4rem 0 2rem 0; position: relative; }
.hero-title {
    font-size: clamp(3rem, 8vw, 6rem); font-weight: 900;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 1rem; line-height: 1.1;
    animation: slideInUp 1s cubic-bezier(.39,.575,.565,1) both;
}
.hero-subtitle {
    font-size: clamp(1.2rem, 3vw, 1.8rem); color: var(--text-secondary-dark);
    margin-bottom: 2rem; font-weight: 400;
    animation: slideInUp 1s cubic-bezier(.39,.575,.565,1) 0.2s both;
}
[data-theme="light"] .hero-title { -webkit-text-fill-color: #23272F; }
[data-theme="light"] .hero-subtitle { color: #23272F; }
@keyframes slideInUp { from { opacity: 0; transform: translateY(50px); } to { opacity: 1; transform: translateY(0); } }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 2rem; margin: 3rem 0; }
.stat-card {
    background: var(--card-bg-dark); backdrop-filter: blur(8px); border-radius: 16px;
    border: 1px solid var(--border-color-dark); padding: 2rem; text-align: center;
    transition: all 0.3s; position: relative; overflow: hidden;
    box-shadow: var(--shadow-lg);
}
[data-theme="light"] .stat-card { background: var(--card-bg-light); border: 1px solid var(--border-color-light); }
.stat-card:hover { background: var(--glass-bg-dark); transform: translateY(-8px) scale(1.02); box-shadow: var(--shadow-xl); }
[data-theme="light"] .stat-card:hover { background: var(--glass-bg-light); }
.stat-icon { font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(90deg, var(--primary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100%{transform:scale(1);} 50%{transform:scale(1.1);} }
.stat-value { font-size: 2.5rem; font-weight: 700; color: var(--text-primary-dark); margin-bottom: 0.5rem; }
[data-theme="light"] .stat-value { color: var(--text-primary-light); }
.stat-label { color: var(--text-secondary-dark); font-size: 1rem; font-weight: 500; }
[data-theme="light"] .stat-label { color: var(--text-secondary-light); }
.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 2rem; margin: 3rem 0; }
.feature-card {
    background: var(--card-bg-dark); backdrop-filter: blur(8px); border-radius: 20px;
    border: 1px solid var(--border-color-dark); padding: 2rem; transition: all 0.3s;
    position: relative; overflow: hidden; box-shadow: var(--shadow-lg);
}
[data-theme="light"] .feature-card { background: var(--card-bg-light); border: 1px solid var(--border-color-light); }
.feature-card:hover { background: var(--glass-bg-dark); transform: translateY(-10px) scale(1.03); box-shadow: var(--shadow-xl); }
[data-theme="light"] .feature-card:hover { background: var(--glass-bg-light); }
.feature-icon { font-size: 3rem; margin-bottom: 1rem; background: linear-gradient(90deg, var(--secondary), var(--accent)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.feature-title { font-size: 1.5rem; font-weight: 600; color: var(--text-primary-dark); margin-bottom: 1rem; }
[data-theme="light"] .feature-title { color: var(--text-primary-light); }
.feature-description { color: var(--text-secondary-dark); line-height: 1.6; }
[data-theme="light"] .feature-description { color: var(--text-secondary-light); }
/* Custom button-style navigation */
.sidebar-radio-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.sidebar-radio-item {
    background: rgba(255, 255, 255, 0.05);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    transition: all 0.2s ease-in-out;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 500;
    color: var(--text-color);
}
.sidebar-radio-item:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateX(5px);
}
.sidebar-radio-selected {
    border: 1px solid var(--accent);
    background: rgba(79, 172, 254, 0.1);
    color: var(--accent);
}
.stRadio > label {
    display: none;
}
</style>
""", unsafe_allow_html=True)


# Initialize counters and logger in session state
if "images_processed_count" not in st.session_state:
    st.session_state["images_processed_count"] = 0
if "logger" not in st.session_state:
    st.session_state["logger"] = ViolationLogger()
if "yolo_transformer" not in st.session_state:
    st.session_state["yolo_transformer"] = None
if "selected_option" not in st.session_state:
    st.session_state["selected_option"] = "📊 Dashboard"


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

# 🎛️ Enhanced Sidebar with Premium Design
with st.sidebar:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0;">🎛️ Detection Options</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom Radio Button replacement with HTML/CSS
    options = ["📊 Dashboard", "📷 Single Image", "📁 Batch Processing", "📹 Real-time Webcam", "📑 Violations Report"]
    icons = ["🧭", "🖼️", "🗂️", "📹", "📄"]
    
    for i, opt in enumerate(options):
        is_selected = " sidebar-radio-selected" if st.session_state["selected_option"] == opt else ""
        if st.markdown(f'<div class="sidebar-radio-item{is_selected}">{icons[i]} {opt}</div>', unsafe_allow_html=True):
            st.session_state["selected_option"] = opt
            st.rerun()

    st.markdown("---")
    
    # --- LIVE STATISTICS ---
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
        <h4 style="color: white; margin: 0;">📈 Live Statistics</h4>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamically get stats from the active webcam transformer and logger
    if st.session_state["yolo_transformer"] and hasattr(st.session_state["yolo_transformer"], 'fps'):
        images_processed = st.session_state["yolo_transformer"].processed_frames
        fps = st.session_state["yolo_transformer"].fps
    else:
        images_processed = st.session_state["images_processed_count"]
        fps = 0 # Default to 0 if not using webcam

    total_violations = len(st.session_state["logger"].get_violations())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Images Processed", f"{images_processed}", "+12%")
    with col2:
        st.metric("Violations Logged", f"{total_violations}", "+2.1%")
    
    st.markdown("---")
    
    # 🔍 Detection Classes Info
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
        <h4 style="color: white; margin: 0;">🔍 Detection Classes</h4>
    </div>
    """, unsafe_allow_html=True)
    
    classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
    
    for i, class_name in enumerate(classes):
        if "NO-" in class_name:
            st.markdown(f"🔴 {class_name} (Violation)")
        else:
            st.markdown(f"🟢 {class_name} (Compliant)")

    # 🚨 Live Violation Log Section
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
        <h4 style="color: white; margin: 0;">🚨 Live Violation Log</h4>
    </div>
    """, unsafe_allow_html=True)
    
    violations = st.session_state["logger"].get_violations()
    total_violations = len(violations)
    
    st.markdown(f"**Total Violations:** `{total_violations}`")
    
    violation_counts = {}
    for v in violations:
        violation_type = v['violation_type']
        violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

    for v_type, count in violation_counts.items():
        st.markdown(f"- **{v_type}**: `{count}`")


# Use the selected option to display the correct page content
option = st.session_state["selected_option"]
if option == "📊 Dashboard":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">AI CCTV Surveillance System</h1>
        <p class="hero-subtitle">
            Advanced PPE Detection for Construction Site Safety<br>
            Portfolio-Ready | Real-Time | Professional UI/UX
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamically get violations for the dashboard chart
    violations = st.session_state["logger"].get_violations()
    
    # Create a DataFrame for the chart data
    violation_counts = {}
    for v in violations:
        violation_type = v['violation_type']
        violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

    chart_data = {
        'Violation Type': list(violation_counts.keys()),
        'Count': list(violation_counts.values())
    }

    # Dynamically get FPS
    if st.session_state["yolo_transformer"] and hasattr(st.session_state["yolo_transformer"], 'fps'):
        fps = st.session_state["yolo_transformer"].fps
    else:
        fps = 0 # Default to 0 if not using webcam

    st.markdown(f"**Performance Metrics**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Processing Speed (FPS)", f"{fps:.2f}")
    with col2:
        st.metric("Total Violations", f"{len(violations)}")

    if not chart_data['Violation Type']:
        st.info("No violations logged yet. Start a session to see real-time data.")
    else:
        fig = px.bar(chart_data, x='Violation Type', y='Count',
                     title="Real-time Violation Analysis",
                     color='Count',
                     color_continuous_scale='viridis')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Real-Time Detection</div>
            <div class="feature-description">Instant PPE and safety violation detection using YOLOv8.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Live Analytics</div>
            <div class="feature-description">Animated charts and statistics for system performance.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Premium UI/UX</div>
            <div class="feature-description">Modern, responsive, and animated interface for portfolio showcase.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🛡️</div>
            <div class="feature-title">Multi-Class Detection</div>
            <div class="feature-description">Detects hardhats, masks, vests, people, vehicles, and more.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 📷 Single Image Upload with Enhanced UI
elif option == "📷 Single Image":
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">📷 Single Image Detection</h2>
    </div>
    """, unsafe_allow_html=True)
    
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
                    # Draw boxes for display
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
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">📁 Batch Image Processing</h2>
    </div>
    """, unsafe_allow_html=True)
    
    image_files = st.file_uploader("Upload multiple images for batch analysis", 
                                  type=["jpg", "jpeg", "png"], 
                                  accept_multiple_files=True,
                                  help="Upload multiple images to batch process")
    
    if image_files:
        st.info(f"📁 Processing {len(image_files)} images...")
        
        # Progress bar
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
                        
                        # Get detection summary
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
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">📹 Real-time Webcam Detection</h2>
    </div>
    """, unsafe_allow_html=True)

    st.info("🎥 Click 'Start Webcam Detection' to begin. Click 'Stop' to end.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
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
    st.markdown("""
    <div style="background: rgba(255,255,255,0.1); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">📑 Session Violations Report</h2>
        <p style="color: #ccc; text-align: center;">View all PPE violations detected during this session and export them to PDF or CSV.</p>
    </div>
    """, unsafe_allow_html=True)

    violations = st.session_state["logger"].get_violations()
    
    if not violations:
        st.info("✅ No violations recorded yet.")
    else:
        # Display violation summary
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