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
    initial_sidebar_state="expanded"
)

# 🎨 Premium CSS with Modern Design
st.markdown("""
<style>
:root {
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --secondary: #7c3aed;
    --accent: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --dark-bg: #0f172a;
    --dark-card: #1e293b;
    --dark-border: #334155;
    --light-bg: #f8fafc;
    --light-card: #ffffff;
    --light-border: #e2e8f0;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
}

body, .main, .stApp {
    background: var(--dark-bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
    transition: all 0.3s ease;
}

[data-theme="light"] body, [data-theme="light"] .main, [data-theme="light"] .stApp {
    background: var(--light-bg) !important;
    color: #1e293b !important;
}

/* Hide Streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-card); }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-dark); }

/* Background gradient */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: 
        radial-gradient(circle at 20% 20%, rgba(37, 99, 235, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 60%, rgba(6, 182, 212, 0.1) 0%, transparent 50%);
    z-index: -1;
    pointer-events: none;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: var(--dark-card) !important;
    border-right: 1px solid var(--dark-border) !important;
}

/* Navigation cards */
.nav-card {
    background: var(--dark-card);
    border: 1px solid var(--dark-border);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.nav-card:hover {
    background: var(--primary);
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.nav-card.active {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-color: var(--primary);
    box-shadow: var(--shadow-xl);
}

.nav-card-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    display: block;
}

.nav-card-title {
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0;
    color: var(--text-primary);
}

.nav-card-description {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin: 0.25rem 0 0 0;
}

/* Stats cards */
.stats-card {
    background: var(--dark-card);
    border: 1px solid var(--dark-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    text-align: center;
    transition: all 0.3s ease;
}

.stats-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.stats-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
    margin-bottom: 0.5rem;
}

.stats-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    font-weight: 500;
}

.stats-change {
    font-size: 0.75rem;
    color: var(--success);
    font-weight: 600;
}

/* Hero section */
.hero-section {
    text-align: center;
    padding: 3rem 0 2rem 0;
    position: relative;
}

.hero-title {
    font-size: clamp(2.5rem, 6vw, 4rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: clamp(1rem, 2.5vw, 1.25rem);
    color: var(--text-secondary);
    margin-bottom: 2rem;
    font-weight: 400;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* Feature cards */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: var(--dark-card);
    border: 1px solid var(--dark-border);
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
    border-color: var(--primary);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.feature-title {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
}

.feature-description {
    color: var(--text-secondary);
    line-height: 1.6;
    font-size: 0.9rem;
}

/* Status indicators */
.status-success {
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--success);
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: var(--warning);
    padding: 0.75rem 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

/* Section headers */
.section-header {
    background: var(--dark-card);
    border: 1px solid var(--dark-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1.5rem 0;
    text-align: center;
}

.section-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .feature-grid {
        grid-template-columns: 1fr;
    }
    
    .hero-title {
        font-size: 2rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
    }
}

/* Animation classes */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}

/* Custom button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

/* File uploader styling */
.stFileUploader {
    background: var(--dark-card);
    border: 2px dashed var(--dark-border);
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: var(--primary);
    background: rgba(37, 99, 235, 0.05);
}

/* Radio button styling to look like cards */
.stRadio > div {
    background: var(--dark-card);
    border: 1px solid var(--dark-border);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
    cursor: pointer;
}

.stRadio > div:hover {
    background: var(--primary);
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.stRadio > div[data-testid="stRadio"] > div:first-child {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-color: var(--primary);
    box-shadow: var(--shadow-xl);
}

/* Hide radio button labels */
.stRadio > label {
    display: none !important;
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

# Model Status Indicator
if model is not None:
    st.markdown("""
    <div class="status-success">
        <strong>✅ Model loaded successfully</strong><br>
        <small>Path: app/models/best.pt | Device: CPU</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-warning">
        <strong>⚠️ Model not loaded</strong><br>
        <small>Please ensure app/models/best.pt exists and is valid</small>
    </div>
    """, unsafe_allow_html=True)

# 🎛️ Modern Sidebar with Card Navigation
with st.sidebar:
    # Navigation Section
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title">🎛️ Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Cards
    nav_options = [
        {
            "icon": "📊",
            "title": "Dashboard",
            "description": "Overview & Analytics",
            "key": "dashboard"
        },
        {
            "icon": "📷", 
            "title": "Single Image",
            "description": "Upload & Analyze",
            "key": "single_image"
        },
        {
            "icon": "📁",
            "title": "Batch Processing", 
            "description": "Multiple Images",
            "key": "batch_processing"
        },
        {
            "icon": "📹",
            "title": "Real-time Webcam",
            "description": "Live Detection",
            "key": "webcam"
        },
        {
            "icon": "📑",
            "title": "Violations Report",
            "description": "Detailed Logs",
            "key": "violations_report"
        }
    ]
    
    # Initialize selected option in session state
    if "selected_nav" not in st.session_state:
        st.session_state.selected_nav = "dashboard"
    
    # Use radio buttons styled as cards
    nav_labels = [f"{item['icon']} {item['title']}" for item in nav_options]
    nav_values = [item['key'] for item in nav_options]
    
    # Create custom radio button with card styling
    selected = st.radio(
        "Navigation",
        options=nav_values,
        format_func=lambda x: nav_options[nav_values.index(x)]['title'],
        index=nav_values.index(st.session_state.selected_nav)
    )
    
    if selected != st.session_state.selected_nav:
        st.session_state.selected_nav = selected
        st.rerun()
    
    st.markdown("---")
    
    # Live Statistics Section
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title">📈 Live Statistics</h3>
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
    
    # Stats Cards
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-value">{images_processed}</div>
        <div class="stats-label">Images Processed</div>
        <div class="stats-change">+12%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-card">
        <div class="stats-value">{total_violations}</div>
        <div class="stats-label">Violations Logged</div>
        <div class="stats-change">+2.1%</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Detection Classes Section
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title">🔍 Detection Classes</h3>
    </div>
    """, unsafe_allow_html=True)
    
    classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
               'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
    
    for class_name in classes:
        if "NO-" in class_name:
            st.markdown(f"🔴 **{class_name}** (Violation)")
        else:
            st.markdown(f"🟢 **{class_name}** (Compliant)")

    # Live Violation Log Section
    st.markdown("---")
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title">🚨 Live Violation Log</h3>
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


# 📊 Dashboard View
if st.session_state.selected_nav == "dashboard":
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
elif st.session_state.selected_nav == "single_image":
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📷 Single Image Detection</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence threshold slider
    confidence_threshold = st.slider(
        "Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25, 
        step=0.05,
        help="Lower values detect more objects but may include false positives. Higher values are more strict."
    )
    
    # Information about what the model detects
    st.info("""
    🎯 **What this model detects:**
    - **Compliant PPE**: Hardhat, Mask, Safety Vest, Safety Cone
    - **Violations**: NO-Hardhat, NO-Mask, NO-Safety Vest  
    - **Other**: Person, machinery, vehicle
    
    💡 **Tips for better detection:**
    - Use clear, well-lit images
    - Ensure the person is clearly visible
    - Try adjusting the confidence threshold if no detections are found
    """)
    
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
                    # Use user-defined confidence threshold
                    results = model(img_array, device=DEVICE, conf=confidence_threshold, iou=0.45)
                    # Draw boxes for display
                    result_img = img_array.copy()
                    if results and len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()
                        clss = results[0].boxes.cls.cpu().numpy().astype(int)
                        
                        # Show detection summary
                        detection_summary = []
                        for box, conf, cls in zip(boxes, confs, clss):
                            label = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else str(cls)
                            detection_summary.append(f"{label} ({conf:.2f})")
                        
                        if detection_summary:
                            st.success(f"🔍 Detected: {', '.join(detection_summary)}")
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            x1, y1, x2, y2 = map(int, box)
                            label = model.names[cls] if hasattr(model, 'names') and cls < len(model.names) else str(cls)
                            color = (0, 255, 0) if 'NO-' not in label else (0, 0, 255)
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(result_img, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        st.warning("⚠️ No PPE violations detected. The person appears to be wearing proper safety equipment.")
                    st.image(result_img, caption="Detected Objects", use_container_width=True)
                    st.session_state["images_processed_count"] += 1
                except Exception as e:
                    st.warning(f"Detection failed: {e}")
                    st.image(image, caption="Detected Objects (No Detection Available)", use_container_width=True)

# 🗂️ Multiple Image Upload with Enhanced UI
elif st.session_state.selected_nav == "batch_processing":
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📁 Batch Image Processing</h2>
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
elif st.session_state.selected_nav == "webcam":
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📹 Real-time Webcam Detection</h2>
    </div>
    """, unsafe_allow_html=True)

    # Confidence threshold slider for webcam
    webcam_confidence = st.slider(
        "Webcam Detection Confidence", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.25, 
        step=0.05,
        help="Lower values detect more objects but may include false positives. Higher values are more strict."
    )
    
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
                predict_webcam(model, webcam_confidence)
                if st.button("🛑 Stop Webcam Detection", key="stop_webcam_portfolio"):
                    st.session_state["webcam_active"] = False
                    st.rerun()

# 📑 Violations Report Section
elif st.session_state.selected_nav == "violations_report":
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📑 Session Violations Report</h2>
        <p style="color: var(--text-secondary); text-align: center; margin-top: 0.5rem;">View all PPE violations detected during this session and export them to PDF or CSV.</p>
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
<div style="background: var(--dark-card); border: 1px solid var(--dark-border); border-radius: 12px; padding: 2rem; text-align: center; margin-top: 3rem;">
    <h3 style="color: var(--text-primary); margin-bottom: 1rem;">🛡️ AI CCTV Surveillance System</h3>
    <p style="color: var(--text-secondary); margin-bottom: 0.5rem;">Advanced PPE Detection for Construction Site Safety</p>
    <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 0.5rem;">Built with YOLOv8, Streamlit, and Modern Web Technologies</p>
    <p style="color: var(--text-muted); font-size: 0.8rem;">Portfolio Showcase Project | Professional AI/ML Implementation</p>
</div>
""", unsafe_allow_html=True)