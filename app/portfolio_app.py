import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import os
import sys
import numpy as np
import plotly.express as px
import time
from src.violation_logger import ViolationLogger
from src.report_generator import PDFReport, CSVReport
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference import load_model, predict_image, predict_webcam, DEVICE
from src.auth import create_db_and_table, verify_user, add_user, get_all_users, delete_user

# --- 1. INITIALIZATION ---
create_db_and_table()

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
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--dark-card); }
::-webkit-scrollbar-thumb { background: var(--primary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary-dark); }
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
.sidebar .sidebar-content {
    background: var(--dark-card) !important;
    border-right: 1px solid var(--dark-border) !important;
}
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
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}
.stButton > button {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    width: 100%;
    margin: 0.25rem 0;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    background: linear-gradient(135deg, var(--primary-dark), var(--primary));
}
div[data-testid="stButton"] > button {
    background: var(--dark-card);
    border: 1px solid var(--dark-border);
    color: var(--text-primary);
    font-weight: 500;
    padding: 1rem;
    border-radius: 12px;
    transition: all 0.3s ease;
    width: 100%;
    margin: 0.25rem 0;
    text-align: left;
}
div[data-testid="stButton"] > button:hover {
    background: var(--primary);
    border-color: var(--primary);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    color: white;
}
div[data-testid="stButton"] > button[aria-pressed="true"] {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    border-color: var(--primary);
    color: white;
    box-shadow: var(--shadow-xl);
}
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
.stRadio > label {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# App State and Model Loading
# -------------------------------------------------------------------

# Initialize session state for login and the rest of the app
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.logger = ViolationLogger()
    st.session_state.selected_nav = "dashboard" # Default page after login
    # Add other initializations
    st.session_state.images_processed_count = 0
    st.session_state.yolo_transformer = None
    st.session_state.last_violation_count = 0
    st.session_state.auto_refresh = True

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

# -------------------------------------------------------------------
# Page View Functions
# -------------------------------------------------------------------

def show_dashboard():
    """Renders the main dashboard page."""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">AI CCTV Surveillance System</h1>
        <p class="hero-subtitle">
            Advanced PPE Detection for Construction Site Safety<br>
            Portfolio-Ready | Real-Time | Professional UI/UX
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    violations = st.session_state["logger"].get_violations()
    
    # Performance Metrics Section
    st.markdown("### 📊 Performance Metrics")
    col1, col2, col3 = st.columns(3)
    fps = st.session_state["yolo_transformer"].fps if st.session_state["yolo_transformer"] else 0
    with col1:
        st.metric("Processing Speed (FPS)", f"{fps:.2f}")
    with col2:
        st.metric("Total Violations", f"{len(violations)}")
    with col3:
        st.metric("Recent Violations", f"{st.session_state['last_violation_count']}")

    # Chart Section
    st.markdown("### 📈 Real-time Violation Analysis")
    
    violation_counts = {v['violation_type']: 0 for v in violations}
    for v in violations:
        violation_counts[v['violation_type']] += 1
        
    chart_data = {
        'Violation Type': list(violation_counts.keys()),
        'Count': list(violation_counts.values())
    }

    if not chart_data['Violation Type']:
        st.info("No violations logged yet. Process an image or start the webcam to see live analytics.")
        chart_data = {
            'Violation Type': ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'],
            'Count': [0, 0, 0]
        }
    
    fig = px.bar(chart_data, x='Violation Type', y='Count',
                 title="Violation Types Breakdown",
                 color='Count',
                 color_continuous_scale=px.colors.sequential.Viridis)
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'), title_font_color='white',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'))
    st.plotly_chart(fig, use_container_width=True)

    # Feature Cards Section
    st.markdown("### 🚀 System Features")
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">⚡</div><div class="feature-title">Real-Time Detection</div>
            <div class="feature-description">Instant PPE and safety violation detection using YOLOv8.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div><div class="feature-title">Live Analytics</div>
            <div class="feature-description">Animated charts and statistics for system performance.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎨</div><div class="feature-title">Premium UI/UX</div>
            <div class="feature-description">Modern, responsive, and animated interface for portfolio showcase.</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🛡️</div><div class="feature-title">Multi-Class Detection</div>
            <div class="feature-description">Detects hardhats, masks, vests, people, vehicles, and more.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_single_image():
    """Renders the single image processing page."""
    st.markdown("""
    <div class="section-header"><h2 class="section-title">📷 Single Image Detection</h2></div>
    """, unsafe_allow_html=True)
    
    confidence_threshold = st.slider(
        "Detection Confidence Threshold", 0.1, 0.9, 0.25, 0.05,
        help="Lower values detect more objects but may include false positives."
    )
    
    image_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    
    if image_file:
        col1, col2 = st.columns(2)
        image = Image.open(image_file)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            if model:
                result_img = predict_image(model, image)
                st.image(result_img, caption="Detected Objects", use_container_width=True)
                st.session_state["images_processed_count"] += 1
            else:
                st.error("Model not loaded. Cannot perform detection.")


def show_batch_processing():
    """Renders the batch image processing page."""
    st.markdown("""
    <div class="section-header"><h2 class="section-title">📁 Batch Image Processing</h2></div>
    """, unsafe_allow_html=True)
    
    image_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if image_files:
        st.info(f"Processing {len(image_files)} images...")
        progress_bar = st.progress(0)
        
        for i, image_file in enumerate(image_files):
            st.markdown(f"### 📸 Image: {image_file.name}")
            col1, col2 = st.columns(2)
            image = Image.open(image_file)
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            with col2:
                if model:
                    result_img = predict_image(model, image)
                    st.image(result_img, caption="Detected", use_container_width=True)
                    st.session_state["images_processed_count"] += 1
                else:
                    st.error("Model not loaded.")
            st.markdown("---")
            progress_bar.progress((i + 1) / len(image_files))
        st.success("Batch processing completed!")


def show_webcam():
    """Renders the real-time webcam detection page."""
    st.markdown("""
    <div class="section-header"><h2 class="section-title">📹 Real-time Webcam Detection</h2></div>
    """, unsafe_allow_html=True)
    
    webcam_confidence = st.slider("Webcam Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    st.info("Click 'Start' to begin live detection from your webcam.")
    
    if model:
        predict_webcam(model, webcam_confidence)
    else:
        st.error("Model not loaded. Cannot start webcam detection.")


def show_violations_report():
    """Renders the violations report page."""
    st.markdown("""
    <div class="section-header">
        <h2 class="section-title">📑 Session Violations Report</h2>
        <p style="color: var(--text-secondary); text-align: center; margin-top: 0.5rem;">
            View all PPE violations detected during this session.
        </p>
    </div>
    """, unsafe_allow_html=True)

    violations = st.session_state["logger"].get_violations()
    
    if not violations:
        st.info("✅ No violations recorded yet.")
        return

    violations.sort(key=lambda x: x['timestamp'], reverse=True)
    
    st.subheader("📊 Violation Summary")
    violation_counts = {}
    for v in violations:
        v_type = v['violation_type']
        violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
    
    st.markdown(f"**Total Violations:** `{len(violations)}`")
    for v_type, count in violation_counts.items():
        st.markdown(f"- **{v_type}**: `{count}` violations")
    
    st.markdown("### 🛠️ Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_report = PDFReport()
        pdf_path = pdf_report.generate(violations)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download PDF Report", f, file_name="violations_report.pdf")
    with col2:
        csv_report = CSVReport()
        csv_path = csv_report.generate(violations)
        with open(csv_path, "rb") as f:
            st.download_button("📊 Download CSV Report", f, file_name="violations_report.csv")
    with col3:
        # --- Role-Based Action ---
        if st.session_state.role == 'Admin':
            if st.button("🗑️ Clear All Violations"):
                st.session_state["logger"].clear()
                st.success("All violations cleared!")
                st.rerun()
            
    st.markdown("---")
    st.subheader("📸 Violation Logs (Newest First)")

    for v in violations:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(v["image_path"], width=120, caption=v['violation_type'])
        with col2:
            st.markdown(f"**Type:** {v['violation_type']}<br>**Time:** {v['timestamp']}", unsafe_allow_html=True)
        st.markdown("---")

# --- NEW FEATURE: ADMIN PANEL PAGE ---
def show_admin_panel():
    st.markdown("<div class='section-header'><h2 class='section-title'>👑 Admin Panel</h2></div>", unsafe_allow_html=True)
    
    st.subheader("Add New User")
    with st.form("add_user_form", clear_on_submit=True):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", ["Manager", "Viewer"])
        submitted = st.form_submit_button("Add User")

        if submitted:
            if not new_username or not new_password:
                st.warning("Please fill in all fields.")
            elif add_user(new_username, new_password, new_role):
                st.success(f"User '{new_username}' added successfully as a {new_role}.")
            else:
                st.error(f"Username '{new_username}' already exists.")
    
    st.markdown("---")
    st.subheader("Existing Users")
    st.dataframe(get_all_users(), use_container_width=True)


# --- MAIN APP LOGIC (Login vs. Main App) ---

def show_login_page():
    """Displays the login form."""
    st.markdown("<h1 class='hero-title'>AI CCTV Surveillance System Login</h1>", unsafe_allow_html=True)
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            role = verify_user(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = role
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password")

def show_main_app():
    """Displays the main application after successful login."""
    # --- Sidebar and Navigation ---
    with st.sidebar:
        st.markdown(f"Welcome, **{st.session_state.username}**!")
        st.markdown(f"Role: **{st.session_state.role}**")
        if st.button("Logout"):
            # Clear session state on logout
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><h3 class='section-title'>🎛️ Navigation</h3></div>", unsafe_allow_html=True)

        # --- Role-Based Navigation ---
        nav_options = {
            "dashboard": {"icon": "📊", "title": "Dashboard", "desc": "Overview & Analytics"},
            "single_image": {"icon": "📷", "title": "Single Image", "desc": "Upload & Analyze"},
            "batch_processing": {"icon": "📁", "title": "Batch Processing", "desc": "Multiple Images"},
            "webcam": {"icon": "📹", "title": "Real-time Webcam", "desc": "Live Detection"},
            "violations_report": {"icon": "📑", "title": "Violations Report", "desc": "Detailed Logs"},
            "admin_panel": {"icon": "👑", "title": "Admin Panel", "desc": "Manage Users"} # New admin page
        }

        # Filter navigation based on role
        role_permissions = {
            "Admin": ["dashboard", "single_image", "batch_processing", "webcam", "violations_report", "admin_panel"],
            "Manager": ["dashboard", "single_image", "batch_processing", "webcam", "violations_report"],
            "Viewer": ["dashboard", "webcam"]
        }
        
        allowed_pages = role_permissions.get(st.session_state.role, [])

        for key in allowed_pages:
            item = nav_options[key]
            if st.button(f"{item['icon']} {item['title']}", key=f"nav_{key}", help=item['desc']):
                st.session_state.selected_nav = key
                st.rerun()

        st.markdown("---")
        st.markdown("<div class='section-header'><h3 class='section-title'>📈 Live Statistics</h3></div>", unsafe_allow_html=True)
        
        total_violations = len(st.session_state["logger"].get_violations())
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-value">{st.session_state['images_processed_count']}</div>
            <div class="stats-label">Images Processed</div>
        </div>
        <div class="stats-card">
            <div class="stats-value">{total_violations}</div>
            <div class="stats-label">Violations Logged</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

    # --- Main Content Area ---
    # Model Status Indicator
    if model:
        st.markdown(f"""
        <div class="status-success"><strong>✅ Model loaded successfully</strong>
        <small> | Path: app/models/best.pt | Device: {DEVICE}</small></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-warning"><strong>⚠️ Model not loaded.</strong> 
        <small>Please ensure app/models/best.pt exists.</small></div>
        """, unsafe_allow_html=True)

    page_router = {
        "dashboard": show_dashboard,
        "single_image": show_single_image,
        "batch_processing": show_batch_processing,
        "webcam": show_webcam,
        "violations_report": show_violations_report,
        "admin_panel": show_admin_panel # New admin page
    }
    
    # Execute the function for the selected page
    page_to_show = page_router.get(st.session_state.selected_nav)
    if page_to_show:
        page_to_show()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="background: var(--dark-card); border: 1px solid var(--dark-border); border-radius: 12px; padding: 2rem; text-align: center; margin-top: 3rem;">
        <h3 style="color: var(--text-primary); margin-bottom: 1rem;">🛡️ AI CCTV Surveillance System</h3>
        <p style="color: var(--text-secondary); margin-bottom: 0.5rem;">Built with YOLOv8 and Streamlit</p>
        <p style="color: var(--text-muted); font-size: 0.8rem;">Portfolio Showcase Project</p>
    </div>
    """, unsafe_allow_html=True)

# --- Final Check: Show Login Page or Main App ---
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()