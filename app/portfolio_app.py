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
from src.inference import load_model, predict_image, predict_webcam, get_detection_summary, YOLOVideoTransformer, DEVICE
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
    box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
    color: white;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# App State and Model Loading
# -------------------------------------------------------------------

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.logger = ViolationLogger()
    st.session_state.selected_nav = "dashboard"
    st.session_state.images_processed_count = 0
    st.session_state.yolo_transformer = None
    st.session_state.last_violation_count = 0

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
    st.markdown("""<div class="hero-section"><h1 class="hero-title">AI CCTV Surveillance System</h1><p class="hero-subtitle">Advanced PPE Detection for Construction Site Safety</p></div>""", unsafe_allow_html=True)
    violations = st.session_state["logger"].get_violations()
    st.markdown("### 📊 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Violations", f"{len(violations)}")
    with col2:
        st.metric("Images Processed", f"{st.session_state['images_processed_count']}")
    # ... Rest of your dashboard code ...

def show_single_image():
    st.markdown("""<div class="section-header"><h2 class="section-title">📷 Single Image Detection</h2></div>""", unsafe_allow_html=True)
    image_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
    if image_file and model:
        image = Image.open(image_file)
        result_img = predict_image(model, image)
        st.image(result_img, caption="Detected Objects", use_container_width=True)
        st.session_state["images_processed_count"] += 1

def show_batch_processing():
    st.markdown("""<div class="section-header"><h2 class="section-title">📁 Batch Image Processing</h2></div>""", unsafe_allow_html=True)
    image_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if image_files and model:
        for image_file in image_files:
            image = Image.open(image_file)
            result_img = predict_image(model, image)
            st.image(result_img, caption=f"Detected: {image_file.name}", use_container_width=True)
            st.session_state["images_processed_count"] += 1

def show_webcam():
    st.markdown("""<div class="section-header"><h2 class="section-title">📹 Real-time Webcam Detection</h2></div>""", unsafe_allow_html=True)
    if model:
        predict_webcam(model, 0.25)
    else:
        st.error("Model not loaded.")

def show_violations_report():
    st.markdown("""<div class="section-header"><h2 class="section-title">📑 Session Violations Report</h2></div>""", unsafe_allow_html=True)
    violations = st.session_state["logger"].get_violations()
    if not violations:
        st.info("✅ No violations recorded yet.")
        return
    # ... Rest of your report logic ...
    if st.session_state.role == 'Admin':
        if st.button("🗑️ Clear All Violations"):
            st.session_state["logger"].clear()
            st.success("All violations cleared!")
            st.rerun()

# --- THIS IS THE CORRECTED FUNCTION ---
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
                st.success(f"User '{new_username}' added successfully.")
                st.rerun()
            else:
                st.error(f"Username '{new_username}' already exists.")
    
    st.markdown("---")
    st.subheader("Existing Users")
    
    users_df = get_all_users()

    # This code replaces the simple table with an interactive list
    # that includes the delete buttons.
    for index, user in users_df.iterrows():
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.write(f"**Username:** {user['username']}")
        with col2:
            st.write(f"**Role:** {user['role']}")
        with col3:
            # Add a delete button for every user EXCEPT the admin
            if user['username'] != 'admin':
                if st.button(f"🗑️ Delete", key=f"delete_{user['username']}"):
                    delete_user(user['username'])
                    st.success(f"User '{user['username']}' has been deleted.")
                    st.rerun()

# --- MAIN APP LOGIC ---
def show_login_page():
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
                st.rerun()
            else:
                st.error("Invalid username or password")

def show_main_app():
    with st.sidebar:
        st.markdown(f"Welcome, **{st.session_state.username}**!")
        st.markdown(f"Role: **{st.session_state.role}**")
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'><h3 class='section-title'>🎛️ Navigation</h3></div>", unsafe_allow_html=True)

        nav_options = {
            "dashboard": "📊 Dashboard", "single_image": "📷 Single Image",
            "batch_processing": "📁 Batch Processing", "webcam": "📹 Real-time Webcam",
            "violations_report": "📑 Violations Report", "admin_panel": "👑 Admin Panel"
        }
        role_permissions = {
            "Admin": list(nav_options.keys()),
            "Manager": ["dashboard", "single_image", "batch_processing", "webcam", "violations_report"],
            "Viewer": ["dashboard", "webcam"]
        }
        allowed_pages = role_permissions.get(st.session_state.role, [])
        for page_key in allowed_pages:
            if st.button(nav_options[page_key], key=f"nav_{page_key}"):
                st.session_state.selected_nav = page_key
                st.rerun()

    page_router = {
        "dashboard": show_dashboard, "single_image": show_single_image,
        "batch_processing": show_batch_processing, "webcam": show_webcam,
        "violations_report": show_violations_report, "admin_panel": show_admin_panel
    }
    
    page_to_show = page_router.get(st.session_state.selected_nav)
    if page_to_show:
        page_to_show()

# Final Check
if not st.session_state.logged_in:
    show_login_page()
else:
    show_main_app()