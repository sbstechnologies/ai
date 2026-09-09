import os
import sys
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, time
from io import BytesIO

# Base directory setup for reliable deployment & path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from database.db_manager import DatabaseManager
from utils.face_recognizer import FaceRecognizer


# =========================================================
# INITIALIZATION & THEME DETECTION
# =========================================================
logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
user_placeholder_path = os.path.join(BASE_DIR, "assets", "user_placeholder.png")

st.set_page_config(
    page_title="Face AI Suite • Mobile Edition",
    page_icon=logo_path if os.path.exists(logo_path) else "📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

if "theme_setting" not in st.session_state:
    st.session_state["theme_setting"] = "Automatic (Based on Time)"

current_hour = datetime.now().hour
is_daytime = 6 <= current_hour < 18

if st.session_state["theme_setting"] == "☀️ Light Mode":
    active_theme = "light"
elif st.session_state["theme_setting"] == "🌙 Dark Mode":
    active_theme = "dark"
else:
    active_theme = "light" if is_daytime else "dark"

# Theme Color Palette Tokens
if active_theme == "light":
    theme_bg = "#f8fafc"
    theme_text = "#0f172a"
    theme_subtext = "#475569"
    sidebar_bg = "#ffffff"
    sidebar_border = "#e2e8f0"
    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    card_shadow = "0 4px 14px rgba(0, 0, 0, 0.05)"
    header_bg = "linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0369a1 100%)"
    header_title = "linear-gradient(90deg, #38bdf8 0%, #60a5fa 100%)"
    badge_bg = "rgba(2, 132, 199, 0.12)"
    badge_fg = "#0284c7"
    badge_border = "rgba(2, 132, 199, 0.3)"
    tab_inactive_bg = "#ffffff"
    tab_inactive_fg = "#475569"
    tab_active_bg = "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)"
    btn_bg = "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)"
    btn_fg = "#ffffff"
    input_bg = "#ffffff"
    input_border = "#cbd5e1"
    input_text = "#0f172a"
    hr_color = "#e2e8f0"
    company_box_bg = "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)"
    company_box_fg = "#f8fafc"
    company_box_link = "#38bdf8"
    mode_icon = "☀️"
    mode_label = "Daytime Light Mode"
else:
    theme_bg = "#0b192c"
    theme_text = "#f8fafc"
    theme_subtext = "#94a3b8"
    sidebar_bg = "#0f172a"
    sidebar_border = "rgba(56, 189, 248, 0.2)"
    card_bg = "#1e293b"
    card_border = "rgba(56, 189, 248, 0.25)"
    card_shadow = "0 6px 20px rgba(0, 0, 0, 0.4)"
    header_bg = "linear-gradient(135deg, #0b192c 0%, #1e3e62 60%, #0f172a 100%)"
    header_title = "linear-gradient(90deg, #38bdf8 0%, #818cf8 100%)"
    badge_bg = "rgba(56, 189, 248, 0.15)"
    badge_fg = "#38bdf8"
    badge_border = "rgba(56, 189, 248, 0.4)"
    tab_inactive_bg = "#1e293b"
    tab_inactive_fg = "#cbd5e1"
    tab_active_bg = "linear-gradient(135deg, #2563eb 0%, #0284c7 100%)"
    btn_bg = "linear-gradient(135deg, #2563eb 0%, #0284c7 100%)"
    btn_fg = "#ffffff"
    input_bg = "#1e293b"
    input_border = "#334155"
    input_text = "#f8fafc"
    hr_color = "#334155"
    company_box_bg = "linear-gradient(135deg, #0b192c 0%, #1e3e62 100%)"
    company_box_fg = "#f8fafc"
    company_box_link = "#38bdf8"
    mode_icon = "🌙"
    mode_label = "Nighttime Dark Mode"


# Inject Universal CSS with Sticky Top Navbar & Mobile Bottom Navigation Bar
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
        background-color: {theme_bg} !important;
        color: {theme_text} !important;
    }}

    /* Main Container Padding to account for Fixed Top Navbar & Fixed Bottom Menu */
    .main .block-container {{
        padding-top: 75px !important;
        padding-bottom: 95px !important;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 100% !important;
    }}

    /* Global Headings & Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stMarkdown, .stMarkdown p {{
        color: {theme_text} !important;
    }}

    label[data-testid="stWidgetLabel"], .stCaption, [data-testid="stCaptionContainer"] {{
        color: {theme_subtext} !important;
        font-weight: 600 !important;
    }}

    /* FIXED TOP NAVBAR */
    .fixed-top-navbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 58px;
        background: {header_bg};
        border-bottom: 2px solid #0284c7;
        z-index: 99998;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }}
    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .nav-title {{
        font-size: 1.15rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin: 0;
    }}
    .nav-status {{
        font-size: 0.78rem;
        font-weight: 700;
        color: #10b981 !important;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 12px;
    }}

    /* FIXED BOTTOM NAVIGATION MENU BAR FOR MOBILEN & ALL DEVICES */
    .bottom-nav-wrapper {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: {theme_bg};
        border-top: 2px solid #0284c7;
        z-index: 99999;
        padding: 8px 12px 12px 12px;
        box-shadow: 0 -8px 30px rgba(0, 0, 0, 0.4);
    }}

    div[data-testid="stRadio"] > div {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-around !important;
        align-items: center !important;
        width: 100% !important;
        gap: 6px !important;
    }}

    div[data-testid="stRadio"] label {{
        background: {tab_inactive_bg};
        color: {tab_inactive_fg} !important;
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 8px 12px;
        font-weight: 700;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        flex: 1;
        white-space: nowrap;
    }}

    div[data-testid="stRadio"] label[data-checked="true"] {{
        background: {tab_active_bg} !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
    }}

    div[data-testid="stRadio"] label[data-checked="true"] span,
    div[data-testid="stRadio"] label[data-checked="true"] p {{
        color: #ffffff !important;
    }}

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}

    section[data-testid="stSidebar"] *, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label {{
        color: {theme_text} !important;
    }}

    button[aria-label="Collapse sidebar"], button[aria-label="Expand sidebar"], [data-testid="stSidebarCollapseButton"] button {{
        color: {theme_text} !important;
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px !important;
    }}

    /* INPUT FIELDS & SELECTBOXES */
    input, select, textarea, div[data-baseweb="input"], div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border-color: {input_border} !important;
        border-radius: 10px !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
    }}

    hr, div[data-testid="stDivider"] {{
        border-color: {hr_color} !important;
    }}

    /* Metric Cards Grid */
    .theme-metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 14px;
        margin-bottom: 20px;
    }}
    .theme-metric-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: {card_shadow};
        transition: transform 0.2s ease;
    }}
    .theme-metric-card:hover {{
        transform: translateY(-2px);
    }}
    .theme-metric-label {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {theme_subtext} !important;
        text-transform: uppercase;
    }}
    .theme-metric-value {{
        font-size: clamp(1.5rem, 3.2vw, 2.1rem);
        font-weight: 800;
        color: {theme_text} !important;
        margin: 4px 0;
    }}

    /* Log Item Card */
    .theme-log-card {{
        background: {card_bg};
        border-left: 5px solid #0284c7;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: {card_shadow};
        border-top: 1px solid {card_border};
        border-right: 1px solid {card_border};
        border-bottom: 1px solid {card_border};
    }}

    /* Buttons */
    .stButton>button {{
        background: {btn_bg} !important;
        color: {btn_fg} !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        height: 48px !important;
    }}

    /* Company Box */
    .theme-company-box {{
        background: {company_box_bg};
        color: {company_box_fg} !important;
        border-radius: 16px;
        padding: 24px;
        margin-top: 24px;
        border-left: 6px solid #38bdf8;
        box-shadow: {card_shadow};
    }}
    .theme-company-box a {{
        color: {company_box_link} !important;
        text-decoration: none;
        font-weight: 600;
    }}
    .theme-company-box p, .theme-company-box h3, .theme-company-box h4, .theme-company-box span {{
        color: {company_box_fg} !important;
    }}

    /* Responsive Device Media Queries */
    @media (max-width: 768px) {{
        div[data-testid="stRadio"] label {{
            padding: 6px 6px !important;
            font-size: 0.72rem !important;
        }}
        .main .block-container {{
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }}
        .theme-metric-grid {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .fixed-top-navbar {{
            padding: 0 12px;
        }}
    }}

    @media (max-width: 480px) {{
        .theme-metric-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
""", unsafe_allow_html=True)


# =========================================================
# DB & MODEL INITIALIZATION
# =========================================================
REGISTERED_FACES_DIR = os.path.join(BASE_DIR, "database", "registered_faces")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "database", "snapshots")
os.makedirs(REGISTERED_FACES_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


def get_secret_val(key, default=""):
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def load_db_credentials():
    """
    Load database credentials from st.secrets if available,
    falling back to local default values.
    """
    secrets_db = {}
    try:
        if "mysql" in st.secrets:
            secrets_db = dict(st.secrets["mysql"])
        elif "database" in st.secrets:
            secrets_db = dict(st.secrets["database"])
    except Exception:
        secrets_db = {}

    db_type = secrets_db.get("db_type") or get_secret_val("db_type", "sqlite")
    host = secrets_db.get("host") or get_secret_val("mysql_host", "localhost")
    user = secrets_db.get("user") or get_secret_val("mysql_user", "root")
    password = secrets_db.get("password") or get_secret_val("mysql_pass", "")
    database = secrets_db.get("database") or get_secret_val("mysql_db", "face_ai")
    port = int(secrets_db.get("port") or get_secret_val("mysql_port", 3306))

    if (secrets_db.get("host") or get_secret_val("mysql_host")) and not get_secret_val("db_type") and "db_type" not in secrets_db:
        db_type = "mysql"

    return db_type, host, user, password, database, port


secret_db_type, secret_host, secret_user, secret_pass, secret_db, secret_port = load_db_credentials()

if "db_type" not in st.session_state:
    st.session_state["db_type"] = secret_db_type
if "mysql_host" not in st.session_state:
    st.session_state["mysql_host"] = secret_host
if "mysql_user" not in st.session_state:
    st.session_state["mysql_user"] = secret_user
if "mysql_pass" not in st.session_state:
    st.session_state["mysql_pass"] = secret_pass
if "mysql_db" not in st.session_state:
    st.session_state["mysql_db"] = secret_db
if "mysql_port" not in st.session_state:
    st.session_state["mysql_port"] = secret_port


@st.cache_resource
def get_db_instance(db_type, host, user, password, database, port=3306):
    sqlite_path = os.path.join(BASE_DIR, "database", "attendance.db")
    return DatabaseManager(
        db_type=db_type,
        host=host,
        user=user,
        password=password,
        database=database,
        sqlite_path=sqlite_path,
        port=port
    )


@st.cache_resource
def load_recognizer():
    model_path = os.path.join(BASE_DIR, "models", "face_model.pt")
    return FaceRecognizer(model_path=model_path)


try:
    db = get_db_instance(
        st.session_state["db_type"],
        st.session_state["mysql_host"],
        st.session_state["mysql_user"],
        st.session_state["mysql_pass"],
        st.session_state["mysql_db"],
        st.session_state.get("mysql_port", 3306)
    )
    if hasattr(db, "connection_warning") and db.connection_warning:
        st.warning(f"⚠️ **Database Notice:** {db.connection_warning}")
except Exception as db_err:
    st.error(f"❌ **Database Initialization Error:** {db_err}")
    sqlite_path = os.path.join(BASE_DIR, "database", "attendance.db")
    db = DatabaseManager(db_type="sqlite", sqlite_path=sqlite_path)

recognizer = load_recognizer()


# =========================================================
# FIXED TOP NAVBAR (HEADER APP BAR)
# =========================================================
now_time = datetime.now().strftime("%I:%M %p")

st.markdown(f"""
    <div class="fixed-top-navbar">
        <div class="nav-brand">
            {'<img src="assets/logo.png" width="30" style="border-radius:6px;">' if os.path.exists(logo_path) else '👤'}
            <span class="nav-title">Face AI Suite</span>
        </div>
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:0.8rem; font-weight:700; color:{theme_subtext};">{now_time}</span>
            <span class="nav-status">⚡ Live YOLO Engine</span>
        </div>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR / SIDE MENU SETTINGS
# =========================================================
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=110)

st.sidebar.markdown("<h3 style='margin:0;'>Face AI Config</h3>", unsafe_allow_html=True)

# Theme Selector in Sidebar
st.sidebar.markdown("#### 🌓 Theme Mode")
selected_theme_setting = st.sidebar.selectbox(
    "Theme Mode Selector",
    ["Automatic (Based on Time)", "☀️ Light Mode", "🌙 Dark Mode"],
    index=["Automatic (Based on Time)", "☀️ Light Mode", "🌙 Dark Mode"].index(st.session_state["theme_setting"])
)

if selected_theme_setting != st.session_state["theme_setting"]:
    st.session_state["theme_setting"] = selected_theme_setting
    st.rerun()

st.sidebar.caption(f"Active: {mode_icon} {mode_label}")

st.sidebar.divider()
st.sidebar.markdown("#### ⚙️ YOLO Controls")
conf_threshold = st.sidebar.slider("YOLO Detection Confidence", 0.10, 1.00, 0.50, 0.05)
tolerance = st.sidebar.slider("Face Distance Tolerance", 0.30, 0.70, 0.45, 0.02)
work_start_time_str = st.sidebar.time_input("Late Cutoff Time", value=time(9, 30))

st.sidebar.divider()
st.sidebar.markdown(f"""
<div style="font-size: 0.82rem; line-height: 1.5; color:{theme_subtext};">
    <b style="color:#0ea5e9;">Built by 🏬 SBS TECHNOLOGIES</b><br>
    💻 Software Company<br>
    📍 Erode, Tamil Nadu, India - 638004.<br>
    🌍 <a href="https://sbstechnologies.in" target="_blank" style="color:#0ea5e9;">sbstechnologies.in</a><br>
    📨 <a href="mailto:hr@sbstechnologies.in" style="color:#0ea5e9;">hr@sbstechnologies.in</a><br><br>
    <b>Babu Subramanian</b> (Founder & CEO)<br>
    📱 +91-81440 65688
</div>
""", unsafe_allow_html=True)


# =========================================================
# FIXED BOTTOM NAVIGATION MENU BAR (MOBILE & ALL DEVICES)
# =========================================================
st.markdown('<div class="bottom-nav-wrapper">', unsafe_allow_html=True)
selected_tab = st.radio(
    "Bottom Menu Navigation",
    [
        "🎥 Live Scan",
        "👤 Register",
        "👥 Members",
        "📊 Reports",
        "⚙️ Settings"
    ],
    index=0,
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# TAB 1: LIVE SCAN
# =========================================================
if selected_tab == "🎥 Live Scan":
    known_users = db.get_user_embeddings()
    summary = db.get_today_summary()

    st.markdown(f"""
        <div class="theme-metric-grid">
            <div class="theme-metric-card">
                <div class="theme-metric-label">Registered Members</div>
                <div class="theme-metric-value">{summary['total_users']}</div>
                <div style="font-size:0.75rem; color:{theme_subtext};">Active Profiles</div>
            </div>
            <div class="theme-metric-card" style="border-left:4px solid #10b981;">
                <div class="theme-metric-label" style="color:#10b981;">Present Today</div>
                <div class="theme-metric-value" style="color:#10b981;">{summary['present']}</div>
                <div style="font-size:0.75rem; color:{theme_subtext};">Checked In On-Time</div>
            </div>
            <div class="theme-metric-card" style="border-left:4px solid #f59e0b;">
                <div class="theme-metric-label" style="color:#f59e0b;">Late Today</div>
                <div class="theme-metric-value" style="color:#f59e0b;">{summary['late']}</div>
                <div style="font-size:0.75rem; color:{theme_subtext};">After {work_start_time_str.strftime('%H:%M')} Cutoff</div>
            </div>
            <div class="theme-metric-card" style="border-left:4px solid #ef4444;">
                <div class="theme-metric-label" style="color:#ef4444;">Absent Today</div>
                <div class="theme-metric-value" style="color:#ef4444;">{summary['absent']}</div>
                <div style="font-size:0.75rem; color:{theme_subtext};">Pending Check-Ins</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if len(known_users) == 0:
        st.warning("💡 No members registered yet. Use **'👤 Register'** in bottom menu to add people.")

    col_feed, col_log = st.columns([1.4, 1])

    with col_feed:
        st.markdown("### 📹 Live Feed Capture")
        source_mode = st.radio("Select Input Source", ["Webcam Snapshot", "Upload Image File"], horizontal=True)

        image_bgr = None
        if source_mode == "Webcam Snapshot":
            cam_photo = st.camera_input("Capture Camera Snapshot")
            if cam_photo:
                bytes_data = cam_photo.getvalue()
                image_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            uploaded_file = st.file_uploader("Upload Image File", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                bytes_data = uploaded_file.getvalue()
                image_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if image_bgr is not None:
            annotated_img, recognized_faces, face_count = recognizer.detect_and_recognize(
                image_bgr,
                known_users=known_users,
                conf_threshold=conf_threshold,
                tolerance=tolerance
            )

            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption=f"YOLO Face AI Analysis ({face_count} face(s) detected)", use_container_width=True)

    with col_log:
        st.markdown("### 📋 Recognition Stream")
        camera_name = st.text_input("Camera Location Tag", value="Cam-Main-Entry")

        if image_bgr is not None:
            db.save_detection(camera_name, face_count, conf_threshold)

            if recognized_faces:
                now = datetime.now()
                date_str = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")
                current_time_obj = now.time()

                status = "Late" if current_time_obj > work_start_time_str else "Present"

                for face in recognized_faces:
                    user_code = face["user_code"]
                    name = face["name"]
                    department = face["department"]
                    conf = face["confidence"]

                    if user_code != "UNKNOWN":
                        snapshot_filename = f"{date_str}_{user_code}_{now.strftime('%H%M%S')}.jpg"
                        snapshot_path = os.path.join(SNAPSHOTS_DIR, snapshot_filename)
                        cv2.imwrite(snapshot_path, image_bgr)

                        result_type, check_in_t = db.mark_attendance(
                            user_code=user_code,
                            name=name,
                            department=department,
                            date_str=date_str,
                            time_str=time_str,
                            status=status,
                            confidence=conf,
                            snapshot_path=snapshot_path
                        )

                        border_color = "#10b981" if status == "Present" else "#f59e0b"
                        st.markdown(f"""
                            <div class="theme-log-card" style="border-left-color:{border_color};">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h4 style="margin:0; font-size:1.05rem; color:{theme_text};">{name}</h4>
                                    <span style="background:rgba(16,185,129,0.2); color:#10b981; border:1px solid rgba(16,185,129,0.4); padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.78rem;">{status.upper()}</span>
                                </div>
                                <p style="margin:4px 0; font-size:0.85rem; color:{theme_subtext};">Code: <code>{user_code}</code> | Dept: {department}</p>
                                <p style="margin:4px 0; font-size:0.85rem; color:#0ea5e9;"><b>Match Score:</b> {conf*100:.1f}% | <b>Time:</b> {time_str}</p>
                            </div>
                        """, unsafe_allow_html=True)

                        if result_type == "marked":
                            st.success(f"✅ Attendance marked for {name}")
                        elif result_type == "updated":
                            st.info(f"ℹ️ Check-out updated for {name}")
                    else:
                        st.markdown(f"""
                            <div class="theme-log-card" style="border-left-color:#ef4444;">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <h4 style="margin:0; font-size:1.05rem; color:#ef4444;">Unknown Face</h4>
                                    <span style="background:rgba(239,68,68,0.2); color:#ef4444; border:1px solid rgba(239,68,68,0.4); padding:3px 8px; border-radius:6px; font-weight:700; font-size:0.75rem;">NOT REGISTERED</span>
                                </div>
                                <p style="margin:4px 0; font-size:0.85rem; color:{theme_subtext};">YOLO Conf: {face['yolo_conf']*100:.1f}%</p>
                            </div>
                        """, unsafe_allow_html=True)


# =========================================================
# TAB 2: REGISTER
# =========================================================
elif selected_tab == "👤 Register":
    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.markdown("### 📝 Member Information")
        user_code = st.text_input("User Code / Employee ID *", placeholder="e.g. EMP-101").strip().upper()
        name = st.text_input("Full Name *", placeholder="e.g. John Doe").strip()
        department = st.selectbox("Department", ["Engineering", "HR", "Sales", "Marketing", "Finance", "Management", "Students", "Other"])
        role = st.text_input("Role / Designation", placeholder="e.g. Software Engineer")
        email = st.text_input("Email Address", placeholder="e.g. john@example.com")

    with col_cam:
        st.markdown("### 📷 Face Photo Capture")
        reg_source = st.radio("Photo Source", ["Webcam Snapshot", "Upload File"], horizontal=True)

        reg_img_bgr = None
        if reg_source == "Webcam Snapshot":
            reg_cam = st.camera_input("Take Member Photo")
            if reg_cam:
                bytes_data = reg_cam.getvalue()
                reg_img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            reg_file = st.file_uploader("Upload Face Photo File", type=["jpg", "jpeg", "png"])
            if reg_file:
                bytes_data = reg_file.getvalue()
                reg_img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.divider()

    if st.button("💾 Save & Complete Registration", use_container_width=True, type="primary"):
        if not user_code or not name:
            st.error("⚠️ User Code / Employee ID and Full Name are required fields!")
        elif reg_img_bgr is None:
            st.error("⚠️ Please capture or upload a face photo for the member!")
        else:
            with st.spinner("Extracting YOLO Face & 128-d Feature Vector..."):
                embedding, face_crop, err_msg = recognizer.extract_single_face_embedding(reg_img_bgr, conf_threshold=conf_threshold)

                if err_msg:
                    st.error(f"❌ Registration Failed: {err_msg}")
                else:
                    photo_filename = f"{user_code}.jpg"
                    photo_path = os.path.join(REGISTERED_FACES_DIR, photo_filename)
                    cv2.imwrite(photo_path, reg_img_bgr)

                    success = db.add_user(
                        user_code=user_code,
                        name=name,
                        department=department,
                        role=role,
                        email=email,
                        embedding=embedding,
                        photo_path=photo_path
                    )

                    if success:
                        st.success(f"🎉 **Member '{name}' (`{user_code}`) successfully registered!**")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.image(cv2.cvtColor(reg_img_bgr, cv2.COLOR_BGR2RGB), caption="Captured Image", width=250)
                        with c2:
                            st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption="Extracted YOLO Face ROI", width=250)
                    else:
                        st.error("❌ Failed to save member to database.")


# =========================================================
# TAB 3: MEMBERS
# =========================================================
elif selected_tab == "👥 Members":
    df_users = db.get_all_users()

    if df_users.empty:
        st.info("No members registered in database yet.")
    else:
        st.markdown(f"### Total Registered Members: `{len(df_users)}`")
        search_query = st.text_input("🔍 Search member by Name, Code, or Department", "").strip().lower()

        filtered_df = df_users.copy()
        if search_query:
            filtered_df = filtered_df[
                filtered_df["name"].str.lower().str.contains(search_query) |
                filtered_df["user_code"].str.lower().str.contains(search_query) |
                filtered_df["department"].str.lower().str.contains(search_query)
            ]

        user_placeholder_path = os.path.join("assets", "user_placeholder.png")

        for _, row in filtered_df.iterrows():
            with st.container():
                col_img, col_details, col_action = st.columns([1, 3.5, 1])

                with col_img:
                    photo_p = row.get("photo_path", "")
                    if photo_p and os.path.exists(photo_p):
                        st.image(photo_p, width=100)
                    elif os.path.exists(user_placeholder_path):
                        st.image(user_placeholder_path, width=90)
                    else:
                        st.image("https://img.icons8.com/color/96/000000/user.png", width=90)

                with col_details:
                    st.markdown(f"#### {row['name']} <span style='font-size:0.85rem; color:#0ea5e9; font-weight:normal;'>(`{row['user_code']}`)</span>", unsafe_allow_html=True)
                    st.markdown(f"🏢 **Department:** {row['department']} | 💼 **Role:** {row['role']}")
                    st.markdown(f"📨 **Email:** {row['email']} | 📅 **Registered:** {row['created_at']}")

                with col_action:
                    if st.button(f"🗑️ Delete", key=f"del_{row['user_code']}"):
                        if db.delete_user(row['user_code']):
                            st.success(f"Deleted `{row['user_code']}`")
                            st.rerun()
                        else:
                            st.error("Failed to delete member.")
                st.divider()


# =========================================================
# TAB 4: REPORTS
# =========================================================
elif selected_tab == "📊 Reports":
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        date_from = st.date_input("From Date", value=datetime.now())
    with col_f2:
        date_to = st.date_input("To Date", value=datetime.now())
    with col_f3:
        dept_filter = st.selectbox("Department Filter", ["All", "Engineering", "HR", "Sales", "Marketing", "Finance", "Management", "Students", "Other"])

    search_user = st.text_input("Search by User ID or Name", "")

    df_att = db.get_attendance_records(
        date_from=date_from,
        date_to=date_to,
        department=dept_filter,
        user_code=search_user
    )

    if not df_att.empty:
        total_records = len(df_att)
        present_records = len(df_att[df_att["status"] == "Present"])
        late_records = len(df_att[df_att["status"] == "Late"])

        st.markdown(f"""
            <div class="theme-metric-grid">
                <div class="theme-metric-card">
                    <div class="theme-metric-label">TOTAL LOGS RECORDED</div>
                    <div class="theme-metric-value">{total_records}</div>
                </div>
                <div class="theme-metric-card" style="border-left:4px solid #10b981;">
                    <div class="theme-metric-label" style="color:#10b981;">PRESENT COUNT</div>
                    <div class="theme-metric-value" style="color:#10b981;">{present_records}</div>
                </div>
                <div class="theme-metric-card" style="border-left:4px solid #f59e0b;">
                    <div class="theme-metric-label" style="color:#f59e0b;">LATE COUNT</div>
                    <div class="theme-metric-value" style="color:#f59e0b;">{late_records}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📋 Attendance Records Table")

        df_display = df_att.copy()
        if "confidence" in df_display.columns:
            df_display["confidence"] = df_display["confidence"].apply(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")

        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("### 📥 Export Options")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            csv_data = df_att.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Export CSV Report",
                data=csv_data,
                file_name=f"Attendance_Report_{date_from}_to_{date_to}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with exp_col2:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_att.to_excel(writer, index=False, sheet_name="Attendance")
            st.download_button(
                label="📊 Export Excel Report (.xlsx)",
                data=buffer.getvalue(),
                file_name=f"Attendance_Report_{date_from}_to_{date_to}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    else:
        st.info("No attendance records found matching the specified criteria.")


# =========================================================
# TAB 5: SETTINGS & BRANDING
# =========================================================
elif selected_tab == "⚙️ Settings":
    st.markdown("### 🗄️ Database Engine Configuration")

    db_choice = st.radio("Select Database Engine", ["SQLite (Local / Built-in)", "MySQL Server"], index=0 if st.session_state["db_type"] == "sqlite" else 1)

    if db_choice == "SQLite (Local / Built-in)":
        st.session_state["db_type"] = "sqlite"
        st.success(f"Active Engine: SQLite Database (`{db.sqlite_path}`)")
    else:
        st.session_state["db_type"] = "mysql"
        st.session_state["mysql_host"] = st.text_input("MySQL Host", value=st.session_state["mysql_host"])
        st.session_state["mysql_port"] = st.number_input("MySQL Port", value=int(st.session_state.get("mysql_port", 3306)), min_value=1, max_value=65535)
        st.session_state["mysql_user"] = st.text_input("MySQL User", value=st.session_state["mysql_user"])
        st.session_state["mysql_pass"] = st.text_input("MySQL Password", value=st.session_state["mysql_pass"], type="password")
        st.session_state["mysql_db"] = st.text_input("MySQL Database Name", value=st.session_state["mysql_db"])

        if st.button("🔌 Test & Save Connection"):
            get_db_instance.clear()
            test_db = get_db_instance(
                st.session_state["db_type"],
                st.session_state["mysql_host"],
                st.session_state["mysql_user"],
                st.session_state["mysql_pass"],
                st.session_state["mysql_db"],
                st.session_state["mysql_port"]
            )
            if hasattr(test_db, "connection_warning") and test_db.connection_warning:
                st.warning(f"⚠️ {test_db.connection_warning}")
            else:
                st.success("Successfully connected to MySQL database!")

    st.divider()
    st.markdown(f"""
        <div class="theme-company-box">
            <h3 style="color:#38bdf8; margin-top:0;">Built by 🏬 SBS TECHNOLOGIES</h3>
            <p style="font-size:1.05rem; color:#93c5fd; margin:4px 0;">💻 <b>Software Company</b></p>
            <p style="margin:4px 0;">📍 <b>Address:</b> 1/166, Vallalar Street, Municipal Colony Road, Erode, Tamil Nadu, India - 638004.</p>
            <div style="margin: 12px 0;">
                <span>🌍 Web: <a href="https://sbstechnologies.in" target="_blank">https://sbstechnologies.in</a></span> |
                <span>📨 Email: <a href="mailto:hr@sbstechnologies.in">hr@sbstechnologies.in</a></span>
            </div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 16px 0;">
            <h4 style="color:#f8fafc; font-size:1.15rem; margin-bottom:6px;">👨‍💻 Founder & CEO</h4>
            <p style="font-size:1.1rem; color:#38bdf8; font-weight:700; margin:0;">Babu Subramanian</p>
            <p style="color:#93c5fd; font-size:0.9rem; margin-top:2px;">MSc (Software Engineering)., Microsoft Certified Professional.</p>
            <p style="margin:4px 0;">📨 Email: <a href="mailto:babu@sbstechnologies.in">babu@sbstechnologies.in</a></p>
            <p style="margin:4px 0;">📱 Mobile: +91-81440 65688 | +91-96985 29560 | +91-98435 44844 | ☎️ +91-424-355-3661</p>
        </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.markdown(f"""
<div style="text-align: center; color: {theme_subtext}; font-size: 0.88rem; padding: 10px 0 20px 0;">
    <b>Face AI Attendance System</b> • {mode_icon} {mode_label}<br>
    Built with ❤️ by 🏬 <b><a href="https://sbstechnologies.in" target="_blank" style="color: #0ea5e9;">SBS TECHNOLOGIES</a></b> | 📨 hr@sbstechnologies.in | 📱 +91-81440 65688
</div>
""", unsafe_allow_html=True)
