import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, time
from io import BytesIO

# Base directory setup for reliable deployment & path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logo_path = os.path.join(BASE_DIR, "assets", "logo.png")
user_placeholder_path = os.path.join(BASE_DIR, "assets", "user_placeholder.png")

st.set_page_config(
    page_title="Face AI Suite • Mobile Edition",
    page_icon=logo_path if os.path.exists(logo_path) else "📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

try:
    import cv2
except ImportError as e:
    st.error(
        f"⚠️ **OpenCV (`cv2`) Import Failed:** `{e}`\n\n"
        "**Solution for Streamlit Cloud / Linux Deployments:**\n"
        "1. Ensure `packages.txt` includes required system shared libraries: `libgl1`, `libglib2.0-0`, `libsm6`, `libxext6`, `ffmpeg`.\n"
        "2. Ensure `requirements.txt` specifies `opencv-python-headless>=4.8.0`."
    )
    st.stop()

from database.db_manager import DatabaseManager
from utils.face_recognizer import FaceRecognizer


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
    theme_bg = "#f1f5f9"
    theme_text = "#0f172a"
    theme_subtext = "#475569"
    sidebar_bg = "#ffffff"
    sidebar_border = "#cbd5e1"
    card_bg = "#ffffff"
    card_border = "#e2e8f0"
    card_shadow = "0 10px 25px -5px rgba(15, 23, 42, 0.08), 0 8px 10px -6px rgba(15, 23, 42, 0.04)"
    header_bg = "rgba(15, 23, 42, 0.95)"
    header_title = "linear-gradient(90deg, #38bdf8 0%, #60a5fa 100%)"
    badge_bg = "rgba(2, 132, 199, 0.12)"
    badge_fg = "#0284c7"
    badge_border = "rgba(2, 132, 199, 0.3)"
    tab_inactive_bg = "#ffffff"
    tab_inactive_fg = "#475569"
    tab_active_bg = "linear-gradient(135deg, #0284c7 0%, #2563eb 100%)"
    btn_bg = "linear-gradient(135deg, #0284c7 0%, #1d4ed8 100%)"
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
    theme_bg = "#070d17"
    theme_text = "#f8fafc"
    theme_subtext = "#94a3b8"
    sidebar_bg = "#0f172a"
    sidebar_border = "rgba(56, 189, 248, 0.2)"
    card_bg = "rgba(17, 24, 39, 0.85)"
    card_border = "rgba(56, 189, 248, 0.25)"
    card_shadow = "0 10px 30px rgba(0, 0, 0, 0.5)"
    header_bg = "rgba(7, 13, 23, 0.95)"
    header_title = "linear-gradient(90deg, #38bdf8 0%, #818cf8 100%)"
    badge_bg = "rgba(56, 189, 248, 0.15)"
    badge_fg = "#38bdf8"
    badge_border = "rgba(56, 189, 248, 0.4)"
    tab_inactive_bg = "#111827"
    tab_inactive_fg = "#94a3b8"
    tab_active_bg = "linear-gradient(135deg, #2563eb 0%, #0284c7 100%)"
    btn_bg = "linear-gradient(135deg, #2563eb 0%, #0284c7 100%)"
    btn_fg = "#ffffff"
    input_bg = "#111827"
    input_border = "#1f2937"
    input_text = "#f8fafc"
    hr_color = "#1f2937"
    company_box_bg = "linear-gradient(135deg, #0b1329 0%, #111e38 100%)"
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
        padding-top: 80px !important;
        padding-bottom: 105px !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
        max-width: 1400px !important;
        margin: 0 auto;
    }}

    /* Global Headings & Typography */
    h1, h2, h3, h4, h5, h6 {{
        color: {theme_text} !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }}

    p, span, div, .stMarkdown, .stMarkdown p {{
        color: {theme_text} !important;
    }}

    label[data-testid="stWidgetLabel"], .stCaption, [data-testid="stCaptionContainer"] {{
        color: {theme_subtext} !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        margin-bottom: 4px !important;
    }}

    /* FIXED TOP NAVBAR WITH GLASSMORPHISM */
    .fixed-top-navbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 62px;
        background: {header_bg};
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid rgba(56, 189, 248, 0.25);
        z-index: 99998;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }}

    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}

    .nav-title {{
        font-size: 1.2rem;
        font-weight: 800;
        background: {header_title};
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin: 0;
    }}

    .nav-status {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #10b981 !important;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
    }}

    .pulse-dot {{
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        animation: pulse-ring 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
    }}

    @keyframes pulse-ring {{
        0% {{
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }}
        70% {{
            box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
        }}
        100% {{
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }}
    }}

    /* FIXED BOTTOM NAVIGATION MENU BAR FOR MOBILE & DESKTOP */
    .bottom-nav-wrapper {{
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: {theme_bg};
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-top: 1px solid {card_border};
        z-index: 99999;
        padding: 10px 16px 14px 16px;
        box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.35);
    }}

    div[data-testid="stRadio"] > div {{
        display: flex !important;
        flex-direction: row !important;
        justify-content: space-around !important;
        align-items: center !important;
        width: 100% !important;
        gap: 8px !important;
    }}

    div[data-testid="stRadio"] label {{
        background: {tab_inactive_bg};
        color: {tab_inactive_fg} !important;
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 10px 14px;
        font-weight: 700;
        font-size: 0.88rem;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        flex: 1;
        white-space: nowrap;
    }}

    div[data-testid="stRadio"] label:hover {{
        border-color: #38bdf8 !important;
        transform: translateY(-1px);
    }}

    div[data-testid="stRadio"] label[data-checked="true"] {{
        background: {tab_active_bg} !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 18px rgba(2, 132, 199, 0.45);
        transform: translateY(-2px);
    }}

    div[data-testid="stRadio"] label[data-checked="true"] span,
    div[data-testid="stRadio"] label[data-checked="true"] p {{
        color: #ffffff !important;
        font-weight: 800 !important;
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
        border-radius: 10px !important;
    }}

    /* INPUT FIELDS & SELECTBOXES */
    input, select, textarea, div[data-baseweb="input"], div[data-baseweb="select"] {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        border-color: {input_border} !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }}

    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {{
        border-color: #38bdf8 !important;
        box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
    }}

    hr, div[data-testid="stDivider"] {{
        border-color: {hr_color} !important;
        margin: 1.5rem 0 !important;
    }}

    /* METRIC CARDS GRID */
    .theme-metric-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }}
    .theme-metric-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        border: 1px solid {card_border};
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: {card_shadow};
        transition: all 0.25s ease;
        position: relative;
        overflow: hidden;
    }}
    .theme-metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.2);
    }}
    .theme-metric-label {{
        font-size: 0.8rem;
        font-weight: 700;
        color: {theme_subtext} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .theme-metric-value {{
        font-size: clamp(1.8rem, 3.5vw, 2.4rem);
        font-weight: 800;
        color: {theme_text} !important;
        margin: 6px 0;
        letter-spacing: -0.03em;
    }}

    /* LOG ITEM CARD */
    .theme-log-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        border-left: 5px solid #0284c7;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: {card_shadow};
        border-top: 1px solid {card_border};
        border-right: 1px solid {card_border};
        border-bottom: 1px solid {card_border};
        transition: transform 0.2s ease;
    }}
    .theme-log-card:hover {{
        transform: translateX(3px);
    }}

    /* BUTTONS */
    .stButton>button {{
        background: {btn_bg} !important;
        color: {btn_fg} !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        height: 48px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.25s ease !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.45) !important;
    }}
    .stButton>button:active {{
        transform: translateY(0) !important;
    }}

    /* COMPANY BRANDING BOX */
    .theme-company-box {{
        background: {company_box_bg};
        color: {company_box_fg} !important;
        border-radius: 20px;
        padding: 28px;
        margin-top: 28px;
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 6px solid #38bdf8;
        box-shadow: {card_shadow};
        position: relative;
        overflow: hidden;
    }}
    .theme-company-box a {{
        color: {company_box_link} !important;
        text-decoration: none;
        font-weight: 600;
    }}
    .theme-company-box a:hover {{
        text-decoration: underline;
    }}
    .theme-company-box p, .theme-company-box h3, .theme-company-box h4, .theme-company-box span {{
        color: {company_box_fg} !important;
    }}

    /* RESPONSIVE BREAKPOINTS */
    @media (max-width: 768px) {{
        div[data-testid="stRadio"] label {{
            padding: 8px 6px !important;
            font-size: 0.76rem !important;
        }}
        .main .block-container {{
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }}
        .theme-metric-grid {{
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}
        .fixed-top-navbar {{
            padding: 0 14px;
        }}
        .nav-title {{
            font-size: 1.05rem;
        }}
    }}

    @media (max-width: 480px) {{
        .theme-metric-grid {{
            grid-template-columns: 1fr;
        }}
        div[data-testid="stRadio"] label {{
            font-size: 0.7rem !important;
            padding: 6px 4px !important;
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

if hasattr(recognizer, "face_recognition_available") and not recognizer.face_recognition_available:
    st.sidebar.warning(f"⚠️ **Engine Notice:** Face feature matching engine (dlib) is limited: {getattr(recognizer, 'face_recognition_error', '')}")



# =========================================================
# FIXED TOP NAVBAR (HEADER APP BAR)
# =========================================================
now_time = datetime.now().strftime("%I:%M %p")

st.markdown(f"""
    <div class="fixed-top-navbar">
        <div class="nav-brand">
            {'<img src="assets/logo.png" width="32" style="border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.2);">' if os.path.exists(logo_path) else '👤'}
            <span class="nav-title">Face AI Suite</span>
        </div>
        <div style="display:flex; align-items:center; gap:12px;">
            <span style="font-size:0.82rem; font-weight:700; color:{theme_subtext};">{now_time}</span>
            <div class="nav-status">
                <span class="pulse-dot"></span>
                <span>Live YOLO Engine</span>
            </div>
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
            <div class="theme-metric-card" style="border-top: 3px solid #0284c7;">
                <div class="theme-metric-label">👥 Registered Members</div>
                <div class="theme-metric-value">{summary['total_users']}</div>
                <div style="font-size:0.78rem; color:{theme_subtext}; font-weight:600;">Active Profiles</div>
            </div>
            <div class="theme-metric-card" style="border-top: 3px solid #10b981;">
                <div class="theme-metric-label" style="color:#10b981;">✅ Present Today</div>
                <div class="theme-metric-value" style="color:#10b981;">{summary['present']}</div>
                <div style="font-size:0.78rem; color:{theme_subtext}; font-weight:600;">Checked In On-Time</div>
            </div>
            <div class="theme-metric-card" style="border-top: 3px solid #f59e0b;">
                <div class="theme-metric-label" style="color:#f59e0b;">⏱️ Late Today</div>
                <div class="theme-metric-value" style="color:#f59e0b;">{summary['late']}</div>
                <div style="font-size:0.78rem; color:{theme_subtext}; font-weight:600;">After {work_start_time_str.strftime('%H:%M')} Cutoff</div>
            </div>
            <div class="theme-metric-card" style="border-top: 3px solid #ef4444;">
                <div class="theme-metric-label" style="color:#ef4444;">🚨 Absent Today</div>
                <div class="theme-metric-value" style="color:#ef4444;">{summary['absent']}</div>
                <div style="font-size:0.78rem; color:{theme_subtext}; font-weight:600;">Pending Check-Ins</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if len(known_users) == 0:
        st.warning("💡 No members registered yet. Use **'👤 Register'** in bottom menu to add people.")

    col_feed, col_log = st.columns([1.4, 1])

    with col_feed:
        st.markdown("### 📹 Live Feed & Multi-Device Camera")
        source_mode = st.radio(
            "Select Input Method",
            ["📷 Web Camera (Live Stream)", "📱 Mobile Native Camera / Upload"],
            horizontal=True,
            key="tab1_source_radio"
        )

        with st.expander("ℹ️ Camera Permission & Multi-Device Tips", expanded=False):
            st.markdown("""
            - **Desktop & Laptops**: Click **"Allow"** when browser prompts for Camera permissions.
            - **Mobile Phones (Android & iPhone)**: 
              - Select **📷 Web Camera** for live stream. Tap the **switch camera icon (↻)** in the camera frame to switch between **Front Selfie** and **Rear Camera**.
              - Select **📱 Mobile Native Camera / Upload** to snap a photo directly using your phone's native camera app.
            - **HTTPS Connection**: Camera requires a secure HTTPS connection (provided automatically on Streamlit Cloud).
            """)

        image_bgr = None
        if source_mode == "📷 Web Camera (Live Stream)":
            cam_photo = st.camera_input(
                "Capture Live Camera Frame",
                key="live_scan_camera_input",
                help="Click Take Photo to process face recognition on live frame."
            )
            if cam_photo:
                bytes_data = cam_photo.getvalue()
                image_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            uploaded_file = st.file_uploader(
                "Take Photo using Phone Camera or Upload Image File",
                type=["jpg", "jpeg", "png"],
                key="live_scan_file_uploader",
                help="On mobile devices, tapping this opens your device's native camera app."
            )
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
        st.markdown("### 📷 Member Photo Capture")
        reg_source = st.radio(
            "Photo Source Method",
            ["📷 Web Camera", "📱 Mobile Native Camera / File"],
            horizontal=True,
            key="tab2_source_radio"
        )

        with st.expander("ℹ️ Registration Camera Guidance", expanded=False):
            st.markdown("""
            - **Mobile Devices**: On Android/iOS, you can use **📷 Web Camera** or select **📱 Mobile Native Camera / File** to snap a photo directly using your device camera app.
            - **Front vs Back Camera**: Use the camera toggle button (↻) inside the camera view on smartphones to switch between front and rear lenses.
            """)

        reg_img_bgr = None
        if reg_source == "📷 Web Camera":
            reg_cam = st.camera_input(
                "Take Member Photo",
                key="reg_camera_input",
                help="Position member's face clearly in center of frame and click Take Photo."
            )
            if reg_cam:
                bytes_data = reg_cam.getvalue()
                reg_img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        else:
            reg_file = st.file_uploader(
                "Take Member Photo or Upload Image File",
                type=["jpg", "jpeg", "png"],
                key="reg_file_uploader",
                help="On smartphones & tablets, tap to take a photo using native camera."
            )
            if reg_file:
                bytes_data = reg_file.getvalue()
                reg_img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    st.divider()

    if st.button("💾 Save & Complete Registration", width="stretch", type="primary"):
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

        st.dataframe(df_display, width="stretch", hide_index=True)

        st.markdown("### 📥 Export Options")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            csv_data = df_att.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Export CSV Report",
                data=csv_data,
                file_name=f"Attendance_Report_{date_from}_to_{date_to}.csv",
                mime="text/csv",
                width="stretch"
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
                width="stretch"
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
