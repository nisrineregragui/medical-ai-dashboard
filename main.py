import streamlit as st
import os

st.set_page_config(
    page_title="Medical AI Diagnosis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Sidebar
with st.sidebar:
    # Brand logo
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-logo-wrap">🏥</div>
        <div class="brand-name">Medical AI</div>
        <div class="brand-tagline">Diagnosis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # Navigation
    PAGES = [
        ("🏠", "Dashboard",                   "Dashboard"),
        ("🎗️", "Breast Cancer Classification", "Breast Cancer"),
        ("🫁", "Pneumonia Detection",           "Pneumonia"),
        ("📊", "Model Performance",             "Performance"),
        ("ℹ️", "About",                         "About"),
    ]

    for icon, label, key in PAGES:
        is_active = st.session_state.current_page == key
        active_css = (
            "background:#eeecff !important;"
            "color:#6c63ff !important;"
            "font-weight:600 !important;"
            "border-left:3px solid #6c63ff !important;"
        ) if is_active else ""

        st.markdown(
            f'<style>[data-testid="stSidebar"] button#btn_{key} {{ {active_css} }}</style>',
            unsafe_allow_html=True,
        )
        if st.button(f"{icon}  {label}", key=f"btn_{key}", use_container_width=True):
            st.session_state.current_page = key
            st.rerun()

    # About the project box
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-about">
        <strong>About the Project</strong>
        AI-powered medical diagnosis system using deep learning models for breast cancer
        classification and pneumonia detection.
    </div>
    """, unsafe_allow_html=True)

    # Tech stack
    st.markdown("""
    <div style="margin:10px 12px 0;">
        <div style="font-size:11px;font-weight:600;color:#8a94a6;margin-bottom:6px;">Tech Stack</div>
        <div class="tech-stack-row">
            <div class="tech-badge">🐍 Python</div>
            <div class="tech-badge">🔥 PyTorch</div>
            <div class="tech-badge">🤖 sklearn</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Page Router
page = st.session_state.current_page

if page == "Dashboard":
    from views.dashboard import show
    show()
elif page == "Breast Cancer":
    st.title("🎗️ Breast Cancer Classification")
    st.info("Full page coming soon. Use the Dashboard for predictions.")
elif page == "Pneumonia":
    st.title("🫁 Pneumonia Detection")
    st.info("Full page coming soon. Use the Dashboard for predictions.")
elif page == "Performance":
    st.title("📊 Model Performance")
    st.info("Performance metrics page coming soon.")
elif page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    ### Medical AI Diagnosis Dashboard
    This dashboard provides AI-powered diagnostic tools for:
    - **Breast Cancer Classification** using an MLP model trained on the Wisconsin Breast Cancer Dataset
    - **Pneumonia Detection** using a FlexCNN model trained on PneumoniaMNIST
    
    > ⚠️ For educational and research purposes only. Not a substitute for medical advice.
    """)
