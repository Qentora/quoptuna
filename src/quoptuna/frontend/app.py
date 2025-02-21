import sys
from pathlib import Path
import base64

import streamlit as st
from streamlit.web import cli as stcli

from quoptuna.frontend.main_page import main_page
from quoptuna.frontend.sidebar import handle_sidebar
from quoptuna.frontend.support import (
    initialize_session_state,
    update_plot,
)

# Get the absolute path to the assets directory
ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.png"


def get_base64_encoded_image(image_path):
    """Get base64 encoded image."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


st.set_page_config(
    page_title="QuOptuna",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/Qentora/quoptuna/issues",
        "Report a bug": "https://github.com/Qentora/quoptuna/issues/new",
        "About": "Quoptuna: Optimizing Quantum Models and generating Governance Reports",
    },
)

# Apply custom CSS for enhanced dark theme with purple accents and hover effects
st.markdown(
    """
    <style>
    /* Global styles */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Header section */
    .header-section {
        text-align: center;
        background: linear-gradient(180deg, rgba(14,17,23,0) 0%, rgba(155,89,182,0.05) 100%);
        border-radius: 15px;
        margin: 0 0 1rem 0;
    }
    
    /* Logo container */
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        padding: 1rem;
    }
    
    .logo-container img {
        max-width: 120px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }
    
    .logo-container img:hover {
        transform: scale(1.05);
    }
    
    /* Title styles */
    div.main-title {
        font-size: 2.2em;
        background: linear-gradient(45deg, #9b59b6, #3498db);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        font-weight: 600;
        letter-spacing: 0.5px;
        font-family: 'Helvetica Neue', sans-serif;
        line-height: 1.2;
    }
    
    div.description {
        font-size: 1.1em;
        color: #a0a0a0;
        text-align: center;
        margin: 0;
        max-width: 800px;
        line-height: 1.4;
    }
    
    /* Button styles */
    .stButton>button {
        background: linear-gradient(135deg, #9b59b6, #8e44ad);
        color: white;
        border: none;
        padding: 8px 20px;
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        background: linear-gradient(135deg, #8e44ad, #9b59b6);
    }
    
    /* Tab styles */
    .stTabs {
        background-color: rgba(26, 31, 44, 0.5);
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0;
        border: 1px solid rgba(155, 89, 182, 0.1);
    }
    
    .stTab {
        color: #9b59b6;
        font-weight: 500;
        padding: 10px 20px;
        transition: all 0.3s ease;
        border-radius: 8px;
        margin: 0 5px;
    }
    
    .stTab:hover {
        color: #8e44ad;
        background-color: rgba(155, 89, 182, 0.1);
    }
    
    /* Sidebar styles */
    section[data-testid="stSidebar"] {
        background-color: #171b26;
        border-right: 1px solid rgba(155, 89, 182, 0.1);
    }
    
    /* Input field styles */
    .stTextInput>div>div>input {
        background-color: #1a1f2c;
        border: 1px solid rgba(155, 89, 182, 0.2);
        color: #e0e0e0;
        border-radius: 8px;
        padding: 8px 12px;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #9b59b6;
        box-shadow: 0 0 0 1px #9b59b6;
        background-color: #1e2433;
    }
    
    /* Select box styles */
    .stSelectbox>div>div>select {
        background-color: #1a1f2c;
        border: 1px solid rgba(155, 89, 182, 0.2);
        color: #e0e0e0;
        border-radius: 8px;
        padding: 8px 12px;
    }
    
    /* Card/container styles */
    .stMarkdown {
        background-color: rgba(26, 31, 44, 0.5);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        border: 1px solid rgba(155, 89, 182, 0.1);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #9b59b6, #8e44ad);
    }
    
    /* File uploader */
    .stUploadedFile {
        background-color: rgba(26, 31, 44, 0.5) !important;
        border: 1px solid rgba(155, 89, 182, 0.2) !important;
        border-radius: 8px !important;
    }
    
    /* Dataframe/table styles */
    .stDataFrame {
        background-color: rgba(26, 31, 44, 0.5);
        border-radius: 8px;
        border: 1px solid rgba(155, 89, 182, 0.1);
    }
    
    .stDataFrame th {
        background-color: rgba(155, 89, 182, 0.1);
        color: #9b59b6;
    }
    
    /* Tooltip styles */
    .stTooltip {
        background-color: #1a1f2c !important;
        border: 1px solid rgba(155, 89, 182, 0.2) !important;
        color: #e0e0e0 !important;
    }
    
    /* Remove default streamlit margins */
    .block-container {
        padding-top: 0;
        padding-bottom: 0;
        margin-top: 0;
    }
    
    div[data-testid="stToolbar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Display header section with logo and title
st.markdown(
    f"""
    <div class="header-section">
        <div class="logo-container">
            <img src="data:image/png;base64,{get_base64_encoded_image(LOGO_PATH)}" alt="Quoptuna Logo">
            <div>
                <div class="main-title">Quoptuna: QML Optimization and Governance</div>
                <div class="description">Welcome to Quoptuna! This app helps you optimize quantum models and generate governance reports. Explore different optimization strategies and visualize results in real-time.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def main():
    if "run" in sys.argv:
        sys.argv = ["streamlit", "run", "src/quoptuna/frontend/app.py"]
        stcli.main()
    if "--start" in sys.argv:
        sys.argv = ["streamlit", "run", "src/quoptuna/frontend/app.py"]
        stcli.main()
    else:
        initialize_session_state()

        # Create tabs for different pages
        tab1, tab2 = st.tabs(["📊 Dashboard", "📈 Plots"])

        with tab1:
            with st.sidebar:
                handle_sidebar()
            update_plot()

        with tab2:
            from quoptuna.frontend.pages.plots import app as plots_app

            plots_app()


if __name__ == "__main__":
    main()
