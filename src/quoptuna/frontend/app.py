import sys

import streamlit as st
from streamlit.web import cli as stcli

from quoptuna.frontend.main_page import main_page
from quoptuna.frontend.sidebar import handle_sidebar
from quoptuna.frontend.support import (
    initialize_session_state,
    update_plot,
)

st.set_page_config(
    page_title="QuOptuna", page_icon=":atom_symbol:", layout="wide"
)  # Move this line to the top

# Apply custom CSS for dark theme with purple accents and hover effects
st.markdown(
    """
    <style>
    .main-title {
        font-size: 3em;
        color: #9b59b6; /* Purple */
        text-align: center;
        margin-top: 20px;
    }
    .description {
        font-size: 1.2em;
        color: #ecf0f1; /* Light grey */
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #9b59b6; /* Purple */
        color: #ecf0f1; /* Light grey */
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #8e44ad; /* Darker Purple */
    }
    </style>
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
        main_page()
        with st.sidebar:
            handle_sidebar()
        # if st.session_state["optimizer"] or st.session_state["start_visualization"]:
        update_plot()


if __name__ == "__main__":
    main()
