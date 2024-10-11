import os
import threading
import time
import webbrowser

import optuna
import pandas as pd
import streamlit as st

from quoptuna import Optimizer
from quoptuna.backend.data import (
    find_free_port,
    preprocess_data,
    start_optuna_dashboard,
)

# Set wide mode and dark mode as default for Streamlit
st.set_page_config(page_title="QuOptuna", page_icon="🌙", layout="wide")


def upload_and_display_data():
    """Handles file upload and displays the data."""
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        # save the  uploaded file to local file system in uploaded_data folder
        if not os.path.exists("./uploaded_data"):  # noqa: PTH110
            os.makedirs("./uploaded_data")  # noqa: PTH103
        with open(f"./uploaded_data/{uploaded_file.name}", "wb") as f:  # noqa: PTH123
            f.write(uploaded_file.getvalue())
        # also keep the uploaded data in session state
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["uploaded_file_name"] = uploaded_file.name
        # file location
        file_location = f"./uploaded_data/{uploaded_file.name}"
        st.session_state["file_location"] = file_location
        return pd.read_csv(file_location)
    return None


def select_columns(data):
    """Allows user to select columns for X and y."""
    x_columns = st.multiselect("Select columns for X", data.columns.tolist())
    y_column = st.selectbox("Select column for y", data.columns.tolist())
    return x_columns, y_column


def run_optimization_in_background(optimizer, n_trials):
    """Run the optimization process in a separate thread."""

    def optimization_task():
        study, best_trials = optimizer.optimize(n_trials=n_trials)
        for trial in best_trials:
            st.write(str(trial))

    # Start the optimization in a new thread
    optimization_thread = threading.Thread(target=optimization_task)
    optimization_thread.start()
    st.write("Optimization started")
    st.session_state["process_running"] = True


def open_dashboard(optimizer, port):
    """Open the Optuna dashboard."""
    optuna_dashboard_url = start_optuna_dashboard(
        storage=optimizer.storage_location, port=port
    )
    webbrowser.open_new_tab(optuna_dashboard_url)


def initialize_session_state():
    """Initialize session state keys."""
    keys = [
        "uploaded_file",
        "uploaded_file_name",
        "file_location",
        "x_columns",
        "y_column",
        "DB_NAME",
        "study_name",
        "n_trials",
        "optimizer",
        "process_running",
    ]
    defaults = [None, None, None, None, None, None, None, 100, None, False]
    for key, default in zip(keys, defaults):
        if key not in st.session_state:
            st.session_state[key] = default


def handle_sidebar():
    """Handle the sidebar interactions."""
    st.title("QuOptuna: Optimizing Quantum Models with Optuna")
    st.write("Please upload your data file below.")
    data = upload_and_display_data()
    if data is not None:
        x_columns, y_column = select_columns(data)
        if x_columns and y_column:
            x = data[x_columns]
            y = data[y_column]
            x_train, x_test, y_train, y_test = preprocess_data(x, y)
            session_state = {
                "x_columns": x_columns,
                "y_column": y_column,
                "DB_NAME": st.text_input("Enter database name"),
                "study_name": st.text_input("Enter study name"),
                "n_trials": st.number_input(
                    "Number of trials", min_value=1, max_value=100, value=100
                ),
            }
            data_dict = {
                "train_x": x_train,
                "test_x": x_test,
                "train_y": y_train,
                "test_y": y_test,
            }
            st.session_state.update(session_state)
            if session_state["DB_NAME"] and all(
                len(data_dict[key]) > 0
                for key in ["train_x", "test_x", "train_y", "test_y"]
            ):
                optimizer = Optimizer(
                    db_name=session_state["DB_NAME"],
                    data=data_dict,
                    study_name=session_state["study_name"],
                )
                st.session_state["optimizer"] = optimizer
                port = find_free_port()
                if st.button("Run Optimization"):
                    run_optimization_in_background(optimizer, session_state["n_trials"])
                if st.button("Open Dashboard"):
                    open_dashboard(optimizer, port)


def update_plot():
    """Update the plot with the latest optimization results."""
    optimizer = st.session_state["optimizer"]
    plot_placeholder = st.empty()
    while True:
        loaded_study = optuna.load_study(
            study_name=optimizer.study_name, storage=optimizer.storage_location
        )
        fig = optuna.visualization.plot_timeline(loaded_study)
        plot_placeholder.plotly_chart(fig)
        time.sleep(10)


def main():
    initialize_session_state()
    with st.sidebar:
        handle_sidebar()
    if st.session_state["optimizer"] and st.session_state["process_running"]:
        update_plot()


if __name__ == "__main__":
    main()
