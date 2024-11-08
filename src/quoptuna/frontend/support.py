import os
import threading
import time

import optuna
import pandas as pd
import streamlit as st


def upload_and_display_data():
    """Handles file upload and displays the data."""
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file:
        if not os.path.exists("./uploaded_data"):  # noqa: PTH110
            os.makedirs("./uploaded_data")  # noqa: PTH103
        with open(f"./uploaded_data/{uploaded_file.name}", "wb") as f:  # noqa: PTH123
            f.write(uploaded_file.getvalue())
        st.session_state["uploaded_file"] = uploaded_file
        st.session_state["uploaded_file_name"] = uploaded_file.name
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

    optimization_thread = threading.Thread(target=optimization_task)
    optimization_thread.start()
    st.success("Optimization started successfully.")
    st.session_state["process_running"] = True


def initialize_session_state():
    """Initialize session state keys."""
    session_defaults = {
        "uploaded_file": None,
        "uploaded_file_name": None,
        "file_location": None,
        "x_columns": None,
        "y_column": None,
        "DB_NAME": None,
        "study_name": None,
        "n_trials": 100,
        "optimizer": None,
        "process_running": False,
        "start_visualization": False,
    }
    for key, default in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def update_plot():
    """Update the plot with the latest optimization results."""
    optimizer, study_name, db_name = handle_input()
    st.toast("Updating Plot")
    st.toast(f"Study Name: {study_name}, DB Name: {db_name}")
    if st.session_state["optimizer"] and st.session_state["study_name"]:
        display_study_control_panel(study_name, optimizer)


def handle_input():
    """Handles input for optimizer, study name, and database name."""
    optimizer = st.session_state["optimizer"]
    study_name = st.session_state.get("study_name", "")  # Get study_name from session state
    db_name = st.session_state.get("DB_NAME", "")  # Get DB_NAME from session state

    with st.expander("Load Optimizer", expanded=(optimizer is None)):
        db_name = st.text_input(
            "Enter database name",
            value=db_name,  # Set the default value to the existing DB name
            help="Name of the database to load optimizer from",
        )
        study_name = st.text_input(
            "Enter study name",
            value=study_name,  # Set the default value to the existing study name
            help="Name of the study to load optimizer from",
        )
        uploaded_db = st.file_uploader("Upload DB file", type=["db"])

        if uploaded_db:
            db_path = f"./db/{uploaded_db.name}"
            if not os.path.exists("./db"):  # noqa: PTH110
                os.makedirs("./db")  # noqa: PTH103
            with open(db_path, "wb") as f:  # noqa: PTH123
                f.write(uploaded_db.getvalue())
            st.session_state["file_location"] = db_path
        if st.button("Load Optimizer") and study_name:
            from quoptuna import Optimizer  # Import Optimizer here to avoid lint error

            optimizer = Optimizer(db_name=db_name, study_name=study_name)
            st.session_state["optimizer"] = optimizer
            st.session_state["DB_NAME"] = db_name
            st.session_state["study_name"] = study_name  # Store study_name in session state
            st.session_state["data_loaded_from_file"] = True
    return optimizer, study_name, db_name


def display_study_control_panel(study_name, optimizer):
    """Displays the study control panel and handles visualization."""
    st.toast("Study Control Panel")
    st.markdown("### Study Control Panel")
    st.text_input("Study name", value=st.session_state["study_name"], disabled=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(
            "Start Visualization",
            on_click=lambda: st.session_state.update({"start_visualization": True}),
            help="Start the visualization with the latest data.",
            key="start_visualization_button",  # Add a unique key
        )
    with col2:
        st.button(
            "Stop Visualization",
            on_click=lambda: st.session_state.update({"start_visualization": False}),
            help="Stop the visualization updates.",
            key="stop_visualization_button",  # Add a unique key
        )

    plot_visualization(optimizer, study_name)


def plot_visualization(optimizer, study_name):
    """Handles the visualization of the optimization results."""
    st.toast("Plotting Visualization")
    plot_placeholder_timeline = st.empty()
    plot_placeholder_importances = st.empty()
    plot_placeholder_optimization_history = st.empty()
    trials_placeholder = st.empty()
    trails_title = st.empty()
    best_trials_placeholder = st.empty()
    best_trials_title = st.empty()
    counter = 0  # Initialize a counter for unique keys
    while st.session_state["start_visualization"]:
        loaded_study = optuna.load_study(study_name=study_name, storage=optimizer.storage_location)
        try:
            fig_timeline = optuna.visualization.plot_timeline(loaded_study)
            plot_placeholder_timeline.plotly_chart(fig_timeline, key=f"timeline_chart_{counter}")
        except ValueError as e:
            st.error(f"Error in plotting timeline: {e}")

        try:
            fig_importances = optuna.visualization.plot_param_importances(loaded_study)
            plot_placeholder_importances.plotly_chart(
                fig_importances, key=f"importances_chart_{counter}"
            )
        except ValueError as e:
            st.error(f"Error in plotting parameter importances: {e}")

        try:
            fig_optimization_history = optuna.visualization.plot_optimization_history(loaded_study)
            plot_placeholder_optimization_history.plotly_chart(
                fig_optimization_history,
                key=f"optimization_history_chart_{counter}",
            )
        except ValueError as e:
            st.error(f"Error in plotting optimization history: {e}")

        trials = loaded_study.get_trials(deepcopy=False)
        trials_data = [
            {
                "Trial Number": trial.number,
                "State": trial.state,
                "Value": trial.value,
                "Params": trial.params,
                "Start Time": trial.datetime_start,
                "End Time": trial.datetime_complete,
                "Classical F1 Score": trial.user_attrs.get(
                    "Classical_f1_score", None
                ),  # Access user attributes
                "Quantum F1 Score": trial.user_attrs.get(
                    "Quantum_f1_score", None
                ),  # Access user attributes
            }
            for trial in trials
        ]
        trails_title.write("Trials Data")
        # make a header and title for the dataframe
        trials_placeholder.dataframe(
            trials_data,
            use_container_width=True,
            column_config={
                "Classical F1 Score": st.column_config.NumberColumn(
                    "Classical F1 Score", help="Classical F1 Score"
                ),
                "Quantum F1 Score": st.column_config.NumberColumn(
                    "Quantum F1 Score", help="Quantum F1 Score"
                ),
            },
        )
        # check if the best trial in study is not none
        if loaded_study.best_trials is not None:
            best_trials_title.write("Best Trials Data")
            best_trial = loaded_study.best_trials
            best_trials_data = [
                {
                    "Trial Number": trial.number,
                    "State": trial.state,
                    "Value": trial.value,
                    "Params": trial.params,
                    "Start Time": trial.datetime_start,
                    "End Time": trial.datetime_complete,
                    "Classical F1 Score": trial.user_attrs.get(
                        "Classical_f1_score", None
                    ),  # Access user attributes
                    "Quantum F1 Score": trial.user_attrs.get(
                        "Quantum_f1_score", None
                    ),  # Access user attributes
                }
                for trial in best_trial
            ]
            best_trials_placeholder.dataframe(best_trials_data, use_container_width=True)
        counter += 1  # Increment the counter for the next iteration
        if not st.session_state["process_running"]:
            break
        time.sleep(10)
