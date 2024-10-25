import streamlit as st

from quoptuna import Optimizer
from quoptuna.backend.data import preprocess_data
from quoptuna.frontend.support import (
    run_optimization_in_background,
    select_columns,
    upload_and_display_data,
)


def select_data(data):
    """Select features from the uploaded data."""
    st.markdown("### Select Features")
    x_columns, y_column = select_columns(data)
    if x_columns and y_column:
        x = data[x_columns]
        y = data[y_column]
        return preprocess_data(x, y), x_columns, y_column
    return None, None, None


def run_optimization(session_state, data_dict):
    """Run the optimization process."""
    if session_state["DB_NAME"] and all(
        len(data_dict[key]) > 0 for key in ["train_x", "test_x", "train_y", "test_y"]
    ):
        optimizer = Optimizer(
            db_name=session_state["DB_NAME"],
            data=data_dict,
            study_name=session_state["study_name"],
        )
        st.session_state["optimizer"] = optimizer
        st.markdown("### Actions")
        if st.button("Run Optimization", help="Start the optimization process"):
            try:
                run_optimization_in_background(optimizer, session_state["n_trials"])
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to start optimization: {e}")
                st.error(f"Exception type: {type(e).__name__}")


def handle_sidebar():
    """Handle the sidebar interactions."""
    st.title("QuOptuna: Optimizing Quantum Models with Optuna")
    st.write("Please upload your data file below.")
    data = upload_and_display_data()
    if data is not None:
        session_state = {
            "x_columns": None,
            "y_column": None,
            "DB_NAME": st.text_input(
                "Enter database name", help="Name of the database to store results"
            ),
            "study_name": st.text_input(
                "Enter study name", help="Name of the study for optimization"
            ),
            "n_trials": st.number_input(
                "Number of trials",
                min_value=1,
                max_value=100,
                value=100,
                help="Number of optimization trials",
            ),
        }
        (x_train, x_test, y_train, y_test), x_columns, y_column = select_data(data)
        if x_train is not None:
            data_dict = {
                "train_x": x_train,
                "test_x": x_test,
                "train_y": y_train,
                "test_y": y_test,
            }
            st.session_state.update(session_state)
            run_optimization(session_state, data_dict)
            if not st.session_state.get("start_visualization", False):
                st.button(
                    "Start Visualization",
                    on_click=lambda: st.session_state.update(
                        {"start_visualization": True}
                    ),
                    help="Start the visualization with the latest data.",
                )
                st.info("Click to start visualization.")
            else:
                st.success("Visualization is already running.")
