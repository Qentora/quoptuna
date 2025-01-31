from __future__ import annotations

import copy

# import optuna frozen trials
from typing import TYPE_CHECKING

import shap
import streamlit as st

from quoptuna.backend.models import create_model
from quoptuna.backend.tuners.optimizer import Optimizer
from quoptuna.frontend.sidebar import select_data
from quoptuna.frontend.support import (
    handle_input,
    initialize_session_state,
    select_columns,
    upload_and_display_data,
)

if TYPE_CHECKING:
    from optuna.trial import FrozenTrial

initialize_session_state()
optimizer, study_name, db_name = handle_input()


def select_data_columns(data):
    """Select x and y columns from the uploaded data."""
    st.markdown("### Select Features")
    x_columns, y_column = select_columns(data)  # Assuming select_columns is available
    if x_columns and y_column:
        x = data[x_columns]
        y = data[y_column]
        return x, y
    return None, None


# get the study
if isinstance(optimizer, Optimizer):
    optimizer.load_study()
    study = optimizer.study
    # Check if study is not None before accessing best_trials
    best_trial: list[FrozenTrial] = []
    if study is not None:
        best_trial = study.best_trials
    else:
        st.error("No study found. Please ensure the optimizer has a valid study.")

    # make a dropdown to select the best trial
    def format_trial(trial):
        quantum_f1_score = trial.user_attrs.get("Quantum_f1_score")
        classical_f1_score = trial.user_attrs.get("Classical_f1_score")
        f1_score = (
            quantum_f1_score
            if quantum_f1_score != 0
            else classical_f1_score
            if classical_f1_score != 0
            else "N/A"
        )
        return f"Trial {trial.number} " f"{trial.params.get('model_type')} " f"F1-Score {f1_score}"

    best_trial_dropdown = st.selectbox(
        "Select Best Trial",
        list(best_trial),
        format_func=format_trial,
    )
    # whenever a selection in drop down is made.
    if best_trial_dropdown:
        with st.expander(label="Selected Parameters"):
            st.write(best_trial_dropdown.params)
        selected_trial = copy.copy(best_trial_dropdown)
        model_parameters = selected_trial.params
        model_type = model_parameters.pop("model_type")
        model = create_model(model_type=model_type, **model_parameters)
        st.write(model)

        # Use the similar method to get data and select x and y columns from the data
        data = upload_and_display_data()  # Assuming this function is available
        if data is not None:
            (x_train, x_test, y_train, y_test), x_columns, y_column = select_data(data)
            if st.button("Train Model"):
                model.fit(X=x_train, y=y_train)
                # Calculate SHAP values

                explainer = shap.Explainer(model, x_train)  # Create an explainer
                shap_values = explainer(x_train)  # Calculate SHAP values

                # Visualize SHAP values
                st.subheader("SHAP Summary Plot")
                shap.summary_plot(shap_values, x_train, show=False)
                st.pyplot(bbox_inches="tight")  # Display the plot in Streamlit

                # Optionally, you can add more SHAP visualizations
                st.subheader("SHAP Dependence Plot")
                feature: str
                feature = st.selectbox("Select feature for dependence plot", x_train.columns)
                shap.dependence_plot(feature, shap_values.values, x_train, show=False)
                st.pyplot(bbox_inches="tight")
