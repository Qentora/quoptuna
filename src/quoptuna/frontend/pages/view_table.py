import time

import optuna
import streamlit as st

optimizer = st.session_state["optimizer"]


# Create a placeholder for the plot
plot_placeholder = st.empty()

while True:
    loaded_study = optuna.load_study(
        study_name=optimizer.study_name, storage=optimizer.storage_location
    )
    fig = optuna.visualization.plot_timeline(loaded_study)
    # Update the existing plot
    plot_placeholder.plotly_chart(fig)
    time.sleep(10)  # update every 60 seconds
