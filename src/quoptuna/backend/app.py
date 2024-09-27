import pandas as pd
import streamlit as st

from quoptuna import Optimizer
from quoptuna.backend.data import (
    preprocess_data,
    find_free_port,
    start_optuna_dashboard,
)


def main():
    st.title("QuOptuna: Optimizing Quantum Models with Optuna")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Select columns for X and y
        X_columns = st.multiselect("Select columns for X", data.columns.tolist())
        y_column = st.selectbox("Select column for y", data.columns.tolist())

        if X_columns and y_column:
            X = data[X_columns]
            y = data[y_column]

            # Split data into train and test
            X_train, X_test, y_train, y_test = preprocess_data(X, y)

            DB_NAME = st.text_input("Enter database name")
            # take input n_trials as int intput with deafault value as set as 100
            n_trials = st.number_input(
                "Number of trials", min_value=1, max_value=100, value=100
            )
            if (
                DB_NAME
                and len(X_train) > 0
                and len(X_test) > 0
                and len(y_train) > 0
                and len(y_test) > 0
            ):
                optimizer = Optimizer(
                    db_name=DB_NAME,
                    train_X=X_train,
                    test_X=X_test,
                    train_y=y_train,
                    test_y=y_test,
                )
                port = find_free_port()
                # start an optuna server
                optuna_dashboard_url = start_optuna_dashboard(
                    storage=optimizer.storage_location, port=port
                )
                # create a hyperlink to optuna_dashboard url
                st.markdown(
                    f"Optuna Dashboard: [{optuna_dashboard_url}]({optuna_dashboard_url})",
                    unsafe_allow_html=True,
                )
                # RUN THE OPTIMIZATION if the user hits a button to run the optimization
                if st.button("Run Optimization"):
                    study = optimizer.optimize(n_trials=n_trials)
                    # find a free port


if __name__ == "__main__":
    main()
