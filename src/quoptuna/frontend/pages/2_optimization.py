"""Optimization Page - Prepare data and run hyperparameter optimization."""

import streamlit as st

from quoptuna import DataPreparation, Optimizer


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "data_dict": None,
        "optimizer": None,
        "study": None,
        "best_trials": None,
        "optimization_complete": False,
        "db_name": None,
        "study_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prepare_data():
    """Prepare data for optimization."""
    st.subheader("🔧 Data Preparation")

    if "file_path" not in st.session_state or not st.session_state["file_path"]:
        st.warning("⚠️ Please complete Dataset Selection first!")
        st.info("👈 Go to the **Dataset Selection** page to load a dataset.")
        return False

    st.success(f"✅ Dataset loaded: {st.session_state['dataset_name']}")
    st.write(f"📁 File path: `{st.session_state['file_path']}`")

    # Show dataset info
    if st.session_state.get("dataset_df") is not None:
        df = st.session_state["dataset_df"]
        with st.expander("📊 Dataset Summary"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Features", len(st.session_state["feature_columns"]))
            with col3:
                target_dist = df["target"].value_counts()
                st.write("**Target Distribution:**")
                st.write(target_dist)

    # Prepare data button
    if st.button("🔨 Prepare Data for Training", type="primary"):
        try:
            with st.spinner("Preparing data..."):
                data_prep = DataPreparation(
                    file_path=st.session_state["file_path"],
                    x_cols=st.session_state["feature_columns"],
                    y_col=st.session_state["target_column"],
                )

                # Get data in the format required by Optimizer
                data_dict = data_prep.get_data(output_type="2")

                # Convert to numpy arrays
                data_dict["train_x"] = data_dict["train_x"].values
                data_dict["test_x"] = data_dict["test_x"].values
                data_dict["train_y"] = data_dict["train_y"].values
                data_dict["test_y"] = data_dict["test_y"].values

                st.session_state["data_dict"] = data_dict

                st.success("✅ Data prepared successfully!")

                # Show split information
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Samples", len(data_dict["train_x"]))
                with col2:
                    st.metric("Test Samples", len(data_dict["test_x"]))

        except Exception as e:
            st.error(f"❌ Error preparing data: {e}")
            return False

    return st.session_state.get("data_dict") is not None


def run_optimization():
    """Run hyperparameter optimization."""
    st.subheader("🎯 Hyperparameter Optimization")

    if st.session_state.get("data_dict") is None:
        st.warning("⚠️ Please prepare data first!")
        return

    # Optimization settings
    col1, col2 = st.columns(2)

    with col1:
        db_name = st.text_input(
            "Database Name:",
            value=st.session_state["dataset_name"],
            help="Name for the Optuna database",
        )

    with col2:
        study_name = st.text_input(
            "Study Name:",
            value=st.session_state["dataset_name"],
            help="Name for this optimization study",
        )

    n_trials = st.slider(
        "Number of Trials:",
        min_value=10,
        max_value=200,
        value=100,
        step=10,
        help="Number of optimization trials to run",
    )

    st.info(
        f"💡 This will run {n_trials} trials to find the best hyperparameters "
        "for both classical and quantum models."
    )

    # Run optimization
    if st.button("🚀 Start Optimization", type="primary"):
        try:
            with st.spinner(f"Running optimization with {n_trials} trials..."):
                # Create optimizer
                optimizer = Optimizer(
                    db_name=db_name,
                    study_name=study_name,
                    data=st.session_state["data_dict"],
                )

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Run optimization
                study, best_trials = optimizer.optimize(n_trials=n_trials)

                progress_bar.progress(100)
                status_text.text("Optimization complete!")

                # Store results
                st.session_state["optimizer"] = optimizer
                st.session_state["study"] = study
                st.session_state["best_trials"] = best_trials
                st.session_state["optimization_complete"] = True
                st.session_state["db_name"] = db_name
                st.session_state["study_name"] = study_name

                st.success("✅ Optimization completed successfully!")

                # Show best trials
                display_results()

        except Exception as e:
            st.error(f"❌ Error during optimization: {e}")
            st.exception(e)


def display_results():
    """Display optimization results."""
    if not st.session_state.get("optimization_complete"):
        return

    st.subheader("🏆 Optimization Results")

    best_trials = st.session_state["best_trials"]

    if not best_trials:
        st.warning("No best trials found.")
        return

    # Format trial information
    def format_trial(trial):
        quantum_f1 = trial.user_attrs.get("Quantum_f1_score", 0)
        classical_f1 = trial.user_attrs.get("Classical_f1_score", 0)
        f1_score = quantum_f1 if quantum_f1 != 0 else classical_f1
        model_type = trial.params.get("model_type", "Unknown")
        return f"Trial {trial.number} - {model_type} - F1 Score: {f1_score:.4f}"

    # Display best trials
    st.write(f"**Found {len(best_trials)} best trial(s):**")

    for i, trial in enumerate(best_trials[:5]):  # Show top 5
        with st.expander(format_trial(trial), expanded=(i == 0)):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Performance:**")
                st.write(f"- Quantum F1: {trial.user_attrs.get('Quantum_f1_score', 0):.4f}")
                st.write(f"- Classical F1: {trial.user_attrs.get('Classical_f1_score', 0):.4f}")

            with col2:
                st.write("**Key Parameters:**")
                st.write(f"- Model Type: {trial.params.get('model_type')}")
                important_params = {
                    k: v
                    for k, v in trial.params.items()
                    if k in ["learning_rate", "n_layers", "batch_size", "C", "gamma"]
                }
                for k, v in important_params.items():
                    st.write(f"- {k}: {v}")

            with st.expander("All Parameters"):
                st.json(trial.params)

    st.info("👉 Proceed to the next page: **Model Training & Evaluation**")


def main():
    """Main function for optimization page."""
    st.set_page_config(
        page_title="Optimization - QuOptuna",
        page_icon="🎯",
        layout="wide",
    )

    initialize_session_state()

    st.title("🎯 Data Preparation & Optimization")
    st.markdown(
        """
        Prepare your data and run hyperparameter optimization to find the best model configuration.
        """
    )

    # Data preparation section
    data_ready = prepare_data()

    if data_ready:
        st.divider()
        # Optimization section
        run_optimization()

        # Display results if available
        if st.session_state.get("optimization_complete"):
            st.divider()
            display_results()


if __name__ == "__main__":
    main()
