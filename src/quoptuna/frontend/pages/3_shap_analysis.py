"""SHAP Analysis & Report Generation Page."""

import base64
import os
from io import BytesIO

import streamlit as st

from quoptuna import XAI
from quoptuna.backend.models import create_model
from quoptuna.backend.xai.xai import PlotType, XAIConfig


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "selected_trial": None,
        "trained_model": None,
        "xai": None,
        "report": None,
        "shap_images": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def select_trial():
    """Select a trial for SHAP analysis."""
    st.subheader("🎯 Select Trial for Analysis")

    if not st.session_state.get("optimization_complete"):
        st.warning("⚠️ Please complete optimization first!")
        st.info("👈 Go to the **Optimization** page to run optimization.")
        return None

    best_trials = st.session_state["best_trials"]

    if not best_trials:
        st.error("No trials found. Please run optimization first.")
        return None

    # Format trial for dropdown
    def format_trial(trial):
        quantum_f1 = trial.user_attrs.get("Quantum_f1_score", 0)
        classical_f1 = trial.user_attrs.get("Classical_f1_score", 0)
        f1_score = quantum_f1 if quantum_f1 != 0 else classical_f1
        model_type = trial.params.get("model_type", "Unknown")
        return f"Trial {trial.number} - {model_type} - F1: {f1_score:.4f}"

    # Trial selection
    selected_trial = st.selectbox(
        "Select a trial:",
        best_trials,
        format_func=format_trial,
    )

    if selected_trial:
        with st.expander("📋 Trial Details", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Performance Metrics:**")
                st.metric(
                    "Quantum F1 Score",
                    f"{selected_trial.user_attrs.get('Quantum_f1_score', 0):.4f}",
                )
                st.metric(
                    "Classical F1 Score",
                    f"{selected_trial.user_attrs.get('Classical_f1_score', 0):.4f}",
                )

            with col2:
                st.write("**Model Information:**")
                st.write(f"**Model Type:** {selected_trial.params.get('model_type')}")
                st.write(f"**Trial Number:** {selected_trial.number}")

            with st.expander("All Parameters"):
                st.json(selected_trial.params)

        st.session_state["selected_trial"] = selected_trial

    return selected_trial


def train_model():
    """Train the model with selected parameters."""
    st.subheader("🎓 Model Training")

    if not st.session_state.get("selected_trial"):
        st.warning("Please select a trial first!")
        return False

    trial = st.session_state["selected_trial"]

    if st.button("🚀 Train Model", type="primary"):
        try:
            with st.spinner("Training model..."):
                # Get model parameters
                model_params = trial.params.copy()

                # Create and train model
                model = create_model(**model_params)

                # Train the model
                data_dict = st.session_state["data_dict"]
                model.fit(data_dict["train_x"], data_dict["train_y"])

                st.session_state["trained_model"] = model

                st.success("✅ Model trained successfully!")

        except Exception as e:
            st.error(f"❌ Error training model: {e}")
            st.exception(e)
            return False

    return st.session_state.get("trained_model") is not None


def run_shap_analysis():
    """Run SHAP analysis on the trained model."""
    st.subheader("🔍 SHAP Analysis")

    if not st.session_state.get("trained_model"):
        st.warning("Please train the model first!")
        return False

    # SHAP configuration
    st.markdown("### Configuration")

    col1, col2 = st.columns(2)

    with col1:
        use_proba = st.checkbox(
            "Use Probability Predictions",
            value=True,
            help="Use probability predictions instead of class predictions",
        )

    with col2:
        onsubset = st.checkbox(
            "Use Subset of Data",
            value=True,
            help="Use a subset of test data for faster computation",
        )

    if onsubset:
        subset_size = st.slider(
            "Subset Size:",
            min_value=10,
            max_value=min(200, len(st.session_state["data_dict"]["test_x"])),
            value=50,
            help="Number of samples to use for SHAP analysis",
        )
    else:
        subset_size = None

    # Run SHAP analysis
    if st.button("🔬 Run SHAP Analysis", type="primary"):
        try:
            with st.spinner("Calculating SHAP values..."):
                # Create XAI config
                config = XAIConfig(
                    use_proba=use_proba,
                    onsubset=onsubset,
                    subset_size=subset_size if onsubset else None,
                )

                # Create XAI instance
                xai = XAI(
                    model=st.session_state["trained_model"],
                    data=st.session_state["data_dict"],
                    config=config,
                )

                st.session_state["xai"] = xai

                st.success("✅ SHAP analysis completed!")

        except Exception as e:
            st.error(f"❌ Error during SHAP analysis: {e}")
            st.exception(e)
            return False

    return st.session_state.get("xai") is not None


def display_shap_plots():
    """Display SHAP visualizations."""
    st.subheader("📊 SHAP Visualizations")

    if not st.session_state.get("xai"):
        st.warning("Please run SHAP analysis first!")
        return

    xai = st.session_state["xai"]

    # Create tabs for different plot types
    tabs = st.tabs(
        [
            "📊 Bar Plot",
            "🐝 Beeswarm Plot",
            "🎻 Violin Plot",
            "🔥 Heatmap",
            "💧 Waterfall Plot",
            "📈 Confusion Matrix",
        ]
    )

    # Bar Plot
    with tabs[0]:
        st.markdown("### Feature Importance (Bar Plot)")
        st.info("Shows the mean absolute SHAP value for each feature")

        try:
            bar_plot = xai.get_plot("bar", max_display=10, class_index=1)
            st.image(bar_plot, use_container_width=True)

            if st.button("💾 Save Bar Plot", key="save_bar"):
                save_plot(bar_plot, "bar_plot.png")

        except Exception as e:
            st.error(f"Error generating bar plot: {e}")

    # Beeswarm Plot
    with tabs[1]:
        st.markdown("### Feature Impact Distribution (Beeswarm Plot)")
        st.info("Shows how feature values affect predictions")

        try:
            beeswarm_plot = xai.get_plot("beeswarm", max_display=10, class_index=1)
            st.image(beeswarm_plot, use_container_width=True)

            if st.button("💾 Save Beeswarm Plot", key="save_beeswarm"):
                save_plot(beeswarm_plot, "beeswarm_plot.png")

        except Exception as e:
            st.error(f"Error generating beeswarm plot: {e}")

    # Violin Plot
    with tabs[2]:
        st.markdown("### Feature Distribution (Violin Plot)")
        st.info("Shows the distribution of SHAP values for each feature")

        try:
            violin_plot = xai.get_plot("violin", max_display=10, class_index=1)
            st.image(violin_plot, use_container_width=True)

            if st.button("💾 Save Violin Plot", key="save_violin"):
                save_plot(violin_plot, "violin_plot.png")

        except Exception as e:
            st.error(f"Error generating violin plot: {e}")

    # Heatmap
    with tabs[3]:
        st.markdown("### Instance-Level Analysis (Heatmap)")
        st.info("Shows SHAP values for individual instances")

        try:
            heatmap_plot = xai.get_plot("heatmap", max_display=50, class_index=1)
            st.image(heatmap_plot, use_container_width=True)

            if st.button("💾 Save Heatmap", key="save_heatmap"):
                save_plot(heatmap_plot, "heatmap_plot.png")

        except Exception as e:
            st.error(f"Error generating heatmap: {e}")

    # Waterfall Plot
    with tabs[4]:
        st.markdown("### Individual Prediction Explanation (Waterfall Plot)")
        st.info("Shows how features contribute to a single prediction")

        sample_idx = st.number_input(
            "Sample Index:",
            min_value=0,
            max_value=min(50, len(xai.x_test) - 1),
            value=0,
        )

        try:
            waterfall_plot = xai.get_plot("waterfall", index=sample_idx, class_index=1)
            st.image(waterfall_plot, use_container_width=True)

            if st.button("💾 Save Waterfall Plot", key="save_waterfall"):
                save_plot(waterfall_plot, f"waterfall_plot_{sample_idx}.png")

        except Exception as e:
            st.error(f"Error generating waterfall plot: {e}")

    # Confusion Matrix
    with tabs[5]:
        st.markdown("### Model Performance (Confusion Matrix)")
        st.info("Shows model classification performance")

        try:
            fig = xai.plot_confusion_matrix()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error generating confusion matrix: {e}")


def generate_report():
    """Generate AI report using LLM."""
    st.subheader("📝 AI-Generated Report")

    if not st.session_state.get("xai"):
        st.warning("Please run SHAP analysis first!")
        return

    st.markdown("### Report Configuration")

    # LLM provider selection
    provider = st.selectbox(
        "LLM Provider:",
        ["google", "openai", "anthropic"],
        help="Select the LLM provider for report generation",
    )

    api_key = st.text_input(
        "API Key:",
        type="password",
        help="Enter your API key for the selected provider",
    )

    model_name = st.text_input(
        "Model Name:",
        value="models/gemini-2.0-flash-exp" if provider == "google" else "",
        help="Enter the model name (e.g., 'models/gemini-2.0-flash-exp' for Google)",
    )

    # Dataset information
    with st.expander("📋 Dataset Information (Optional)", expanded=False):
        dataset_url = st.text_input("Dataset URL:", value="")
        dataset_description = st.text_area("Dataset Description:", value="")

    # Generate report
    if st.button("✨ Generate Report", type="primary"):
        if not api_key:
            st.error("Please provide an API key!")
            return

        try:
            with st.spinner("Generating report with AI... This may take a minute."):
                xai = st.session_state["xai"]

                # Prepare dataset info
                dataset_info = {
                    "Name": st.session_state.get("dataset_name", "Unknown"),
                    "URL": dataset_url if dataset_url else "N/A",
                    "Description": dataset_description if dataset_description else "N/A",
                    "Features": st.session_state.get("feature_columns", []),
                    "Target": st.session_state.get("target_column", "target"),
                }

                # Get basic report
                report_data = xai.get_report()

                # Generate images for report
                images = {}
                plot_types: list[PlotType] = ["bar", "beeswarm", "violin", "heatmap"]

                for plot_type in plot_types:
                    try:
                        images[plot_type] = xai.get_plot(plot_type, class_index=1)
                    except Exception as e:
                        st.warning(f"Could not generate {plot_type} plot: {e}")

                # Add confusion matrix
                try:
                    fig = xai.plot_confusion_matrix()
                    img_buf = BytesIO()
                    fig.savefig(img_buf, format="png")
                    img_buf.seek(0)
                    img_base64 = base64.b64encode(img_buf.getvalue()).decode("utf-8")
                    images["confusion_matrix"] = f"data:image/png;base64,{img_base64}"
                except Exception as e:
                    st.warning(f"Could not generate confusion matrix: {e}")

                # Generate final report with LLM
                report = xai.generate_report_with_langchain(
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    dataset_info=dataset_info,
                )

                st.session_state["report"] = report
                st.session_state["shap_images"] = images

                st.success("✅ Report generated successfully!")

        except Exception as e:
            st.error(f"❌ Error generating report: {e}")
            st.exception(e)

    # Display report
    if st.session_state.get("report"):
        st.divider()
        st.markdown("### 📄 Generated Report")

        # Download button
        report_text = st.session_state["report"]
        st.download_button(
            label="📥 Download Report (Markdown)",
            data=report_text,
            file_name=f"{st.session_state['dataset_name']}_report.md",
            mime="text/markdown",
        )

        # Display report
        st.markdown(report_text)


def save_plot(plot_data, filename):
    """Save plot to file."""
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)

    # If plot_data is base64 encoded image
    if isinstance(plot_data, str) and plot_data.startswith("data:image"):
        img_data = plot_data.split(",")[1]
        with open(filepath, "wb") as f:
            f.write(base64.b64decode(img_data))
    else:
        # Assume it's a bytes object or similar
        with open(filepath, "wb") as f:
            f.write(plot_data)

    st.success(f"✅ Saved to {filepath}")


def main():
    """Main function for SHAP analysis page."""
    st.set_page_config(
        page_title="SHAP Analysis - QuOptuna",
        page_icon="🔍",
        layout="wide",
    )

    initialize_session_state()

    st.title("🔍 SHAP Analysis & Report Generation")
    st.markdown(
        """
        Analyze model behavior with SHAP (SHapley Additive exPlanations) and generate
        comprehensive AI-powered reports.
        """
    )

    # Trial selection
    trial = select_trial()

    if trial:
        st.divider()

        # Model training
        model_trained = train_model()

        if model_trained:
            st.divider()

            # SHAP analysis
            shap_ready = run_shap_analysis()

            if shap_ready:
                st.divider()

                # Display SHAP plots
                display_shap_plots()

                st.divider()

                # Generate report
                generate_report()


if __name__ == "__main__":
    main()
