import streamlit as st


def main_page():
    """This is the main page of the app."""
    st.markdown(
        '<div class="main-title">QuOptuna: Quantum-Enhanced ML Optimization</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="description">
        Welcome to QuOptuna! A comprehensive platform for quantum-enhanced machine learning
        with automated hyperparameter optimization, model training, and explainable AI.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Introduction
    st.markdown("## 🚀 Getting Started")
    st.markdown(
        """
        QuOptuna provides a complete workflow for training and analyzing quantum and classical machine learning models:

        1. **📊 Dataset Selection** - Load datasets from UCI ML Repository or upload your own
        2. **🎯 Optimization** - Automated hyperparameter tuning with Optuna
        3. **🔍 SHAP Analysis** - Explainable AI with comprehensive visualizations
        4. **📝 Report Generation** - AI-powered analysis reports
        """
    )

    # Feature highlights
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🎯 Optimization")
        st.markdown(
            """
            - Multi-objective optimization
            - Support for quantum & classical models
            - Automatic trial management
            - Real-time visualization
            """
        )

    with col2:
        st.markdown("### 🔍 Explainability")
        st.markdown(
            """
            - SHAP value analysis
            - Multiple visualization types
            - Feature importance ranking
            - Individual prediction explanations
            """
        )

    with col3:
        st.markdown("### 📊 Analytics")
        st.markdown(
            """
            - Performance metrics
            - Confusion matrices
            - Model comparisons
            - AI-generated reports
            """
        )

    # Quick start guide
    st.markdown("## 📖 Quick Start Guide")

    with st.expander("🎓 How to use QuOptuna", expanded=True):
        st.markdown(
            """
            ### Step-by-Step Workflow:

            #### 1. Dataset Selection
            - Navigate to **📊 Dataset Selection** in the sidebar
            - Choose a dataset from UCI ML Repository or upload your own CSV
            - Configure target and feature columns
            - Transform target values to -1 and 1 for binary classification

            #### 2. Data Preparation & Optimization
            - Go to **🎯 Optimization** page
            - Prepare your data (automatic train/test split)
            - Configure optimization parameters (number of trials, database name)
            - Run hyperparameter optimization
            - Review best performing models

            #### 3. SHAP Analysis & Reporting
            - Navigate to **🔍 SHAP Analysis** page
            - Select a trial from the best performers
            - Train the model with optimized parameters
            - Run SHAP analysis to understand feature importance
            - Generate visualizations (bar, beeswarm, violin, heatmap, waterfall)
            - Create AI-powered analysis reports with your preferred LLM provider

            ### Supported Models:

            **Quantum Models:**
            - Data Reuploading Classifier
            - Circuit-Centric Classifier
            - Quantum Kitchen Sinks
            - Quantum Metric Learner
            - Dressed Quantum Circuit Classifier

            **Classical Models:**
            - Support Vector Classifier (SVC)
            - Multi-Layer Perceptron (MLP)
            - Perceptron
            """
        )

    # Tips and best practices
    with st.expander("💡 Tips & Best Practices"):
        st.markdown(
            """
            ### Optimization Tips:
            - Start with 50-100 trials for initial exploration
            - Use more trials (100-200) for fine-tuning
            - Monitor both quantum and classical model performance

            ### SHAP Analysis Tips:
            - Use subset of data (50-100 samples) for faster computation
            - Generate multiple plot types for comprehensive understanding
            - Use waterfall plots to explain individual predictions

            ### Report Generation Tips:
            - Provide dataset context for better AI-generated insights
            - Use faster models (like Gemini Flash) for quick reports
            - Use advanced models (like GPT-4 or Gemini Pro) for detailed analysis
            """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
        <p>Built with ❤️ by the QuOptuna Team |
        <a href="https://github.com/Qentora/quoptuna" target="_blank">GitHub</a> |
        <a href="https://github.com/Qentora/quoptuna/issues" target="_blank">Report Issues</a>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
