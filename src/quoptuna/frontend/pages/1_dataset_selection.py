"""Dataset Selection Page - Select UCI dataset or upload custom data."""

import os

import pandas as pd
import streamlit as st
from ucimlrepo import fetch_ucirepo

from quoptuna.backend.utils.data_utils.data import mock_csv_data


def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "dataset_loaded": False,
        "dataset_df": None,
        "dataset_name": None,
        "dataset_metadata": None,
        "file_path": None,
        "target_column": None,
        "feature_columns": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def fetch_uci_dataset():
    """Fetch dataset from UCI ML Repository."""
    st.subheader("📊 UCI ML Repository")

    # Popular datasets for quick access
    popular_datasets = {
        "Statlog (Australian Credit Approval)": 143,
        "Blood Transfusion Service Center": 176,
        "Banknote Authentication": 267,
        "Heart Disease": 45,
        "Ionosphere": 225,
    }

    dataset_choice = st.radio(
        "Choose dataset source:",
        ["Popular Datasets", "Custom UCI ID"],
        horizontal=True,
    )

    dataset_id = None
    dataset_name = None

    if dataset_choice == "Popular Datasets":
        selected = st.selectbox(
            "Select a dataset:",
            list(popular_datasets.keys()),
        )
        if selected:
            dataset_id = popular_datasets[selected]
            dataset_name = selected
    else:
        dataset_id = st.number_input(
            "Enter UCI Dataset ID:",
            min_value=1,
            max_value=1000,
            value=143,
            help="Find dataset IDs at https://archive.ics.uci.edu/datasets",
        )
        dataset_name = f"UCI_Dataset_{dataset_id}"

    if st.button("Load UCI Dataset", type="primary"):
        try:
            with st.spinner(f"Fetching dataset {dataset_id}..."):
                dataset = fetch_ucirepo(id=dataset_id)

                # Combine features and targets
                X = dataset.data.features
                y = dataset.data.targets
                df = pd.concat([X, y], axis=1)

                # Store in session state
                st.session_state["dataset_df"] = df
                st.session_state["dataset_name"] = dataset_name
                st.session_state["dataset_metadata"] = dataset.metadata
                st.session_state["dataset_loaded"] = True

                st.success(f"✅ Dataset '{dataset_name}' loaded successfully!")

                # Display metadata
                with st.expander("📋 Dataset Metadata", expanded=True):
                    metadata = dataset.metadata
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Instances", metadata.get("num_instances", "N/A"))
                        st.metric("Features", metadata.get("num_features", "N/A"))
                    with col2:
                        st.metric("Area", metadata.get("area", "N/A"))
                        st.metric("Tasks", ", ".join(metadata.get("tasks", [])))

                    st.write("**Abstract:**", metadata.get("abstract", "N/A"))

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")


def upload_custom_dataset():
    """Upload custom CSV dataset."""
    st.subheader("📁 Upload Custom Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file with your dataset",
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Store in session state
            st.session_state["dataset_df"] = df
            st.session_state["dataset_name"] = uploaded_file.name.replace(".csv", "")
            st.session_state["dataset_loaded"] = True

            st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")


def configure_dataset():
    """Configure target and feature columns."""
    if not st.session_state["dataset_loaded"]:
        return

    df = st.session_state["dataset_df"]

    st.subheader("⚙️ Configure Dataset")

    # Show data preview
    with st.expander("👁️ Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing)

    # Column selection
    st.markdown("### Select Target Column")
    target_col = st.selectbox(
        "Target Column (y):",
        df.columns.tolist(),
        help="Select the column you want to predict",
    )

    st.markdown("### Feature Columns")
    feature_cols = st.multiselect(
        "Feature Columns (X):",
        [col for col in df.columns if col != target_col],
        default=[col for col in df.columns if col != target_col],
        help="Select the features to use for prediction",
    )

    # Target transformation
    st.markdown("### Target Transformation")
    st.info("⚠️ QuOptuna requires binary classification targets to be encoded as -1 and 1")

    unique_values = df[target_col].unique()
    st.write(f"Current unique values in target: {unique_values}")

    transform_target = st.checkbox(
        "Transform target values to -1 and 1",
        value=True,
        help="Automatically transform binary targets to -1 and 1",
    )

    if transform_target and len(unique_values) == 2:
        col1, col2 = st.columns(2)
        with col1:
            negative_value = st.selectbox("Map to -1:", unique_values)
        with col2:
            positive_value = st.selectbox(
                "Map to 1:",
                [v for v in unique_values if v != negative_value],
            )

    # Save configuration
    if st.button("💾 Save Configuration", type="primary"):
        if not feature_cols:
            st.error("Please select at least one feature column!")
            return

        # Apply transformations
        processed_df = df.copy()

        # Handle missing values
        if processed_df.isnull().sum().sum() > 0:
            st.warning("⚠️ Removing rows with missing values...")
            processed_df = processed_df.dropna()

        # Transform target if needed
        if transform_target and len(unique_values) == 2:
            processed_df[target_col] = processed_df[target_col].replace(
                {negative_value: -1, positive_value: 1}
            )
            st.success(f"✅ Transformed target: {negative_value} → -1, {positive_value} → 1")

        # Rename target column to 'target'
        if target_col != "target":
            processed_df = processed_df.rename(columns={target_col: "target"})
            target_col = "target"

        # Keep only selected columns
        processed_df = processed_df[feature_cols + [target_col]]

        # Save to file
        data_dir = "data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        file_path = mock_csv_data(
            processed_df,
            tmp_path=data_dir,
            file_name=st.session_state["dataset_name"],
        )

        # Update session state
        st.session_state["file_path"] = file_path
        st.session_state["target_column"] = target_col
        st.session_state["feature_columns"] = feature_cols
        st.session_state["dataset_df"] = processed_df

        st.success(f"✅ Configuration saved! Data saved to: {file_path}")
        st.info("👉 Proceed to the next page: **Data Preparation & Optimization**")


def main():
    """Main function for dataset selection page."""
    st.set_page_config(
        page_title="Dataset Selection - QuOptuna",
        page_icon="📊",
        layout="wide",
    )

    initialize_session_state()

    st.title("📊 Dataset Selection")
    st.markdown(
        """
        Select a dataset from the UCI ML Repository or upload your own CSV file.
        This is the first step in the QuOptuna workflow.
        """
    )

    # Dataset source selection
    tab1, tab2 = st.tabs(["🌐 UCI ML Repository", "📁 Upload Custom Dataset"])

    with tab1:
        fetch_uci_dataset()

    with tab2:
        upload_custom_dataset()

    # Show configuration if dataset is loaded
    if st.session_state["dataset_loaded"]:
        st.divider()
        configure_dataset()


if __name__ == "__main__":
    main()
