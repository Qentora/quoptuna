"""Shared test fixtures for quoptuna tests."""


import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def sample_classification_dataset():
    """Create a small sample classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    # Convert to binary labels: 1 and -1
    y = np.where(y == 0, 1, -1)
    return X, y


@pytest.fixture
def sample_dataframe_dataset():
    """Create a sample DataFrame dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    # Convert to binary labels: 1 and -1
    y = np.where(y == 0, 1, -1)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    return df


@pytest.fixture
def train_test_split_data(sample_classification_dataset):
    """Create train/test split data."""
    X, y = sample_classification_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {
        "train_x": X_train,
        "test_x": X_test,
        "train_y": y_train,
        "test_y": y_test,
    }


@pytest.fixture
def train_test_split_dataframes(sample_dataframe_dataset):
    """Create train/test split data as DataFrames."""
    df = sample_dataframe_dataset

    # Split features and target
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return {
        "x_train": X_train_scaled,
        "x_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def trained_classifier(train_test_split_data):
    """Create a trained classifier for testing."""
    model = LogisticRegression(random_state=42)
    model.fit(train_test_split_data["train_x"], train_test_split_data["train_y"])
    return model


@pytest.fixture
def temp_csv_file(tmp_path, sample_dataframe_dataset):
    """Create a temporary CSV file for testing."""
    df = sample_dataframe_dataset
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    return tmp_path / "test_db.db"


@pytest.fixture
def mock_optuna_study_data():
    """Create mock data for Optuna study testing."""
    return {
        "study_name": "test_study",
        "n_trials": 5,
        "direction": "maximize",
    }
