"""Shared test fixtures for quoptuna tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    # Convert to binary labels (-1, 1) as expected by some models
    y = np.where(y == 0, -1, 1)
    return X, y


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing data preparation."""
    np.random.seed(42)
    data = {
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "feature3": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_dataframe, tmp_path):
    """Create a temporary CSV file with sample data."""
    file_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def temp_csv_with_crystal_column(tmp_path):
    """Create a temporary CSV file with 'Crystal' column for data utils testing."""
    np.random.seed(42)
    data = {
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "Crystal": np.random.choice(["A", "B"], 50),
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "crystal_data.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def trained_classifier(sample_data):
    """Create a simple trained classifier for testing."""
    X, y = sample_data
    # Use binary labels (0, 1) for sklearn models
    y_binary = np.where(y == -1, 0, 1)
    model = LogisticRegression(random_state=42)
    model.fit(X, y_binary)
    return model


@pytest.fixture
def mock_study_database(tmp_path):
    """Create a temporary database path for Optuna studies."""
    db_path = tmp_path / "test_study.db"
    return str(db_path)


@pytest.fixture
def sample_split_data(sample_data):
    """Create pre-split training and testing data."""
    from sklearn.model_selection import train_test_split

    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    return {
        "train_x": X_train,
        "test_x": X_test,
        "train_y": y_train,
        "test_y": y_test,
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
