"""Unit tests for data utilities."""

import numpy as np
import pandas as pd

from quoptuna.backend.utils.data_utils.data import (
    find_free_port,
    load_data,
    mock_csv_data,
    preprocess_data,
)


def test_load_data(tmp_path):
    """Test loading data from CSV file."""
    # Create a CSV with Crystal column
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "Crystal": ["A", "B", "A", "B", "A"],
        }
    )
    file_path = tmp_path / "test_data_crystal.csv"
    df.to_csv(file_path, index=False)

    # Load the data
    x, y = load_data(str(file_path))

    # Verify x is a DataFrame without the target column
    assert isinstance(x, pd.DataFrame)
    assert "Crystal" not in x.columns
    assert len(x.columns) > 0

    # Verify y is a Series
    assert isinstance(y, pd.Series)
    assert len(y) == len(x)


def test_load_data_with_crystal_column(tmp_path):
    """Test loading data when CSV has a Crystal column."""
    # Create a CSV with Crystal column
    df = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [5, 4, 3, 2, 1],
            "Crystal": ["A", "B", "A", "B", "A"],
        }
    )
    file_path = tmp_path / "crystal_data.csv"
    df.to_csv(file_path, index=False)

    # Load the data
    x, y = load_data(str(file_path))

    # Verify Crystal column is in y
    assert isinstance(x, pd.DataFrame)
    assert "Crystal" not in x.columns
    assert isinstance(y, pd.Series)
    assert y.name == "Crystal"


def test_preprocess_data(sample_classification_dataset):
    """Test preprocessing data."""
    X, y = sample_classification_dataset

    # Convert to DataFrame and Series
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X_df, y_series)

    # Verify train/test split
    assert len(X_train) + len(X_test) == len(X_df)
    assert len(y_train) + len(y_test) == len(y_series)

    # Verify scaling was applied (mean should be close to 0, std close to 1)
    assert np.abs(X_train.mean()) < 0.5
    assert np.abs(X_train.std() - 1.0) < 0.5

    # Verify labels are converted to 1 and -1
    assert set(np.unique(y_train)).issubset({1, -1})
    assert set(np.unique(y_test)).issubset({1, -1})


def test_preprocess_data_maintains_shape():
    """Test that preprocessing maintains correct data shapes."""
    # Create test data
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], 100))

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Verify shapes
    assert X_train.shape[1] == X.shape[1]
    assert X_test.shape[1] == X.shape[1]
    assert len(y_train.shape) == 1
    assert len(y_test.shape) == 1


def test_preprocess_data_with_multiclass():
    """Test preprocessing with multiclass labels."""
    # Create test data with 3 classes
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1, 2], 100))

    # Preprocess
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # First class should be mapped to 1, others to -1
    unique_train = set(np.unique(y_train))
    unique_test = set(np.unique(y_test))

    assert unique_train.issubset({1, -1})
    assert unique_test.issubset({1, -1})


def test_find_free_port():
    """Test finding a free port."""
    port = find_free_port()

    # Verify port is in expected range
    assert port is not None
    assert 6000 <= port < 7000


def test_mock_csv_data(tmp_path):
    """Test creating mock CSV data."""
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0],
    }

    # Create CSV file
    file_path = mock_csv_data(data, tmp_path, file_name="test_mock")

    # Verify file was created
    assert file_path.exists()
    assert file_path.name == "test_mock.csv"

    # Verify content
    df = pd.read_csv(file_path)
    assert len(df) == 5
    assert list(df.columns) == ["feature1", "feature2", "target"]


def test_mock_csv_data_default_name(tmp_path):
    """Test creating mock CSV data with default filename."""
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}

    # Create CSV file without specifying name
    file_path = mock_csv_data(data, tmp_path)

    # Verify file was created with default name
    assert file_path.exists()
    assert file_path.name == "mock_csv.csv"


def test_preprocess_data_random_state_consistency():
    """Test that preprocessing produces consistent results with fixed random state."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.choice([0, 1], 100))

    # Run preprocessing twice
    X_train1, X_test1, y_train1, y_test1 = preprocess_data(X, y)
    X_train2, X_test2, y_train2, y_test2 = preprocess_data(X, y)

    # Verify results are identical (arrays are returned)
    np.testing.assert_array_equal(X_train1, X_train2)
    np.testing.assert_array_equal(X_test1, X_test2)
    np.testing.assert_array_equal(y_train1, y_train2)
    np.testing.assert_array_equal(y_test1, y_test2)


def test_load_data_file_structure(tmp_path):
    """Test load_data with different file structures."""
    # Create a CSV with multiple feature columns
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "x3": [7.0, 8.0, 9.0],
            "Crystal": ["A", "B", "C"],
        }
    )
    file_path = tmp_path / "multi_feature.csv"
    df.to_csv(file_path, index=False)

    x, y = load_data(str(file_path))

    # Verify all feature columns are included
    assert x.shape[1] == 3
    assert "x1" in x.columns
    assert "x2" in x.columns
    assert "x3" in x.columns
    assert "Crystal" not in x.columns
