"""Tests for data utility functions."""

import socket

import numpy as np
import pandas as pd

from quoptuna.backend.utils.data_utils.data import (
    find_free_port,
    load_data,
    mock_csv_data,
    preprocess_data,
)


def test_load_data(temp_csv_with_crystal_column):
    """Test loading data from CSV file with Crystal column."""
    x, y = load_data(str(temp_csv_with_crystal_column))

    # Check that data was loaded correctly
    assert isinstance(x, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(x) == 50  # noqa: PLR2004
    assert len(y) == 50  # noqa: PLR2004
    assert "Crystal" not in x.columns
    assert y.name == "Crystal"


def test_preprocess_data():
    """Test preprocessing data with scaling and binary label conversion."""
    # Create sample data
    np.random.seed(42)
    x = pd.DataFrame({"feature1": np.random.randn(100), "feature2": np.random.randn(100)})
    y = pd.Series(np.random.choice(["A", "B"], 100))

    # Preprocess the data
    x_train, x_test, y_train, y_test = preprocess_data(x, y)

    # Check shapes
    assert len(x_train) + len(x_test) == 100  # noqa: PLR2004
    assert len(y_train) + len(x_test) == 100  # noqa: PLR2004

    # Check that labels are binary (-1, 1)
    assert set(np.unique(y_train)).issubset({-1, 1})
    assert set(np.unique(y_test)).issubset({-1, 1})

    # Check that features are scaled (mean should be close to 0)
    assert np.abs(x_train.mean()) < 0.5
    assert np.abs(x_test.mean()) < 0.5


def test_preprocess_data_with_different_classes():
    """Test preprocessing with different class labels."""
    np.random.seed(42)
    x = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50)})
    y = pd.Series(np.random.choice(["ClassX", "ClassY"], 50))

    x_train, x_test, y_train, y_test = preprocess_data(x, y)

    # Verify binary conversion
    assert set(np.unique(y_train)).issubset({-1, 1})
    assert set(np.unique(y_test)).issubset({-1, 1})


def test_find_free_port():
    """Test finding a free port."""
    port = find_free_port()

    # Check that a port was found
    assert port is not None
    assert isinstance(port, int)
    assert 6000 <= port < 7000  # noqa: PLR2004

    # Verify the port is actually free
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(("localhost", port))
    sock.close()
    # Result should be non-zero (connection refused) if port is free
    assert result != 0


def test_mock_csv_data(tmp_path):
    """Test creating mock CSV data."""
    data = {
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
        "target": [0, 1, 0],
    }

    file_path = mock_csv_data(data, tmp_path)

    # Check that file was created
    assert file_path.exists()
    assert file_path.suffix == ".csv"

    # Load and verify data
    df = pd.read_csv(file_path)
    assert len(df) == 3  # noqa: PLR2004
    assert list(df.columns) == ["col1", "col2", "target"]
    assert df["col1"].tolist() == [1, 2, 3]


def test_mock_csv_data_with_custom_filename(tmp_path):
    """Test creating mock CSV data with a custom filename."""
    data = {"x": [1, 2], "y": [3, 4]}
    file_path = mock_csv_data(data, tmp_path, file_name="custom_name")

    # Check filename
    assert file_path.name == "custom_name.csv"
    assert file_path.exists()


def test_preprocess_data_deterministic():
    """Test that preprocessing produces deterministic results."""
    np.random.seed(100)
    x = pd.DataFrame({"feature1": np.random.randn(50), "feature2": np.random.randn(50)})
    y = pd.Series(np.random.choice(["A", "B"], 50))

    # Run preprocessing twice
    result1 = preprocess_data(x, y)
    result2 = preprocess_data(x, y)

    # Results should be identical due to fixed random_state
    np.testing.assert_array_equal(result1[0], result2[0])  # x_train
    np.testing.assert_array_equal(result1[1], result2[1])  # x_test
    np.testing.assert_array_equal(result1[2], result2[2])  # y_train
    np.testing.assert_array_equal(result1[3], result2[3])  # y_test
