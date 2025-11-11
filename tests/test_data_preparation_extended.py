"""Extended tests for DataPreparation class covering edge cases."""

import numpy as np
import pandas as pd

from quoptuna.backend.utils.data_utils.prepare import DataPreparation


def test_data_preparation_from_dataset_dict():
    """Test DataPreparation initialization from dataset dictionary."""
    x = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    assert data_prep.dataset is not None
    assert "x" in data_prep.dataset
    assert "y" in data_prep.dataset


def test_data_preparation_select_columns():
    """Test selecting columns from dataset."""
    x = pd.DataFrame({"feat1": [1, 2, 3], "feat2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    x_selected, y_selected = data_prep.select_columns()

    assert x_selected.shape == (3, 2)
    assert len(y_selected) == 3  # noqa: PLR2004


def test_data_preparation_prepare_data_split():
    """Test data preparation and train/test split."""
    np.random.seed(42)
    x = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
    })
    y = pd.Series(np.random.choice([0, 1], 100))

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    # The prepare_data is already called in __init__
    x_train, x_test, y_train, y_test = (
        data_prep.x_train,
        data_prep.x_test,
        data_prep.y_train,
        data_prep.y_test,
    )

    # Check that data was split
    assert len(x_train) + len(x_test) == 100  # noqa: PLR2004
    assert len(y_train) + len(y_test) == 100  # noqa: PLR2004
    # Default train_size is 0.75, so test size is 0.25
    assert len(x_test) == 25  # noqa: PLR2004


def test_data_preparation_get_data_type1():
    """Test get_data method with output_type='1'."""
    x = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    data = data_prep.get_data(output_type="1")

    assert "x_train" in data
    assert "x_test" in data
    assert "y_train" in data
    assert "y_test" in data


def test_data_preparation_get_data_type2():
    """Test get_data method with output_type='2'."""
    x = pd.DataFrame({"feat1": [1, 2, 3, 4, 5], "feat2": [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    data = data_prep.get_data(output_type="2")

    assert "train_x" in data
    assert "test_x" in data
    assert "train_y" in data
    assert "test_y" in data


def test_data_preparation_with_numeric_column_names():
    """Test handling of numeric column names."""
    x = pd.DataFrame({
        "1": [1, 2, 3],
        "2": [4, 5, 6],
        "normal": [7, 8, 9],
    })
    y = pd.Series([0, 1, 0])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["1", "2", "normal"],
        y_col="target",
    )

    # Check that single-digit column names were updated
    assert "feat: 1" in data_prep.dataset["x"].columns
    assert "feat: 2" in data_prep.dataset["x"].columns
    assert "normal" in data_prep.dataset["x"].columns


def test_data_preparation_from_csv(temp_csv_file):
    """Test DataPreparation from CSV file."""
    data_prep = DataPreparation(
        file_path=str(temp_csv_file),
        x_cols=["feature1", "feature2", "feature3"],
        y_col="target",
    )

    assert data_prep.dataset is not None
    assert len(data_prep.dataset["x"]) == 50  # noqa: PLR2004


def test_data_preparation_with_mixed_column_names():
    """Test handling of mixed column name types."""
    x = pd.DataFrame({
        "a": [1, 2, 3],
        "1": [4, 5, 6],
        "feature_3": [7, 8, 9],
        "2": [10, 11, 12],
    })
    y = pd.Series([0, 1, 0])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["a", "1", "feature_3", "2"],
        y_col="target",
    )

    columns = data_prep.dataset["x"].columns.tolist()

    # Check that all single-character names were changed
    assert "feat: a" in columns  # Single letter changed
    assert "feat: 1" in columns  # Single digit changed
    assert "feature_3" in columns  # Normal name not changed
    assert "feat: 2" in columns  # Single digit changed


def test_data_preparation_empty_dataset():
    """Test DataPreparation with minimal data."""
    x = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
    y = pd.Series([0, 1])

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    assert len(data_prep.dataset["x"]) == 2  # noqa: PLR2004
    assert len(data_prep.dataset["y"]) == 2  # noqa: PLR2004


def test_data_preparation_default_split():
    """Test DataPreparation with default train/test split."""
    np.random.seed(42)
    x = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
    })
    y = pd.Series(np.random.choice([0, 1], 100))

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": x, "y": y},
        x_cols=["feat1", "feat2"],
        y_col="target",
    )

    # Default train_size is 0.75
    assert len(data_prep.x_test) == 25  # noqa: PLR2004
    assert len(data_prep.x_train) == 75  # noqa: PLR2004
