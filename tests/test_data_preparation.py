# test_data_preparation.py
import pandas as pd
import pytest

from quoptuna.backend.utils.data_utils.prepare import DataPreparation


# Mock sample data
@pytest.fixture
def mock_csv_data(tmp_path):
    data = {"feature1": [1, 2, 3, 4, 5], "feature2": [5, 4, 3, 2, 1], "target": [0, 1, 0, 1, 0]}
    dataframe = pd.DataFrame(data)
    file_path = tmp_path / "mock_data.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


# Define a constant for the number of samples
NUM_SAMPLES = 5


def test_data_preparation_with_csv(mock_csv_data):
    """Test DataPreparation with a mock CSV file."""
    data_prep = DataPreparation(
        file_path=str(mock_csv_data), x_cols=["feature1", "feature2"], y_col="target"
    )

    # Test if dataset is created correctly
    assert data_prep.dataset["x"].shape[0] == NUM_SAMPLES
    assert data_prep.dataset["y"].shape[0] == NUM_SAMPLES

    # Test select_columns method
    x, y = data_prep.select_columns()
    assert x.shape[0] == NUM_SAMPLES
    assert y.shape[0] == NUM_SAMPLES

    # Test preprocess method
    x_train, x_test, y_train, y_test = data_prep.prepare_data()
    assert x_train.shape[0] + x_test.shape[0] == NUM_SAMPLES
    assert y_train.shape[0] + y_test.shape[0] == NUM_SAMPLES

    # Test get_data method with different output types
    data_type1 = data_prep.get_data(output_type="1")
    assert "x_train" in data_type1
    assert "x_test" in data_type1
    assert "y_train" in data_type1
    assert "y_test" in data_type1

    data_type2 = data_prep.get_data(output_type="2")
    assert "train_x" in data_type2
    assert "test_x" in data_type2
    assert "train_y" in data_type2
    assert "test_y" in data_type2


def test_update_column_names():
    """Test column name updating functionality."""
    data = {"1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
    data_frame = pd.DataFrame(data)

    data_prep = DataPreparation(
        file_path=None,
        dataset={"x": data_frame[["1", "feature2"]], "y": data_frame["target"]},
        x_cols=["1", "feature2"],
        y_col="target",
    )

    # Test if single-character column names are updated
    assert "feat: 1" in data_prep.dataset["x"].columns
    assert "feature2" in data_prep.dataset["x"].columns
