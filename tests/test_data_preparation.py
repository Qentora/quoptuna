# test_data_preparation.py
import pytest
import pandas as pd
from quoptuna.backend.data_processing.prepare import DataPreparation


# Mock sample data
@pytest.fixture
def mock_csv_data(tmp_path):
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0]
    }
    dataframe = pd.DataFrame(data)
    file_path = tmp_path / "mock_data.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


def test_data_preparation_with_csv(mock_csv_data):
    """Test DataPreparation with a mock CSV file."""
    data_prep = DataPreparation(file_path=str(mock_csv_data),
                                x_cols=["feature1", "feature2"],
                                y_col="target")

    # Test if dataset is created correctly
    assert data_prep.dataset["x_train"].shape[0] == 5
    assert data_prep.dataset["y_train"].shape[0] == 5

    # Test select_columns method
    x, y = data_prep.select_columns()
    assert x.shape[0] == 5
    assert y.shape[0] == 5

    # Test preprocess method
    x_train, x_test, y_train, y_test = data_prep.prepare_data()
    assert x_train.shape[0] + x_test.shape[0] == 5
    assert y_train.shape[0] + y_test.shape[0] == 5
