"""Extended unit tests for DataPreparation class."""

import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from quoptuna.backend.utils.data_utils.prepare import DataPreparation


def test_data_preparation_from_dataset(sample_dataframe_dataset):
    """Test DataPreparation initialization from dataset."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    # Verify dataset is set correctly
    assert data_prep.dataset is not None
    assert "x" in data_prep.dataset
    assert "y" in data_prep.dataset

    # Verify preprocessing was done
    assert data_prep.x_train is not None
    assert data_prep.x_test is not None
    assert data_prep.y_train is not None
    assert data_prep.y_test is not None


def test_data_preparation_from_file(temp_csv_file):
    """Test DataPreparation initialization from file."""
    # Get column names from the file
    df = pd.read_csv(temp_csv_file)
    x_cols = [col for col in df.columns if col != "target"]

    data_prep = DataPreparation(
        file_path=str(temp_csv_file),
        x_cols=x_cols,
        y_col="target",
    )

    # Verify data was loaded
    assert data_prep.dataset is not None
    assert data_prep.x_train is not None
    assert data_prep.x_test is not None


def test_data_preparation_requires_dataset_or_file():
    """Test that DataPreparation requires either dataset or file_path."""
    with pytest.raises(ValueError, match="Either dataset or file_path must be provided"):
        DataPreparation(dataset=None, file_path=None)


def test_data_preparation_file_requires_columns(temp_csv_file):
    """Test that file_path initialization requires x_cols and y_col."""
    with pytest.raises(ValueError, match="x_cols and y_col must be provided"):
        DataPreparation(file_path=str(temp_csv_file), x_cols=None, y_col=None)

    with pytest.raises(ValueError, match="x_cols and y_col must be provided"):
        DataPreparation(file_path=str(temp_csv_file), x_cols=["feature_0"], y_col=None)


def test_select_columns(sample_dataframe_dataset):
    """Test select_columns method."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    x, y = data_prep.select_columns()

    assert x is not None
    assert y is not None
    assert len(x) == len(y)


def test_update_column_names_single_char():
    """Test update_column_names with single character columns."""
    data = {
        "1": [1, 2, 3],
        "A": [4, 5, 6],
        "feature2": [7, 8, 9],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    X = df[["1", "A", "feature2"]]
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = ["1", "A", "feature2"]
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    # Verify single-character columns were renamed
    assert "feat: 1" in data_prep.dataset["x"].columns
    assert "feat: A" in data_prep.dataset["x"].columns
    assert "feature2" in data_prep.dataset["x"].columns


def test_update_column_names_no_single_char():
    """Test update_column_names with no single character columns."""
    data = {
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
        "feature3": [7, 8, 9],
        "target": [0, 1, 0],
    }
    df = pd.DataFrame(data)
    X = df[["feature1", "feature2", "feature3"]]
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = ["feature1", "feature2", "feature3"]
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    # Verify no columns were renamed
    assert "feature1" in data_prep.dataset["x"].columns
    assert "feature2" in data_prep.dataset["x"].columns
    assert "feature3" in data_prep.dataset["x"].columns


def test_preprocess_method(sample_dataframe_dataset):
    """Test preprocess method."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    x_train, x_test, y_train, y_test = data_prep.preprocess(X, y, train_size=0.75)

    # Verify train/test split
    total_samples = len(X)
    train_samples = len(x_train)
    test_samples = len(x_test)

    assert train_samples + test_samples == total_samples
    assert train_samples == int(total_samples * 0.75)

    # Verify labels are binary (1, -1)
    assert set(y_train[y_train.columns[0]].unique()).issubset({1, -1})
    assert set(y_test[y_test.columns[0]].unique()).issubset({1, -1})


def test_prepare_data_dict(sample_dataframe_dataset):
    """Test prepare_data_dict method."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    result = data_prep.prepare_data_dict(
        data_prep.x_train,
        data_prep.y_train,
        data_prep.x_test,
        data_prep.y_test,
    )

    assert "x_train" in result
    assert "x_test" in result
    assert "y_train" in result
    assert "y_test" in result


def test_get_data_output_type_1(sample_dataframe_dataset):
    """Test get_data with output_type='1'."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    data = data_prep.get_data(output_type="1")

    assert "x_train" in data
    assert "x_test" in data
    assert "y_train" in data
    assert "y_test" in data


def test_get_data_output_type_2(sample_dataframe_dataset):
    """Test get_data with output_type='2'."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    data = data_prep.get_data(output_type="2")

    assert "train_x" in data
    assert "test_x" in data
    assert "train_y" in data
    assert "test_y" in data


def test_get_data_invalid_output_type(sample_dataframe_dataset):
    """Test get_data with invalid output_type."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    data = data_prep.get_data(output_type="3")

    assert data is None


def test_custom_scaler(sample_dataframe_dataset):
    """Test DataPreparation with custom scaler."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    custom_scaler = MinMaxScaler()

    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target", scaler=custom_scaler)

    # Verify custom scaler was used
    assert isinstance(data_prep.scaler, MinMaxScaler)

    # Verify data is scaled between 0 and 1
    assert data_prep.x_train.min().min() >= 0
    assert data_prep.x_train.max().max() <= 1


def test_read_csv_static_method(temp_csv_file):
    """Test read_csv static method."""
    df = DataPreparation.read_csv(str(temp_csv_file))

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_set_x_cols(sample_dataframe_dataset):
    """Test set_x_cols method."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    new_cols = ["new_col1", "new_col2"]
    data_prep.set_x_cols(new_cols)

    assert data_prep.x_cols == new_cols


def test_set_y_col(sample_dataframe_dataset):
    """Test set_y_col method."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    data_prep.set_y_col("new_target")

    assert data_prep.y_col == "new_target"


def test_create_dataset(sample_dataframe_dataset, temp_csv_file):
    """Test create_dataset method."""
    df = pd.read_csv(temp_csv_file)
    x_cols = [col for col in df.columns if col != "target"]

    data_prep_temp = DataPreparation(
        file_path=str(temp_csv_file),
        x_cols=x_cols,
        y_col="target",
    )

    result = data_prep_temp.create_dataset(df, x_cols, "target")

    assert "x" in result
    assert "y" in result
    assert len(result["x"]) == len(df)
    assert len(result["y"]) == len(df)


def test_data_preparation_train_test_split_ratio(sample_dataframe_dataset):
    """Test that train/test split maintains correct ratio."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    total = len(X)
    train_size = len(data_prep.x_train)
    test_size = len(data_prep.x_test)

    # Default train_size is 0.75
    assert train_size == int(total * 0.75)
    assert train_size + test_size == total


def test_data_preparation_data_types(sample_dataframe_dataset):
    """Test that DataPreparation maintains correct data types."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    # All should be DataFrames
    assert isinstance(data_prep.x_train, pd.DataFrame)
    assert isinstance(data_prep.x_test, pd.DataFrame)
    assert isinstance(data_prep.y_train, pd.DataFrame)
    assert isinstance(data_prep.y_test, pd.DataFrame)


def test_data_preparation_column_preservation(sample_dataframe_dataset):
    """Test that column names are preserved through preprocessing."""
    df = sample_dataframe_dataset
    X = df.drop(columns=["target"])
    y = df["target"]

    original_columns = list(X.columns)

    dataset = {"x": X, "y": y}
    x_cols = list(X.columns)
    data_prep = DataPreparation(dataset=dataset, x_cols=x_cols, y_col="target")

    # Check that column names match (may have some renamed if single char)
    assert len(data_prep.x_train.columns) == len(original_columns)
    assert len(data_prep.x_test.columns) == len(original_columns)
