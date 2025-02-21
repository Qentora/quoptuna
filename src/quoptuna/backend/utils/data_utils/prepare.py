from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from quoptuna.backend.typing.data_typing import DataSet


class DataPreparation:
    def __init__(
        self,
        dataset: DataSet | None = None,
        file_path: str | None = None,
        x_cols: list[str] | None = None,
        y_col: str | None = None,
        scaler=None,
    ):
        self.x_cols = x_cols
        self.y_col = y_col
        self.scaler = scaler or StandardScaler()
        if dataset is not None:
            x = self.update_column_names(dataset.get("x"))
            self.set_x_cols(x.columns)
            self.dataset = {"x": x, "y": dataset.get("y")}
        elif file_path is not None:
            if x_cols is None or y_col is None:
                msg = "x_cols and y_col must be provided when file_path is used"
                raise ValueError(msg)
            self.dataset = self.create_dataset(self.read_csv(file_path), x_cols, y_col)
        else:
            msg = "Either dataset or file_path must be provided"
            raise ValueError(msg)
        self.x_train, self.x_test, self.y_train, self.y_test = self.prepare_data()

    def select_columns(self):
        """Selects specified columns and splits the dataset into features and target."""
        if self.x_cols is None or self.y_col is None:
            msg = "x_cols and y_col must be provided"
            raise ValueError(msg)
        x = self.dataset.get("x")
        y = self.dataset.get("y")
        return x, y

    def update_column_names(self, dataframe: pd.DataFrame | None = None):
        """Update column names in x_cols if they are single length after conversion to string.
        Also updates the corresponding DataFrame if provided.
        """
        if self.x_cols is not None:
            for i, col in enumerate(self.x_cols):
                if len(str(col)) == 1:
                    self.x_cols[i] = f"feat: {col}"
                    if dataframe is not None and col in dataframe.columns:
                        dataframe = dataframe.rename(columns={col: f"feat: {col}"})
        return dataframe

    def preprocess(self, x: pd.DataFrame, y: pd.Series, train_size: float = 0.75):
        """Preprocess the features and target."""
        x = pd.DataFrame(self.scaler.fit_transform(x), columns=x.columns)
        classes = np.unique(y)
        y = pd.DataFrame(
            np.where(y == classes[0], 1, -1),
            columns=[self.y_col] if not isinstance(self.y_col, list) else self.y_col,
        )
        return train_test_split(x, y, train_size=train_size, random_state=42)

    def prepare_data(self):
        """Selects columns and preprocesses the data."""
        if self.x_cols is None or self.y_col is None:
            msg = "x_cols and y_col must be provided"
            raise ValueError(msg)
        x, y = self.select_columns()
        return self.preprocess(x, y)

    @staticmethod
    def read_csv(file_path: str) -> pd.DataFrame:
        """Reads a CSV file and returns a raw dataset."""
        return pd.read_csv(file_path)

    def set_x_cols(self, x_cols: list[str]):
        self.x_cols = x_cols

    def set_y_col(self, y_col: str):
        self.y_col = y_col

    def create_dataset(self, raw_data: pd.DataFrame, x_cols: list[str], y_col: str) -> DataSet:
        """Creates a dataset from raw data."""
        x = raw_data[x_cols]
        y = raw_data[y_col]
        x = self.update_column_names(x)
        self.set_x_cols(x.columns)
        return {"x": x, "y": y}

    def prepare_data_dict(self, x_train, y_train, x_test, y_test):
        return {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

    def get_data(self, output_type: Literal["1", "2"] = "1"):
        if output_type == "1":
            return {
                "x_train": self.x_train,
                "x_test": self.x_test,
                "y_train": self.y_train,
                "y_test": self.y_test,
            }
        if output_type == "2":
            return {
                "train_x": self.x_train,
                "train_y": self.y_train,
                "test_x": self.x_test,
                "test_y": self.y_test,
            }
        return None

    # Additional methods for SHAP preparation can be added here
    # Ensure pickle is imported for serialization
