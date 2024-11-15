from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from quoptuna.backend.data_typing import DataSet


class DataPreparation:

    def __init__(self,
                 dataset: DataSet = None,
                 file_path: str | None = None,
                 x_cols: list[str] | None = None,
                 y_col: str | None = None,
                 scaler=None):
        if dataset is not None:
            self.dataset = dataset
        elif file_path is not None:
            self.dataset = self.create_dataset(self.read_csv(file_path),
                                               x_cols, y_col)
        else:
            msg = "Either dataset or file_path must be provided"
            raise ValueError(msg)

        self.x_cols = x_cols
        self.y_col = y_col
        self.scaler = scaler or StandardScaler()

    def select_columns(self):
        """Selects specified columns and splits the dataset into features and target."""
        if self.x_cols is None or self.y_col is None:
            msg = "x_cols and y_col must be provided"
            raise ValueError(msg)
        x = self.dataset["x_train"]
        y = self.dataset["y_train"]
        return x, y

    def preprocess(self, x, y):
        """Preprocess the features and target."""
        x = self.scaler.fit_transform(x)
        classes = np.unique(y)
        y = np.where(y == classes[0], 1, -1)
        return train_test_split(x, y, random_state=42)

    def prepare_data_dict(self, x_train, y_train, x_test, y_test):
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        }

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

    @staticmethod
    def create_dataset(raw_data: pd.DataFrame, x_cols: list[str],
                       y_col: str) -> DataSet:
        """Creates a dataset from raw data."""
        x = raw_data[x_cols]
        y = raw_data[y_col]
        return {"x_train": x, "y_train": y}

    # Additional methods for SHAP preparation can be added here
