from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from quoptuna.backend.typing.data_typing import DataSet

MAX_ONEHOT_CATEGORIES = 10
MAX_ORDINAL_CATEGORIES = 100


def encode_features(x: pd.DataFrame, method: str = "ordinal") -> tuple[pd.DataFrame, dict]:
    """Make a feature frame numeric: impute NaN and encode categorical columns.

    - Numeric columns keep their name; NaN imputed with the column median.
    - Bool columns are cast to float.
    - Object/categorical columns get NaN imputed as "missing" and encoded per
      ``method``:
      - ``"ordinal"`` (default): sorted-category integer codes scaled to
        [0, 1] — one column per feature, so quantum circuit width does not
        grow. Cap ``MAX_ORDINAL_CATEGORIES``.
      - ``"onehot"``: ``<col>_<category>`` 0/1 columns. Cap
        ``MAX_ONEHOT_CATEGORIES`` — unbounded growth blows up qubit width.

    Encoded columns are already well-scaled and must NOT be re-scaled by
    StandardScaler downstream (see ``DataPreparation(passthrough_columns=...)``).

    Row order and count are preserved (positional-index alignment with the raw
    dataframe is relied upon downstream, e.g. by the fairness audit).

    Returns the encoded frame and per-column metadata
    ``{column: {"kind", "categories"?, "imputed"}}``. The metadata's non-numeric
    entries identify the passthrough (encoded) output columns.
    """
    if method not in ("ordinal", "onehot"):
        msg = f"Unknown categorical encoding method: {method!r}"
        raise ValueError(msg)

    parts: list[pd.DataFrame] = []
    encoding: dict[str, dict] = {}
    for col in x.columns:
        series = x[col]
        n_missing = int(series.isna().sum())
        if pd.api.types.is_bool_dtype(series):
            parts.append(series.astype(float).to_frame())
            encoding[str(col)] = {"kind": "numeric", "imputed": 0}
        elif pd.api.types.is_numeric_dtype(series):
            if n_missing:
                series = series.fillna(series.median())
            parts.append(series.astype(float).to_frame())
            encoding[str(col)] = {"kind": "numeric", "imputed": n_missing}
        else:
            series = series.astype(str).where(~series.isna(), "missing")
            categories = sorted(series.unique().tolist())
            if method == "onehot":
                if len(categories) > MAX_ONEHOT_CATEGORIES:
                    msg = (
                        f"Column '{col}' has {len(categories)} unique categories "
                        f"(max {MAX_ONEHOT_CATEGORIES} for one-hot encoding). "
                        "Use ordinal encoding or drop the column."
                    )
                    raise ValueError(msg)
                dummies = pd.get_dummies(series, prefix=str(col), dtype=float)
                parts.append(dummies)
                encoding[str(col)] = {
                    "kind": "onehot",
                    "categories": categories,
                    "imputed": n_missing,
                    "columns": [str(c) for c in dummies.columns],
                }
            else:
                if len(categories) > MAX_ORDINAL_CATEGORIES:
                    msg = (
                        f"Column '{col}' has {len(categories)} unique categories "
                        f"(max {MAX_ORDINAL_CATEGORIES} for ordinal encoding). "
                        "It looks like an identifier — drop it before optimizing."
                    )
                    raise ValueError(msg)
                code_of = {cat: i for i, cat in enumerate(categories)}
                denom = float(max(len(categories) - 1, 1))
                codes = series.map(code_of).astype(float) / denom
                parts.append(codes.to_frame(name=str(col)))
                encoding[str(col)] = {
                    "kind": "ordinal",
                    "categories": categories,
                    "imputed": n_missing,
                    "columns": [str(col)],
                }
    encoded = pd.concat(parts, axis=1) if parts else x.copy()
    encoded.index = x.index
    return encoded, encoding


def encoded_passthrough_columns(encoding: dict) -> list[str]:
    """Output columns produced by categorical encoding (already in [0, 1])."""
    passthrough: list[str] = []
    for meta in encoding.values():
        if meta["kind"] in ("ordinal", "onehot"):
            passthrough.extend(meta["columns"])
    return passthrough


class DataPreparation:
    def __init__(  # noqa: PLR0913
        self,
        dataset: DataSet | None = None,
        file_path: str | None = None,
        x_cols: list[str] | None = None,
        y_col: str | None = None,
        scaler=None,
        passthrough_columns: list[str] | None = None,
    ):
        self.x_cols = x_cols
        self.y_col = y_col
        self.scaler = scaler or StandardScaler()
        # Columns that bypass the scaler — encoded categoricals are already in
        # [0, 1]; z-scoring 0/1 dummies distorts quantum feature maps.
        self.passthrough_columns = passthrough_columns or []
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
        passthrough = [c for c in x.columns if c in set(self.passthrough_columns)]
        to_scale = [c for c in x.columns if c not in set(passthrough)]
        if to_scale:
            scaled = pd.DataFrame(self.scaler.fit_transform(x[to_scale]), columns=to_scale)
        else:
            scaled = pd.DataFrame(index=range(len(x)))
        if passthrough:
            kept = x[passthrough].reset_index(drop=True)
            x = pd.concat([scaled, kept], axis=1)[list(x.columns)]
        else:
            x = scaled
        # Index is a fresh RangeIndex either way, so split indices remain
        # positional row numbers into the raw dataframe.
        classes = np.unique(y)
        # Labels already encoded to {-1, 1} (e.g. an explicit user mapping applied
        # upstream) must pass through unchanged — re-encoding would invert them.
        # Multiclass targets (K>2) also pass through: they arrive pre-encoded to
        # 0..K-1 codes, and the binary re-encode would silently collapse them.
        y_values: np.ndarray
        if set(classes.tolist()) <= {-1, 1} or len(classes) != 2:  # noqa: PLR2004
            y_values = np.asarray(y).ravel()
        else:
            y_values = np.where(np.asarray(y).ravel() == classes[0], 1, -1)
        y = pd.DataFrame(
            y_values,
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
