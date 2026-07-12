"""Resolve a raw dataset's sensitive-attribute column against a train/test split.

``DataPreparation.preprocess`` resets the feature index to a RangeIndex before
its seeded ``train_test_split``, so split indices are positional row numbers
into the raw dataframe (post feature-selection, which only selects columns).
Both the post-hoc fairness audit and the fairness-aware search rely on this
positional alignment, so it lives here as the single implementation.
"""

from __future__ import annotations

import pandas as pd

from quoptuna.backend.xai.fairness import MAX_SENSITIVE_GROUPS
from quoptuna.server.services import dataset_registry


class SensitiveColumnError(ValueError):
    """Raised when the sensitive column cannot be resolved or aligned."""


def resolve_sensitive_series(
    dataset_id: str,
    column: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return the sensitive column split into (train, test) series.

    Raises ``SensitiveColumnError`` with a user-facing message when the dataset
    is missing, the column is absent, the row counts no longer match the split,
    or the column has too many unique groups.
    """
    record = dataset_registry.get(dataset_id)
    if not record or not record.get("file_path"):
        raise SensitiveColumnError("Dataset file not found in registry")

    raw_df = pd.read_csv(record["file_path"]).reset_index(drop=True)
    if column not in raw_df.columns:
        raise SensitiveColumnError(f"Column '{column}' not in dataset")

    n_split = len(x_train) + len(x_test)
    if len(raw_df) != n_split:
        raise SensitiveColumnError(
            f"Dataset rows ({len(raw_df)}) do not match the optimization split "
            f"({n_split}); the dataset file may have changed since the run"
        )

    series = raw_df[column]
    if series.nunique() > MAX_SENSITIVE_GROUPS:
        raise SensitiveColumnError(
            f"Column '{column}' has {series.nunique()} unique values "
            f"(max {MAX_SENSITIVE_GROUPS}); pick a categorical column"
        )
    return series.iloc[x_train.index], series.iloc[x_test.index]
