"""Resolve a raw dataset's sensitive-attribute column against a train/test split.

``DataPreparation.preprocess`` resets the feature index to a RangeIndex before
its seeded ``train_test_split``, so split indices are positional row numbers
into the raw dataframe (post feature-selection, which only selects columns).
Both the post-hoc fairness audit and the fairness-aware search rely on this
positional alignment, so it lives here as the single implementation.

When TRAIN resampling is used (see ``data_utils.resampling``), the resampled
train frame's index is no longer positional into the raw file — duplicated or
dropped rows break that mapping. The train side is then resolved once, at
resampling time, from the raw file (while the index is still positional) and
resampled in lockstep with x_train/y_train; see
``WorkflowExecutor._execute_train_test_split``. Everything downstream of that
either receives the already-resampled series or, for the TEST split (never
resampled), can resolve positionally as before via ``resolve_sensitive_test_series``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quoptuna.backend.xai.fairness import MAX_SENSITIVE_GROUPS
from quoptuna.server.services import dataset_registry

if TYPE_CHECKING:
    import pandas as pd


class SensitiveColumnError(ValueError):
    """Raised when the sensitive column cannot be resolved or aligned."""


def _load_sensitive_column(dataset_id: str, column: str) -> pd.Series:
    record = dataset_registry.get(dataset_id)
    if not record or not record.get("file_path"):
        raise SensitiveColumnError("Dataset file not found in registry")

    import pandas as pd

    raw_df = pd.read_csv(record["file_path"]).reset_index(drop=True)
    if column not in raw_df.columns:
        raise SensitiveColumnError(f"Column '{column}' not in dataset")

    series = raw_df[column]
    if series.nunique() > MAX_SENSITIVE_GROUPS:
        raise SensitiveColumnError(
            f"Column '{column}' has {series.nunique()} unique values "
            f"(max {MAX_SENSITIVE_GROUPS}); pick a categorical column"
        )
    return series


def resolve_sensitive_series(
    dataset_id: str,
    column: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return the sensitive column split into (train, test) series.

    ``x_train``/``x_test`` indices must both still be positional into the raw
    dataset file (i.e. NOT a resampled train frame — resolve that side via
    ``resample_train_split``'s ``sensitive_train`` instead, before resampling).

    Raises ``SensitiveColumnError`` with a user-facing message when the dataset
    is missing, the column is absent, the row counts no longer match the split,
    or the column has too many unique groups.
    """
    series = _load_sensitive_column(dataset_id, column)

    n_split = len(x_train) + len(x_test)
    if len(series) != n_split:
        raise SensitiveColumnError(
            f"Dataset rows ({len(series)}) do not match the optimization split "
            f"({n_split}); the dataset file may have changed since the run"
        )
    return series.iloc[x_train.index], series.iloc[x_test.index]


def resolve_sensitive_test_series(dataset_id: str, column: str, x_test: pd.DataFrame) -> pd.Series:
    """Return the sensitive column aligned to the TEST split only.

    Unlike ``resolve_sensitive_series``, this does not require the train split
    (which may already be resampled and so no longer positional into the raw
    file) — only ``x_test.index``, which resampling never touches.
    """
    series = _load_sensitive_column(dataset_id, column)
    try:
        return series.iloc[x_test.index]
    except IndexError as exc:
        raise SensitiveColumnError(
            f"Dataset rows ({len(series)}) do not cover the test split's row "
            "indices; the dataset file may have changed since the run"
        ) from exc
