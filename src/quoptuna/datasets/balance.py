"""Classification-target distribution summaries for catalog and preview UIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    import pandas as pd

MAX_CLASSIFICATION_CLASSES = 20
BALANCED_MINORITY_FRACTION = 0.40
BINARY_N_CLASSES = 2

MODERATE_IMBALANCE_MINORITY_FRACTION = 0.20


class TargetClassCount(TypedDict):
    label: str
    count: int


class TargetBalanceProfile(TypedDict):
    kind: Literal["binary", "multiclass"]
    label: Literal["balanced", "moderate_imbalance", "imbalanced", "multiclass"]
    classes: list[TargetClassCount]
    minority_fraction: float


def target_balance_profile(target: pd.Series) -> TargetBalanceProfile | None:
    """Summarize a supported classification target, or return ``None``.

    The label describes class prevalence for the selected target only. It is
    not a fairness conclusion and is intentionally unavailable for degenerate
    or high-cardinality columns that cannot be classification targets.
    """
    counts = target.value_counts(dropna=True)
    n_classes = len(counts)
    if not BINARY_N_CLASSES <= n_classes <= MAX_CLASSIFICATION_CLASSES:
        return None

    total = int(counts.sum())
    minority_fraction = int(counts.min()) / total
    classes = [{"label": str(value), "count": int(count)} for value, count in counts.items()]
    if n_classes > BINARY_N_CLASSES:
        label: Literal["balanced", "moderate_imbalance", "imbalanced", "multiclass"] = "multiclass"
        kind: Literal["binary", "multiclass"] = "multiclass"
    elif minority_fraction >= BALANCED_MINORITY_FRACTION:
        label = "balanced"
        kind = "binary"
    elif minority_fraction >= MODERATE_IMBALANCE_MINORITY_FRACTION:
        label = "moderate_imbalance"
        kind = "binary"
    else:
        label = "imbalanced"
        kind = "binary"

    return {
        "kind": kind,
        "label": label,
        "classes": classes,
        "minority_fraction": minority_fraction,
    }
