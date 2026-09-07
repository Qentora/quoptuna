"""Train-split class-imbalance resampling.

Quantum models train via a raw JAX loss (``qml_benchmarks.model_utils.train``)
that has no ``sample_weight``/``class_weight`` hook, unlike the sklearn
classifiers in ``MODEL_CONSTRUCTORS`` (SVC/SVClinear/Perceptron). Resampling
the train split before it reaches ``Optimizer`` rebalances classes for every
model type uniformly, quantum included.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

if TYPE_CHECKING:
    import pandas as pd

RESAMPLING_STRATEGIES = ("none", "oversample", "undersample")

# Column name the sensitive series is temporarily appended under so imblearn
# resamples it in lockstep with x_train (same duplicated/dropped rows), then
# split back out. Namespaced to avoid colliding with a real feature column.
_SENSITIVE_COL = "__quoptuna_sensitive__"


def resample_train_split(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    seed: int = 42,
    sensitive_train: pd.Series | None = None,
):
    """Rebalance ``x_train``/``y_train`` by class; test split is untouched.

    ``strategy``: "none" (pass through), "oversample" (duplicate minority-class
    rows to match the majority, via ``RandomOverSampler``), or "undersample"
    (drop majority-class rows to match the minority, via ``RandomUnderSampler``).

    When ``sensitive_train`` is given, it is resampled in lockstep with
    ``x_train`` (appended as an extra column before resampling, then split
    back out) so the fairness audit's per-row sensitive-attribute alignment
    survives duplication/dropping; the third return value is then the
    resampled sensitive series, else it is ``sensitive_train`` unchanged (i.e.
    ``None`` when not given).
    """
    if strategy == "none":
        return x_train, y_train, sensitive_train
    if strategy not in RESAMPLING_STRATEGIES:
        msg = f"Unknown resampling strategy: {strategy!r} (expected one of {RESAMPLING_STRATEGIES})"
        raise ValueError(msg)

    sampler = (
        RandomOverSampler(random_state=seed)
        if strategy == "oversample"
        else RandomUnderSampler(random_state=seed)
    )

    if sensitive_train is None:
        x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
        return x_resampled, y_resampled, None

    x_with_sensitive = x_train.copy()
    x_with_sensitive[_SENSITIVE_COL] = sensitive_train.to_numpy()
    x_resampled, y_resampled = sampler.fit_resample(x_with_sensitive, y_train)
    sensitive_resampled = x_resampled.pop(_SENSITIVE_COL)
    return x_resampled, y_resampled, sensitive_resampled
