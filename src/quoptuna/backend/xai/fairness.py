"""Fairness auditing utilities built on fairlearn.

Metrics are computed per sensitive-feature group with ``MetricFrame`` and the
group plots reuse the same base64 PNG data-URL convention as the SHAP plots so
they can be surfaced by the API and fed to the LLM report unchanged.
"""

from __future__ import annotations

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MAX_SENSITIVE_GROUPS = 20

# Disparity metrics that can drive the fairness-aware search (see
# ``compute_disparity``). All are scalars where 0.0 means perfect parity.
FAIRNESS_METRICS = (
    "equal_opportunity_difference",
    "disparate_impact",
    "demographic_parity_difference",
)

# Assigned when a trial's disparity cannot be computed (e.g. a sensitive group
# with a single class); also the natural upper bound of every metric above.
WORST_DISPARITY = 1.0

# Quantum/classical models in this project use labels in {-1, 1}; fairlearn's
# rate metrics assume a {0, 1} positive class, so map at this boundary.
_POS_LABEL = 1


def binarize_favorable(y, favorable: int = _POS_LABEL) -> np.ndarray:
    """Indicator of the favorable outcome, for fairlearn's binary rate metrics.

    Binary tasks keep the default (+1 is the positive class). Multiclass tasks
    pass the encoded favorable class code, reducing the audit to
    "favorable class vs rest" — selection rates and TPRs of the favorable
    outcome per group remain well-defined for any K.
    """
    arr = np.ravel(np.asarray(y))
    return (arr == favorable).astype(int)


def _to_binary(y, favorable: int = _POS_LABEL) -> np.ndarray:
    return binarize_favorable(y, favorable)


def _figure_to_data_url(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _build_metric_frame(
    y_true, y_pred, sensitive: pd.Series, favorable: int = _POS_LABEL
) -> MetricFrame:
    metrics = {
        "accuracy": accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "selection_rate": selection_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "count": lambda yt, yp: len(yt),  # noqa: ARG005
    }
    return MetricFrame(
        metrics=metrics,
        y_true=_to_binary(y_true, favorable),
        y_pred=_to_binary(y_pred, favorable),
        sensitive_features=np.asarray(sensitive).ravel(),
    )


def compute_fairness(y_true, y_pred, sensitive: pd.Series, favorable: int = _POS_LABEL) -> dict:
    """Group fairness metrics + disparity summary for one set of predictions.

    For multiclass tasks pass ``favorable`` (the encoded favorable-class code);
    the audit is then computed on the binarized favorable-vs-rest outcome.
    """
    yt, yp = _to_binary(y_true, favorable), _to_binary(y_pred, favorable)
    sf = np.asarray(sensitive).ravel()
    frame = _build_metric_frame(yt, yp, pd.Series(sf))

    by_group = {
        metric: {str(group): float(value) for group, value in series.items()}
        for metric, series in frame.by_group.items()
    }
    overall = {metric: float(value) for metric, value in frame.overall.items()}
    disparities = {
        "demographic_parity_difference": float(
            demographic_parity_difference(yt, yp, sensitive_features=sf)
        ),
        "demographic_parity_ratio": float(demographic_parity_ratio(yt, yp, sensitive_features=sf)),
        "equalized_odds_difference": float(
            equalized_odds_difference(yt, yp, sensitive_features=sf)
        ),
        # Disparate Impact is the selection-rate ratio (four-fifths rule);
        # kept under its dissertation name alongside demographic_parity_ratio.
        "disparate_impact": float(demographic_parity_ratio(yt, yp, sensitive_features=sf)),
        "equal_opportunity_difference": float(
            equal_opportunity_difference(yt, yp, sensitive_features=sf)
        ),
    }
    return {"by_group": by_group, "overall": overall, "disparities": disparities}


def compute_disparity(
    y_true, y_pred, sensitive, metric: str, favorable: int = _POS_LABEL
) -> float:
    """Scalar disparity to MINIMIZE during search; 0.0 means perfect parity.

    ``disparate_impact`` (a ratio where 1.0 is parity) is mapped to
    ``max(0, 1 - ratio)`` so every metric shares the same direction; the
    difference metrics are returned as-is (fairlearn guarantees them
    non-negative).
    """
    yt, yp = _to_binary(y_true, favorable), _to_binary(y_pred, favorable)
    sf = np.asarray(sensitive).ravel()
    if metric == "equal_opportunity_difference":
        return float(equal_opportunity_difference(yt, yp, sensitive_features=sf))
    if metric == "disparate_impact":
        ratio = float(demographic_parity_ratio(yt, yp, sensitive_features=sf))
        if np.isnan(ratio):
            return WORST_DISPARITY
        return max(0.0, 1.0 - ratio)
    if metric == "demographic_parity_difference":
        return float(demographic_parity_difference(yt, yp, sensitive_features=sf))
    msg = f"Unknown fairness metric: {metric!r}. Expected one of {FAIRNESS_METRICS}."
    raise ValueError(msg)


def plot_group_metrics(fairness: dict, title_prefix: str = "") -> dict[str, str]:
    """One bar chart per metric across sensitive groups, as data URLs."""
    plots: dict[str, str] = {}
    for metric, groups in fairness["by_group"].items():
        if metric == "count":
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        names = list(groups.keys())
        values = [groups[name] for name in names]
        ax.bar(names, values, color="#6366f1")
        ax.axhline(
            fairness["overall"].get(metric, 0.0),
            color="#9ca3af",
            lw=1,
            linestyle="--",
            label="overall",
        )
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"{title_prefix}{metric.replace('_', ' ').title()} by group")
        ax.set_ylim(0, max(1.0, max(values) * 1.1) if values else 1.0)
        ax.legend(loc="best")
        max_label_len = 8
        if max(len(str(n)) for n in names) > max_label_len:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plots[metric] = _figure_to_data_url(fig)
    return plots


def plot_mitigation_comparison(before: dict, after: dict) -> str:
    """Grouped bar chart of disparity measures before vs after mitigation."""
    labels = list(before["disparities"].keys())
    before_vals = [before["disparities"][k] for k in labels]
    after_vals = [after["disparities"][k] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, before_vals, width, label="Before", color="#f59e0b")
    ax.bar(x + width / 2, after_vals, width, label="After (ThresholdOptimizer)", color="#10b981")
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("_", " ") for label in labels], fontsize=8)
    ax.set_title("Disparity before vs after mitigation")
    ax.legend()
    return _figure_to_data_url(fig)


class _BinaryLabelAdapter:
    """Expose a {-1,1}-label model to fairlearn as a {0,1}-label estimator.

    Outputs are coerced to numpy: quantum models built on JAX return
    ``jaxlib.ArrayImpl``, which fairlearn rejects with
    "Unexpected data type ... encountered".
    """

    def __init__(self, model):
        self._model = model
        self.classes_ = np.array([0, 1])

    def predict(self, x):
        return _to_binary(np.asarray(self._model.predict(x)))

    def predict_proba(self, x):
        return np.asarray(self._model.predict_proba(x))

    def fit(self, *args, **kwargs):  # noqa: ARG002  # pragma: no cover - prefit, never called
        return self


def mitigate_with_threshold_optimizer(  # noqa: PLR0913
    model,
    x_train,
    y_train,
    sensitive_train: pd.Series,
    x_test,
    y_test,
    sensitive_test: pd.Series,
    constraint: str = "equalized_odds",
) -> dict:
    """Post-process predictions with fairlearn's ThresholdOptimizer.

    Returns before/after fairness metrics plus a comparison plot.
    """
    predict_method = "predict_proba" if hasattr(model, "predict_proba") else "predict"
    optimizer = ThresholdOptimizer(
        estimator=_BinaryLabelAdapter(model),
        constraints=constraint,
        objective="accuracy_score",
        prefit=True,
        predict_method=predict_method,
    )
    optimizer.fit(
        np.asarray(x_train),
        _to_binary(y_train),
        sensitive_features=np.asarray(sensitive_train).ravel(),
    )
    mitigated_pred = optimizer.predict(
        np.asarray(x_test),
        sensitive_features=np.asarray(sensitive_test).ravel(),
        random_state=42,
    )

    before = compute_fairness(y_test, model.predict(x_test), sensitive_test)
    after = compute_fairness(_to_binary(y_test), mitigated_pred, sensitive_test)
    return {
        "constraint": constraint,
        "before": before,
        "after": after,
        "comparison_plot": plot_mitigation_comparison(before, after),
    }
