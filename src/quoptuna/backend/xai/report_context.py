"""The evidence bundle ("research dump") behind every generated report.

Reports used to be drafted from ``str(report_dict)`` over six keys: the metrics,
the feature importances, the confusion matrix, the curves and the task type.
Everything else the run knew - which sampler and pruner were used, which model
families were searched, the trial history, the Pareto front of a fairness-aware
search, the resampling and encoding choices, the analysis configuration - never
reached the agent, so the agent could not write about it.

This module closes that gap. :func:`build_context` collects *every* recorded
fact about a run into one JSON-serialisable structure with a stable schema, and
:func:`render_markdown` renders it as the deterministic markdown tables the
agents read. The same structure is what the API serves as ``context.json`` and
zips into the downloadable bundle, so the report, the download and the agent
prompt are all views of one artifact.

Design rules:

* **No base64 in the context.** Images are represented by a figure manifest
  (id, title, group, in-bundle path); bytes stay in the snapshot payload.
* **Absence is evidence.** Anything unavailable is listed in ``omissions`` so
  the agent states "not available in this run" instead of inventing it.
* **Nothing is truncated silently.** Trial rows are capped, but the counts and
  aggregates are computed over the full history and the cap is recorded.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "1.0"

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")

# Outside this magnitude band a plain decimal is unreadable, so switch to
# scientific notation (log loss can be tiny; support counts can be large).
_SCI_LOWER = 1e-4
_SCI_UPPER = 1e6

#: Plot keys that come out of the SHAP explainer; everything else in
#: ``payload["plots"]`` is a performance curve.
_SHAP_PLOT_KEYS = frozenset(
    {"bar", "beeswarm", "violin", "heatmap", "waterfall", "force", "decision"}
)

_FIGURE_TITLES = {
    "shap_bar": "SHAP mean absolute importance (bar)",
    "shap_beeswarm": "SHAP beeswarm",
    "shap_violin": "SHAP violin",
    "shap_heatmap": "SHAP heatmap",
    "shap_waterfall": "SHAP waterfall (single instance)",
    "shap_force": "SHAP force plot",
    "shap_decision": "SHAP decision plot",
    "roc_curve": "ROC curve",
    "pr_curve": "Precision-recall curve",
    "confusion_matrix": "Confusion matrix",
    "study_optimization_history": "Optimization history",
    "study_param_importances": "Hyperparameter importances",
    "study_parallel_coordinate": "Parallel coordinate plot",
    "study_slice": "Slice plot",
    "study_timeline": "Trial timeline",
    "fairness_mitigation_comparison": "Disparity before vs after mitigation",
}

#: Free-text guidance attached to each figure so the agent knows what the plot
#: can and cannot support - the difference between describing and interpreting.
_FIGURE_NOTES = {
    "shap_bar": "Ranks features by mean |SHAP|; magnitude only, no direction.",
    "shap_beeswarm": "Per-instance SHAP values coloured by feature value; shows direction.",
    "shap_violin": "Distribution of SHAP values per feature.",
    "shap_heatmap": "Instances x features SHAP matrix; reveals interaction clusters.",
    "shap_waterfall": "Additive local explanation for one inspected instance.",
    "roc_curve": "TPR vs FPR across thresholds; AUC is threshold-independent.",
    "pr_curve": "Precision vs recall; preferred over ROC under class imbalance.",
    "confusion_matrix": "Counts of predicted vs true class on the test split.",
    "study_optimization_history": "Objective per trial plus best-so-far trace.",
    "study_param_importances": "fANOVA-style importance of each hyperparameter.",
    "study_timeline": "Wall-clock start/end of every trial, including pruned ones.",
    "fairness_mitigation_comparison": (
        "Disparity measures before and after ThresholdOptimizer post-processing."
    ),
}

_METRIC_MEANING = {
    "f1_score": "Harmonic mean of precision and recall on the test split.",
    "precision": "Share of predicted positives that are correct.",
    "recall": "Share of actual positives that were found.",
    "accuracy": "Share of all predictions that are correct.",
    "roc_auc_score": "Threshold-free ranking quality; 0.5 is random.",
    "average_precision_score": "Area under the precision-recall curve.",
    "mcc": "Matthews correlation coefficient; balanced, -1 to 1.",
    "cohens_kappa": "Agreement above chance; 0 is chance-level.",
    "log_loss": "Probability calibration penalty; lower is better.",
}

_DISPARITY_MEANING = {
    "demographic_parity_difference": (
        "Largest gap in favourable-outcome rate between groups; 0 is parity."
    ),
    "demographic_parity_ratio": ("Ratio of the lowest to the highest selection rate; 1 is parity."),
    "disparate_impact": (
        "Selection-rate ratio under the four-fifths rule; below 0.8 is adverse impact."
    ),
    "equalized_odds_difference": ("Largest gap in TPR or FPR between groups; 0 is parity."),
    "equal_opportunity_difference": (
        "Largest gap in true-positive rate between groups; 0 is parity."
    ),
}

_SEARCH_OPTION_MEANING = {
    "sampler": "How Optuna proposes hyperparameters (TPE is Bayesian, random is uniform).",
    "sampler_seed": "Seed for the sampler; set for a reproducible search.",
    "pruner": "Early-stopping rule for unpromising trials.",
    "pruner_min_resource": "Reports a trial must make before it can be pruned (rung 0).",
    "pruner_reduction_factor": "How aggressively each rung culls survivors.",
    "intermediate_metric": "What iterative models report for pruning decisions.",
    "num_trials": "Trial budget requested for the search.",
    "model_types": "Model families the search was allowed to sample.",
    "max_steps": "Cap on training steps per trial for iterative models.",
    "convergence_interval": "Steps between flat-loss convergence checks and pruning reports.",
    "max_vmap": "Circuit evaluations vectorised per JAX call; must divide the batch size.",
    "dev_type": "PennyLane simulator backing the quantum circuits.",
    "categorical_encoding": "How categorical feature columns were encoded.",
    "resampling": "Class-imbalance handling applied to the TRAIN split only.",
    "database_name": "Optuna storage database holding the study.",
    "study_name": "Optuna study name (needed to reload this search).",
    "dataset_id": "Registry id of the dataset the run read.",
    "dataset_source": "Where the dataset came from (UCI fetch or user upload).",
    "selected_features": "Columns fed to the model as features.",
    "target_column": "Column predicted.",
    "label_mapping": "Which raw target values were mapped to the negative/positive class.",
    "favorable_class": "Class treated as the favourable outcome for fairness framing.",
    "sensitive_feature": "Protected attribute used for group analysis (not necessarily a feature).",
    "search_space": "Explicit hyperparameter grid; absent means the full default space.",
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _snake(name: str) -> str:
    return _CAMEL.sub("_", str(name)).lower()


def figure_id_for_plot(key: str) -> str:
    """Canonical figure id for a ``payload["plots"]`` key.

    The same rule is implemented in the frontend (``lib/figures.ts``) so a
    report's ``figures/<id>.png`` reference resolves both to a file inside the
    downloaded bundle and to the in-memory data URL in the browser.

    An index suffix is preserved: the legacy XAI path emits one waterfall plot
    per inspected instance (``waterfall_0``, ``waterfall_1``, ...), and those are
    still SHAP figures.
    """
    snake = _snake(key)
    base, _, suffix = snake.rpartition("_")
    if base and suffix.isdigit() and base in _SHAP_PLOT_KEYS:
        return f"shap_{snake}"
    return f"shap_{snake}" if snake in _SHAP_PLOT_KEYS else snake


def figure_images(payload: dict, figures: Sequence[dict]) -> dict[str, str]:
    """Figure id -> image data (data URL or object-store link) from a payload.

    The inverse of :func:`collect_figures`: each manifest entry records the
    payload path it came from, so this resolves them without re-deriving names.
    """
    out: dict[str, str] = {}
    for figure in figures:
        if figure.get("kind") != "image":
            continue
        current: Any = payload
        for part in (figure.get("source") or "").split("."):
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)
        if isinstance(current, str) and current:
            out[figure["id"]] = current
    return out


def _number(value: Any, digits: int = 6) -> Any:
    """JSON-safe number: rounded floats, NaN/inf mapped to ``None``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _fmt(value: Any, digits: int = 4) -> str:  # noqa: PLR0911
    """Human/agent-readable cell text."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        if value and (abs(value) < _SCI_LOWER or abs(value) >= _SCI_UPPER):
            return f"{value:.{digits}e}"
        return f"{round(value, digits):g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item, digits) for item in value) or "n/a"
    if isinstance(value, dict):
        return "; ".join(f"{k}={_fmt(v, digits)}" for k, v in value.items()) or "n/a"
    text = str(value).strip()
    return text.replace("|", "\\|") if text else "n/a"


def _code(value: Any) -> str:
    text = _fmt(value)
    return text if text == "n/a" else f"`{text}`"


def _table(
    headers: Sequence[str], rows: Iterable[Sequence[Any]], numeric: Sequence[bool] = ()
) -> str:
    """Render a GFM pipe table; empty row sets yield an explicit placeholder."""
    body = [[_fmt(cell) for cell in row] for row in rows]
    if not body:
        return "_No rows recorded._"
    flags = list(numeric) + [False] * (len(headers) - len(numeric))
    delimiter = ["---:" if flag else "---" for flag in flags]
    # Headers carry notation like "Mean |SHAP|"; an unescaped pipe there would
    # split the header row and make every data row look ragged.
    headers = [str(header).replace("|", "\\|") for header in headers]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(delimiter) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _kv_table(items: Iterable[tuple[str, Any]], meaning: dict[str, str] | None = None) -> str:
    rows = []
    for key, value in items:
        row = [f"`{key}`", _fmt(value)]
        if meaning is not None:
            row.append(meaning.get(key, ""))
        rows.append(row)
    headers = ["Setting", "Value"] + (["Why it matters"] if meaning is not None else [])
    return _table(headers, rows)


def _section(title: str, body: str) -> str:
    return f"### {title}\n\n{body.rstrip()}\n"


# --------------------------------------------------------------------------
# Inclusion flags
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportInclusions:
    """Which evidence families enter the bundle sent to the agents.

    Defaults mirror :data:`quoptuna.backend.xai.prompts.REPORT_PROMPT_SETTINGS`,
    which is what the Settings page renders and documents.
    """

    fairness: bool = True
    fairness_search: bool = True
    pareto: bool = True
    shap_detail: bool = True
    trial_history: bool = True
    study_plots: bool = True
    figures: bool = True
    max_trial_rows: int = 40

    def as_dict(self) -> dict[str, Any]:
        return {
            "fairness": self.fairness,
            "fairness_search": self.fairness_search,
            "pareto": self.pareto,
            "shap_detail": self.shap_detail,
            "trial_history": self.trial_history,
            "study_plots": self.study_plots,
            "figures": self.figures,
            "max_trial_rows": self.max_trial_rows,
        }


@dataclass
class _Omissions:
    """Collector for "this is not available" notes handed to the agent."""

    items: list[str] = field(default_factory=list)

    def add(self, what: str, why: str) -> None:
        self.items.append(f"{what}: {why}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _figure(  # noqa: PLR0913
    fid: str,
    group: str,
    *,
    kind: str = "image",
    source: str = "",
    title: str = "",
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": fid,
        "title": title or _FIGURE_TITLES.get(fid, fid.replace("_", " ").title()),
        "group": group,
        "kind": kind,
        "path": f"figures/{fid}.png" if kind == "image" else f"study_plots/{fid}.json",
        "note": note or _FIGURE_NOTES.get(fid, ""),
        "source": source,
    }


def collect_figures(  # noqa: C901
    payload: dict, inclusions: ReportInclusions
) -> list[dict[str, Any]]:
    """Figure manifest for a snapshot payload, in report-reading order."""
    figures: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        if entry["id"] not in seen:
            seen.add(entry["id"])
            figures.append(entry)

    if payload.get("confusion_matrix_plot"):
        add(_figure("confusion_matrix", "performance", source="confusion_matrix_plot"))
    for key, value in (payload.get("plots") or {}).items():
        if not value:
            continue
        fid = figure_id_for_plot(key)
        group = "shap" if fid.startswith("shap_") else "performance"
        add(_figure(fid, group, source=f"plots.{key}"))
    if inclusions.fairness:
        fairness = payload.get("fairness") or {}
        attribute = fairness.get("sensitive_feature") or "the protected attribute"
        for key, value in (fairness.get("plots") or {}).items():
            if value:
                metric = str(key).replace("_", " ")
                add(
                    _figure(
                        f"fairness_{_snake(key)}",
                        "fairness",
                        source=f"fairness.plots.{key}",
                        title=f"{metric.capitalize()} by {attribute} group",
                        note=(
                            f"Per-group {metric} with the overall value as a dashed reference "
                            "line; the gap between bars is the disparity."
                        ),
                    )
                )
        mitigation = fairness.get("mitigation") or {}
        if mitigation.get("comparison_plot"):
            add(
                _figure(
                    "fairness_mitigation_comparison",
                    "fairness",
                    source="fairness.mitigation.comparison_plot",
                )
            )
    if inclusions.study_plots:
        for key, value in (payload.get("study_plots") or {}).items():
            if value:
                add(
                    _figure(
                        f"study_{_snake(key)}",
                        "study",
                        kind="plotly",
                        source=f"study_plots.{key}",
                    )
                )
    # Performance figures first, then explainability, fairness, diagnostics -
    # the order the report skeleton walks through them.
    order = {"performance": 0, "shap": 1, "fairness": 2, "study": 3}
    figures.sort(key=lambda item: (order.get(item["group"], 9), item["id"]))
    return figures


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _configuration(request: dict) -> dict[str, Any]:
    """Every option chosen for the run, grouped the way the report reads it."""
    return {
        "data": {
            "dataset_id": request.get("dataset_id"),
            "dataset_source": request.get("dataset_source"),
            "selected_features": list(request.get("selected_features") or []),
            "target_column": request.get("target_column"),
            "label_mapping": request.get("label_mapping"),
            "favorable_class": request.get("favorable_class"),
            "sensitive_feature": request.get("sensitive_feature"),
            "categorical_encoding": request.get("categorical_encoding"),
            "resampling": request.get("resampling"),
        },
        "search": {
            "study_name": request.get("study_name"),
            "database_name": request.get("database_name"),
            "num_trials": request.get("num_trials"),
            "sampler": request.get("sampler"),
            "sampler_seed": request.get("sampler_seed"),
            "pruner": request.get("pruner"),
            "pruner_min_resource": request.get("pruner_min_resource"),
            "pruner_reduction_factor": request.get("pruner_reduction_factor"),
            "intermediate_metric": request.get("intermediate_metric"),
            "model_types": request.get("model_types"),
            "search_space": request.get("search_space"),
        },
        "training": {
            "max_steps": request.get("max_steps"),
            "convergence_interval": request.get("convergence_interval"),
            "max_vmap": request.get("max_vmap"),
            "dev_type": request.get("dev_type"),
        },
        "fairness_search": {
            "mode": request.get("fairness_mode") or "off",
            "metric": request.get("fairness_metric"),
            "threshold": request.get("fairness_threshold"),
        },
    }


# --------------------------------------------------------------------------
# Trials
# --------------------------------------------------------------------------

_TRIAL_ATTR_KEYS = (
    "val_f1_score",
    "fairness_disparity",
    "fairness_metric",
    "training_time",
    "n_steps",
    "converged",
    "pruned_at_step",
    "decision_threshold",
    "error",
)


def _trial_row(trial: dict) -> dict[str, Any]:
    attrs = trial.get("user_attrs") or {}
    params = dict(trial.get("params") or {})
    values = trial.get("values")
    return {
        "trial": trial.get("trial"),
        "state": trial.get("state"),
        "model_type": params.get("model_type"),
        "objective": _number(trial.get("value")),
        "values": [_number(v) for v in values] if values else None,
        "params": {key: _number(value) for key, value in params.items() if key != "model_type"},
        "attrs": {key: _number(attrs[key]) for key in _TRIAL_ATTR_KEYS if key in attrs},
    }


def _model_family_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = row.get("model_type") or "unknown"
        entry = families.setdefault(
            name,
            {
                "model_type": name,
                "trials": 0,
                "complete": 0,
                "pruned": 0,
                "failed": 0,
                "best_objective": None,
                "mean_objective": None,
                "mean_training_time": None,
                "_values": [],
                "_times": [],
            },
        )
        entry["trials"] += 1
        state = (row.get("state") or "").upper()
        if state == "COMPLETE":
            entry["complete"] += 1
        elif state == "PRUNED":
            entry["pruned"] += 1
        elif state == "FAIL":
            entry["failed"] += 1
        if row.get("objective") is not None:
            entry["_values"].append(row["objective"])
        training_time = (row.get("attrs") or {}).get("training_time")
        if training_time is not None:
            entry["_times"].append(training_time)
    summary = []
    for entry in families.values():
        values = entry.pop("_values")
        times = entry.pop("_times")
        if values:
            entry["best_objective"] = _number(max(values))
            entry["mean_objective"] = _number(sum(values) / len(values))
        if times:
            entry["mean_training_time"] = _number(sum(times) / len(times))
        summary.append(entry)
    summary.sort(key=lambda item: (item["best_objective"] is None, -(item["best_objective"] or 0)))
    return summary


def _pareto_knee(points: list[dict[str, Any]]) -> int | None:
    """Index of the Pareto point closest to the ideal (max F1, min disparity).

    Both objectives are min-max normalised over the front first, so the choice
    does not depend on their very different scales.
    """
    usable = [
        (index, point["values"])
        for index, point in enumerate(points)
        if point.get("values") and len(point["values"]) >= 2  # noqa: PLR2004
    ]
    if len(usable) < 2:  # noqa: PLR2004
        return usable[0][0] if usable else None
    f1s = [values[0] for _, values in usable]
    disparities = [values[1] for _, values in usable]
    f1_span = (max(f1s) - min(f1s)) or 1.0
    disparity_span = (max(disparities) - min(disparities)) or 1.0
    best_index, best_distance = None, None
    for index, values in usable:
        # Ideal corner: best F1 seen on the front, lowest disparity seen on it.
        gap_f1 = (max(f1s) - values[0]) / f1_span
        gap_disparity = (values[1] - min(disparities)) / disparity_span
        distance = math.hypot(gap_f1, gap_disparity)
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


# --------------------------------------------------------------------------
# Study-plot extraction
# --------------------------------------------------------------------------


def _extract_param_importances(figure: dict | None) -> list[dict[str, Any]]:
    """Read hyperparameter importances straight out of the Plotly figure JSON."""
    if not isinstance(figure, dict):
        return []
    out: list[dict[str, Any]] = []
    for trace in figure.get("data") or []:
        names, values = trace.get("y"), trace.get("x")
        if not isinstance(names, list) or not isinstance(values, list):
            continue
        for name, value in zip(names, values):
            number = _number(value)
            if isinstance(name, str) and isinstance(number, (int, float)):
                out.append({"parameter": name, "importance": number})
    out.sort(key=lambda item: -item["importance"])
    return out


def _extract_history(figure: dict | None) -> dict[str, Any]:
    """Objective-per-trial and best-so-far traces from the history figure."""
    if not isinstance(figure, dict):
        return {}
    traces: dict[str, list[Any]] = {}
    for trace in figure.get("data") or []:
        name = str(trace.get("name") or trace.get("mode") or "trace")
        y = trace.get("y")
        if isinstance(y, list) and y:
            traces[name] = [_number(value) for value in y]
    summary: dict[str, Any] = {}
    for name, series in traces.items():
        finite = [value for value in series if isinstance(value, (int, float))]
        if not finite:
            continue
        summary[name] = {
            "n_points": len(series),
            "first": finite[0],
            "last": finite[-1],
            "best": max(finite),
        }
    return summary


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


def build_context(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    optimization_id: str,
    snapshot: dict,
    run: dict | None = None,
    dataset: dict | None = None,
    trials: Sequence[dict] | None = None,
    pareto_trials: Sequence[dict] | None = None,
    dataset_description: str | None = None,
    inclusions: ReportInclusions | None = None,
    prompts: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the full evidence bundle for one analysis snapshot.

    Every argument except ``snapshot`` is optional: a run whose in-memory job
    was lost to a restart still produces a valid (smaller) bundle, with the
    missing pieces recorded under ``omissions`` rather than omitted silently.
    """
    inclusions = inclusions or ReportInclusions()
    payload = snapshot.get("payload") or {}
    request = dict((run or {}).get("request") or {})
    omissions = _Omissions()

    configuration = _configuration(request)
    if not request:
        omissions.add(
            "Run configuration",
            "the optimization job record was unavailable, so search and preprocessing "
            "options could not be recovered",
        )

    fairness_mode = configuration["fairness_search"]["mode"]

    # ---- trials ---------------------------------------------------------
    trial_rows = [_trial_row(trial) for trial in (trials or [])]
    state_counts: dict[str, int] = {}
    for row in trial_rows:
        state = (row.get("state") or "UNKNOWN").upper()
        state_counts[state] = state_counts.get(state, 0) + 1
    ranked = sorted(
        trial_rows,
        key=lambda row: (row["objective"] is None, -(row["objective"] or 0.0)),
    )
    cap = max(0, int(inclusions.max_trial_rows))
    included_rows = ranked[:cap] if inclusions.trial_history else []
    if inclusions.trial_history and cap and len(ranked) > cap:
        omissions.add(
            "Trial history",
            f"only the {cap} best of {len(ranked)} trials are listed (row cap); "
            "state counts and per-family aggregates cover all of them",
        )
    if not inclusions.trial_history and trial_rows:
        omissions.add("Trial history", "excluded by the report settings")
    if not trial_rows:
        omissions.add("Trial history", "no serialised trials were available for this run")

    best_trial: dict[str, Any] = {
        "trial": (run or {}).get("best_trial_number"),
        "objective": _number((run or {}).get("best_value")),
        "params": {
            key: _number(value) for key, value in ((run or {}).get("best_params") or {}).items()
        },
    }
    if best_trial["trial"] is None and ranked:
        best_trial["trial"] = ranked[0]["trial"]
    # The snapshot's analysed_model block is authoritative: it records the
    # trial the analysis actually retrained, with "best" already resolved to a
    # concrete number. Fall back to the request config for older snapshots
    # written before that block existed.
    analysed_model = (payload or {}).get("analysed_model") or {}
    analysed_trial = analysed_model.get("trial_number")
    if analysed_trial is None:
        analysed_trial = (snapshot.get("config") or {}).get("trial_number")
    best_trial["analysed_trial"] = analysed_trial
    best_trial["analysed_is_best"] = analysed_trial is None or analysed_trial == best_trial["trial"]
    best_trial["analysed_model_type"] = analysed_model.get("model_type")
    best_trial["analysed_selected_by"] = analysed_model.get("selected_by")
    best_trial["analysed_training_budget"] = analysed_model.get("training_budget") or {}
    best_trial["analysis_revision"] = snapshot.get("revision")

    # ---- pareto front ---------------------------------------------------
    pareto: dict[str, Any] = {"present": False, "included": False}
    if pareto_trials:
        points = [
            {
                "trial": point.get("trial"),
                "values": [_number(value) for value in (point.get("values") or [])],
                "params": {
                    key: _number(value) for key, value in (point.get("params") or {}).items()
                },
            }
            for point in pareto_trials
        ]
        points.sort(key=lambda point: -(point["values"][0] if point["values"] else 0.0))
        knee = _pareto_knee(points)
        pareto = {
            "present": True,
            "included": inclusions.pareto,
            "objective_names": [
                "f1_score (maximise)",
                f"{configuration['fairness_search']['metric'] or 'disparity'} (minimise)",
            ],
            "n_points": len(points),
            "points": points if inclusions.pareto else [],
            "knee_trial": points[knee]["trial"] if knee is not None else None,
            "reported_trial": best_trial["trial"],
        }
        if not inclusions.pareto:
            omissions.add(
                "Pareto front",
                f"{len(points)} Pareto-optimal trials exist but were excluded by the "
                "report settings",
            )
    elif fairness_mode == "multi_objective":
        omissions.add(
            "Pareto front",
            "the run used multi-objective fairness search but no Pareto front was recorded",
        )

    # ---- fairness -------------------------------------------------------
    fairness_payload = payload.get("fairness") or {}
    fairness: dict[str, Any] = {
        "audit_available": bool(fairness_payload.get("metrics")),
        "audit_included": bool(fairness_payload.get("metrics")) and inclusions.fairness,
        "protected_attribute": fairness_payload.get("sensitive_feature")
        or request.get("sensitive_feature"),
        "search": {
            "included": inclusions.fairness_search,
            **configuration["fairness_search"],
        },
    }
    if fairness["search"]["included"] and fairness_mode != "off":
        threshold = configuration["fairness_search"]["threshold"]
        metric = configuration["fairness_search"]["metric"]
        fairness["search"]["direction"] = (
            "feasible when the disparate-impact ratio >= threshold (four-fifths rule)"
            if metric == "disparate_impact"
            else "feasible when the disparity difference <= threshold"
        )
        fairness["search"]["threshold_effective"] = (
            threshold if threshold is not None else (0.8 if metric == "disparate_impact" else 0.1)
        )
        disparities = [
            row["attrs"]["fairness_disparity"]
            for row in trial_rows
            if row["attrs"].get("fairness_disparity") is not None
        ]
        if disparities:
            limit = (
                1.0 - fairness["search"]["threshold_effective"]
                if metric == "disparate_impact"
                else fairness["search"]["threshold_effective"]
            )
            fairness["search"]["trial_disparities"] = {
                "n_trials_scored": len(disparities),
                "best": _number(min(disparities)),
                "worst": _number(max(disparities)),
                "mean": _number(sum(disparities) / len(disparities)),
                "n_feasible": sum(1 for value in disparities if value <= limit),
                "feasibility_limit_in_disparity_space": _number(limit),
            }
    if fairness_mode != "off" and not inclusions.fairness_search:
        omissions.add(
            "Fairness-aware search",
            f"the run used fairness_mode='{fairness_mode}' but its search details were "
            "excluded by the report settings",
        )
    if fairness["audit_available"] and inclusions.fairness:
        metrics = fairness_payload.get("metrics") or {}
        by_group = metrics.get("by_group") or {}
        groups = sorted({group for values in by_group.values() for group in values})
        fairness.update(
            {
                "task_type": fairness_payload.get("task_type"),
                "favorable_class": fairness_payload.get("favorable_class"),
                "groups": groups,
                "by_group": by_group,
                "overall": metrics.get("overall") or {},
                "disparities": metrics.get("disparities") or {},
            }
        )
        counts = by_group.get("count") or {}
        accuracies = by_group.get("accuracy") or {}
        recalls = by_group.get("recall") or {}
        if accuracies:
            worst = min(accuracies, key=lambda group: accuracies[group])
            fairness["most_disadvantaged_group"] = {
                "group": worst,
                "accuracy": _number(accuracies[worst]),
                "recall": _number(recalls.get(worst)),
                "count": _number(counts.get(worst)),
            }
        mitigation = fairness_payload.get("mitigation") or {}
        if mitigation:
            before = (mitigation.get("before") or {}).get("disparities") or {}
            after = (mitigation.get("after") or {}).get("disparities") or {}
            fairness["mitigation"] = {
                "constraint": mitigation.get("constraint"),
                "before": {key: _number(value) for key, value in before.items()},
                "after": {key: _number(value) for key, value in after.items()},
                "delta": {key: _number(after[key] - before[key]) for key in before if key in after},
                "accuracy_before": _number(
                    ((mitigation.get("before") or {}).get("overall") or {}).get("accuracy")
                ),
                "accuracy_after": _number(
                    ((mitigation.get("after") or {}).get("overall") or {}).get("accuracy")
                ),
            }
        else:
            omissions.add(
                "Fairness mitigation",
                "no ThresholdOptimizer comparison was computed for this snapshot",
            )
    elif fairness["audit_available"]:
        omissions.add("Fairness audit", "computed for this run but excluded by the report settings")
    elif fairness["protected_attribute"]:
        omissions.add(
            "Fairness audit",
            f"a protected attribute ('{fairness['protected_attribute']}') was configured but "
            "no audit is stored in this analysis snapshot",
        )
    else:
        omissions.add(
            "Fairness audit",
            "no protected attribute was selected for this run, so no group analysis exists; "
            "do not infer group harms",
        )

    # ---- explainability -------------------------------------------------
    explainability: dict[str, Any] = {
        "feature_importance": [
            {"feature": item.get("feature"), "mean_abs_shap": _number(item.get("importance"))}
            for item in (payload.get("feature_importance") or [])
        ],
        "inspected_sample_index": (snapshot.get("config") or {}).get("sample_index"),
        "class_index": (snapshot.get("config") or {}).get("class_index"),
    }
    shap_data = payload.get("shap_data") or {}
    if inclusions.shap_detail and shap_data.get("feature_names"):
        explainability["shap_statistics"] = _shap_statistics(shap_data)
        explainability["shap_n_samples"] = shap_data.get("n_samples")
        explainability["shap_base_value"] = _number(shap_data.get("base_value"))
    elif not shap_data.get("feature_names"):
        omissions.add("Per-feature SHAP statistics", "raw SHAP values were not stored")
    else:
        omissions.add("Per-feature SHAP statistics", "excluded by the report settings")

    # ---- performance ----------------------------------------------------
    metrics = payload.get("metrics") or {}
    performance: dict[str, Any] = {
        "headline": {
            key: _number(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float, bool))
        },
        "roc_auc": _number(payload.get("roc_auc")),
        "average_precision": _number(payload.get("average_precision")),
        "per_class": metrics.get("per_class"),
        "classification_report": metrics.get("classification_report"),
        "confusion_matrix": payload.get("confusion_data")
        or (
            {"matrix": metrics.get("confusion_matrix")} if metrics.get("confusion_matrix") else None
        ),
        "curves": _curve_summary(payload.get("curves_data")),
        "task_type": payload.get("task_type"),
        "class_labels": payload.get("class_labels"),
    }
    if not performance["confusion_matrix"]:
        omissions.add("Confusion matrix", "not recorded in this snapshot")
    if not performance["curves"]:
        omissions.add("ROC/PR curve data", "not computed for this snapshot")
    # A metric that failed to compute is stored as its error string (see
    # ``XAI.get_report``). Surface that rather than letting the metric look absent.
    for key, value in metrics.items():
        if isinstance(value, str) and key not in ("classification_report",):
            omissions.add(f"Metric '{key}'", f"could not be computed ({value})")

    # ---- study diagnostics ----------------------------------------------
    study_plots = payload.get("study_plots") or {}
    diagnostics: dict[str, Any] = {"included": inclusions.study_plots}
    if inclusions.study_plots and study_plots:
        diagnostics["param_importances"] = _extract_param_importances(
            study_plots.get("param_importances")
        )
        diagnostics["optimization_history"] = _extract_history(
            study_plots.get("optimization_history")
        )
        diagnostics["available_figures"] = sorted(
            key for key, value in study_plots.items() if value
        )
        unavailable = sorted(key for key, value in study_plots.items() if not value)
        if unavailable:
            omissions.add(
                "Study diagnostics",
                "these Optuna figures could not be produced for this study: "
                + ", ".join(unavailable),
            )
    elif not study_plots:
        omissions.add("Study diagnostics", "no Optuna study figures were stored")
    else:
        omissions.add("Study diagnostics", "excluded by the report settings")

    figures = collect_figures(payload, inclusions) if inclusions.figures else []
    if not inclusions.figures:
        omissions.add(
            "Figures",
            "figure images were not sent to the model; do not reference any figure",
        )

    warnings = payload.get("warnings") or {}
    for section, detail in warnings.items():
        omissions.add(f"Analysis section '{section}' failed", str(detail))

    started_at = (run or {}).get("started_at")
    completed_at = (run or {}).get("completed_at")
    context: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),  # noqa: DTZ005
        "run": {
            "optimization_id": optimization_id,
            "status": (run or {}).get("status"),
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": _duration(started_at, completed_at),
            "study_name": configuration["search"]["study_name"],
            "database_name": configuration["search"]["database_name"],
            "requested_trials": configuration["search"]["num_trials"],
            "finished_trials": sum(state_counts.values()) or None,
        },
        "dataset": {
            "id": (dataset or {}).get("id") or request.get("dataset_id"),
            "name": (dataset or {}).get("name"),
            "source": (dataset or {}).get("source") or request.get("dataset_source"),
            "rows": (dataset or {}).get("rows"),
            "columns": (dataset or {}).get("columns"),
            "selected_features": configuration["data"]["selected_features"],
            "target_column": configuration["data"]["target_column"],
            "description": dataset_description,
        },
        "task": {
            "kind": payload.get("task_type"),
            "class_labels": payload.get("class_labels"),
            "label_mapping": configuration["data"]["label_mapping"],
            "favorable_class": configuration["data"]["favorable_class"],
        },
        "configuration": configuration,
        "analysis": {
            "snapshot_id": snapshot.get("id"),
            "revision": snapshot.get("revision"),
            "completed_at": snapshot.get("completed_at"),
            "config": snapshot.get("config"),
            "storage_backend": snapshot.get("storage_backend"),
        },
        "optimization": {
            "best_trial": best_trial,
            "state_counts": state_counts,
            "n_trials_recorded": len(trial_rows),
            "trials": included_rows,
            "trial_row_cap": cap if inclusions.trial_history else 0,
            "model_families": _model_family_summary(trial_rows),
        },
        "pareto_front": pareto,
        "performance": performance,
        "explainability": explainability,
        "fairness": fairness,
        "diagnostics": diagnostics,
        "figures": figures,
        "warnings": warnings,
        "omissions": omissions.items,
        "report_settings": {
            "inclusions": inclusions.as_dict(),
            "prompts": prompts or {},
        },
    }
    return context


def _duration(started_at: Any, completed_at: Any) -> Any:
    try:
        start = datetime.fromisoformat(str(started_at))
        end = datetime.fromisoformat(str(completed_at))
    except (TypeError, ValueError):
        return None
    return _number((end - start).total_seconds(), 1)


def _shap_statistics(shap_data: dict) -> list[dict[str, Any]]:
    """Per-feature SHAP magnitude, direction and value range."""
    names = [str(name) for name in shap_data.get("feature_names") or []]
    values = shap_data.get("values") or []
    data = shap_data.get("data") or []
    stats: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        column = [row[index] for row in values if index < len(row) and row[index] is not None]
        raw = [row[index] for row in data if index < len(row) and row[index] is not None]
        if not column:
            continue
        stats.append(
            {
                "feature": name,
                "mean_abs_shap": _number(sum(abs(v) for v in column) / len(column)),
                "mean_shap": _number(sum(column) / len(column)),
                "min_shap": _number(min(column)),
                "max_shap": _number(max(column)),
                "feature_min": _number(min(raw)) if raw else None,
                "feature_max": _number(max(raw)) if raw else None,
            }
        )
    stats.sort(key=lambda item: -(item["mean_abs_shap"] or 0.0))
    return stats


def _curve_summary(curves: dict | None) -> dict[str, Any] | None:
    """Headline numbers from the raw curve payloads (the points stay in the zip)."""
    if not curves:
        return None
    summary: dict[str, Any] = {"task_type": curves.get("task_type")}
    roc = curves.get("roc") or {}
    pr = curves.get("pr") or {}
    if roc.get("auc") is not None:
        summary["roc_auc"] = _number(roc["auc"])
    if roc.get("fpr"):
        summary["roc_points"] = len(roc["fpr"])
    if roc.get("macro_auc") is not None:
        summary["roc_macro_auc"] = _number(roc["macro_auc"])
    if roc.get("per_class"):
        summary["roc_per_class"] = [
            {"label": entry.get("label"), "auc": _number(entry.get("auc"))}
            for entry in roc["per_class"]
        ]
    if pr.get("average_precision") is not None:
        summary["average_precision"] = _number(pr["average_precision"])
    if pr.get("per_class"):
        summary["pr_per_class"] = [
            {
                "label": entry.get("label"),
                "average_precision": _number(entry.get("average_precision")),
            }
            for entry in pr["per_class"]
        ]
    return summary or None


# --------------------------------------------------------------------------
# Markdown rendering (what the agents actually read)
# --------------------------------------------------------------------------


def figure_manifest_markdown(figures: Sequence[dict]) -> str:
    """The figure table the analyst must take its ``figures/<id>.png`` paths from."""
    if not figures:
        return "No figures are available for this report. Do not reference any figure."
    rows = [
        [figure["id"], figure["title"], figure["group"], figure["path"], figure.get("note") or ""]
        for figure in figures
    ]
    return _table(["figure_id", "Title", "Group", "Reference path", "What it shows"], rows)


def render_markdown(context: dict) -> str:  # noqa: C901, PLR0912, PLR0915
    """Render the evidence bundle as the markdown the agents are prompted with."""
    parts: list[str] = ["## Run identity"]
    run = context.get("run") or {}
    dataset = context.get("dataset") or {}
    task = context.get("task") or {}
    parts.append(
        _kv_table(
            [
                ("optimization_id", run.get("optimization_id")),
                ("status", run.get("status")),
                ("study_name", run.get("study_name")),
                ("database_name", run.get("database_name")),
                ("started_at", run.get("started_at")),
                ("completed_at", run.get("completed_at")),
                ("duration_seconds", run.get("duration_seconds")),
                ("analysis_snapshot_id", (context.get("analysis") or {}).get("snapshot_id")),
                ("analysis_revision", (context.get("analysis") or {}).get("revision")),
            ]
        )
    )

    parts.append("## Dataset and task")
    parts.append(
        _kv_table(
            [
                ("dataset_name", dataset.get("name")),
                ("dataset_id", dataset.get("id")),
                ("source", dataset.get("source")),
                ("rows", dataset.get("rows")),
                ("n_columns", len(dataset.get("columns") or []) or None),
                ("target_column", dataset.get("target_column")),
                ("selected_features", dataset.get("selected_features")),
                ("task_kind", task.get("kind")),
                ("class_labels", task.get("class_labels")),
                ("label_mapping", task.get("label_mapping")),
                ("favorable_class", task.get("favorable_class")),
                ("user_description", dataset.get("description")),
            ]
        )
    )

    configuration = context.get("configuration") or {}
    parts.append("## Configuration chosen for this run")
    for group, title in (
        ("data", "Data preparation"),
        ("search", "Search strategy"),
        ("training", "Training and simulator"),
    ):
        section = configuration.get(group) or {}
        items = [(key, value) for key, value in section.items() if key != "search_space"]
        parts.append(_section(title, _kv_table(items, _SEARCH_OPTION_MEANING)))
    search_space = (configuration.get("search") or {}).get("search_space")
    if search_space:
        parts.append(
            _section(
                "Search space override",
                _table(
                    ["Hyperparameter", "Values searched"],
                    [[key, value] for key, value in search_space.items()],
                ),
            )
        )
    else:
        parts.append(
            _section(
                "Search space override",
                "None - the full default QuOptuna search space was used.",
            )
        )

    optimization = context.get("optimization") or {}
    parts.append("## Search outcome")
    best = optimization.get("best_trial") or {}
    parts.append(
        _kv_table(
            [
                ("best_trial", best.get("trial")),
                ("best_objective_f1", best.get("objective")),
                ("analysed_trial", best.get("analysed_trial")),
                ("analysed_trial_is_best", best.get("analysed_is_best")),
                ("analysed_model_type", best.get("analysed_model_type")),
                ("analysed_trial_selected_by", best.get("analysed_selected_by")),
                ("analysis_revision", best.get("analysis_revision")),
                ("trials_recorded", optimization.get("n_trials_recorded")),
                ("trial_states", optimization.get("state_counts")),
            ]
        )
    )
    analysed_type = best.get("analysed_model_type")
    if analysed_type:
        scope = (
            "which is also the best trial of the search"
            if best.get("analysed_is_best")
            else f"which is NOT the best trial (best is #{best.get('trial')})"
        )
        budget = best.get("analysed_training_budget") or {}
        budget_note = (
            f" It was retrained with {', '.join(f'{k}={v}' for k, v in sorted(budget.items()))}."
            if budget
            else ""
        )
        parts.append(
            f"All SHAP values, metrics, curves and fairness figures in this report "
            f"describe a **{analysed_type}** model from trial "
            f"**#{best.get('analysed_trial')}**, {scope}.{budget_note}"
        )
    if best.get("params"):
        parts.append(
            _section(
                "Best trial hyperparameters",
                _table(
                    ["Hyperparameter", "Value"],
                    [[key, value] for key, value in best["params"].items()],
                ),
            )
        )
    families = optimization.get("model_families") or []
    if families:
        parts.append(
            _section(
                "Per-model-family aggregates (all trials)",
                _table(
                    [
                        "Model type",
                        "Trials",
                        "Complete",
                        "Pruned",
                        "Failed",
                        "Best F1",
                        "Mean F1",
                        "Mean train s",
                    ],
                    [
                        [
                            item["model_type"],
                            item["trials"],
                            item["complete"],
                            item["pruned"],
                            item["failed"],
                            item["best_objective"],
                            item["mean_objective"],
                            item["mean_training_time"],
                        ]
                        for item in families
                    ],
                    numeric=[False, True, True, True, True, True, True, True],
                ),
            )
        )
    trials = optimization.get("trials") or []
    if trials:
        parts.append(
            _section(
                f"Trial history (top {len(trials)} by objective)",
                _table(
                    [
                        "Trial",
                        "State",
                        "Model type",
                        "F1",
                        "Objectives",
                        "Disparity",
                        "Train s",
                        "Hyperparameters",
                    ],
                    [
                        [
                            row["trial"],
                            row["state"],
                            row["model_type"],
                            row["objective"],
                            row["values"],
                            (row["attrs"] or {}).get("fairness_disparity"),
                            (row["attrs"] or {}).get("training_time"),
                            row["params"],
                        ]
                        for row in trials
                    ],
                    numeric=[True, False, False, True, False, True, True, False],
                ),
            )
        )

    diagnostics = context.get("diagnostics") or {}
    importances = diagnostics.get("param_importances") or []
    if importances:
        parts.append(
            _section(
                "Hyperparameter importances (from the Optuna study)",
                _table(
                    ["Hyperparameter", "Importance"],
                    [[item["parameter"], item["importance"]] for item in importances],
                    numeric=[False, True],
                ),
            )
        )
    history = diagnostics.get("optimization_history") or {}
    if history:
        parts.append(
            _section(
                "Optimization history summary",
                _table(
                    ["Trace", "Points", "First", "Last", "Best"],
                    [
                        [name, item["n_points"], item["first"], item["last"], item["best"]]
                        for name, item in history.items()
                    ],
                    numeric=[False, True, True, True, True],
                ),
            )
        )

    fairness = context.get("fairness") or {}
    search = fairness.get("search") or {}
    if search.get("included") and search.get("mode") and search["mode"] != "off":
        parts.append("## Fairness-aware search")
        parts.append(
            _kv_table(
                [
                    ("fairness_mode", search.get("mode")),
                    ("fairness_metric", search.get("metric")),
                    ("fairness_threshold", search.get("threshold")),
                    ("threshold_effective", search.get("threshold_effective")),
                    ("feasibility_rule", search.get("direction")),
                    ("protected_attribute", fairness.get("protected_attribute")),
                ]
            )
        )
        trial_disparities = search.get("trial_disparities")
        if trial_disparities:
            parts.append(
                _section(
                    "Per-trial disparity over the search",
                    _kv_table(list(trial_disparities.items())),
                )
            )

    pareto = context.get("pareto_front") or {}
    if pareto.get("present") and pareto.get("included"):
        parts.append("## Pareto front (multi-objective search)")
        parts.append(
            _kv_table(
                [
                    ("objectives", pareto.get("objective_names")),
                    ("n_pareto_points", pareto.get("n_points")),
                    ("knee_trial", pareto.get("knee_trial")),
                    ("trial_reported_in_this_report", pareto.get("reported_trial")),
                ]
            )
        )
        parts.append(
            _section(
                "Pareto-optimal trials",
                _table(
                    ["Trial", "F1", "Disparity", "Hyperparameters"],
                    [
                        [
                            point["trial"],
                            (point["values"] or [None])[0],
                            (point["values"] or [None, None])[1]
                            if len(point["values"] or []) > 1
                            else None,
                            point["params"],
                        ]
                        for point in pareto.get("points") or []
                    ],
                    numeric=[True, True, True, False],
                ),
            )
        )

    performance = context.get("performance") or {}
    parts.append("## Predictive performance (test split)")
    headline = performance.get("headline") or {}
    parts.append(
        _table(
            ["Metric", "Value", "Meaning"],
            [[key, value, _METRIC_MEANING.get(key, "")] for key, value in headline.items()],
            numeric=[False, True, False],
        )
    )
    curves = performance.get("curves") or {}
    if curves:
        parts.append(_section("Curve summary", _kv_table(list(curves.items()))))
    confusion = performance.get("confusion_matrix") or {}
    matrix = confusion.get("matrix")
    if matrix:
        labels = (
            confusion.get("labels")
            or performance.get("class_labels")
            or [str(index) for index in range(len(matrix))]
        )
        parts.append(
            _section(
                "Confusion matrix (rows = true class, columns = predicted class)",
                _table(
                    ["True \\ Predicted", *labels],
                    [[labels[index], *row] for index, row in enumerate(matrix)],
                    numeric=[False, *[True] * len(labels)],
                ),
            )
        )
    per_class = performance.get("per_class")
    if isinstance(per_class, dict) and per_class:
        parts.append(
            _section(
                "Per-class metrics",
                _table(
                    ["Class", "Precision", "Recall", "F1", "Support"],
                    [
                        [
                            name,
                            _number(values.get("precision")),
                            _number(values.get("recall")),
                            _number(values.get("f1-score")),
                            _number(values.get("support")),
                        ]
                        for name, values in per_class.items()
                        if isinstance(values, dict)
                    ],
                    numeric=[False, True, True, True, True],
                ),
            )
        )

    explainability = context.get("explainability") or {}
    parts.append("## Explainability (SHAP)")
    parts.append(
        _kv_table(
            [
                ("inspected_sample_index", explainability.get("inspected_sample_index")),
                ("class_index_explained", explainability.get("class_index")),
                ("shap_samples", explainability.get("shap_n_samples")),
                ("shap_base_value", explainability.get("shap_base_value")),
            ]
        )
    )
    importance = explainability.get("feature_importance") or []
    if importance:
        parts.append(
            _section(
                "Global feature importance (mean |SHAP|, descending)",
                _table(
                    ["Rank", "Feature", "Mean |SHAP|"],
                    [
                        [index + 1, item["feature"], item["mean_abs_shap"]]
                        for index, item in enumerate(importance)
                    ],
                    numeric=[True, False, True],
                ),
            )
        )
    statistics = explainability.get("shap_statistics") or []
    if statistics:
        parts.append(
            _section(
                "Per-feature SHAP statistics",
                _table(
                    [
                        "Feature",
                        "Mean |SHAP|",
                        "Mean SHAP (signed)",
                        "Min SHAP",
                        "Max SHAP",
                        "Feature min",
                        "Feature max",
                    ],
                    [
                        [
                            item["feature"],
                            item["mean_abs_shap"],
                            item["mean_shap"],
                            item["min_shap"],
                            item["max_shap"],
                            item["feature_min"],
                            item["feature_max"],
                        ]
                        for item in statistics
                    ],
                    numeric=[False, True, True, True, True, True, True],
                ),
            )
        )

    if fairness.get("audit_included"):
        parts.append("## Fairness audit")
        parts.append(
            _kv_table(
                [
                    ("protected_attribute", fairness.get("protected_attribute")),
                    ("audit_task_type", fairness.get("task_type")),
                    ("favorable_class", fairness.get("favorable_class")),
                    ("groups", fairness.get("groups")),
                ]
            )
        )
        by_group = fairness.get("by_group") or {}
        groups = fairness.get("groups") or []
        metric_names = [name for name in by_group if name != "count"]
        if groups and metric_names:
            parts.append(
                _section(
                    "Metrics by group",
                    _table(
                        ["Group", "Count", *[name.replace("_", " ") for name in metric_names]],
                        [
                            [
                                group,
                                _number((by_group.get("count") or {}).get(group)),
                                *[_number(by_group[name].get(group)) for name in metric_names],
                            ]
                            for group in groups
                        ],
                        numeric=[False, True, *[True] * len(metric_names)],
                    ),
                )
            )
        overall = fairness.get("overall") or {}
        if overall:
            parts.append(
                _section(
                    "Overall (ungrouped) values",
                    _table(
                        ["Metric", "Value"],
                        [[key, _number(value)] for key, value in overall.items()],
                        numeric=[False, True],
                    ),
                )
            )
        disparities = fairness.get("disparities") or {}
        if disparities:
            parts.append(
                _section(
                    "Disparity summary",
                    _table(
                        ["Measure", "Value", "Interpretation"],
                        [
                            [key, _number(value), _DISPARITY_MEANING.get(key, "")]
                            for key, value in disparities.items()
                        ],
                        numeric=[False, True, False],
                    ),
                )
            )
        worst = fairness.get("most_disadvantaged_group")
        if worst:
            parts.append(
                _section("Least-advantaged group by accuracy", _kv_table(list(worst.items())))
            )
        mitigation = fairness.get("mitigation")
        if mitigation:
            before = mitigation.get("before") or {}
            after = mitigation.get("after") or {}
            parts.append(
                _section(
                    f"Mitigation - ThresholdOptimizer (constraint: {mitigation.get('constraint')})",
                    _table(
                        ["Measure", "Before", "After", "Change"],
                        [
                            [
                                key,
                                before.get(key),
                                after.get(key),
                                (mitigation.get("delta") or {}).get(key),
                            ]
                            for key in before
                        ],
                        numeric=[False, True, True, True],
                    )
                    + "\n\n"
                    + _kv_table(
                        [
                            ("accuracy_before", mitigation.get("accuracy_before")),
                            ("accuracy_after", mitigation.get("accuracy_after")),
                        ]
                    ),
                )
            )

    parts.append("## Figure manifest")
    parts.append(figure_manifest_markdown(context.get("figures") or []))

    omissions = context.get("omissions") or []
    parts.append("## Not available in this run (do not speculate about these)")
    if omissions:
        parts.append("\n".join(f"- {item}" for item in omissions))
    else:
        parts.append("- Nothing: every evidence family listed above is present.")

    return "\n\n".join(part.rstrip() for part in parts) + "\n"


# --------------------------------------------------------------------------
# Tabular exports for the downloadable bundle
# --------------------------------------------------------------------------


def context_tables(context: dict) -> dict[str, list[list[Any]]]:  # noqa: C901
    """CSV-ready tables (header row first) for the downloadable bundle."""
    tables: dict[str, list[list[Any]]] = {}

    metrics = (context.get("performance") or {}).get("headline") or {}
    if metrics:
        tables["metrics"] = [["metric", "value"], *[[k, v] for k, v in metrics.items()]]

    importance = (context.get("explainability") or {}).get("feature_importance") or []
    if importance:
        tables["feature_importance"] = [
            ["rank", "feature", "mean_abs_shap"],
            *[[i + 1, row["feature"], row["mean_abs_shap"]] for i, row in enumerate(importance)],
        ]

    statistics = (context.get("explainability") or {}).get("shap_statistics") or []
    if statistics:
        header = list(statistics[0].keys())
        tables["shap_statistics"] = [header, *[[row[key] for key in header] for row in statistics]]

    trials = (context.get("optimization") or {}).get("trials") or []
    if trials:
        param_keys = sorted({key for row in trials for key in (row.get("params") or {})})
        header = ["trial", "state", "model_type", "objective", "values", *param_keys]
        tables["trials"] = [
            header,
            *[
                [
                    row["trial"],
                    row["state"],
                    row["model_type"],
                    row["objective"],
                    ";".join(str(v) for v in row["values"] or []),
                    *[(row.get("params") or {}).get(key) for key in param_keys],
                ]
                for row in trials
            ],
        ]

    families = (context.get("optimization") or {}).get("model_families") or []
    if families:
        header = list(families[0].keys())
        tables["model_families"] = [header, *[[row[key] for key in header] for row in families]]

    pareto = context.get("pareto_front") or {}
    if pareto.get("points"):
        param_keys = sorted(
            {key for point in pareto["points"] for key in (point.get("params") or {})}
        )
        header = ["trial", "f1", "disparity", *param_keys]
        tables["pareto_front"] = [
            header,
            *[
                [
                    point["trial"],
                    (point["values"] or [None])[0],
                    (point["values"] or [None, None])[1]
                    if len(point["values"] or []) > 1
                    else None,
                    *[(point.get("params") or {}).get(key) for key in param_keys],
                ]
                for point in pareto["points"]
            ],
        ]

    confusion = (context.get("performance") or {}).get("confusion_matrix") or {}
    if confusion.get("matrix"):
        labels = confusion.get("labels") or [str(i) for i in range(len(confusion["matrix"]))]
        tables["confusion_matrix"] = [
            ["true_class", *labels],
            *[[labels[i], *row] for i, row in enumerate(confusion["matrix"])],
        ]

    fairness = context.get("fairness") or {}
    by_group = fairness.get("by_group") or {}
    if by_group:
        metric_names = list(by_group)
        groups = fairness.get("groups") or sorted(
            {group for values in by_group.values() for group in values}
        )
        tables["fairness_by_group"] = [
            ["group", *metric_names],
            *[[group, *[by_group[name].get(group) for name in metric_names]] for group in groups],
        ]
    if fairness.get("disparities"):
        tables["fairness_disparities"] = [
            ["measure", "value"],
            *[[k, v] for k, v in fairness["disparities"].items()],
        ]
    mitigation = fairness.get("mitigation") or {}
    if mitigation.get("before"):
        tables["fairness_mitigation"] = [
            ["measure", "before", "after", "change"],
            *[
                [
                    key,
                    mitigation["before"].get(key),
                    (mitigation.get("after") or {}).get(key),
                    (mitigation.get("delta") or {}).get(key),
                ]
                for key in mitigation["before"]
            ],
        ]

    importances = (context.get("diagnostics") or {}).get("param_importances") or []
    if importances:
        tables["param_importances"] = [
            ["parameter", "importance"],
            *[[row["parameter"], row["importance"]] for row in importances],
        ]

    figures = context.get("figures") or []
    if figures:
        tables["figures"] = [
            ["figure_id", "title", "group", "kind", "path", "source"],
            *[
                [f["id"], f["title"], f["group"], f["kind"], f["path"], f.get("source", "")]
                for f in figures
            ],
        ]
    return tables
