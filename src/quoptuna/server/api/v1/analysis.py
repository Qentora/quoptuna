"""
Analysis endpoints (SHAP, metrics, AI reports).
"""

import asyncio
import base64
import io
import logging
import threading
import time
import zipfile
from contextvars import ContextVar
from datetime import datetime
from typing import Any, List, Optional, cast

import matplotlib as mpl

# Use a non-interactive backend; these endpoints render figures in FastAPI's
# threadpool where no GUI backend is available.
mpl.use("Agg")

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from quoptuna.backend.utils.storage import DEFAULT_DB_NAME
from quoptuna.backend.xai import prompts as report_prompts
from quoptuna.backend.xai import report_context, shap_progress

# Access optimization results stored by the optimize module.
from quoptuna.server.api.v1.optimize import (
    OptimizationRequest,
    build_workflow,
    get_job,
    serialize_study_trials,
)
from quoptuna.server.services import analysis_store, dataset_registry, research_bundle
from quoptuna.server.services.storage import optuna_storage_url
from quoptuna.server.services.workflow_service import (
    WorkflowExecutor,
    build_xai,
    study_best_trial,
)

logger = logging.getLogger(__name__)

router = APIRouter()

NON_CLASS_PLOTS = {"bar", "beeswarm", "violin", "heatmap"}
_job_xai: ContextVar[Any | None] = ContextVar("analysis_job_xai", default=None)


def _analysis_xai(opt_result: dict, **kwargs):
    """Reuse the single XAI instance while a durable analysis job is running."""
    return _job_xai.get() or build_xai(opt_result, **kwargs)


class SHAPRequest(BaseModel):
    optimization_id: str
    plot_types: List[str] = ["bar", "beeswarm", "violin", "heatmap", "waterfall"]
    trial_number: Optional[int] = None
    sample_index: int = 0
    use_proba: bool = True
    subset_size: int = 50
    # Background rows the masker marginalises over, and per-row masked
    # evaluations. None keeps the XAIConfig defaults (25 / 3 permutations);
    # both trade SHAP precision for wall-clock roughly linearly.
    background_size: Optional[int] = None
    max_evals: Optional[int] = None
    # Which class's SHAP values to slice/plot when values are per-class
    # (multiclass). None keeps the default (positive class for binary,
    # class 0 for multiclass).
    class_index: Optional[int] = None


class MetricsRequest(BaseModel):
    optimization_id: str
    trial_number: Optional[int] = None
    use_proba: bool = True
    subset_size: int = 50
    background_size: Optional[int] = None
    max_evals: Optional[int] = None


class ReportInclusionOptions(BaseModel):
    """Which evidence families reach the report agents.

    Defaults mirror ``report_prompts.REPORT_PROMPT_SETTINGS``, which is what the
    Settings page renders and documents, so an older client that sends none of
    these still gets the full bundle.
    """

    # Include a fairness audit in the report when a protected attribute is
    # available (stored with the run or given here).
    include_fairness: bool = True
    # Search-time fairness configuration: mode, disparity metric, threshold and
    # per-trial disparities. Without it a constrained run reads as unconstrained.
    include_fairness_search: bool = True
    # Pareto front of a multi-objective run (the front *is* the result).
    include_pareto: bool = True
    include_shap_detail: bool = True
    include_trial_history: bool = True
    include_study_plots: bool = True
    # Attach rendered figures to the request (off for text-only/cheap models).
    attach_figures: bool = True
    max_trial_rows: int = Field(default=40, ge=0, le=500)

    def inclusions(self) -> report_context.ReportInclusions:
        return report_context.ReportInclusions(
            fairness=self.include_fairness,
            fairness_search=self.include_fairness_search,
            pareto=self.include_pareto,
            shap_detail=self.include_shap_detail,
            trial_history=self.include_trial_history,
            study_plots=self.include_study_plots,
            figures=self.attach_figures,
            max_trial_rows=self.max_trial_rows,
        )


class ReportRequest(ReportInclusionOptions):
    # "model_name" collides with Pydantic's protected "model_" namespace; opt out.
    model_config = ConfigDict(protected_namespaces=())

    optimization_id: str
    analysis_snapshot_id: str
    analysis_revision: int
    trial_number: Optional[int] = None
    llm_provider: str = "google"
    api_key: str
    model_name: str = "gpt-4o"
    dataset_description: Optional[str] = None
    sensitive_feature: Optional[str] = None
    # Prompt overrides from Settings; empty falls back to the built-in prompts.
    analyst_instructions: Optional[str] = None
    reviewer_instructions: Optional[str] = None
    # The reviewer pass is what catches ungrounded numbers and broken tables.
    enable_review: bool = True


class AnalysisJobRequest(BaseModel):
    optimization_id: str
    trial_number: Optional[int] = None
    use_proba: bool = True
    subset_size: int = 50
    background_size: Optional[int] = None
    max_evals: Optional[int] = None
    class_index: int = 0
    sample_index: int = 0


class SnapshotFairnessRequest(BaseModel):
    sensitive_feature: Optional[str] = None
    mitigate: bool = False
    constraint: str = "equalized_odds"


class StudyPlotsRequest(BaseModel):
    optimization_id: str


class FairnessRequest(BaseModel):
    optimization_id: str
    # Falls back to the sensitive_feature persisted with the optimization request.
    sensitive_feature: Optional[str] = None
    trial_number: Optional[int] = None
    mitigate: bool = False
    constraint: str = "equalized_odds"


def _rehydrate_result(job: dict) -> dict:
    """Re-derive the analysis result for a completed run after a restart.

    The train/test split is deterministic (fixed random_state), so re-running
    only the data-prep nodes from the persisted request reproduces the exact
    DataFrames; best value/params are reloaded from the Optuna study on disk.
    """
    request = OptimizationRequest(**job["request"])
    workflow = build_workflow(job["id"], request, include_optimize=False)
    node_results = WorkflowExecutor(workflow).execute()["node_results"]
    # The optuna-config node's output merges all upstream data-prep outputs
    # (x_train/x_test/y_train/y_test/x_columns/y_column + study/db config).
    result = dict(node_results["optuna"])

    from optuna import load_study

    study = load_study(
        storage=optuna_storage_url(request.database_name), study_name=request.study_name
    )
    best_trial = study_best_trial(study)
    result.update(
        {
            "type": "optimization_result",
            "best_value": best_trial.values[0],
            "best_params": best_trial.params,
            "best_trial_number": best_trial.number,
            "model_name": request.model_name,
        }
    )
    return result


def _get_completed_result(optimization_id: str) -> dict:
    job = get_job(optimization_id)
    # A backend restart marks in-flight runs 'interrupted', but the Optuna
    # study on disk may already hold completed trials — those runs are
    # perfectly analyzable (rehydration below reloads the study; it fails
    # with a clear error if no trial ever completed).
    if job["status"] not in ("completed", "interrupted", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Current status: {job['status']}",
        )
    result = job.get("result")
    if not result:
        try:
            result = _rehydrate_result(job)
        except Exception as e:
            logger.exception("Failed to rehydrate result for %s", optimization_id)
            detail = (
                f"Optimization has no completed trials to analyze (status: {job['status']}): {e!s}"
                if job["status"] != "completed"
                else f"Optimization result not found: {e!s}"
            )
            raise HTTPException(status_code=400, detail=detail)
        job["result"] = result  # cache for subsequent analysis calls
    return result


#: Largest |trial F1 - analysis F1| treated as reproduction noise. Models seed
#: their init (``random_state=42``), so a faithful refit on the same frame
#: reproduces the trial's own test metrics almost exactly; anything above this
#: means the analysed model is not the model the search selected.
MAX_REFIT_METRIC_DRIFT = 0.02


def _refit_consistency(
    opt_result: dict,
    trial_number: int | None,
    analysis_f1,
    decision_threshold: float | None = None,
) -> dict | None:
    """Compare the analysis F1 against the F1 the trial recorded for itself.

    The search already scores every trial on the test split and stores it
    (``Quantum_f1_score`` / ``Classical_f1_score``). Analyze retrains that
    trial and recomputes the same number, so the two are the same quantity
    measured twice — they must agree.

    Every silent-divergence bug this pipeline has had would have shown up
    here on the first analysis: a validation split contaminated by resampled
    duplicates, a refit on the wrong frame, a mis-shaped training target, a
    decision threshold replayed onto an incompatible probability scale. None
    of them raised; all of them moved this delta.

    The comparison must use the same decision rule on both sides. The trial's
    headline attrs are unthresholded, so when the analysis applied a tuned
    ``decision_threshold`` the trial's ``f1_score_thresholded`` — recorded at
    that same cutoff — is the like-for-like number. Comparing against the
    unthresholded one instead reports the threshold's effect as drift, which
    is a property of the classifier, not a divergence.
    """
    from optuna import load_study

    if analysis_f1 is None:
        return None
    try:
        study = load_study(
            storage=optuna_storage_url(str(opt_result.get("db_name") or DEFAULT_DB_NAME)),
            study_name=opt_result.get("study_name"),
        )
        if trial_number is None:
            trial = study_best_trial(study)
        else:
            trial = next((t for t in study.trials if t.number == trial_number), None)
        if trial is None:
            return None
        attrs = trial.user_attrs
        if decision_threshold is not None and attrs.get("f1_score_thresholded") is not None:
            recorded = attrs["f1_score_thresholded"]
            rule = f"threshold={decision_threshold}"
        else:
            # Exactly one of the two families is non-zero for a given trial.
            recorded = attrs.get("Quantum_f1_score") or attrs.get("Classical_f1_score")
            rule = "argmax"
        if recorded is None:
            return None
    except Exception:  # a consistency check must never fail the analysis
        logger.warning("Could not load the trial's recorded metrics", exc_info=True)
        return None

    drift = abs(float(analysis_f1) - float(recorded))
    return {
        "trial_test_f1": float(recorded),
        "analysis_test_f1": float(analysis_f1),
        "decision_rule": rule,
        "drift": drift,
        "within_tolerance": drift <= MAX_REFIT_METRIC_DRIFT,
        "tolerance": MAX_REFIT_METRIC_DRIFT,
    }


def _analysed_model(opt_result: dict, xai, trial_number: int | None) -> dict:
    """Provenance for the model a snapshot explains.

    ``trial_number`` is what the request asked for; ``None`` means "the study's
    best trial", which ``build_xai`` resolved. Resolve it again here so the
    stored record names a concrete trial rather than "best", which would drift
    if the study is extended later.
    """
    from optuna import load_study

    resolved = trial_number
    params: dict = {}
    try:
        study = load_study(
            storage=optuna_storage_url(str(opt_result.get("db_name") or DEFAULT_DB_NAME)),
            study_name=opt_result.get("study_name"),
        )
        if resolved is None:
            trial = study_best_trial(study)
        else:
            trial = next((t for t in study.trials if t.number == resolved), None)
        if trial is not None:
            resolved = trial.number
            params = dict(trial.params)
    except Exception:  # provenance must never fail a completed analysis
        logger.warning("Could not resolve analysed trial for provenance", exc_info=True)

    budget = {
        "max_steps": opt_result.get("max_steps"),
        "convergence_interval": opt_result.get("convergence_interval"),
        "dev_type": opt_result.get("dev_type"),
    }
    return {
        "trial_number": resolved,
        "requested_trial": trial_number,
        "selected_by": "best_trial" if trial_number is None else "explicit",
        "best_trial_number": opt_result.get("best_trial_number"),
        "is_best_trial": resolved is not None and resolved == opt_result.get("best_trial_number"),
        "model_type": params.get("model_type"),
        "params": {k: v for k, v in params.items() if k != "model_type"},
        "training_budget": {k: v for k, v in budget.items() if v is not None},
        # The cutoff every label-based metric below was produced at; None
        # means the model's own predict() (argmax, or 0.5 for binary proba).
        # ``decision_threshold_discarded`` explains a None that the trial did
        # record a threshold for.
        "decision_threshold": getattr(xai, "decision_threshold", None),
        "decision_threshold_discarded": getattr(xai, "threshold_discarded", None),
        "retrained_at": datetime.now().isoformat(),
    }


def _warm_xai_caches(xai, use_proba: bool) -> None:
    """Populate the lazy caches the parallel sections read.

    ``XAI.shap_values`` / ``predictions`` / ``predictions_proba`` memoise into
    unguarded attributes. Computing them once up front makes the later
    concurrent readers side-effect free. Each is best-effort: a model without
    ``predict_proba`` must not fail the whole analysis here, because the
    sections that need it already degrade on their own.
    """
    for label, get in (
        ("shap_values", lambda: xai.shap_values),
        ("predictions", lambda: xai.predictions),
        *((("predictions_proba", lambda: xai.predictions_proba),) if use_proba else ()),
    ):
        try:
            get()
        except Exception:
            logger.debug("Could not pre-compute %s; sections will handle it", label)


def _figure_to_data_url(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _plot_class_index(xai, requested: Optional[int] = None) -> int:
    """Class slice to use for SHAP plots when values are per-class (ndim > 2).

    Many quantum/classical models emit per-class SHAP values shaped
    ``(samples, features, classes)``; SHAP's plots need a single 2-D slice.
    An explicit ``requested`` index (from the API) wins; otherwise pick the
    positive (last) class for binary and the first class for multiclass.
    Returns -1 when values are already 2-D (no slicing needed).
    """
    try:
        shap_values = xai.shap_values
        if getattr(shap_values, "values", None) is None or shap_values.values.ndim <= 2:
            return -1
        classes = list(xai.get_classes())
        # Honor an explicit class only for multiclass: the UI always sends
        # class_index=0 and hides the picker for binary, so honoring it there
        # would silently flip binary SHAP plots to the negative class.
        if len(classes) > 2 and requested is not None and 0 <= requested < len(classes):
            return int(requested)
        return int(classes[-1] if len(classes) <= 2 else classes[0])
    except Exception:
        return -1


def _task_spec(opt_result: dict) -> Optional[dict]:
    """Class-structure spec for this run (see TaskSpec.to_dict), if known.

    Present on the in-process/rehydrated result (threaded through the split
    node); falls back to the study user_attrs written at optimize time.
    """
    spec = opt_result.get("task_spec")
    if spec:
        return spec
    try:
        from optuna import load_study

        study = load_study(
            storage=optuna_storage_url(str(opt_result.get("db_name") or DEFAULT_DB_NAME)),
            study_name=opt_result.get("study_name"),
        )
        return study.user_attrs.get("task_spec")
    except Exception:
        return None


def _spec_display_labels(spec: Optional[dict], encoded: list) -> list[str]:
    """Original class names for encoded label values, falling back to str()."""
    if not spec:
        return [str(v) for v in encoded]
    labels = [str(c) for c in spec.get("class_labels", [])]
    if spec.get("kind") == "binary":
        mapping = {-1: labels[0], 1: labels[1]} if len(labels) == 2 else {}
        return [mapping.get(int(v), str(v)) for v in encoded]
    out = []
    for v in encoded:
        code = int(v)
        out.append(labels[code] if 0 <= code < len(labels) else str(v))
    return out


def _positive_proba(proba):
    """Reduce a probability array/frame to the 1-D positive-class column."""
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return np.ravel(arr)


MAX_CURVE_POINTS = 500


def _downsample_indices(n: int, max_points: int = MAX_CURVE_POINTS) -> np.ndarray:
    """Evenly spaced indices (keeping both endpoints) capping a curve at max_points."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).round().astype(int))


def _roc_payload(y_test, proba) -> dict[str, Any]:
    """Raw ROC curve points (downsampled) + AUC, from the same inputs as the PNG plot."""
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_test, proba)
    # roc_curve on a single-class y returns NaN arrays with only a warning;
    # NaN in the payload would 500 the whole response (json allow_nan=False).
    fpr, tpr = np.asarray(fpr), np.asarray(tpr)
    finite = np.isfinite(fpr) & np.isfinite(tpr)
    if not finite.any():
        msg = "ROC curve undefined (evaluation labels contain a single class)"
        raise ValueError(msg)
    fpr, tpr = fpr[finite], tpr[finite]
    idx = _downsample_indices(len(fpr))
    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = None
    return {
        "fpr": fpr[idx].tolist(),
        "tpr": tpr[idx].tolist(),
        "auc": auc,
    }


def _pr_payload(y_test, proba) -> dict[str, Any]:
    """Raw precision-recall points (downsampled) + AP, matching the PNG plot inputs."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_test, proba)
    idx = _downsample_indices(len(precision))
    try:
        avg_prec = float(average_precision_score(y_test, proba))
    except Exception:
        avg_prec = None
    return {
        "precision": np.asarray(precision)[idx].tolist(),
        "recall": np.asarray(recall)[idx].tolist(),
        "average_precision": avg_prec,
    }


def _per_class_curve_payloads(y_test, proba, spec: dict, model_classes=None) -> tuple[dict, dict]:
    """One-vs-rest ROC and PR payloads per class for a multiclass task.

    Binarizes the encoded labels per class and reuses the binary payload
    helpers on each (indicator, class-probability-column) pair.

    ``model_classes`` (the fitted model's ``classes_``) maps encoded class
    codes to probability columns: models fit their class list from y_train,
    so a class absent from the (unstratified) train split shifts every later
    proba column. Without the mapping we'd attribute curves to wrong names.
    """
    from sklearn.preprocessing import label_binarize

    n_classes = int(spec["n_classes"])
    encoded = list(range(n_classes))
    names = _spec_display_labels(spec, encoded)
    proba = np.asarray(proba)
    y_bin = label_binarize(np.asarray(y_test).ravel(), classes=encoded)
    col_of = {int(c): i for i, c in enumerate(model_classes)} if model_classes is not None else None

    roc_classes, pr_classes = [], []
    for k in range(n_classes):
        col = col_of.get(k) if col_of is not None else k
        if col is None or col >= proba.shape[1]:
            logger.warning(
                "No probability column for class %s (absent from training data); skipping",
                names[k],
            )
            continue
        if len(np.unique(y_bin[:, k])) < 2:
            logger.warning(
                "Class %s absent from the evaluation subset; skipping its curves", names[k]
            )
            continue
        try:
            roc_classes.append({"label": names[k], **_roc_payload(y_bin[:, k], proba[:, col])})
        except Exception as exc:
            logger.warning("ROC curve failed for class %s: %s", names[k], exc)
        try:
            pr_classes.append({"label": names[k], **_pr_payload(y_bin[:, k], proba[:, col])})
        except Exception as exc:
            logger.warning("PR curve failed for class %s: %s", names[k], exc)

    aucs = [c["auc"] for c in roc_classes if c.get("auc") is not None]
    macro_auc = float(np.mean(aucs)) if len(aucs) == n_classes else None
    return (
        {"per_class": roc_classes, "macro_auc": macro_auc},
        {"per_class": pr_classes},
    )


def _confusion_matrix_payload(matrix, labels) -> dict[str, Any]:
    """Counts + row-normalized confusion matrix with string class labels."""
    cm = np.asarray(matrix, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    return {
        "labels": [str(label) for label in labels],
        "matrix": cm.astype(int).tolist(),
        "normalized": normalized.tolist(),
    }


def _feature_importance_from_xai(xai) -> list[dict[str, Any]]:
    shap_values = xai.shap_values
    importance: list[dict[str, Any]] = []
    if hasattr(shap_values, "values"):
        values = np.abs(shap_values.values)
        # Collapse a trailing per-class axis: (samples, features, classes) -> (samples, features).
        if values.ndim > 2:
            values = values.mean(axis=-1)
        mean_abs = values.mean(axis=0)
        feature_names = xai.feature_names or [f"feature_{i}" for i in range(len(mean_abs))]
        for i, feature in enumerate(feature_names):
            value = mean_abs[i]
            importance.append(
                {
                    "feature": feature,
                    "importance": float(value if np.ndim(value) == 0 else np.mean(value)),
                }
            )
        importance.sort(key=lambda item: item["importance"], reverse=True)
    return importance


MAX_SHAP_SAMPLES = 200


def _shap_data_payload(
    shap_values, class_idx: int, max_samples: int = MAX_SHAP_SAMPLES
) -> dict[str, Any]:
    """JSON-safe raw SHAP data from a shap.Explanation-like object.

    Slices a trailing per-class axis (values ``(samples, features, classes)``
    -> ``[:, :, class_idx]``, matching the plotting slice), evenly subsamples
    rows to ``max_samples`` (values and data kept row-aligned), and converts
    everything to plain Python floats with NaN/inf mapped to ``None``.
    """
    values = np.asarray(shap_values.values, dtype=float)
    if values.ndim > 2:
        idx = max(class_idx, 0)
        values = values[:, :, idx]

    data = getattr(shap_values, "data", None)
    data = np.asarray(data, dtype=float) if data is not None else None

    n = values.shape[0]
    row_idx = _downsample_indices(n, max_samples)
    values = values[row_idx]
    if data is not None:
        data = data[row_idx]

    base_values = getattr(shap_values, "base_values", None)
    base_value = None
    if base_values is not None:
        bv = np.asarray(base_values, dtype=float)
        if bv.ndim == 0:
            base_value = float(bv)
        else:
            bv = bv[row_idx[0]] if bv.shape[0] == n else bv
            bv = np.asarray(bv, dtype=float)
            if bv.ndim >= 1:  # per-class base values -> same class slice
                bv = bv.flat[max(class_idx, 0)]
            base_value = float(bv)
        if base_value is not None and not np.isfinite(base_value):
            base_value = None

    def _safe_rows(arr) -> list[list[Optional[float]]]:
        return [
            [float(v) if np.isfinite(v) else None for v in row]
            for row in np.asarray(arr, dtype=float)
        ]

    feature_names = list(getattr(shap_values, "feature_names", None) or []) or [
        f"feature_{i}" for i in range(values.shape[1])
    ]

    return {
        "feature_names": [str(f) for f in feature_names],
        "values": _safe_rows(values),
        "data": _safe_rows(data) if data is not None else [],
        "base_value": base_value,
        "n_samples": int(values.shape[0]),
    }


def _shap_progress_sink(job_id: str, min_interval: float = 1.0):
    """Persist SHAP's row progress, throttled to one write per second.

    A write per explained row would cost more than the explanation itself on a
    fast model. The final row always writes, so the UI never stalls one short
    of complete.
    """
    last = 0.0

    def sink(done: int, total: int) -> None:
        nonlocal last
        now = time.monotonic()
        if done < total and now - last < min_interval:
            return
        last = now
        analysis_store.update_job(job_id, progress_done=done, progress_total=total)
        # Called once per explained row, which makes it the finest-grained
        # place to honour a stop request inside the longest step.
        _stop_if_cancelled(job_id)

    return sink


class _JobCancelled(BaseException):
    """Raised inside a running analysis once the client asks it to stop.

    Deliberately a ``BaseException``: the job body and SHAP's progress hook
    both swallow ``Exception`` broadly, and a cancellation that gets caught
    there would be recorded as a failure instead of a stop.
    """


#: Job ids the client has asked to stop. In-process by design — an analysis is
#: an in-process background task, so nothing outside this process can be
#: running one.
_cancelled_jobs: set[str] = set()
_cancelled_lock = threading.Lock()


def _request_cancel(job_id: str) -> None:
    with _cancelled_lock:
        _cancelled_jobs.add(job_id)


def _is_cancelled(job_id: str) -> bool:
    with _cancelled_lock:
        return job_id in _cancelled_jobs


def _clear_cancel(job_id: str) -> None:
    with _cancelled_lock:
        _cancelled_jobs.discard(job_id)


def _stop_if_cancelled(job_id: str) -> None:
    if _is_cancelled(job_id):
        raise _JobCancelled(job_id)


#: Analysis jobs render through pyplot's global figure state, and until they
#: moved to the threadpool the event loop serialized them for free. Two
#: interleaving in different threads would corrupt each other's figures, so
#: keep the old one-at-a-time behaviour explicitly.
_analysis_job_lock = threading.Lock()


def _run_analysis_job(job_id: str, request: AnalysisJobRequest) -> None:
    """Run one analysis off the event loop.

    Every step below is synchronous CPU work — rehydration, the model refit,
    SHAP, figure rendering — and an ``async def`` background task runs *on* the
    event loop. That left the server unable to answer the client's progress
    polls for the whole job: the UI sat on "Starting analysis" and then showed
    whichever section happened to be current once the loop freed up, which was
    always SHAP. Per-row SHAP progress was unreadable for the same reason.

    Starlette runs a non-async background task in its threadpool, so making
    this synchronous keeps the loop free and lets progress arrive as it happens.
    """
    with _analysis_job_lock:
        asyncio.run(_run_analysis_job_async(job_id, request))


async def _run_analysis_job_async(job_id: str, request: AnalysisJobRequest) -> None:
    """Compute and persist one complete analysis bundle."""
    config = analysis_store.normalize_config(request.model_dump(exclude={"optimization_id"}))
    trial = config["trial_number"]
    metrics_request = MetricsRequest(
        optimization_id=request.optimization_id,
        trial_number=trial,
        use_proba=config["use_proba"],
        subset_size=config["subset_size"],
        background_size=config.get("background_size"),
        max_evals=config.get("max_evals"),
    )
    shap_request = SHAPRequest(
        optimization_id=request.optimization_id,
        trial_number=trial,
        sample_index=config["sample_index"],
        use_proba=config["use_proba"],
        subset_size=config["subset_size"],
        background_size=config.get("background_size"),
        max_evals=config.get("max_evals"),
        class_index=config["class_index"],
    )
    warnings: dict[str, str] = {}

    async def optional(section: str, call):
        _stop_if_cancelled(job_id)
        analysis_store.update_job(job_id, current_section=section)
        try:
            return await call
        except Exception as exc:  # optional visual sections must not discard core results
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            warnings[section] = str(detail)
            logger.warning("Analysis section %s failed: %s", section, detail)
            return None

    async def gather_optional(section: str, calls: dict[str, Any]) -> dict[str, Any]:
        """Run independent sections concurrently, recording failures per name.

        Safe only once the shared ``XAI``'s lazy caches are warm: the section
        coroutines read ``shap_values`` / ``predictions*`` off one instance and
        those properties are not synchronised, so racing them would recompute
        (or interleave) the same work. ``_warm_xai_caches`` populates them
        first, leaving these calls as pure readers.

        The coroutines are async but CPU-bound, so this overlaps their awaits
        rather than their compute; it is ordering, not true parallelism. It
        still removes the serialised section-by-section stalls and keeps one
        failure from cancelling its siblings.
        """
        analysis_store.update_job(job_id, current_section=section)
        names = list(calls)
        settled = await asyncio.gather(*(calls[name] for name in names), return_exceptions=True)
        results: dict[str, Any] = {}
        for name, outcome in zip(names, settled, strict=True):
            if isinstance(outcome, BaseException):
                detail = outcome.detail if isinstance(outcome, HTTPException) else str(outcome)
                warnings[name] = str(detail)
                logger.warning("Analysis section %s failed: %s", name, detail)
                results[name] = None
            else:
                results[name] = outcome
        return results

    token = None
    try:
        # Two substantial steps run before SHAP and both used to report as
        # "shap", which made the job look stuck. Rehydration re-runs data prep
        # for a run whose in-memory job was lost, and build_xai refits the
        # trial's model - on a variational model that is the slowest step of
        # the whole analysis.
        _stop_if_cancelled(job_id)
        analysis_store.update_job(job_id, status="running", current_section="preparing")
        opt_result = _get_completed_result(request.optimization_id)
        _stop_if_cancelled(job_id)
        analysis_store.update_job(job_id, current_section="training")
        shared_xai = build_xai(
            opt_result,
            trial_number=trial,
            use_proba=config["use_proba"],
            subset_size=config["subset_size"],
            background_size=config.get("background_size"),
            max_evals=config.get("max_evals"),
        )
        token = _job_xai.set(shared_xai)
        analysis_store.update_job(job_id, current_section="shap", progress_done=0, progress_total=0)
        # SHAP and metrics are the required core sections. Existing endpoint
        # functions remain the compatibility implementation for now; this job
        # owns orchestration and persistence.
        #
        # SHAP explains row by row and is the longest step here; report that
        # progress to the job so the browser sees what the server's terminal
        # already shows.
        with shap_progress.report_progress(_shap_progress_sink(job_id)):
            shap = await generate_shap_analysis(shap_request)
        analysis_store.update_job(job_id, progress_done=None, progress_total=None)
        metrics = await generate_metrics(metrics_request)
        # Publish the core sections now: the derived ones below can take a
        # while, and there is no reason to withhold finished SHAP and metrics
        # until they land.
        analysis_store.publish_partial(
            job_id,
            {
                "feature_importance": shap.get("feature_importance"),
                "plots": dict(shap.get("plots") or {}),
                "metrics": metrics.get("metrics"),
                "confusion_matrix_plot": metrics.get("confusion_matrix_plot"),
                "task_type": metrics.get("task_type"),
                "class_labels": metrics.get("class_labels"),
            },
        )

        # Everything below only reads the shared XAI. Warm its lazy caches so
        # the concurrent sections cannot race on first computation.
        _warm_xai_caches(shared_xai, config["use_proba"])

        derived = await gather_optional(
            "derived",
            {
                "curves": generate_curves(metrics_request),
                "curves_data": generate_curves_data(metrics_request),
                "confusion_matrix_data": generate_confusion_matrix_data(metrics_request),
                "feature_importance_data": generate_feature_importance_data(metrics_request),
                "shap_data": generate_shap_data(shap_request),
            },
        )
        curves = derived["curves"]
        curves_data = derived["curves_data"]
        confusion_data = derived["confusion_matrix_data"]
        importance_data = derived["feature_importance_data"]
        shap_data = derived["shap_data"]

        # Study plots read the Optuna study, not the XAI, and fairness needs
        # its own refit, so they stay off the shared-instance group above.
        job = get_job(request.optimization_id)
        sensitive = (job.get("request") or {}).get("sensitive_feature")
        tail: dict[str, Any] = {
            "study_plots": generate_study_plots(
                StudyPlotsRequest(optimization_id=request.optimization_id)
            )
        }
        if sensitive:
            tail["fairness"] = generate_fairness(
                FairnessRequest(
                    optimization_id=request.optimization_id,
                    trial_number=trial,
                    sensitive_feature=sensitive,
                )
            )
        tail_results = await gather_optional("study_plots", tail)
        study = tail_results["study_plots"]
        fairness = tail_results.get("fairness")

        plots = dict(shap.get("plots") or {})
        if curves and curves.get("roc_curve_plot"):
            plots["rocCurve"] = curves["roc_curve_plot"]
        if curves and curves.get("pr_curve_plot"):
            plots["prCurve"] = curves["pr_curve_plot"]
        analysed_model = _analysed_model(opt_result, shared_xai, trial)
        # The search and the analysis measure the same quantity on the same
        # split; a disagreement means the analysed model is not the selected
        # one. Surfaced as a warning so it reaches the UI and the report agent
        # instead of only a log line.
        consistency = _refit_consistency(
            opt_result,
            analysed_model.get("trial_number"),
            (metrics.get("metrics") or {}).get("f1_score"),
            decision_threshold=analysed_model.get("decision_threshold"),
        )
        analysed_model["refit_consistency"] = consistency
        if consistency and not consistency["within_tolerance"]:
            warnings["refit_consistency"] = (
                f"The analysed model scores F1 {consistency['analysis_test_f1']:.3f} on the test "
                f"split, but trial {analysed_model.get('trial_number')} recorded "
                f"{consistency['trial_test_f1']:.3f} for itself during the search "
                f"(drift {consistency['drift']:.3f}). The analysis is not describing the model "
                "that was selected; treat these metrics as unreliable."
            )
            logger.warning(warnings["refit_consistency"])
        payload = {
            "feature_importance": shap.get("feature_importance"),
            "plots": plots,
            "study_plots": (study or {}).get("plots"),
            "metrics": metrics.get("metrics"),
            "confusion_matrix_plot": metrics.get("confusion_matrix_plot"),
            "roc_auc": (curves or {}).get("roc_auc"),
            "average_precision": (curves or {}).get("average_precision"),
            "fairness": fairness,
            "curves_data": curves_data,
            "confusion_data": confusion_data,
            "importance_data": importance_data,
            "shap_data": shap_data,
            "task_type": metrics.get("task_type"),
            "class_labels": metrics.get("class_labels"),
            "warnings": warnings,
            # Which model this snapshot actually explains. Recorded at
            # analysis time so a stored snapshot (and any report built from
            # it) stays attributable after the study or request changes.
            "analysed_model": analysed_model,
        }
        analysis_store.complete_job(job_id, payload)
    except _JobCancelled:
        logger.info("Analysis job %s stopped at the client's request", job_id)
        analysis_store.update_job(
            job_id,
            status="cancelled",
            current_section=None,
            progress_done=None,
            progress_total=None,
            completed_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        logger.exception("Analysis job %s failed", job_id)
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        analysis_store.update_job(
            job_id, status="failed", error=str(detail), completed_at=datetime.now().isoformat()
        )
    finally:
        _clear_cancel(job_id)
        if token is not None:
            _job_xai.reset(token)


@router.post("/jobs")
async def start_analysis_job(request: AnalysisJobRequest, background_tasks: BackgroundTasks):
    """Start analysis only after an explicit client request."""
    _get_completed_result(request.optimization_id)
    job = analysis_store.create_job(request.optimization_id, request.model_dump())
    if job.pop("created"):
        background_tasks.add_task(_run_analysis_job, job["id"], request)
    return job


@router.get("/jobs")
async def find_active_analysis_job(optimization_id: str):
    """The analysis still running for this optimization, or ``null``.

    Declared before ``/jobs/{job_id}`` for readability only; the paths do not
    overlap. Lets a client that lost its job id — a browser refresh — reattach
    to work that is still going rather than starting a duplicate run.
    """
    return {"job": analysis_store.find_active_job(optimization_id)}


@router.post("/jobs/{job_id}/cancel")
async def cancel_analysis_job(job_id: str):
    """Ask a running analysis to stop.

    Cooperative: the job checks between sections and once per explained SHAP
    row, so a stop lands within a row rather than instantly. A job that has
    already finished is left alone.
    """
    job = analysis_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if job["status"] not in {"pending", "running"}:
        return {"id": job_id, "status": job["status"], "cancelled": False}
    _request_cancel(job_id)
    return {"id": job_id, "status": "cancelling", "cancelled": True}


@router.get("/jobs/{job_id}")
async def get_analysis_job(job_id: str):
    job = analysis_store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job


@router.get("/snapshots")
async def list_analysis_snapshots(optimization_id: str):
    get_job(optimization_id)
    snapshots = analysis_store.list_snapshots(optimization_id)
    return {"snapshots": snapshots}


@router.get("/snapshots/{snapshot_id}")
async def get_analysis_snapshot(snapshot_id: str):
    snapshot = analysis_store.get_snapshot(snapshot_id)
    if not snapshot or snapshot["revision"] < 1:
        raise HTTPException(status_code=404, detail="Completed analysis snapshot not found")
    return snapshot


@router.get("/snapshots/{snapshot_id}/revisions")
async def list_analysis_revisions(snapshot_id: str):
    """Analysis history for a snapshot, newest first."""
    if analysis_store.get_snapshot(snapshot_id, hydrate=False) is None:
        raise HTTPException(status_code=404, detail="Analysis snapshot not found")
    return {"revisions": analysis_store.list_revisions(snapshot_id)}


@router.get("/snapshots/{snapshot_id}/revisions/{revision}")
async def get_analysis_revision(snapshot_id: str, revision: int):
    found = analysis_store.get_revision(snapshot_id, revision)
    if found is None:
        raise HTTPException(status_code=404, detail="Analysis revision not found")
    return found


@router.get("/snapshots/{snapshot_id}/artifacts/{filename}")
async def get_analysis_artifact(snapshot_id: str, filename: str):
    """Return a short-lived S3 URL for an analysis artifact."""
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid artifact filename")
    url = analysis_store.artifact_url(snapshot_id, filename)
    if url:
        return {"snapshot_id": snapshot_id, "filename": filename, "url": url}
    snapshot = analysis_store.get_snapshot(snapshot_id, hydrate=False)
    if not snapshot:
        raise HTTPException(status_code=404, detail="Analysis snapshot not found")
    raise HTTPException(status_code=404, detail="Artifact is not stored in object storage")


@router.get("/snapshots/{snapshot_id}/reports")
async def list_snapshot_reports(snapshot_id: str):
    if not analysis_store.get_snapshot(snapshot_id, hydrate=False):
        raise HTTPException(status_code=404, detail="Analysis snapshot not found")
    return {"reports": analysis_store.list_reports(snapshot_id)}


@router.post("/snapshots/{snapshot_id}/fairness")
async def update_snapshot_fairness(snapshot_id: str, request: SnapshotFairnessRequest):
    """Explicitly compute and persist a fairness audit or mitigation revision."""
    snapshot = analysis_store.get_snapshot(snapshot_id)
    if not snapshot or not snapshot.get("payload"):
        raise HTTPException(status_code=404, detail="Completed analysis snapshot not found")
    config = snapshot["config"]
    opt_result = _get_completed_result(snapshot["optimization_id"])
    try:
        xai = _analysis_xai(opt_result, trial_number=config.get("trial_number"))
        fairness = _compute_fairness_payload(
            snapshot["optimization_id"],
            request.sensitive_feature,
            xai,
            mitigate=request.mitigate,
            constraint=request.constraint,
            task_spec=_task_spec(opt_result),
            persisted_sensitive_train=opt_result.get("sensitive_train"),
        )
        payload = dict(snapshot["payload"])
        payload["fairness"] = {
            "optimization_id": snapshot["optimization_id"],
            "status": "completed",
            **fairness,
        }
        job = analysis_store.create_revision_job(snapshot_id)
        completed = analysis_store.complete_job(job["id"], payload)
        return {"fairness": payload["fairness"], **completed}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update fairness: {exc!s}")


@router.post("/shap/data")
async def generate_shap_data(request: SHAPRequest):
    """Raw SHAP values/data as JSON (for frontend charting).

    Same computation path as ``/shap`` (identical model loading, test split
    and class slicing); rows are evenly subsampled to at most 200.
    """
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )
        class_index = _plot_class_index(xai, request.class_index)
        payload = _shap_data_payload(xai.shap_values, class_index)
        spec = _task_spec(opt_result)
        return {
            "optimization_id": request.optimization_id,
            **payload,
            "class_index": class_index,
            "n_classes": int(spec["n_classes"]) if spec else 2,
            "class_labels": [str(c) for c in spec["class_labels"]] if spec else None,
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute SHAP data: {e!s}")


@router.post("/shap")
async def generate_shap_analysis(request: SHAPRequest):
    """Generate SHAP plots and real feature importance for a chosen trial."""
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )

        # Per-class SHAP values (ndim > 2) must be sliced to one class for plotting.
        class_index = _plot_class_index(xai, request.class_index)

        plots: dict[str, str] = {}
        for plot_type in request.plot_types:
            try:
                if plot_type in NON_CLASS_PLOTS:
                    plots[plot_type] = xai.get_plot(plot_type, class_index=class_index)
                elif plot_type == "waterfall":
                    plots[plot_type] = xai.get_waterfall_plot(
                        index=request.sample_index, class_index=class_index
                    )
            except Exception as exc:
                logger.error("Failed to generate %s plot: %s", plot_type, exc)

        return {
            "optimization_id": request.optimization_id,
            "feature_importance": _feature_importance_from_xai(xai),
            "plots": plots,
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP analysis: {e!s}")


@router.post("/metrics")
async def generate_metrics(request: MetricsRequest):
    """Compute classification metrics and a confusion-matrix plot for a trial."""
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )

        from sklearn.metrics import average_precision_score, roc_auc_score

        fig = xai.plot_confusion_matrix()
        confusion_plot = _figure_to_data_url(fig)

        spec = _task_spec(opt_result)
        multiclass = bool(spec and spec.get("kind") == "multiclass")
        average = "macro" if multiclass else "binary"

        metrics: dict[str, Any] = {}

        def _safe(name: str, func) -> None:
            try:
                value = func()
                metrics[name] = float(value) if np.ndim(value) == 0 else value
            except Exception as exc:
                logger.warning("Metric %s failed: %s", name, exc)

        _safe("f1_score", lambda: xai.get_f1_score(average=average))
        _safe("precision", lambda: xai.get_precision(average=average))
        _safe("recall", lambda: xai.get_recall(average=average))
        if multiclass:
            # Full (n, K) probability matrix with one-vs-rest macro averaging.
            _safe(
                "roc_auc_score",
                lambda: roc_auc_score(
                    xai.y_test,
                    np.asarray(xai.predictions_proba),
                    multi_class="ovr",
                    average="macro",
                ),
            )
        else:
            # ROC AUC / AP need the 1-D positive-class probability, not the (n, 2) array.
            _safe(
                "roc_auc_score",
                lambda: roc_auc_score(xai.y_test, _positive_proba(xai.predictions_proba)),
            )
            _safe(
                "average_precision_score",
                lambda: average_precision_score(xai.y_test, _positive_proba(xai.predictions_proba)),
            )
        _safe("mcc", xai.get_mcc)
        _safe("cohens_kappa", xai.get_cohens_kappa)
        _safe("log_loss", xai.get_log_loss)

        if multiclass:
            try:
                from sklearn.metrics import classification_report

                encoded = sorted(np.unique(np.asarray(xai.y_test)).tolist())
                names = _spec_display_labels(spec, encoded)
                report = classification_report(
                    xai.y_test,
                    xai.predictions,
                    labels=encoded,
                    target_names=names,
                    output_dict=True,
                    zero_division=0,
                )
                metrics["per_class"] = {name: report[name] for name in names if name in report}
            except Exception as exc:
                logger.warning("per_class metrics failed: %s", exc)

        try:
            from sklearn.metrics import accuracy_score

            metrics["accuracy"] = float(accuracy_score(xai.y_test, xai.predictions))
        except Exception as exc:
            logger.warning("accuracy failed: %s", exc)

        try:
            metrics["classification_report"] = xai.get_classification_report()
        except Exception as exc:
            logger.warning("classification_report failed: %s", exc)

        try:
            metrics["confusion_matrix"] = xai.get_confusion_matrix().tolist()
        except Exception as exc:
            logger.warning("confusion_matrix failed: %s", exc)

        return {
            "optimization_id": request.optimization_id,
            "confusion_matrix_plot": confusion_plot,
            "metrics": metrics,
            "task_type": (spec or {}).get("kind", "binary"),
            "n_classes": int(spec["n_classes"]) if spec else 2,
            "class_labels": [str(c) for c in spec["class_labels"]] if spec else None,
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {e!s}")


@router.post("/curves")
async def generate_curves(request: MetricsRequest):
    """ROC and precision-recall curve plots for a chosen trial.

    Each curve is rendered independently; a failure on one (e.g. multiclass or
    missing probabilities) yields ``null`` rather than failing the request.
    """
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build XAI: {e!s}")

    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    roc_plot = None
    pr_plot = None
    roc_auc = None
    avg_prec = None

    spec = _task_spec(opt_result)
    multiclass_spec = spec if spec is not None and spec.get("kind") == "multiclass" else None
    y_test = xai.y_test

    if multiclass_spec is not None:
        # One-vs-rest: one line per class on each plot.
        roc_data, pr_data = _per_class_curve_payloads(
            y_test,
            xai.predictions_proba,
            multiclass_spec,
            model_classes=getattr(xai.model, "classes_", None),
        )
        roc_auc = roc_data.get("macro_auc")
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            for c in roc_data["per_class"]:
                lbl = f"{c['label']} (AUC = {c['auc']:.3f})" if c["auc"] is not None else c["label"]
                ax.plot(c["fpr"], c["tpr"], lw=2, label=lbl)
            ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, linestyle="--", label="Chance")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            title = "ROC Curves (one-vs-rest)"
            if roc_auc is not None:
                title += f" — macro AUC = {roc_auc:.3f}"
            ax.set_title(title)
            ax.legend(loc="lower right", fontsize=8)
            roc_plot = _figure_to_data_url(fig)
        except Exception as exc:
            logger.warning("Multiclass ROC plot failed: %s", exc)
        try:
            fig, ax = plt.subplots(figsize=(5, 4))
            for c in pr_data["per_class"]:
                ap = c.get("average_precision")
                lbl = f"{c['label']} (AP = {ap:.3f})" if ap is not None else c["label"]
                ax.plot(c["recall"], c["precision"], lw=2, label=lbl)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curves (one-vs-rest)")
            ax.legend(loc="lower left", fontsize=8)
            pr_plot = _figure_to_data_url(fig)
        except Exception as exc:
            logger.warning("Multiclass PR plot failed: %s", exc)
        return {
            "optimization_id": request.optimization_id,
            "roc_curve_plot": roc_plot,
            "pr_curve_plot": pr_plot,
            "roc_auc": roc_auc,
            "average_precision": None,
            "task_type": "multiclass",
            "status": "completed",
        }

    # Per-class proba -> 1-D positive class, as sklearn curve helpers require.
    proba = _positive_proba(xai.predictions_proba)

    try:
        fpr, tpr, _ = roc_curve(y_test, proba)
        try:
            roc_auc = float(roc_auc_score(y_test, proba))
        except Exception:
            roc_auc = None
        fig, ax = plt.subplots(figsize=(5, 4))
        label = f"ROC (AUC = {roc_auc:.3f})" if roc_auc is not None else "ROC"
        ax.plot(fpr, tpr, color="#6366f1", lw=2, label=label)
        ax.plot([0, 1], [0, 1], color="#9ca3af", lw=1, linestyle="--", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        roc_plot = _figure_to_data_url(fig)
    except Exception as exc:
        logger.warning("ROC curve failed: %s", exc)

    try:
        precision, recall, _ = precision_recall_curve(y_test, proba)
        try:
            avg_prec = float(average_precision_score(y_test, proba))
        except Exception:
            avg_prec = None
        fig, ax = plt.subplots(figsize=(5, 4))
        label = f"PR (AP = {avg_prec:.3f})" if avg_prec is not None else "PR"
        ax.plot(recall, precision, color="#10b981", lw=2, label=label)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        pr_plot = _figure_to_data_url(fig)
    except Exception as exc:
        logger.warning("PR curve failed: %s", exc)

    return {
        "optimization_id": request.optimization_id,
        "roc_curve_plot": roc_plot,
        "pr_curve_plot": pr_plot,
        "roc_auc": roc_auc,
        "average_precision": avg_prec,
        "task_type": "binary",
        "status": "completed",
    }


@router.post("/curves/data")
async def generate_curves_data(request: MetricsRequest):
    """Raw ROC and PR curve points as JSON (for frontend charting).

    Uses the exact same model loading / test split / probability reduction as
    the PNG ``/curves`` endpoint. Each curve is independently fault-tolerant
    and yields ``null`` on failure (e.g. multiclass or missing probabilities).
    Curves are downsampled to at most 500 points.
    """
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build XAI: {e!s}")

    y_test = xai.y_test
    spec = _task_spec(opt_result)
    multiclass_spec = spec if spec is not None and spec.get("kind") == "multiclass" else None

    roc = None
    pr = None
    if multiclass_spec is not None:
        try:
            roc, pr = _per_class_curve_payloads(
                y_test,
                xai.predictions_proba,
                multiclass_spec,
                model_classes=getattr(xai.model, "classes_", None),
            )
        except Exception as exc:
            logger.warning("Multiclass curve data failed: %s", exc)
    else:
        proba = _positive_proba(xai.predictions_proba)
        try:
            roc = _roc_payload(y_test, proba)
        except Exception as exc:
            logger.warning("ROC curve data failed: %s", exc)
        try:
            pr = _pr_payload(y_test, proba)
        except Exception as exc:
            logger.warning("PR curve data failed: %s", exc)

    return {
        "optimization_id": request.optimization_id,
        "roc": roc,
        "pr": pr,
        "task_type": "multiclass" if multiclass_spec is not None else "binary",
        "status": "completed",
    }


@router.post("/confusion-matrix/data")
async def generate_confusion_matrix_data(request: MetricsRequest):
    """Raw confusion matrix (counts + row-normalized) as JSON.

    Same computation path as the PNG plot in ``/metrics``
    (``xai.get_confusion_matrix()`` on the identical test split).
    """
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )
        cm = xai.get_confusion_matrix()
        try:
            labels = list(xai.get_classes())
        except Exception:
            labels = list(range(len(cm)))
        # Map encoded label values back to the original class names when known.
        labels = _spec_display_labels(_task_spec(opt_result), labels)
        return {
            "optimization_id": request.optimization_id,
            **_confusion_matrix_payload(cm, labels),
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute confusion matrix: {e!s}")


@router.post("/feature-importance/data")
async def generate_feature_importance_data(request: MetricsRequest):
    """SHAP mean-|value| feature importances as parallel arrays for charting.

    Reuses the same importance computation as ``/shap`` (mean absolute SHAP
    value per feature, per-class axis averaged), sorted descending.
    """
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
            background_size=request.background_size,
            max_evals=request.max_evals,
        )
        importance = _feature_importance_from_xai(xai)
        return {
            "optimization_id": request.optimization_id,
            "features": [item["feature"] for item in importance],
            "importances": [item["importance"] for item in importance],
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute feature importance: {e!s}")


@router.post("/study-plots")
async def generate_study_plots(request: StudyPlotsRequest):
    """Interactive Optuna study plots as Plotly figure JSON.

    Each plot is independently fault-tolerant (e.g. param importances needs
    >= 2 trials and >= 2 distinct params) and yields ``null`` on failure.
    """
    opt_result = _get_completed_result(request.optimization_id)

    import json

    import optuna.visualization as ov
    from optuna import load_study

    db_name = str(opt_result.get("db_name") or DEFAULT_DB_NAME)
    study_name = opt_result.get("study_name")
    try:
        # Storage string mirrors workflow_service.build_xai exactly.
        study = load_study(storage=optuna_storage_url(db_name), study_name=study_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load study: {e!s}")

    plot_funcs = {
        "optimization_history": ov.plot_optimization_history,
        "param_importances": ov.plot_param_importances,
        "parallel_coordinate": ov.plot_parallel_coordinate,
        "slice": ov.plot_slice,
        "timeline": ov.plot_timeline,
    }

    # Multi-objective studies need an explicit target (F1); single-objective
    # plots reject the kwarg-less form otherwise.
    plot_kwargs: dict[str, Any] = {}
    if len(study.directions) > 1:
        plot_kwargs = {"target": lambda t: t.values[0], "target_name": "F1"}

    plots: dict[str, Any] = {}
    for name, func in plot_funcs.items():
        try:
            kwargs = plot_kwargs if name != "timeline" else {}
            fig = cast("Any", func)(study, **kwargs)
            plots[name] = json.loads(fig.to_json())
        except Exception as exc:
            logger.warning("%s plot failed: %s", name, exc)
            plots[name] = None

    return {
        "optimization_id": request.optimization_id,
        "plots": plots,
        "status": "completed",
    }


MAX_SENSITIVE_GROUPS = 20


def _resolve_sensitive_series(
    optimization_id: str,
    sensitive_feature: Optional[str],
    xai,
    persisted_sensitive_train=None,
):
    """Load the raw dataset column and align it to the analysed splits.

    ``DataPreparation.preprocess`` resets the feature index to a RangeIndex
    before its seeded ``train_test_split``, so split indices are positional
    row numbers into the raw dataframe (post feature-selection, which only
    selects columns). The TEST split preserves that property — nothing
    downstream reorders or resamples it — so it always resolves positionally.

    The train side does not: ``xai.data["x_train"]`` is the inner training
    frame (validation carved out, then resampled), so neither its length nor
    its index maps onto the raw file. It is therefore taken from
    ``persisted_sensitive_train`` — the series carried through both steps in
    lockstep by ``WorkflowExecutor._execute_train_test_split``. It is only
    needed for mitigation; a missing one degrades that single feature rather
    than failing the audit.
    """
    from quoptuna.server.services.sensitive import (
        SensitiveColumnError,
        resolve_sensitive_test_series,
    )

    job = get_job(optimization_id)
    request = OptimizationRequest(**job["request"])
    column = sensitive_feature or request.sensitive_feature
    if not column:
        raise HTTPException(
            status_code=400,
            detail="No sensitive_feature provided or stored with this optimization",
        )

    try:
        sens_test = resolve_sensitive_test_series(request.dataset_id, column, xai.x_test)
    except SensitiveColumnError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return column, persisted_sensitive_train, sens_test


def _compute_fairness_payload(
    optimization_id: str,
    sensitive_feature: Optional[str],
    xai,
    *,
    mitigate: bool = False,
    constraint: str = "equalized_odds",
    task_spec: Optional[dict] = None,
    persisted_sensitive_train=None,
) -> dict:
    from quoptuna.backend.xai import fairness as fairness_mod

    column, sens_train, sens_test = _resolve_sensitive_series(
        optimization_id, sensitive_feature, xai, persisted_sensitive_train
    )

    # Multiclass tasks are audited on the favorable-class-vs-rest outcome.
    multiclass_spec = (
        task_spec if task_spec is not None and task_spec.get("kind") == "multiclass" else None
    )
    favorable = 1
    favorable_class = None
    if multiclass_spec is not None:
        if multiclass_spec.get("favorable_code") is None:
            raise HTTPException(
                status_code=400,
                detail="Fairness on a multiclass target requires a favorable_class "
                "(selected at optimization setup)",
            )
        favorable = int(multiclass_spec["favorable_code"])
        favorable_class = multiclass_spec.get("favorable_class")

    metrics = fairness_mod.compute_fairness(
        xai.y_test, xai.predictions, sens_test, favorable=favorable
    )
    plots = fairness_mod.plot_group_metrics(metrics)

    mitigation = None
    if mitigate and multiclass_spec is not None:
        # ThresholdOptimizer adjusts a favorable-vs-rest decision threshold,
        # which cannot be soundly mapped back onto an argmax over K classes.
        # The audit above remains valid; mitigation is binary-only for now.
        logger.info("Fairness mitigation skipped: unsupported for multiclass targets")
    elif mitigate and sens_train is None:
        # Mitigation refits on the training split, which needs its sensitive
        # values row-aligned; runs configured without a sensitive_feature
        # never recorded them. The audit above still stands.
        logger.info("Fairness mitigation skipped: no sensitive values recorded for the train split")
    elif mitigate:
        mitigation = fairness_mod.mitigate_with_threshold_optimizer(
            xai.model,
            xai.data.get("x_train"),
            xai.data.get("y_train"),
            sens_train,
            xai.x_test,
            xai.y_test,
            sens_test,
            constraint=constraint,
        )

    return {
        "sensitive_feature": column,
        "metrics": metrics,
        "plots": plots,
        "mitigation": mitigation,
        "task_type": "multiclass" if multiclass_spec is not None else "binary",
        "favorable_class": favorable_class,
    }


@router.post("/fairness")
async def generate_fairness(request: FairnessRequest):
    """Fairness audit (fairlearn) for a chosen trial, grouped by a protected attribute."""
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = _analysis_xai(opt_result, trial_number=request.trial_number)
        payload = _compute_fairness_payload(
            request.optimization_id,
            request.sensitive_feature,
            xai,
            mitigate=request.mitigate,
            constraint=request.constraint,
            task_spec=_task_spec(opt_result),
            persisted_sensitive_train=opt_result.get("sensitive_train"),
        )
        return {
            "optimization_id": request.optimization_id,
            "status": "completed",
            **payload,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute fairness: {e!s}")


def _run_view(optimization_id: str) -> Optional[dict]:
    """The optimization job as a plain dict, or ``None`` when it is unknown.

    The report path must degrade rather than fail: a snapshot can outlive the
    run record it came from (a wiped hot cache plus a pruned store), and a
    report grounded in the snapshot alone is still worth producing.
    """
    try:
        job = get_job(optimization_id)
    except HTTPException:
        return None
    result = job.get("result") or {}
    return {**job, "best_trial_number": result.get("best_trial_number")}


def _run_trials(run: Optional[dict]) -> list[dict]:
    """Trial history for a run, read back from the Optuna study when needed."""
    if not run:
        return []
    trials = run.get("trials")
    if trials:
        return list(trials)
    request_data = run.get("request") or {}
    study_name = request_data.get("study_name")
    if not study_name:
        return []
    try:
        return serialize_study_trials(
            str(request_data.get("database_name") or DEFAULT_DB_NAME), study_name
        )
    except Exception as exc:
        logger.warning("Could not serialize trials for the report context: %s", exc)
        return []


def _run_pareto(run: Optional[dict]) -> list[dict]:
    """Pareto front of a multi-objective run, recomputed if not cached.

    ``pareto_trials`` is only set on the job by the background task that ran the
    search, so it is absent for any run resumed after a restart even though the
    study on disk still holds the front.
    """
    if not run:
        return []
    cached = run.get("pareto_trials")
    if cached:
        return list(cached)
    request_data = run.get("request") or {}
    study_name = request_data.get("study_name")
    if not study_name:
        return []
    try:
        from optuna import load_study

        study = load_study(
            storage=optuna_storage_url(str(request_data.get("database_name") or DEFAULT_DB_NAME)),
            study_name=study_name,
            sampler=None,
        )
        if len(study.directions) <= 1:
            return []
        return [
            {"trial": trial.number, "values": list(trial.values), "params": trial.params}
            for trial in study.best_trials
        ]
    except Exception as exc:
        logger.warning("Could not load the Pareto front for the report context: %s", exc)
        return []


def _report_context(
    snapshot: dict,
    *,
    options: ReportInclusionOptions,
    dataset_description: Optional[str] = None,
    prompt_names: Optional[dict[str, str]] = None,
) -> tuple[dict, Optional[dict], list[dict], list[dict]]:
    """Build the evidence bundle for a snapshot, plus the raw run data it used."""
    optimization_id = snapshot["optimization_id"]
    run = _run_view(optimization_id)
    trials = _run_trials(run)
    pareto = _run_pareto(run)
    dataset_id = ((run or {}).get("request") or {}).get("dataset_id")
    dataset = dataset_registry.get(dataset_id) if dataset_id else None
    context = report_context.build_context(
        optimization_id=optimization_id,
        snapshot=snapshot,
        run=run,
        dataset=dict(dataset) if dataset else None,
        trials=trials,
        pareto_trials=pareto,
        dataset_description=dataset_description,
        inclusions=options.inclusions(),
        prompts=prompt_names or {},
    )
    return context, run, trials, pareto


def _completed_snapshot(snapshot_id: str, optimization_id: Optional[str] = None) -> dict:
    snapshot = analysis_store.get_snapshot(snapshot_id)
    if not snapshot or snapshot.get("revision", 0) < 1 or not snapshot.get("payload"):
        raise HTTPException(status_code=404, detail="Completed analysis snapshot not found")
    if optimization_id and snapshot["optimization_id"] != optimization_id:
        raise HTTPException(status_code=409, detail="A completed analysis snapshot is required")
    return snapshot


@router.get("/report-prompts")
async def get_report_prompts():
    """Default agent prompts and the documented settings the UI exposes."""
    return {
        "prompts": report_prompts.default_prompts(),
        "settings": report_prompts.REPORT_PROMPT_SETTINGS,
        "markdown_contract": report_prompts.MARKDOWN_CONTRACT,
        "report_skeleton": report_prompts.REPORT_SKELETON,
    }


class BulkResearchBundleRequest(BaseModel):
    optimization_ids: List[str] = Field(min_length=1, max_length=100)


def _build_research_bundle(snapshot: dict) -> tuple[bytes, str]:
    """Build one complete research dump from a completed analysis snapshot."""
    context, run, trials, pareto = _report_context(snapshot, options=ReportInclusionOptions())
    reports = [
        report
        for report in analysis_store.list_reports(snapshot["id"])
        if report.get("snapshot_revision") == snapshot["revision"]
    ]
    try:
        archive = research_bundle.build_zip(
            context=context,
            payload=snapshot.get("payload") or {},
            reports=reports,
            evidence_markdown=report_context.render_markdown(context),
            prompts={
                f"{name}.default": text for name, text in report_prompts.default_prompts().items()
            },
            run=run,
            trials=trials,
            pareto_trials=pareto,
        )
    except Exception as exc:
        logger.exception("Failed to build the research bundle for snapshot %s", snapshot["id"])
        raise HTTPException(status_code=500, detail=f"Failed to build the bundle: {exc!s}")
    return archive, research_bundle.bundle_filename(context)


def _build_run_data_bundle(optimization_id: str) -> tuple[bytes, str]:
    """Build a durable metadata/trial archive for a run without analysis."""
    run = get_job(optimization_id)
    trials = _run_trials(run)
    pareto_trials = _run_pareto(run)
    return (
        research_bundle.build_run_data_zip(
            run=run, trials=trials, pareto_trials=pareto_trials
        ),
        research_bundle.run_data_filename(run),
    )


@router.post("/bundles/bulk")
async def download_bulk_research_bundles(request: BulkResearchBundleRequest):
    """Download each selected run's latest completed research dump in one ZIP."""
    requested_ids = (run_id.strip() for run_id in request.optimization_ids)
    optimization_ids = tuple(dict.fromkeys(run_id for run_id in requested_ids if run_id))
    if not optimization_ids:
        raise HTTPException(status_code=422, detail="Select at least one optimization run")

    bundles: list[tuple[str, bytes]] = []
    for optimization_id in optimization_ids:
        snapshots = analysis_store.list_snapshots(optimization_id)
        if snapshots:
            snapshot = _completed_snapshot(snapshots[0]["id"], optimization_id)
            bundle, filename = _build_research_bundle(snapshot)
        else:
            bundle, filename = _build_run_data_bundle(optimization_id)
        bundles.append((filename, bundle))

    output = io.BytesIO()
    used_filenames: set[str] = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, bundle in bundles:
            stem = filename.removesuffix(".zip")
            candidate = filename
            suffix = 2
            while candidate in used_filenames:
                candidate = f"{stem}-{suffix}.zip"
                suffix += 1
            used_filenames.add(candidate)
            archive.writestr(f"runs/{candidate}", bundle)
        lines = [
            "# QuOptuna bulk run export",
            "",
            "Each selected run has one ZIP in `runs/`.",
            "Completed analysis snapshots contain full research dumps; other runs contain metadata and available trial history.",
            f"Included: {len(bundles)} run(s).",
        ]
        archive.writestr("README.md", "\n".join(lines) + "\n")
    filename = f"quoptuna-research-dumps-{datetime.now():%Y%m%d-%H%M%S}.zip"
    return Response(
        content=output.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/snapshots/{snapshot_id}/context")
async def get_report_context(snapshot_id: str, include_evidence_markdown: bool = True):
    """The structured evidence bundle the report agents are given.

    Served without any base64: figures are described by the manifest and
    downloaded through the bundle endpoint.
    """
    snapshot = _completed_snapshot(snapshot_id)
    context, _, _, _ = _report_context(snapshot, options=ReportInclusionOptions())
    body: dict[str, Any] = {"snapshot_id": snapshot_id, "context": context}
    if include_evidence_markdown:
        body["evidence_markdown"] = report_context.render_markdown(context)
    return body


@router.get("/snapshots/{snapshot_id}/bundle")
async def download_research_bundle(snapshot_id: str):
    """One-click research dump: context, evidence, figures, tables and reports.

    Built with the *default* inclusions rather than the caller's report settings:
    the dump is the archive of what the run produced, so it must stay complete
    even when a report deliberately left some of it out.
    """
    archive, filename = _build_research_bundle(_completed_snapshot(snapshot_id))
    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/report")
async def generate_ai_report(request: ReportRequest):
    """Generate and persist a report strictly from a completed snapshot."""
    if not request.api_key:
        raise HTTPException(status_code=400, detail="An LLM api_key is required")

    snapshot = _completed_snapshot(request.analysis_snapshot_id, request.optimization_id)
    if snapshot["revision"] < request.analysis_revision:
        # The client claims a revision the server never produced.
        raise HTTPException(
            status_code=409,
            detail="The analysis snapshot changed; reload it before generating the report",
        )
    if snapshot["revision"] != request.analysis_revision:
        # A newer analysis completed after the client loaded the page. Report
        # against the latest rather than a stale revision; the report records
        # the revision it used so the output stays attributable.
        logger.info(
            "Report for %s requested revision %s; using latest revision %s",
            request.analysis_snapshot_id,
            request.analysis_revision,
            snapshot["revision"],
        )
    payload = snapshot.get("payload") or {}
    if not payload.get("metrics"):
        raise HTTPException(status_code=409, detail="The analysis snapshot is incomplete")

    report_id = analysis_store.create_report(
        snapshot, request.llm_provider, request.model_name, request.dataset_description
    )

    try:
        from quoptuna.backend.xai import report_agent

        context, _, _, _ = _report_context(
            snapshot,
            options=request,
            dataset_description=request.dataset_description,
            prompt_names={
                "analyst": "custom" if (request.analyst_instructions or "").strip() else "default",
                "reviewer": "custom"
                if (request.reviewer_instructions or "").strip()
                else "default",
            },
        )
        images = (
            research_bundle.image_map(payload, context.get("figures") or [])
            if request.attach_figures
            else {}
        )

        result = await report_agent.generate_report(
            context=context,
            images=images,
            api_key=request.api_key,
            model_name=request.model_name,
            provider=request.llm_provider,
            analyst_instructions=request.analyst_instructions,
            reviewer_instructions=request.reviewer_instructions,
            enable_review=request.enable_review,
        )
        markdown = result["markdown"]
        analysis_store.complete_report(report_id, markdown)

        return {
            "optimization_id": request.optimization_id,
            "report_id": report_id,
            "status": "completed",
            "report_markdown": markdown,
            # The revision the report was actually grounded in, which is not
            # always the one requested: a newer analysis completing mid-session
            # is used instead (see above). The UI labels the report with this.
            "analysis_revision": snapshot["revision"],
            # Diagnostics the UI surfaces: which figures the report actually
            # references, what was dropped as invented, and residual markdown
            # issues the normalizer could not repair.
            "referenced_figures": result.get("referenced_figures") or [],
            "dropped_figures": result.get("dropped_figures") or [],
            "markdown_issues": result.get("lint") or [],
            "reviewed": result.get("reviewed", False),
            "context_summary": {
                "figures": len(context.get("figures") or []),
                "trials_included": len((context.get("optimization") or {}).get("trials") or []),
                "trials_recorded": (context.get("optimization") or {}).get("n_trials_recorded"),
                "fairness_audit_included": (context.get("fairness") or {}).get("audit_included"),
                "fairness_mode": ((context.get("fairness") or {}).get("search") or {}).get("mode"),
                "pareto_points": (context.get("pareto_front") or {}).get("n_points") or 0,
                "omissions": len(context.get("omissions") or []),
            },
        }
    except HTTPException as exc:
        analysis_store.fail_report(report_id, str(exc.detail))
        raise
    except Exception as e:
        analysis_store.fail_report(report_id, str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e!s}")
