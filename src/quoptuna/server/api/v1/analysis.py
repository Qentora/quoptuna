"""
Analysis endpoints (SHAP, metrics, AI reports).
"""

import base64
import io
import logging
from contextvars import ContextVar
from datetime import datetime
from typing import Any, List, Optional, cast

import matplotlib as mpl

# Use a non-interactive backend; these endpoints render figures in FastAPI's
# threadpool where no GUI backend is available.
mpl.use("Agg")

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict

from quoptuna.backend.utils.storage import DEFAULT_DB_NAME

# Access optimization results stored by the optimize module.
from quoptuna.server.api.v1.optimize import (
    OptimizationRequest,
    build_workflow,
    get_job,
)
from quoptuna.server.services import analysis_store
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
    # Which class's SHAP values to slice/plot when values are per-class
    # (multiclass). None keeps the default (positive class for binary,
    # class 0 for multiclass).
    class_index: Optional[int] = None


class MetricsRequest(BaseModel):
    optimization_id: str
    trial_number: Optional[int] = None
    use_proba: bool = True
    subset_size: int = 50


class ReportRequest(BaseModel):
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
    # Include a fairness audit in the report when a protected attribute is
    # available (stored with the run or given here).
    sensitive_feature: Optional[str] = None
    include_fairness: bool = True


class AnalysisJobRequest(BaseModel):
    optimization_id: str
    trial_number: Optional[int] = None
    use_proba: bool = True
    subset_size: int = 50
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


async def _run_analysis_job(job_id: str, request: AnalysisJobRequest) -> None:
    """Compute and persist one complete analysis bundle."""
    config = analysis_store.normalize_config(request.model_dump(exclude={"optimization_id"}))
    trial = config["trial_number"]
    metrics_request = MetricsRequest(
        optimization_id=request.optimization_id,
        trial_number=trial,
        use_proba=config["use_proba"],
        subset_size=config["subset_size"],
    )
    shap_request = SHAPRequest(
        optimization_id=request.optimization_id,
        trial_number=trial,
        sample_index=config["sample_index"],
        use_proba=config["use_proba"],
        subset_size=config["subset_size"],
        class_index=config["class_index"],
    )
    warnings: dict[str, str] = {}

    async def optional(section: str, call):
        analysis_store.update_job(job_id, current_section=section)
        try:
            return await call
        except Exception as exc:  # optional visual sections must not discard core results
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            warnings[section] = str(detail)
            logger.warning("Analysis section %s failed: %s", section, detail)
            return None

    token = None
    try:
        analysis_store.update_job(job_id, status="running", current_section="shap")
        opt_result = _get_completed_result(request.optimization_id)
        shared_xai = build_xai(
            opt_result,
            trial_number=trial,
            use_proba=config["use_proba"],
            subset_size=config["subset_size"],
        )
        token = _job_xai.set(shared_xai)
        # SHAP and metrics are the required core sections. Existing endpoint
        # functions remain the compatibility implementation for now; this job
        # owns orchestration and persistence.
        shap = await generate_shap_analysis(shap_request)
        metrics = await generate_metrics(metrics_request)
        curves = await optional("curves", generate_curves(metrics_request))
        curves_data = await optional("curves_data", generate_curves_data(metrics_request))
        confusion_data = await optional(
            "confusion_matrix_data", generate_confusion_matrix_data(metrics_request)
        )
        importance_data = await optional(
            "feature_importance_data", generate_feature_importance_data(metrics_request)
        )
        shap_data = await optional("shap_data", generate_shap_data(shap_request))
        study = await optional(
            "study_plots",
            generate_study_plots(StudyPlotsRequest(optimization_id=request.optimization_id)),
        )
        fairness = None
        job = get_job(request.optimization_id)
        sensitive = (job.get("request") or {}).get("sensitive_feature")
        if sensitive:
            fairness = await optional(
                "fairness",
                generate_fairness(
                    FairnessRequest(
                        optimization_id=request.optimization_id,
                        trial_number=trial,
                        sensitive_feature=sensitive,
                    )
                ),
            )

        plots = dict(shap.get("plots") or {})
        if curves and curves.get("roc_curve_plot"):
            plots["rocCurve"] = curves["roc_curve_plot"]
        if curves and curves.get("pr_curve_plot"):
            plots["prCurve"] = curves["pr_curve_plot"]
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
        }
        analysis_store.complete_job(job_id, payload)
    except Exception as exc:
        logger.exception("Analysis job %s failed", job_id)
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        analysis_store.update_job(
            job_id, status="failed", error=str(detail), completed_at=datetime.now().isoformat()
        )
    finally:
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


def _resolve_sensitive_series(optimization_id: str, sensitive_feature: Optional[str], xai):
    """Load the raw dataset column and align it to the train/test split.

    ``DataPreparation.preprocess`` resets the feature index to a RangeIndex
    before its seeded ``train_test_split``, so split indices are positional
    row numbers into the raw dataframe (post feature-selection, which only
    selects columns).
    """
    from quoptuna.server.services.sensitive import SensitiveColumnError, resolve_sensitive_series

    job = get_job(optimization_id)
    request = OptimizationRequest(**job["request"])
    column = sensitive_feature or request.sensitive_feature
    if not column:
        raise HTTPException(
            status_code=400,
            detail="No sensitive_feature provided or stored with this optimization",
        )

    try:
        sens_train, sens_test = resolve_sensitive_series(
            request.dataset_id, column, xai.data.get("x_train"), xai.x_test
        )
    except SensitiveColumnError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return column, sens_train, sens_test


def _compute_fairness_payload(
    optimization_id: str,
    sensitive_feature: Optional[str],
    xai,
    *,
    mitigate: bool = False,
    constraint: str = "equalized_odds",
    task_spec: Optional[dict] = None,
) -> dict:
    from quoptuna.backend.xai import fairness as fairness_mod

    column, sens_train, sens_test = _resolve_sensitive_series(
        optimization_id, sensitive_feature, xai
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


@router.post("/report")
async def generate_ai_report(request: ReportRequest):
    """Generate and persist a report strictly from a completed snapshot."""
    if not request.api_key:
        raise HTTPException(status_code=400, detail="An LLM api_key is required")

    snapshot = analysis_store.get_snapshot(request.analysis_snapshot_id)
    if not snapshot or snapshot["optimization_id"] != request.optimization_id:
        raise HTTPException(status_code=409, detail="A completed analysis snapshot is required")
    if snapshot["revision"] != request.analysis_revision:
        raise HTTPException(
            status_code=409,
            detail="The analysis snapshot changed; reload it before generating the report",
        )
    payload = snapshot.get("payload") or {}
    if not payload.get("metrics"):
        raise HTTPException(status_code=409, detail="The analysis snapshot is incomplete")

    report_id = analysis_store.create_report(
        snapshot, request.llm_provider, request.model_name, request.dataset_description
    )

    try:
        from quoptuna.backend.xai import report_agent

        evidence: dict[str, Any] = {
            "metrics": payload["metrics"],
            "feature_importance": payload.get("feature_importance"),
            "confusion_matrix": payload.get("confusion_data"),
            "curves": payload.get("curves_data"),
            "task_type": payload.get("task_type"),
            "class_labels": payload.get("class_labels"),
        }
        if request.include_fairness and payload.get("fairness"):
            fairness = payload["fairness"]
            evidence["fairness_metrics"] = fairness.get("metrics")
            evidence["fairness_mitigation"] = fairness.get("mitigation")

        images = dict(payload.get("plots") or {})
        if payload.get("confusion_matrix_plot"):
            images["confusion_matrix"] = payload["confusion_matrix_plot"]
        fairness = payload.get("fairness") or {}
        if request.include_fairness:
            images.update(
                {f"fairness_{name}": value for name, value in (fairness.get("plots") or {}).items()}
            )
            mitigation = fairness.get("mitigation") or {}
            if mitigation.get("comparison_plot"):
                images["fairness_mitigation_comparison"] = mitigation["comparison_plot"]

        markdown = await report_agent.generate_report(
            report=evidence,
            images=images,
            api_key=request.api_key,
            model_name=request.model_name,
            provider=request.llm_provider,
        )

        if request.dataset_description:
            markdown = f"> Dataset: {request.dataset_description}\n\n{markdown}"

        analysis_store.complete_report(report_id, markdown)

        return {
            "optimization_id": request.optimization_id,
            "report_id": report_id,
            "status": "completed",
            "report_markdown": markdown,
        }
    except HTTPException as exc:
        analysis_store.fail_report(report_id, str(exc.detail))
        raise
    except Exception as e:
        analysis_store.fail_report(report_id, str(e))
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e!s}")
