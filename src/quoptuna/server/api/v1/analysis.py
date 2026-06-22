"""
Analysis endpoints (SHAP, metrics, AI reports).
"""

import base64
import io
import logging
from typing import Any, List, Optional

import matplotlib as mpl

# Use a non-interactive backend; these endpoints render figures in FastAPI's
# threadpool where no GUI backend is available.
mpl.use("Agg")

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict

# Access optimization results stored by the optimize module.
from quoptuna.server.api.v1.optimize import optimization_jobs
from quoptuna.server.services.workflow_service import build_xai

logger = logging.getLogger(__name__)

router = APIRouter()

NON_CLASS_PLOTS = {"bar", "beeswarm", "violin", "heatmap"}


class SHAPRequest(BaseModel):
    optimization_id: str
    plot_types: List[str] = ["bar", "beeswarm", "violin", "heatmap", "waterfall"]
    trial_number: Optional[int] = None
    sample_index: int = 0
    use_proba: bool = True
    subset_size: int = 50


class MetricsRequest(BaseModel):
    optimization_id: str
    trial_number: Optional[int] = None
    use_proba: bool = True
    subset_size: int = 50


class ReportRequest(BaseModel):
    # "model_name" collides with Pydantic's protected "model_" namespace; opt out.
    model_config = ConfigDict(protected_namespaces=())

    optimization_id: str
    trial_number: Optional[int] = None
    llm_provider: str = "google"
    api_key: str
    model_name: str = "gpt-4o"
    dataset_description: Optional[str] = None


class StudyPlotsRequest(BaseModel):
    optimization_id: str


def _get_completed_result(optimization_id: str) -> dict:
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")
    job = optimization_jobs[optimization_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Current status: {job['status']}",
        )
    result = job.get("result")
    if not result:
        raise HTTPException(status_code=500, detail="Optimization result not found")
    return result


def _figure_to_data_url(fig) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _plot_class_index(xai) -> int:
    """Class slice to use for SHAP plots when values are per-class (ndim > 2).

    Many quantum/classical models emit per-class SHAP values shaped
    ``(samples, features, classes)``; SHAP's plots need a single 2-D slice.
    Pick the positive (last) class for binary, otherwise the first. Returns -1
    when values are already 2-D (no slicing needed).
    """
    try:
        shap_values = xai.shap_values
        if getattr(shap_values, "values", None) is None or shap_values.values.ndim <= 2:
            return -1
        classes = list(xai.get_classes())
        return int(classes[-1] if len(classes) <= 2 else classes[0])
    except Exception:
        return -1


def _positive_proba(proba):
    """Reduce a probability array/frame to the 1-D positive-class column."""
    arr = np.asarray(proba)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        return arr[:, 1]
    return np.ravel(arr)


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


@router.post("/shap")
async def generate_shap_analysis(request: SHAPRequest):
    """Generate SHAP plots and real feature importance for a chosen trial."""
    opt_result = _get_completed_result(request.optimization_id)

    try:
        xai = build_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
        )

        # Per-class SHAP values (ndim > 2) must be sliced to one class for plotting.
        class_index = _plot_class_index(xai)

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
        xai = build_xai(
            opt_result,
            trial_number=request.trial_number,
            use_proba=request.use_proba,
            subset_size=request.subset_size,
        )

        from sklearn.metrics import average_precision_score, roc_auc_score

        fig = xai.plot_confusion_matrix()
        confusion_plot = _figure_to_data_url(fig)

        metrics: dict[str, Any] = {}

        def _safe(name: str, func) -> None:
            try:
                value = func()
                metrics[name] = float(value) if np.ndim(value) == 0 else value
            except Exception as exc:
                logger.warning("Metric %s failed: %s", name, exc)

        _safe("f1_score", xai.get_f1_score)
        _safe("precision", xai.get_precision)
        _safe("recall", xai.get_recall)
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
        xai = build_xai(
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

    # Per-class proba -> 1-D positive class, as sklearn curve helpers require.
    proba = _positive_proba(xai.predictions_proba)
    y_test = xai.y_test

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
        "status": "completed",
    }


@router.post("/study-plots")
async def generate_study_plots(request: StudyPlotsRequest):
    """Optuna study-level plots: optimization history and parameter importances.

    Each plot is independently fault-tolerant (param importances needs >= 2
    trials and >= 2 distinct params) and yields ``null`` on failure.
    """
    opt_result = _get_completed_result(request.optimization_id)

    import optuna.visualization.matplotlib as ov
    from optuna import load_study

    db_name = opt_result.get("db_name")
    study_name = opt_result.get("study_name")
    try:
        # Storage string mirrors workflow_service.build_xai exactly.
        study = load_study(storage=f"sqlite:///db/{db_name}.db", study_name=study_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load study: {e!s}")

    history_plot = None
    importances_plot = None

    try:
        ax = ov.plot_optimization_history(study)
        history_plot = _figure_to_data_url(ax.figure)
    except Exception as exc:
        logger.warning("optimization_history plot failed: %s", exc)

    try:
        ax = ov.plot_param_importances(study)
        importances_plot = _figure_to_data_url(ax.figure)
    except Exception as exc:
        logger.warning("param_importances plot failed: %s", exc)

    return {
        "optimization_id": request.optimization_id,
        "optimization_history_plot": history_plot,
        "param_importances_plot": importances_plot,
        "status": "completed",
    }


@router.post("/report")
async def generate_ai_report(request: ReportRequest):
    """Generate a Markdown report using a multimodal LLM (real call)."""
    opt_result = _get_completed_result(request.optimization_id)

    if not request.api_key:
        raise HTTPException(status_code=400, detail="An LLM api_key is required")

    try:
        xai = build_xai(opt_result, trial_number=request.trial_number)

        markdown = xai.generate_report_with_langchain(
            api_key=request.api_key,
            model_name=request.model_name,
            provider=request.llm_provider,
        )

        if request.dataset_description:
            markdown = f"> Dataset: {request.dataset_description}\n\n{markdown}"

        return {
            "optimization_id": request.optimization_id,
            "status": "completed",
            "report_markdown": markdown,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {e!s}")
