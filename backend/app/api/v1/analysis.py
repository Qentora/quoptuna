"""
Analysis endpoints (SHAP, metrics, AI reports).
"""

import base64
import io
import logging
from typing import Any, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Access optimization results stored by the optimize module.
from app.api.v1.optimize import optimization_jobs
from app.services.workflow_service import build_xai

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
    optimization_id: str
    trial_number: Optional[int] = None
    llm_provider: str = "google"
    api_key: str
    model_name: str = "gpt-4o"
    dataset_description: Optional[str] = None


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


def _feature_importance_from_xai(xai) -> list[dict[str, Any]]:
    shap_values = xai.shap_values
    importance: list[dict[str, Any]] = []
    if hasattr(shap_values, "values"):
        mean_abs = np.abs(shap_values.values).mean(axis=0)
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

        plots: dict[str, str] = {}
        for plot_type in request.plot_types:
            try:
                if plot_type in NON_CLASS_PLOTS:
                    plots[plot_type] = xai.get_plot(plot_type)
                elif plot_type == "waterfall":
                    plots[plot_type] = xai.get_waterfall_plot(index=request.sample_index)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to generate %s plot: %s", plot_type, exc)

        return {
            "optimization_id": request.optimization_id,
            "feature_importance": _feature_importance_from_xai(xai),
            "plots": plots,
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate SHAP analysis: {str(e)}")


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

        fig = xai.plot_confusion_matrix()
        confusion_plot = _figure_to_data_url(fig)

        metrics: dict[str, Any] = {}

        def _safe(name: str, func) -> None:
            try:
                value = func()
                metrics[name] = float(value) if np.ndim(value) == 0 else value
            except Exception as exc:  # noqa: BLE001
                logger.warning("Metric %s failed: %s", name, exc)

        _safe("f1_score", xai.get_f1_score)
        _safe("precision", xai.get_precision)
        _safe("recall", xai.get_recall)
        _safe("roc_auc_score", xai.get_roc_auc_score)
        _safe("average_precision_score", xai.get_average_precision_score)
        _safe("mcc", xai.get_mcc)
        _safe("cohens_kappa", xai.get_cohens_kappa)
        _safe("log_loss", xai.get_log_loss)

        try:
            from sklearn.metrics import accuracy_score

            metrics["accuracy"] = float(accuracy_score(xai.y_test, xai.predictions))
        except Exception as exc:  # noqa: BLE001
            logger.warning("accuracy failed: %s", exc)

        try:
            metrics["classification_report"] = xai.get_classification_report()
        except Exception as exc:  # noqa: BLE001
            logger.warning("classification_report failed: %s", exc)

        try:
            metrics["confusion_matrix"] = xai.get_confusion_matrix().tolist()
        except Exception as exc:  # noqa: BLE001
            logger.warning("confusion_matrix failed: %s", exc)

        return {
            "optimization_id": request.optimization_id,
            "confusion_matrix_plot": confusion_plot,
            "metrics": metrics,
            "status": "completed",
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to compute metrics: {str(e)}")


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
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")
