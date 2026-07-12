"""Headless (CLI / test) runner for the exact UI optimization pipeline.

Builds the same ``OptimizationRequest`` the frontend sends, runs it through
``build_workflow`` + ``WorkflowExecutor`` (identical node graph, encoding,
optimizer wiring), and optionally computes the same analysis payloads the
Analyze tab requests. Used by ``quoptuna optimize`` and the integration tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _resolve_dataset(
    csv_path: str | None,
    uci_id: str | None,
    upload_dir: str = "./uploads",
) -> tuple[str, str, pd.DataFrame, str | None]:
    """Load + register the dataset like the UI does.

    Returns (dataset_id, dataset_source, dataframe, default_target).
    """
    from quoptuna.server.services import dataset_registry

    if csv_path:
        df = pd.read_csv(csv_path)
        dataset_id = f"cli_{uuid.uuid4().hex[:8]}"
        dataset_registry.register(
            {
                "id": dataset_id,
                "name": Path(csv_path).stem,
                "source": "upload",
                "file_path": str(csv_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        )
        return dataset_id, "upload", df, None

    if uci_id:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=int(uci_id))
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        # Persist to a CSV and register — same shape as the UI's UCI load.
        out_dir = Path(upload_dir)
        out_dir.mkdir(exist_ok=True)
        file_path = out_dir / f"uci_{uci_id}.csv"
        df.to_csv(file_path, index=False)
        dataset_registry.register(
            {
                "id": str(uci_id),
                "name": getattr(dataset.metadata, "name", f"uci-{uci_id}"),
                "source": "uci",
                "file_path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        )
        default_target = list(dataset.data.targets.columns)[0]
        return str(uci_id), "uci", df, default_target

    msg = "Provide either csv_path or uci_id"
    raise ValueError(msg)


def run_headless_optimization(
    *,
    csv_path: str | None = None,
    uci_id: str | None = None,
    target: str | None = None,
    features: list[str] | None = None,
    n_trials: int = 3,
    model_types: list[str] | None = None,
    search_space: dict | None = None,
    label_neg: str | None = None,
    label_pos: str | None = None,
    favorable_class: str | None = None,
    sensitive_feature: str | None = None,
    categorical_encoding: str = "ordinal",
    sampler: str = "random",
    sampler_seed: int | None = 0,
    pruner: str = "none",
    fairness_mode: str = "off",
    fairness_metric: str = "equal_opportunity_difference",
    fairness_threshold: float | None = None,
    max_steps: int | None = None,
    convergence_interval: int | None = None,
    max_vmap: int | None = None,
    study_name: str | None = None,
    db_name: str = "cli_runs",
    analyze: bool = True,
    subset_size: int = 30,
) -> dict[str, Any]:
    """Run one optimization exactly as the UI would, synchronously.

    Returns a JSON-safe summary: best trial, per-trial values, task spec, and
    (with ``analyze=True``) the same metrics/curves/confusion payloads the
    Analyze tab fetches.
    """
    from quoptuna.server.api.v1.optimize import LabelMapping, OptimizationRequest, build_workflow
    from quoptuna.server.services.workflow_service import WorkflowExecutor, build_xai

    dataset_id, dataset_source, df, default_target = _resolve_dataset(csv_path, uci_id)
    target = target or default_target or df.columns[-1]
    if target not in df.columns:
        msg = f"Target column '{target}' not in dataset columns {list(df.columns)}"
        raise ValueError(msg)
    features = features or [c for c in df.columns if c != target]

    label_mapping = None
    if label_neg is not None and label_pos is not None:
        label_mapping = LabelMapping(neg=label_neg, pos=label_pos)

    request = OptimizationRequest(
        dataset_id=dataset_id,
        dataset_source=dataset_source,
        selected_features=features,
        target_column=target,
        study_name=study_name or f"cli_{uuid.uuid4().hex[:8]}",
        database_name=db_name,
        num_trials=n_trials,
        label_mapping=label_mapping,
        favorable_class=favorable_class,
        sensitive_feature=sensitive_feature,
        categorical_encoding=categorical_encoding,  # type: ignore[arg-type]
        model_types=model_types,
        search_space=search_space,
        sampler=sampler,  # type: ignore[arg-type]
        sampler_seed=sampler_seed,
        pruner=pruner,  # type: ignore[arg-type]
        fairness_mode=fairness_mode,  # type: ignore[arg-type]
        fairness_metric=fairness_metric,  # type: ignore[arg-type]
        fairness_threshold=fairness_threshold,
        max_steps=max_steps,
        convergence_interval=convergence_interval,
        max_vmap=max_vmap,
    )

    workflow = build_workflow(f"cli_{uuid.uuid4().hex[:8]}", request)
    result = WorkflowExecutor(workflow).execute()
    opt = result["node_results"]["optimize"]

    summary: dict[str, Any] = {
        "study_name": request.study_name,
        "db_name": request.database_name,
        "best_value": opt["best_value"],
        "best_params": opt["best_params"],
        "best_trial_number": opt["best_trial_number"],
        "task_spec": opt.get("task_spec"),
        "pareto_trials": opt.get("pareto_trials"),
    }

    if analyze:
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        from quoptuna.server.api.v1 import analysis as analysis_mod

        try:
            xai = build_xai(opt, subset_size=subset_size)
        except TypeError:
            # Models without predict_proba (e.g. SVC without probability=True):
            # same fallback the UI offers via its "use probabilities" toggle.
            xai = build_xai(opt, use_proba=False, subset_size=subset_size)
        spec = opt.get("task_spec")
        multiclass = bool(spec and spec.get("kind") == "multiclass")
        average = "macro" if multiclass else "binary"
        pred = xai.predictions
        metrics: dict[str, Any] = {
            "f1_score": float(f1_score(xai.y_test, pred, average=average)),
            "accuracy": float(accuracy_score(xai.y_test, pred)),
        }
        try:
            proba = np.asarray(xai.predictions_proba)
            metrics["roc_auc_score"] = float(
                roc_auc_score(xai.y_test, proba, multi_class="ovr", average="macro")
                if multiclass
                else roc_auc_score(xai.y_test, proba[:, 1])
            )
        except Exception:
            metrics["roc_auc_score"] = None

        cm = xai.get_confusion_matrix()
        encoded = sorted(np.unique(np.asarray(xai.y_test)).tolist())
        labels = analysis_mod._spec_display_labels(spec, encoded)  # noqa: SLF001
        curves = None
        if multiclass:
            try:
                roc, pr = analysis_mod._per_class_curve_payloads(  # noqa: SLF001
                    xai.y_test,
                    xai.predictions_proba,
                    spec,
                    model_classes=getattr(xai.model, "classes_", None),
                )
                curves = {
                    "per_class_auc": {c["label"]: c["auc"] for c in roc["per_class"]},
                    "macro_auc": roc["macro_auc"],
                    "n_pr_curves": len(pr["per_class"]),
                }
            except Exception:
                curves = None

        summary["analysis"] = {
            "metrics": metrics,
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
            "curves": curves,
        }

        if sensitive_feature:
            try:
                from quoptuna.backend.xai.fairness import compute_fairness
                from quoptuna.server.services.sensitive import resolve_sensitive_series

                _, sens_test = resolve_sensitive_series(
                    dataset_id, sensitive_feature, xai.data.get("x_train"), xai.x_test
                )
                # Mirror the API guard (_compute_fairness_payload): a multiclass
                # audit is meaningless without a designated favorable class —
                # falling back to code 1 would silently audit an arbitrary class.
                if multiclass and (not spec or spec.get("favorable_code") is None):
                    summary["analysis"]["fairness_disparities"] = (
                        "skipped: multiclass fairness requires --favorable-class"
                    )
                else:
                    favorable = (
                        int(spec["favorable_code"])
                        if multiclass and spec and spec.get("favorable_code") is not None
                        else 1
                    )
                    fair = compute_fairness(xai.y_test, pred, sens_test, favorable=favorable)
                    summary["analysis"]["fairness_disparities"] = fair["disparities"]
                    if spec:
                        # Disclose which class was audited as the favorable outcome.
                        labels_list = [str(c) for c in spec.get("class_labels", [])]
                        if multiclass and 0 <= favorable < len(labels_list):
                            summary["analysis"]["fairness_favorable_class"] = labels_list[favorable]
                        elif not multiclass and len(labels_list) == 2:
                            summary["analysis"]["fairness_favorable_class"] = labels_list[1]
            except Exception as exc:
                summary["analysis"]["fairness_disparities"] = f"failed: {exc}"

    return summary
