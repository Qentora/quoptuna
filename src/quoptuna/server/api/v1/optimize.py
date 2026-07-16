"""
Optimization endpoints
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from optuna.trial import TrialState
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quoptuna.server.services import dataset_registry, run_store
from quoptuna.server.services.storage import optuna_storage_url
from quoptuna.server.services.workflow_service import WorkflowExecutor

logger = logging.getLogger(__name__)

router = APIRouter()

# Hot cache of optimization jobs; the durable copy lives in run_store (SQLite).
optimization_jobs: Dict[str, Dict[str, Any]] = {}

# Runs left 'running'/'pending' by a previous process can never finish.
run_store.mark_stale_runs_interrupted()


class LabelMapping(BaseModel):
    neg: Any
    pos: Any


class OptimizationRequest(BaseModel):
    # Several fields start with "model_", which Pydantic reserves; opt out so it
    # does not emit protected-namespace warnings.
    model_config = ConfigDict(protected_namespaces=())

    dataset_id: str
    dataset_source: str  # 'uci' or 'upload'
    selected_features: List[str]
    target_column: str
    study_name: str
    # Optuna storage database; a global app setting on the frontend side.
    database_name: str = "results"
    num_trials: int
    model_name: str = "DataReuploading"
    label_mapping: Optional[LabelMapping] = None
    # Multiclass targets: the class treated as the favorable outcome for
    # fairness auditing and report framing (favorable vs rest). Required when
    # fairness is used on a K>2 target; ignored for binary targets.
    favorable_class: Optional[Any] = None
    # Protected attribute column (may be outside selected_features) used for
    # fairness auditing; rows are aligned to the split via positional indices.
    sensitive_feature: Optional[str] = None
    # How categorical feature columns are encoded: "ordinal" (1 column per
    # feature — fast, quantum-friendly) or "onehot" (1 column per category).
    categorical_encoding: Literal["ordinal", "onehot"] = "ordinal"
    # Optional overrides to shrink Optuna's search (defaults use the full space).
    model_types: Optional[List[str]] = None
    search_space: Optional[Dict[str, List[Any]]] = None
    # Search strategy: sampler ("tpe"/"random"/"grid") and pruner
    # ("asha"/"hyperband"/"none") for early-stopping unpromising trials.
    sampler: Literal["tpe", "random", "grid"] = "tpe"
    sampler_seed: Optional[int] = None
    pruner: Literal["none", "asha", "hyperband"] = "asha"
    pruner_min_resource: int = 1
    pruner_reduction_factor: int = 3
    # Intermediate value iterative models report for pruning decisions.
    intermediate_metric: Literal["accuracy", "neg_loss"] = "accuracy"
    # Optional cap on training steps for iterative models.
    max_steps: Optional[int] = Field(default=None, ge=1)
    # Optional override of the flat-loss convergence window (also the cadence
    # of pruning reports).
    convergence_interval: Optional[int] = Field(default=None, ge=1)
    # Optional override of circuit-evaluation vectorization width; must divide
    # the batch size (32 in the default search space).
    max_vmap: Optional[int] = Field(default=None, ge=1)
    # PennyLane simulator for quantum models; lightning.qubit (C++ state
    # vector) is usually faster. Falls back to default.qubit if unavailable.
    dev_type: Literal["default.qubit", "lightning.qubit"] = "default.qubit"
    # Fairness-aware search: "constrained" adds a TPE feasibility constraint on
    # the disparity; "multi_objective" searches the F1-vs-disparity Pareto front.
    fairness_mode: Literal["off", "constrained", "multi_objective"] = "off"
    fairness_metric: Literal[
        "equal_opportunity_difference", "disparate_impact", "demographic_parity_difference"
    ] = "equal_opportunity_difference"
    # Feasibility threshold: difference metrics are feasible when disparity <=
    # threshold (default 0.1); disparate_impact when the DI ratio >= threshold
    # (default 0.8, the four-fifths rule).
    fairness_threshold: Optional[float] = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_fairness(self):
        if self.fairness_mode == "off":
            return self
        if not self.sensitive_feature:
            msg = f"fairness_mode='{self.fairness_mode}' requires sensitive_feature"
            raise ValueError(msg)
        if self.fairness_mode == "constrained" and self.sampler != "tpe":
            msg = "fairness_mode='constrained' requires sampler='tpe' (constraints are TPE-only)"
            raise ValueError(msg)
        if self.fairness_mode == "multi_objective" and self.pruner != "none":
            # Coerce rather than reject: pruning is genuinely unsupported on
            # multi-objective studies, and rejecting would 422 requests that
            # simply left the pruner at its "asha" default.
            logger.warning(
                "pruner=%r is unsupported with fairness_mode='multi_objective'; using 'none'.",
                self.pruner,
            )
            self.pruner = "none"
        return self


class OptimizationStatus(BaseModel):
    id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    current_trial: int
    total_trials: int
    best_value: Optional[float]
    best_params: Optional[Dict[str, Any]]
    trials: Optional[List[Dict[str, Any]]]
    started_at: str
    completed_at: Optional[str]
    error: Optional[str]


def build_workflow(
    job_id: str, request: OptimizationRequest, include_optimize: bool = True
) -> Dict:
    """Build the node-graph workflow for a request.

    With ``include_optimize=False`` the graph stops after model/optuna config,
    which re-derives the data-prep outputs (train/test splits) without running
    any trials — used to rehydrate analysis after a backend restart.
    """
    # Resolve the dataset to a persisted CSV via the registry. Uploads and
    # UCI loads both register a file_path, so we can always read by file.
    record = dataset_registry.get(request.dataset_id)
    if record and record.get("file_path"):
        data_node = {
            "id": "data",
            "data": {
                "type": "data-upload",
                "config": {"file_path": record["file_path"]},
            },
        }
    elif request.dataset_source == "uci":
        # Fallback: fetch directly from UCI by numeric id.
        data_node = {
            "id": "data",
            "data": {"type": "data-uci", "config": {"dataset_id": request.dataset_id}},
        }
    else:
        raise ValueError(
            f"Dataset '{request.dataset_id}' is not registered. Upload or load it first."
        )

    label_config: Dict[str, Any] = {}
    if request.label_mapping is not None:
        label_config["label_mapping"] = {
            "neg": request.label_mapping.neg,
            "pos": request.label_mapping.pos,
        }
    if request.favorable_class is not None:
        label_config["favorable_class"] = request.favorable_class

    nodes = [
        data_node,
        {
            "id": "features",
            "data": {
                "type": "feature-selection",
                "config": {
                    "x_columns": request.selected_features,
                    "y_column": request.target_column,
                    "categorical_encoding": request.categorical_encoding,
                },
            },
        },
        {"id": "split", "data": {"type": "train-test-split", "config": label_config}},
        {"id": "label_encode", "data": {"type": "label-encoding", "config": label_config}},
        {
            "id": "model",
            "data": {"type": "quantum-model", "config": {"model_name": request.model_name}},
        },
        {
            "id": "optuna",
            "data": {
                "type": "optuna-config",
                "config": {
                    "study_name": request.study_name,
                    "n_trials": request.num_trials,
                    "db_name": request.database_name,
                    "model_types": request.model_types,
                    "search_space": request.search_space,
                    "sampler": request.sampler,
                    "sampler_seed": request.sampler_seed,
                    "pruner": request.pruner,
                    "pruner_min_resource": request.pruner_min_resource,
                    "pruner_reduction_factor": request.pruner_reduction_factor,
                    "intermediate_metric": request.intermediate_metric,
                    "max_steps": request.max_steps,
                    "convergence_interval": request.convergence_interval,
                    "max_vmap": request.max_vmap,
                    "dev_type": request.dev_type,
                    "fairness_mode": request.fairness_mode,
                    "fairness_metric": request.fairness_metric,
                    "fairness_threshold": request.fairness_threshold,
                    "sensitive_feature": request.sensitive_feature,
                    "dataset_id": request.dataset_id,
                },
            },
        },
    ]
    edges = [
        {"source": "data", "target": "features"},
        {"source": "features", "target": "split"},
        {"source": "split", "target": "label_encode"},
        {"source": "label_encode", "target": "model"},
        {"source": "model", "target": "optuna"},
    ]
    if include_optimize:
        nodes.append({"id": "optimize", "data": {"type": "optimization", "config": {}}})
        edges.append({"source": "optuna", "target": "optimize"})

    return {"id": job_id, "name": request.study_name, "nodes": nodes, "edges": edges}


def serialize_study_trials(db_name: str, study_name: str) -> list:
    """Serialize finished trials from the Optuna study; [] if the study can't be loaded."""
    try:
        from optuna import load_study

        study = load_study(storage=optuna_storage_url(db_name), study_name=study_name, sampler=None)
        return [
            {
                "trial": trial.number,
                # PRUNED trials store their last *intermediate* report as
                # value; expose None so it is never mistaken for a final F1.
                # Multi-objective trials have value=None; values[0] is F1.
                "value": (
                    trial.values[0] if trial.state == TrialState.COMPLETE and trial.values else None
                ),
                "values": list(trial.values)
                if trial.state == TrialState.COMPLETE and trial.values
                else None,
                "params": trial.params,
                "state": trial.state.name,
                "user_attrs": trial.user_attrs,
            }
            for trial in study.trials
            if trial.state.is_finished()
        ]
    except Exception:
        return []


def run_optimization_background(job_id: str, request: OptimizationRequest):
    """Background task to run optimization"""
    try:
        optimization_jobs[job_id]["status"] = "running"
        run_store.update_run(job_id, status="running")

        workflow = build_workflow(job_id, request)

        # Execute workflow
        executor = WorkflowExecutor(workflow)
        result = executor.execute()

        # Extract optimization results
        opt_result = result["node_results"]["optimize"]

        # Trial history: prefer what the workflow returned, otherwise read the
        # real per-trial history back from the Optuna study. Never synthesize
        # trials from best_params — that stamps every row with the winning
        # model and makes the other family's trials vanish on completion.
        trials = opt_result.get("trials") or serialize_study_trials(
            request.database_name, request.study_name
        )

        completed_at = datetime.now().isoformat()
        optimization_jobs[job_id].update(
            {
                "status": "completed",
                "current_trial": request.num_trials,
                "best_value": opt_result["best_value"],
                "best_params": opt_result["best_params"],
                "trials": trials,
                "pareto_trials": opt_result.get("pareto_trials"),
                "completed_at": completed_at,
                "result": opt_result,  # Store full result for SHAP
            }
        )
        run_store.update_run(
            job_id,
            status="completed",
            current_trial=request.num_trials,
            best_value=opt_result["best_value"],
            best_params=opt_result["best_params"],
            completed_at=completed_at,
        )

    except Exception as e:
        completed_at = datetime.now().isoformat()
        optimization_jobs[job_id].update(
            {"status": "failed", "error": str(e), "completed_at": completed_at}
        )
        run_store.update_run(job_id, status="failed", error=str(e), completed_at=completed_at)


def get_job(job_id: str) -> Dict[str, Any]:
    """Resolve a job from the hot cache, rehydrating from run_store if needed.

    Raises 404 when the job is unknown to both. Rehydrated jobs are cached
    back into ``optimization_jobs`` so subsequent lookups are cheap.
    """
    job = optimization_jobs.get(job_id)
    if job is not None:
        return job

    run = run_store.get_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Optimization not found")

    job = {
        "id": run["job_id"],
        "status": run["status"],
        "current_trial": run.get("current_trial") or 0,
        "total_trials": run.get("total_trials") or 0,
        "best_value": run.get("best_value"),
        "best_params": run.get("best_params"),
        "trials": None,
        "started_at": run.get("started_at") or "",
        "completed_at": run.get("completed_at"),
        "error": run.get("error"),
        "request": run.get("request") or {},
    }
    optimization_jobs[job_id] = job
    return job


@router.post("", response_model=Dict[str, str])
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start a new optimization study"""
    job_id = f"opt_{uuid.uuid4().hex[:8]}"

    optimization_jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "current_trial": 0,
        "total_trials": request.num_trials,
        "best_value": None,
        "best_params": None,
        "trials": None,
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "error": None,
        "request": request.dict(),  # Store request for later use
    }
    run_store.save_run(optimization_jobs[job_id])

    background_tasks.add_task(run_optimization_background, job_id, request)

    return {"id": job_id, "status": "pending"}


@router.get("")
async def list_optimizations():
    """List all persisted optimization runs (newest first)."""
    runs = run_store.list_runs()
    summaries = []
    for run in runs:
        # Prefer live in-memory state for jobs running in this process.
        live = optimization_jobs.get(run["job_id"])
        request_data = run.get("request") or {}
        record = dataset_registry.get(request_data.get("dataset_id", ""))
        summaries.append(
            {
                "id": run["job_id"],
                "study_name": run.get("study_name"),
                "db_name": run.get("db_name"),
                "status": live["status"] if live else run.get("status"),
                "started_at": run.get("started_at"),
                "completed_at": (live or run).get("completed_at"),
                "best_value": (live or run).get("best_value"),
                "current_trial": (live or run).get("current_trial"),
                "total_trials": run.get("total_trials"),
                "dataset_name": (record or {}).get("name") or request_data.get("dataset_id"),
            }
        )
    return {"runs": summaries}


@router.get("/{optimization_id}/detail")
async def get_optimization_detail(optimization_id: str):
    """Full rehydration payload: run summary plus the original request config."""
    job = get_job(optimization_id)
    request_data = job.get("request") or {}
    record = dataset_registry.get(request_data.get("dataset_id", ""))
    # The stored best_value can be stale (e.g. a dev-server reload killed the
    # background task before its final write). The Optuna study on disk is the
    # source of truth — recompute the best over finished trials on read.
    trials = serialize_study_trials(
        request_data.get("database_name", "results"),
        request_data.get("study_name", "workflow_study"),
    )
    completed = [t for t in trials if t["value"] is not None]
    if completed:
        best = max(completed, key=lambda t: t["value"])
        if job.get("best_value") != best["value"]:
            job["best_value"] = best["value"]
            job["best_params"] = best["params"]
            run_store.update_run(
                optimization_id, best_value=best["value"], best_params=best["params"]
            )
    return {
        "id": job["id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "completed_at": job.get("completed_at"),
        "best_value": job.get("best_value"),
        "best_params": job.get("best_params"),
        "current_trial": job.get("current_trial"),
        "total_trials": job.get("total_trials"),
        "error": job.get("error"),
        "request": request_data,
        "dataset": record,
    }


@router.get("/{optimization_id}", response_model=OptimizationStatus)
async def get_optimization_status(optimization_id: str):
    """Get optimization status"""
    job = get_job(optimization_id)

    # The background task only writes current_trial at completion; while the
    # run is live, read real progress from the Optuna study on disk.
    if job["status"] == "running":
        try:
            from optuna import load_study

            request_data = job.get("request", {})
            study = load_study(
                storage=optuna_storage_url(request_data.get("database_name", "results")),
                study_name=request_data.get("study_name", "workflow_study"),
                sampler=None,
            )
            # Count only finished trials — a trial enters study.trials as
            # RUNNING the moment it starts, which would advance progress early.
            job["current_trial"] = sum(1 for t in study.trials if t.state.is_finished())
        except Exception:  # noqa: S110
            pass  # study not created yet — keep the stored value
    return OptimizationStatus(
        id=job["id"],
        status=job["status"],
        current_trial=job["current_trial"],
        total_trials=job["total_trials"],
        best_value=job["best_value"],
        best_params=job["best_params"],
        trials=job.get("trials"),
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error"),
    )


@router.get("/{optimization_id}/trials")
async def get_optimization_trials(optimization_id: str):
    """Get trial history from Optuna database in real-time"""
    job = get_job(optimization_id)

    # Try to fetch live data from Optuna database
    try:
        from optuna import load_study

        request_data = job.get("request", {})
        db_name = request_data.get("database_name", "workflow_optimization.db")
        study_name = request_data.get("study_name", "workflow_study")
        storage_location = optuna_storage_url(db_name)

        # Load study and get all trials
        study = load_study(storage=storage_location, study_name=study_name, sampler=None)

        trials = []
        best_value = None
        best_params = None

        finished_count = 0
        for trial in study.trials:
            # Include RUNNING trials so the UI can show live training progress
            # (intermediate pruning reports); skip WAITING/other states.
            is_running = trial.state == TrialState.RUNNING
            if not (trial.state.is_finished() or is_running):
                continue
            if trial.state.is_finished():
                finished_count += 1
            # FAILED trials keep value=None (coercing to 0.0 made broken
            # configurations look like real F1=0 runs); the recorded "error"
            # user attr says why. PRUNED/RUNNING trials also expose None: their
            # stored value is the last intermediate report, not a final F1.
            is_complete = trial.state == TrialState.COMPLETE
            intermediate = trial.intermediate_values
            # Multi-objective trials have trial.value=None; values[0] is F1 in
            # all modes, so "best" below stays "best F1 so far".
            f1_value = trial.values[0] if is_complete and trial.values else None
            trial_data = {
                "trial": trial.number,
                "value": f1_value,
                "values": list(trial.values) if is_complete and trial.values else None,
                "params": trial.params,
                "state": trial.state.name,
                "user_attrs": trial.user_attrs,
                # Live pruning telemetry: how many intermediate reports the
                # trial has made and the latest reported value.
                "n_reports": len(intermediate),
                "last_intermediate_value": (
                    intermediate[max(intermediate)] if intermediate else None
                ),
            }
            trials.append(trial_data)

            # Track best F1 (completed trials only)
            if f1_value is not None and (best_value is None or f1_value > best_value):
                best_value = f1_value
                best_params = trial.params

        # Update job with latest data (progress counts finished trials only —
        # a RUNNING row would advance the progress bar prematurely).
        job["trials"] = trials
        job["current_trial"] = finished_count
        if best_value is not None:
            job["best_value"] = best_value
            job["best_params"] = best_params
        progress: Dict[str, Any] = {"current_trial": finished_count}
        if best_value is not None:
            progress.update(best_value=best_value, best_params=best_params)
        run_store.update_run(optimization_id, **progress)

        return {
            "trials": trials,
            "best_trial": {"value": best_value, "params": best_params}
            if best_value is not None
            else None,
            "pareto_trials": job.get("pareto_trials"),
        }
    except Exception:
        # Fallback to stored trials
        return {
            "trials": job.get("trials", []),
            "best_trial": {"value": job.get("best_value"), "params": job.get("best_params")}
            if job.get("best_value")
            else None,
        }


@router.delete("/{optimization_id}")
async def cancel_optimization(optimization_id: str):
    """Cancel a running optimization, or delete a finished run's record.

    The Optuna study database itself is left untouched.
    """
    job = get_job(optimization_id)
    if job["status"] in ("running", "pending"):
        job["status"] = "cancelled"
        job["completed_at"] = datetime.now().isoformat()
        run_store.update_run(optimization_id, status="cancelled", completed_at=job["completed_at"])
        return {"message": "Optimization cancelled"}

    from quoptuna.server.services import analysis_store

    analysis_store.delete_for_run(optimization_id)
    run_store.delete_run(optimization_id)
    optimization_jobs.pop(optimization_id, None)
    return {"message": "Optimization run deleted"}
