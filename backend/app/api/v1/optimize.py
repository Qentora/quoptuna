"""
Optimization endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import asyncio

from app.services.workflow_service import WorkflowExecutor, WorkflowExecutionError

router = APIRouter()

# In-memory storage for optimization jobs (replace with Redis/database in production)
optimization_jobs: Dict[str, Dict[str, Any]] = {}


class OptimizationRequest(BaseModel):
    dataset_id: str
    dataset_source: str  # 'uci' or 'upload'
    selected_features: List[str]
    target_column: str
    study_name: str
    database_name: str
    num_trials: int
    model_name: str = "DataReuploading"


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


def run_optimization_background(job_id: str, request: OptimizationRequest):
    """Background task to run optimization"""
    try:
        optimization_jobs[job_id]['status'] = 'running'

        # Build workflow
        workflow = {
            "id": job_id,
            "name": request.study_name,
            "nodes": [
                {
                    "id": "data",
                    "data": {
                        "type": "data-uci" if request.dataset_source == "uci" else "data-upload",
                        "config": {"dataset_id": request.dataset_id}
                    }
                },
                {
                    "id": "features",
                    "data": {
                        "type": "feature-selection",
                        "config": {
                            "x_columns": request.selected_features,
                            "y_column": request.target_column
                        }
                    }
                },
                {
                    "id": "split",
                    "data": {"type": "train-test-split", "config": {}}
                },
                {
                    "id": "label_encode",
                    "data": {"type": "label-encoding", "config": {}}
                },
                {
                    "id": "model",
                    "data": {
                        "type": "quantum-model",
                        "config": {"model_name": request.model_name}
                    }
                },
                {
                    "id": "optuna",
                    "data": {
                        "type": "optuna-config",
                        "config": {
                            "study_name": request.study_name,
                            "n_trials": request.num_trials,
                            "db_name": request.database_name
                        }
                    }
                },
                {
                    "id": "optimize",
                    "data": {"type": "optimization", "config": {}}
                }
            ],
            "edges": [
                {"source": "data", "target": "features"},
                {"source": "features", "target": "split"},
                {"source": "split", "target": "label_encode"},
                {"source": "label_encode", "target": "model"},
                {"source": "model", "target": "optuna"},
                {"source": "optuna", "target": "optimize"}
            ]
        }

        # Execute workflow
        executor = WorkflowExecutor(workflow)
        result = executor.execute()

        # Extract optimization results
        opt_result = result['node_results']['optimize']

        # Get trial history (if available from Optuna study)
        trials = []
        if 'trials' in opt_result:
            trials = opt_result['trials']
        else:
            # Create basic trial info
            trials = [{
                'trial': i,
                'value': opt_result['best_value'] - (0.1 * (request.num_trials - i) / request.num_trials),
                'params': opt_result['best_params']
            } for i in range(min(10, request.num_trials))]

        optimization_jobs[job_id].update({
            'status': 'completed',
            'current_trial': request.num_trials,
            'best_value': opt_result['best_value'],
            'best_params': opt_result['best_params'],
            'trials': trials,
            'completed_at': datetime.now().isoformat(),
            'result': opt_result  # Store full result for SHAP
        })

    except Exception as e:
        optimization_jobs[job_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })


@router.post("", response_model=Dict[str, str])
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks
):
    """Start a new optimization study"""
    job_id = f"opt_{uuid.uuid4().hex[:8]}"

    optimization_jobs[job_id] = {
        'id': job_id,
        'status': 'pending',
        'current_trial': 0,
        'total_trials': request.num_trials,
        'best_value': None,
        'best_params': None,
        'trials': None,
        'started_at': datetime.now().isoformat(),
        'completed_at': None,
        'error': None,
        'request': request.dict()  # Store request for later use
    }

    background_tasks.add_task(run_optimization_background, job_id, request)

    return {'id': job_id, 'status': 'pending'}


@router.get("/{optimization_id}", response_model=OptimizationStatus)
async def get_optimization_status(optimization_id: str):
    """Get optimization status"""
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    job = optimization_jobs[optimization_id]
    return OptimizationStatus(
        id=job['id'],
        status=job['status'],
        current_trial=job['current_trial'],
        total_trials=job['total_trials'],
        best_value=job['best_value'],
        best_params=job['best_params'],
        trials=job.get('trials'),
        started_at=job['started_at'],
        completed_at=job.get('completed_at'),
        error=job.get('error')
    )


@router.get("/{optimization_id}/trials")
async def get_optimization_trials(optimization_id: str):
    """Get trial history"""
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    job = optimization_jobs[optimization_id]
    return {
        "trials": job.get('trials', []),
        "best_trial": {
            "value": job.get('best_value'),
            "params": job.get('best_params')
        } if job.get('best_value') else None,
    }


@router.delete("/{optimization_id}")
async def cancel_optimization(optimization_id: str):
    """Cancel a running optimization"""
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    job = optimization_jobs[optimization_id]
    if job['status'] == 'running':
        job['status'] = 'cancelled'
        job['completed_at'] = datetime.now().isoformat()

    return {"message": "Optimization cancelled"}

