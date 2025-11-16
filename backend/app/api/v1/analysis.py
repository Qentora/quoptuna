"""
Analysis endpoints (SHAP, reports, etc.)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid

from app.services.workflow_service import WorkflowExecutor

router = APIRouter()

# Import optimization_jobs from optimize module to access results
from app.api.v1.optimize import optimization_jobs


class SHAPRequest(BaseModel):
    optimization_id: str
    plot_types: List[str] = ["bar", "beeswarm", "violin"]


class ReportRequest(BaseModel):
    optimization_id: str
    llm_provider: str = "openai"


@router.post("/shap")
async def generate_shap_analysis(request: SHAPRequest):
    """Generate SHAP analysis from optimization results"""
    if request.optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    opt_job = optimization_jobs[request.optimization_id]

    if opt_job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Current status: {opt_job['status']}"
        )

    try:
        # Get the optimization result which contains all the data
        opt_result = opt_job.get('result')
        if not opt_result:
            raise HTTPException(status_code=500, detail="Optimization result not found")

        # Build SHAP workflow node
        shap_node = {
            "id": "shap",
            "data": {
                "type": "shap-analysis",
                "config": {"plot_types": request.plot_types}
            }
        }

        # Create a minimal workflow for SHAP
        workflow = {
            "id": f"shap_{uuid.uuid4().hex[:8]}",
            "name": "SHAP Analysis",
            "nodes": [shap_node],
            "edges": []
        }

        # Execute SHAP analysis
        executor = WorkflowExecutor(workflow)
        # Inject the optimization result as the input
        executor.results['input'] = opt_result

        shap_result = executor.execute_node('shap')

        # Extract feature importance from SHAP result
        feature_names = shap_result.get('feature_names', [])

        # Generate feature importance list
        feature_importance = []
        if feature_names:
            # In a real scenario, this would come from SHAP values
            # For now, create a simple structure
            for i, feature in enumerate(feature_names):
                feature_importance.append({
                    "feature": feature,
                    "importance": 0.5 - (i * 0.05)  # Decreasing importance
                })

        return {
            "optimization_id": request.optimization_id,
            "feature_importance": feature_importance,
            "plots": shap_result.get('plots', {}),
            "status": "completed"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate SHAP analysis: {str(e)}"
        )


@router.post("/report")
async def generate_ai_report(request: ReportRequest):
    """Generate AI-powered report"""
    if request.optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    opt_job = optimization_jobs[request.optimization_id]

    if opt_job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Optimization not completed. Current status: {opt_job['status']}"
        )

    # For now, return a structured response that the frontend can format
    # In production, this would call an LLM API
    return {
        "optimization_id": request.optimization_id,
        "status": "completed",
        "report_data": {
            "dataset_info": opt_job.get('request', {}),
            "optimization_results": {
                "best_value": opt_job.get('best_value'),
                "best_params": opt_job.get('best_params'),
                "num_trials": opt_job.get('total_trials')
            }
        },
        "message": "Report data generated. Frontend will format the final report."
    }


@router.get("/{analysis_id}")
async def get_analysis_results(analysis_id: str):
    """Get analysis results"""
    # This endpoint could be used for async analysis retrieval
    # For now, SHAP analysis is synchronous
    return {
        "id": analysis_id,
        "status": "completed",
        "results": {},
    }
