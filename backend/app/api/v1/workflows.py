"""
Workflow management endpoints
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.workflow_service import WorkflowExecutor, WorkflowExecutionError

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory storage (for demo - replace with database in production)
workflows_db: Dict[str, Dict] = {}
executions_db: Dict[str, Dict] = {}


class WorkflowCreate(BaseModel):
    name: str
    description: str = ""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None


class WorkflowExecute(BaseModel):
    """Request body for executing a workflow directly (without saving)"""
    name: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


def execute_workflow_task(execution_id: str, workflow: Dict):
    """Background task to execute workflow"""
    try:
        logger.info(f"Starting workflow execution {execution_id}")
        executions_db[execution_id]["status"] = "running"

        executor = WorkflowExecutor(workflow)
        result = executor.execute()

        executions_db[execution_id].update({
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "result": result,
        })

    except Exception as e:
        logger.error(f"Workflow execution {execution_id} failed: {str(e)}")
        executions_db[execution_id].update({
            "status": "failed",
            "completed_at": datetime.utcnow().isoformat(),
            "error": str(e),
        })


@router.post("")
async def create_workflow(workflow: WorkflowCreate):
    """Create a new workflow"""
    workflow_id = f"workflow-{len(workflows_db) + 1}"

    workflow_data = {
        "id": workflow_id,
        "name": workflow.name,
        "description": workflow.description,
        "nodes": workflow.nodes,
        "edges": workflow.edges,
        "status": "draft",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }

    workflows_db[workflow_id] = workflow_data

    return workflow_data


@router.get("")
async def list_workflows():
    """List all workflows"""
    return {
        "workflows": list(workflows_db.values()),
        "total": len(workflows_db),
    }


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str):
    """Get a specific workflow"""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    return workflows_db[workflow_id]


@router.put("/{workflow_id}")
async def update_workflow(workflow_id: str, workflow: WorkflowUpdate):
    """Update a workflow"""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    existing = workflows_db[workflow_id]

    if workflow.name is not None:
        existing["name"] = workflow.name
    if workflow.description is not None:
        existing["description"] = workflow.description
    if workflow.nodes is not None:
        existing["nodes"] = workflow.nodes
    if workflow.edges is not None:
        existing["edges"] = workflow.edges

    existing["updated_at"] = datetime.utcnow().isoformat()

    return existing


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    del workflows_db[workflow_id]
    return {"message": "Workflow deleted successfully"}


@router.post("/execute")
async def execute_workflow_direct(
    workflow: WorkflowExecute,
    background_tasks: BackgroundTasks,
):
    """Execute a workflow directly without saving it first"""
    execution_id = f"exec-{len(executions_db) + 1}"

    workflow_data = {
        "id": f"temp-{execution_id}",
        "name": workflow.name,
        "nodes": workflow.nodes,
        "edges": workflow.edges,
    }

    execution = {
        "id": execution_id,
        "workflow_id": workflow_data["id"],
        "workflow_name": workflow.name,
        "status": "pending",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    executions_db[execution_id] = execution

    # Start execution in background
    background_tasks.add_task(execute_workflow_task, execution_id, workflow_data)

    return {
        "execution_id": execution_id,
        "status": "pending",
        "message": "Workflow execution started",
    }


@router.post("/{workflow_id}/run")
async def run_workflow(workflow_id: str, background_tasks: BackgroundTasks):
    """Execute a saved workflow"""
    if workflow_id not in workflows_db:
        raise HTTPException(status_code=404, detail="Workflow not found")

    workflow = workflows_db[workflow_id]
    execution_id = f"exec-{len(executions_db) + 1}"

    execution = {
        "id": execution_id,
        "workflow_id": workflow_id,
        "workflow_name": workflow["name"],
        "status": "pending",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "result": None,
        "error": None,
    }

    executions_db[execution_id] = execution

    # Start execution in background
    background_tasks.add_task(execute_workflow_task, execution_id, workflow)

    return {
        "execution_id": execution_id,
        "workflow_id": workflow_id,
        "status": "pending",
        "message": "Workflow execution started",
    }


@router.get("/executions/{execution_id}")
async def get_execution_status(execution_id: str):
    """Get execution status and results"""
    if execution_id not in executions_db:
        raise HTTPException(status_code=404, detail="Execution not found")

    return executions_db[execution_id]


@router.get("/executions")
async def list_executions():
    """List all workflow executions"""
    return {
        "executions": list(executions_db.values()),
        "total": len(executions_db),
    }
