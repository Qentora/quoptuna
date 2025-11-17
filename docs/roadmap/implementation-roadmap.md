# Implementation Roadmap: Connect Optimizer UI to Real Backend

This document outlines the steps needed to replace the simulated optimization/SHAP in the UI with real backend services.

---

## Phase 1: Implement Backend API Endpoints

### Task 1.1: Implement Optimization Endpoints

**File:** `/backend/app/api/v1/optimize.py`

```python
# Current: All TODOs
# Needed: Real implementation

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
from pathlib import Path

from app.services.workflow_service import WorkflowExecutor

router = APIRouter()

# In-memory storage (replace with Redis/database in production)
optimization_jobs = {}

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
    best_value: float | None
    best_params: Dict[str, Any] | None
    started_at: str
    completed_at: str | None
    error: str | None

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
                {"source": "split", "target": "model"},
                {"source": "model", "target": "optuna"},
                {"source": "optuna", "target": "optimize"}
            ]
        }

        # Execute workflow
        executor = WorkflowExecutor(workflow)
        result = executor.execute()

        # Extract optimization results
        opt_result = result['node_results']['optimize']

        optimization_jobs[job_id].update({
            'status': 'completed',
            'current_trial': request.num_trials,
            'best_value': opt_result['best_value'],
            'best_params': opt_result['best_params'],
            'completed_at': datetime.now().isoformat(),
            'result': opt_result
        })

    except Exception as e:
        optimization_jobs[job_id].update({
            'status': 'failed',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })

@router.post("", response_model=dict)
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
        'started_at': datetime.now().isoformat(),
        'completed_at': None,
        'error': None
    }

    background_tasks.add_task(run_optimization_background, job_id, request)

    return {'id': job_id, 'status': 'pending'}

@router.get("/{optimization_id}", response_model=OptimizationStatus)
async def get_optimization_status(optimization_id: str):
    """Get optimization status"""
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    return optimization_jobs[optimization_id]
```

### Task 1.2: Implement SHAP Analysis Endpoints

**File:** `/backend/app/api/v1/analysis.py`

```python
# Add real SHAP implementation

@router.post("/shap")
async def generate_shap_analysis(
    optimization_id: str,
    plot_types: List[str] = ["bar", "beeswarm"]
):
    """Generate SHAP analysis from optimization results"""
    if optimization_id not in optimization_jobs:
        raise HTTPException(status_code=404, detail="Optimization not found")

    opt_job = optimization_jobs[optimization_id]
    if opt_job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Optimization not completed")

    # Build SHAP workflow
    opt_result = opt_job['result']

    workflow = {
        "id": f"shap_{uuid.uuid4().hex[:8]}",
        "name": "SHAP Analysis",
        "nodes": [{
            "id": "shap",
            "data": {
                "type": "shap-analysis",
                "config": {"plot_types": plot_types}
            }
        }],
        "edges": []
    }

    # Execute SHAP
    executor = WorkflowExecutor(workflow)
    # Inject optimization result as input
    executor.results['input'] = opt_result

    shap_result = executor.execute_node('shap')

    return {
        "feature_importance": extract_feature_importance(shap_result),
        "plots": shap_result.get('plots', {})
    }
```

---

## Phase 2: Update Frontend to Use Real APIs

### Task 2.1: Create Optimizer API Client

**File:** `/frontend/src/lib/api.ts`

```typescript
export interface OptimizationRequest {
  dataset_id: string;
  dataset_source: 'uci' | 'upload';
  selected_features: string[];
  target_column: string;
  study_name: string;
  database_name: string;
  num_trials: number;
  model_name?: string;
}

export interface OptimizationStatus {
  id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_trial: number;
  total_trials: number;
  best_value: number | null;
  best_params: Record<string, any> | null;
  started_at: string;
  completed_at: string | null;
  error: string | null;
}

export async function startOptimization(
  request: OptimizationRequest
): Promise<{ id: string }> {
  const response = await fetch(`${API_BASE_URL}/api/v1/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to start optimization');
  }

  return response.json();
}

export async function getOptimizationStatus(
  optimizationId: string
): Promise<OptimizationStatus> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/optimize/${optimizationId}`
  );

  if (!response.ok) {
    throw new Error('Failed to get optimization status');
  }

  return response.json();
}

export async function pollOptimization(
  optimizationId: string,
  onUpdate: (status: OptimizationStatus) => void,
  intervalMs: number = 2000
): Promise<OptimizationStatus> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getOptimizationStatus(optimizationId);
        onUpdate(status);

        if (status.status === 'completed' || status.status === 'failed') {
          resolve(status);
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (error) {
        reject(error);
      }
    };

    poll();
  });
}

export async function generateSHAP(
  optimizationId: string,
  plotTypes: string[] = ['bar', 'beeswarm']
): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/api/v1/analysis/shap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ optimization_id: optimizationId, plot_types: plotTypes }),
  });

  if (!response.ok) {
    throw new Error('Failed to generate SHAP analysis');
  }

  return response.json();
}
```

### Task 2.2: Update OptimizeStep to Use Real API

**File:** `/frontend/src/pages/Optimizer.tsx`

Replace the simulated `startOptimization()` function:

```typescript
function OptimizeStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTrial, setCurrentTrial] = useState(0);
  const [optimizationId, setOptimizationId] = useState<string | null>(null);

  const startOptimization = async () => {
    if (!workflowData.dataset) return;

    setIsRunning(true);
    setProgress(0);
    setCurrentTrial(0);

    try {
      // Start real optimization
      const { id } = await startOptimization({
        dataset_id: workflowData.dataset.id,
        dataset_source: workflowData.dataset.source,
        selected_features: workflowData.features.selectedFeatures,
        target_column: workflowData.features.targetColumn!,
        study_name: workflowData.configuration.studyName,
        database_name: workflowData.configuration.databaseName,
        num_trials: workflowData.configuration.numTrials,
      });

      setOptimizationId(id);

      // Poll for status updates
      const finalStatus = await pollOptimization(id, (status) => {
        setCurrentTrial(status.current_trial);
        setProgress((status.current_trial / status.total_trials) * 100);
      });

      // Update workflow data with real results
      setWorkflowData((prev) => ({
        ...prev,
        optimization: {
          executionId: id,
          status: 'completed',
          results: {
            bestValue: finalStatus.best_value,
            bestParams: finalStatus.best_params,
            trials: [], // Could fetch trial history separately
          },
        },
      }));

      setIsRunning(false);
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Optimization failed');
      setIsRunning(false);
    }
  };

  // Rest of component...
}
```

### Task 2.3: Update AnalyzeStep to Use Real API

```typescript
const generateSHAP = async () => {
  if (!workflowData.optimization.executionId) return;

  setIsGenerating(true);

  try {
    // Call real SHAP API
    const shapResult = await generateSHAP(
      workflowData.optimization.executionId,
      ['bar', 'beeswarm', 'waterfall']
    );

    setWorkflowData((prev) => ({
      ...prev,
      analysis: {
        shapData: {
          featureImportance: shapResult.feature_importance,
          plots: shapResult.plots,
          generated: true,
        },
      },
    }));

    setIsGenerating(false);
  } catch (error) {
    setError(error instanceof Error ? error.message : 'SHAP generation failed');
    setIsGenerating(false);
  }
};
```

---

## Phase 3: Add Progress Tracking with WebSockets (Optional but Recommended)

### Why WebSockets?

Long-running optimizations (10+ minutes) benefit from real-time updates instead of polling.

### Implementation

**Backend:** `backend/app/api/v1/optimize.py`

```python
from fastapi import WebSocket

@router.websocket("/ws/{optimization_id}")
async def optimization_websocket(websocket: WebSocket, optimization_id: str):
    await websocket.accept()

    while True:
        if optimization_id in optimization_jobs:
            status = optimization_jobs[optimization_id]
            await websocket.send_json(status)

            if status['status'] in ['completed', 'failed']:
                break

        await asyncio.sleep(1)

    await websocket.close()
```

**Frontend:** Use WebSocket for live updates

```typescript
const ws = new WebSocket(`ws://localhost:8000/api/v1/optimize/ws/${id}`);

ws.onmessage = (event) => {
  const status = JSON.parse(event.data);
  setCurrentTrial(status.current_trial);
  setProgress((status.current_trial / status.total_trials) * 100);

  if (status.status === 'completed') {
    // Handle completion
  }
};
```

---

## Phase 4: Testing Checklist

### Backend Tests

- [ ] `/api/v1/optimize` starts optimization correctly
- [ ] `/api/v1/optimize/{id}` returns accurate status
- [ ] Optimization completes with real Optuna results
- [ ] `/api/v1/analysis/shap` generates real SHAP plots
- [ ] Error handling for invalid inputs
- [ ] Database persistence works

### Frontend Tests

- [ ] UI calls `/api/v1/optimize` instead of mock
- [ ] Progress bar shows real trial progress
- [ ] Results match backend response
- [ ] SHAP analysis displays real feature importance
- [ ] Error states are handled properly
- [ ] Loading states work correctly

### Integration Tests

- [ ] End-to-end: Select dataset → Optimize → SHAP → Report
- [ ] Test with different datasets (UCI, uploaded)
- [ ] Test with different trial counts (10, 100, 1000)
- [ ] Test error scenarios (invalid features, missing data)
- [ ] Test cancellation of running optimization

---

## Phase 5: Performance Optimization

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_uci_dataset(dataset_id: int):
    return fetch_ucirepo(id=dataset_id)
```

### Async Processing

- Use Celery or RQ for background task management
- Store results in Redis for fast access
- Implement job queue for multiple concurrent optimizations

### Database

Replace in-memory `optimization_jobs` dict with:
- PostgreSQL for production
- Redis for fast access to status
- MongoDB for storing large result objects

---

## Timeline Estimates

| Phase | Estimated Time | Priority |
|-------|---------------|----------|
| Phase 1.1: Optimize API | 4-6 hours | High |
| Phase 1.2: SHAP API | 2-3 hours | High |
| Phase 2.1: API Client | 1-2 hours | High |
| Phase 2.2: Update OptimizeStep | 2-3 hours | High |
| Phase 2.3: Update AnalyzeStep | 1-2 hours | High |
| Phase 3: WebSockets | 3-4 hours | Medium |
| Phase 4: Testing | 4-6 hours | High |
| Phase 5: Optimization | 3-4 hours | Low |

**Total:** ~20-30 hours for complete implementation

---

## Current vs Future Flow

### Current (Simulated)
```
User clicks "Start" → setTimeout loop → Show mock results
```

### Future (Real)
```
User clicks "Start"
    ↓
POST /api/v1/optimize (start background job)
    ↓
Poll GET /api/v1/optimize/{id} every 2 seconds
    ↓
Update progress bar with real trial count
    ↓
When status === 'completed'
    ↓
Display real optimization results
    ↓
POST /api/v1/analysis/shap
    ↓
Display real SHAP analysis
```

---

**Next Action:** Start with Phase 1.1 (Implement Optimize API)
