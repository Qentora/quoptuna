/**
 * API client for communicating with the QuOptuna Next backend
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface WorkflowExecuteRequest {
  name: string;
  nodes: any[];
  edges: any[];
}

export interface WorkflowExecuteResponse {
  execution_id: string;
  status: string;
  message: string;
}

export interface ExecutionStatusResponse {
  id: string;
  workflow_id: string;
  workflow_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  started_at: string;
  completed_at?: string;
  result?: any;
  error?: string;
}

/**
 * Execute a workflow directly without saving it
 */
export async function executeWorkflow(
  workflow: WorkflowExecuteRequest
): Promise<WorkflowExecuteResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/workflows/execute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(workflow),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to execute workflow');
  }

  return response.json();
}

/**
 * Get execution status and results
 */
export async function getExecutionStatus(
  executionId: string
): Promise<ExecutionStatusResponse> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/workflows/executions/${executionId}`
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get execution status');
  }

  return response.json();
}

/**
 * Poll execution status until complete or failed
 */
export async function pollExecutionStatus(
  executionId: string,
  onUpdate?: (status: ExecutionStatusResponse) => void,
  intervalMs: number = 2000
): Promise<ExecutionStatusResponse> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getExecutionStatus(executionId);

        if (onUpdate) {
          onUpdate(status);
        }

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

export interface DatasetUploadResponse {
  message: string;
  filename: string;
  id: string;
  file_path: string;
  rows: number;
  columns: string[];
}

/**
 * Upload a CSV dataset
 */
export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/data/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to upload dataset');
  }

  return response.json();
}

export interface UCIDataset {
  id: number;
  name: string;
  description?: string;
  num_instances?: number;
  num_features?: number;
}

/**
 * List available UCI datasets
 */
export async function listUCIDatasets(): Promise<UCIDataset[]> {
  const response = await fetch(`${API_BASE_URL}/api/v1/data/uci`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to list UCI datasets');
  }

  return response.json();
}

export interface UCIDatasetResponse {
  message: string;
  dataset_id: string;
  name: string;
  rows: number;
  columns: string[];
}

/**
 * Fetch a specific UCI dataset
 */
export async function fetchUCIDataset(datasetId: number): Promise<UCIDatasetResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/data/uci/${datasetId}`);

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to fetch UCI dataset');
  }

  return response.json();
}

// Optimization API

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
  trials: Array<{
    trial: number;
    value: number;
    params: Record<string, any>;
  }> | null;
  started_at: string;
  completed_at: string | null;
  error: string | null;
}

/**
 * Start an optimization study
 */
export async function startOptimization(
  request: OptimizationRequest
): Promise<{ id: string; status: string }> {
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

/**
 * Get optimization status
 */
export async function getOptimizationStatus(
  optimizationId: string
): Promise<OptimizationStatus> {
  const response = await fetch(
    `${API_BASE_URL}/api/v1/optimize/${optimizationId}`
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to get optimization status');
  }

  return response.json();
}

/**
 * Fetch optimization trials history
 */
export async function fetchOptimizationTrials(optimizationId: string): Promise<{
  trials: Array<{
    trial: number;
    value: number;
    params: Record<string, any>;
    state: string;
    user_attrs?: Record<string, any>;
  }>;
  best_trial: {
    value: number;
    params: Record<string, any>;
  } | null;
}> {
  const response = await fetch(`${API_BASE_URL}/api/v1/optimize/${optimizationId}/trials`);

  if (!response.ok) {
    throw new Error('Failed to fetch trials');
  }

  return response.json();
}

/**
 * Poll optimization status until complete
 */
export async function pollOptimization(
  optimizationId: string,
  onUpdate: (status: OptimizationStatus, trials?: any) => void,
  intervalMs: number = 2000
): Promise<OptimizationStatus> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const [status, trialsData] = await Promise.all([
          getOptimizationStatus(optimizationId),
          fetchOptimizationTrials(optimizationId).catch(() => null)
        ]);

        onUpdate(status, trialsData);

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

// Analysis API

export interface SHAPResponse {
  optimization_id: string;
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
  plots: Record<string, any>;
  status: string;
}

/**
 * Generate SHAP analysis
 */
export async function generateSHAP(
  optimizationId: string,
  plotTypes: string[] = ['bar', 'beeswarm']
): Promise<SHAPResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/analysis/shap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      optimization_id: optimizationId,
      plot_types: plotTypes,
    }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to generate SHAP analysis');
  }

  return response.json();
}
