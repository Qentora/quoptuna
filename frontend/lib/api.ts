/**
 * API client for communicating with the QuOptuna backend (FastAPI).
 */

// Default to same-origin so the statically-exported UI (served by FastAPI from
// the wheel) calls the co-served API. In dev, set NEXT_PUBLIC_API_URL in
// frontend/.env.local (e.g. http://localhost:8000) to point at a separate backend.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const error = await response.json();
      detail = error.detail || detail;
    } catch {
      // response had no JSON body
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export interface DatasetUploadResponse {
  message: string;
  filename: string;
  id: string;
  file_path: string;
  rows: number;
  columns: string[];
}

export async function uploadDataset(file: File): Promise<DatasetUploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  return request<DatasetUploadResponse>('/api/v1/data/upload', {
    method: 'POST',
    body: formData,
  });
}

export interface UCIDataset {
  id: number;
  name: string;
  description?: string;
  num_instances?: number;
  num_features?: number;
}

export async function listUCIDatasets(): Promise<UCIDataset[]> {
  const data = await request<{ datasets: UCIDataset[] }>('/api/v1/data/uci');
  return data.datasets;
}

export interface UCILoadResponse {
  message: string;
  id: string;
  dataset_id: number;
  name: string;
  file_path: string;
  rows: number;
  columns: string[];
}

/** Fetch a UCI dataset, persist it server-side and register it for reuse. */
export async function loadUCIDataset(datasetId: number): Promise<UCILoadResponse> {
  return request<UCILoadResponse>(`/api/v1/data/uci/${datasetId}/load`, {
    method: 'POST',
  });
}

export interface DatasetPreview {
  id: string;
  columns: string[];
  dtypes: Record<string, string>;
  head: Array<Record<string, any>>;
  num_rows: number;
  missing: Record<string, number>;
  target_values_by_column: Record<string, Array<string | number>>;
}

export async function getDatasetPreview(datasetId: string): Promise<DatasetPreview> {
  return request<DatasetPreview>(`/api/v1/data/${datasetId}/preview`);
}

// ---------------------------------------------------------------------------
// Optimization
// ---------------------------------------------------------------------------

export interface LabelMapping {
  neg: string | number;
  pos: string | number;
}

export interface OptimizationRequest {
  dataset_id: string;
  dataset_source: 'uci' | 'upload';
  selected_features: string[];
  target_column: string;
  study_name: string;
  database_name: string;
  num_trials: number;
  model_name?: string;
  label_mapping?: LabelMapping;
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

export interface OptimizationTrial {
  trial: number;
  value: number;
  params: Record<string, any>;
  state: string;
  user_attrs?: Record<string, any>;
}

export interface OptimizationTrials {
  trials: OptimizationTrial[];
  best_trial: { value: number; params: Record<string, any> } | null;
}

export async function startOptimization(
  body: OptimizationRequest
): Promise<{ id: string; status: string }> {
  return request('/api/v1/optimize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function getOptimizationStatus(id: string): Promise<OptimizationStatus> {
  return request<OptimizationStatus>(`/api/v1/optimize/${id}`);
}

export async function fetchOptimizationTrials(id: string): Promise<OptimizationTrials> {
  return request<OptimizationTrials>(`/api/v1/optimize/${id}/trials`);
}

export async function pollOptimization(
  optimizationId: string,
  onUpdate: (status: OptimizationStatus, trials?: OptimizationTrials | null) => void,
  intervalMs = 2000
): Promise<OptimizationStatus> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const [status, trialsData] = await Promise.all([
          getOptimizationStatus(optimizationId),
          fetchOptimizationTrials(optimizationId).catch(() => null),
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

// ---------------------------------------------------------------------------
// Analysis (SHAP / metrics / report)
// ---------------------------------------------------------------------------

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface SHAPResponse {
  optimization_id: string;
  feature_importance: FeatureImportance[];
  plots: Record<string, string>;
  status: string;
}

export interface SHAPRequest {
  optimization_id: string;
  plot_types?: string[];
  trial_number?: number;
  sample_index?: number;
  use_proba?: boolean;
  subset_size?: number;
}

export async function generateSHAP(body: SHAPRequest): Promise<SHAPResponse> {
  return request<SHAPResponse>('/api/v1/analysis/shap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      plot_types: ['bar', 'beeswarm', 'violin', 'heatmap', 'waterfall'],
      ...body,
    }),
  });
}

export interface MetricsResponse {
  optimization_id: string;
  confusion_matrix_plot: string | null;
  metrics: Record<string, any>;
  status: string;
}

export async function getMetrics(
  optimizationId: string,
  trialNumber?: number
): Promise<MetricsResponse> {
  return request<MetricsResponse>('/api/v1/analysis/metrics', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ optimization_id: optimizationId, trial_number: trialNumber }),
  });
}

export interface CurvesResponse {
  optimization_id: string;
  roc_curve_plot: string | null;
  pr_curve_plot: string | null;
  roc_auc: number | null;
  average_precision: number | null;
  status: string;
}

export async function getCurves(
  optimizationId: string,
  trialNumber?: number,
  useProba = true,
  subsetSize = 50
): Promise<CurvesResponse> {
  return request<CurvesResponse>('/api/v1/analysis/curves', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      optimization_id: optimizationId,
      trial_number: trialNumber,
      use_proba: useProba,
      subset_size: subsetSize,
    }),
  });
}

export interface StudyPlotsResponse {
  optimization_id: string;
  optimization_history_plot: string | null;
  param_importances_plot: string | null;
  status: string;
}

export async function getStudyPlots(optimizationId: string): Promise<StudyPlotsResponse> {
  return request<StudyPlotsResponse>('/api/v1/analysis/study-plots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ optimization_id: optimizationId }),
  });
}

export interface ReportRequest {
  optimization_id: string;
  trial_number?: number;
  llm_provider: 'google' | 'openai';
  api_key: string;
  model_name: string;
  dataset_description?: string;
}

export interface ReportResponse {
  optimization_id: string;
  status: string;
  report_markdown: string;
}

export async function generateReport(body: ReportRequest): Promise<ReportResponse> {
  return request<ReportResponse>('/api/v1/analysis/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

export interface SystemInfo {
  version: string;
  quantum_models: number;
  classical_models: number;
  total_models: number;
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>('/api/v1/info');
}
