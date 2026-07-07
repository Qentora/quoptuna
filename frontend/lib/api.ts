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
  unique_counts: Record<string, number>;
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
  sensitive_feature?: string;
  categorical_encoding?: 'ordinal' | 'onehot';
}

export type RunStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'interrupted';

export interface OptimizationStatus {
  id: string;
  status: RunStatus;
  current_trial: number;
  total_trials: number;
  best_value: number | null;
  best_params: Record<string, any> | null;
  trials: Array<{
    trial: number;
    value: number | null;
    params: Record<string, any>;
    state?: string;
    user_attrs?: Record<string, any>;
  }> | null;
  started_at: string;
  completed_at: string | null;
  error: string | null;
}

export interface OptimizationTrial {
  trial: number;
  // null for FAILED trials; user_attrs.error records the failure reason.
  value: number | null;
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

export interface PastRun {
  id: string;
  study_name: string | null;
  db_name: string | null;
  status: RunStatus;
  started_at: string | null;
  completed_at: string | null;
  best_value: number | null;
  current_trial: number | null;
  total_trials: number | null;
  dataset_name: string | null;
}

export async function listOptimizations(): Promise<PastRun[]> {
  const data = await request<{ runs: PastRun[] }>('/api/v1/optimize');
  return data.runs;
}

export interface OptimizationDetail {
  id: string;
  status: RunStatus;
  started_at: string;
  completed_at: string | null;
  best_value: number | null;
  best_params: Record<string, any> | null;
  current_trial: number | null;
  total_trials: number | null;
  error: string | null;
  request: Partial<OptimizationRequest>;
  dataset: {
    id: string;
    name: string;
    source: 'upload' | 'uci';
    file_path?: string;
    rows?: number;
    columns?: string[];
  } | null;
}

export async function getOptimizationDetail(id: string): Promise<OptimizationDetail> {
  return request<OptimizationDetail>(`/api/v1/optimize/${id}/detail`);
}

/** Cancels a running optimization or deletes a finished run's record. */
export async function deleteOptimization(id: string): Promise<{ message: string }> {
  return request<{ message: string }>(`/api/v1/optimize/${id}`, { method: 'DELETE' });
}

const POLL_MAX_CONSECUTIVE_FAILURES = 5;

export async function pollOptimization(
  optimizationId: string,
  onUpdate: (status: OptimizationStatus, trials?: OptimizationTrials | null) => void,
  intervalMs = 2000
): Promise<OptimizationStatus> {
  return new Promise((resolve, reject) => {
    let failures = 0;
    const poll = async () => {
      try {
        const [status, trialsData] = await Promise.all([
          getOptimizationStatus(optimizationId),
          fetchOptimizationTrials(optimizationId).catch(() => null),
        ]);
        failures = 0;
        onUpdate(status, trialsData);
        if (status.status !== 'pending' && status.status !== 'running') {
          resolve(status);
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (error) {
        // Heavy trials can starve the backend and make a poll time out;
        // keep polling through transient failures instead of giving up.
        failures += 1;
        if (failures >= POLL_MAX_CONSECUTIVE_FAILURES) {
          reject(error);
        } else {
          setTimeout(poll, intervalMs);
        }
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

export interface MetricsRequest {
  optimization_id: string;
  trial_number?: number;
  use_proba?: boolean;
  subset_size?: number;
}

export interface CurvesData {
  roc: { fpr: number[]; tpr: number[]; auc: number } | null;
  pr: { precision: number[]; recall: number[]; average_precision: number } | null;
}

export async function getCurvesData(body: MetricsRequest): Promise<CurvesData> {
  return request<CurvesData>('/api/v1/analysis/curves/data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
  normalized: number[][];
}

export async function getConfusionMatrixData(body: MetricsRequest): Promise<ConfusionMatrixData> {
  return request<ConfusionMatrixData>('/api/v1/analysis/confusion-matrix/data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export interface FeatureImportanceData {
  features: string[];
  importances: number[];
}

export async function getFeatureImportanceData(
  body: MetricsRequest
): Promise<FeatureImportanceData> {
  return request<FeatureImportanceData>('/api/v1/analysis/feature-importance/data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export interface PlotlyFigureJSON {
  data: Array<Record<string, any>>;
  layout: Record<string, any>;
}

export interface StudyPlotsResponse {
  optimization_id: string;
  // Plotly figure JSON keyed by plot name: optimization_history,
  // param_importances, parallel_coordinate, slice, timeline.
  plots: Record<string, PlotlyFigureJSON | null>;
  status: string;
}

export async function getStudyPlots(optimizationId: string): Promise<StudyPlotsResponse> {
  return request<StudyPlotsResponse>('/api/v1/analysis/study-plots', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ optimization_id: optimizationId }),
  });
}

export interface FairnessMetrics {
  by_group: Record<string, Record<string, number>>;
  overall: Record<string, number>;
  disparities: Record<string, number>;
}

export interface FairnessResponse {
  optimization_id: string;
  status: string;
  sensitive_feature: string;
  metrics: FairnessMetrics;
  plots: Record<string, string>;
  mitigation: {
    constraint: string;
    before: FairnessMetrics;
    after: FairnessMetrics;
    comparison_plot: string;
  } | null;
}

export async function generateFairness(body: {
  optimization_id: string;
  sensitive_feature?: string;
  trial_number?: number;
  mitigate?: boolean;
  constraint?: string;
}): Promise<FairnessResponse> {
  return request<FairnessResponse>('/api/v1/analysis/fairness', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export interface ReportRequest {
  optimization_id: string;
  trial_number?: number;
  llm_provider: 'google' | 'openai' | 'anthropic';
  api_key: string;
  model_name: string;
  dataset_description?: string;
  sensitive_feature?: string;
  include_fairness?: boolean;
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
