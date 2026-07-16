/**
 * API client for communicating with the QuOptuna backend (FastAPI).
 */

// Default to same-origin so the statically-exported UI (served by FastAPI from
// the wheel) calls the co-served API. In dev, set NEXT_PUBLIC_API_URL in
// frontend/.env.local (e.g. http://localhost:8000) to point at a separate backend.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

/** Thrown on 401 so callers can distinguish "not logged in" from other errors. */
export class UnauthorizedError extends Error {
  constructor(detail: string) {
    super(detail);
    this.name = 'UnauthorizedError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // credentials: 'include' sends the Auth0 session cookie cross-origin in dev
  // (UI on :3000, API on :8000); it's a no-op for the co-served production build.
  const response = await fetch(`${API_BASE_URL}${path}`, { credentials: 'include', ...init });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const error = await response.json();
      detail = error.detail || detail;
    } catch {
      // response had no JSON body
    }
    if (response.status === 401) {
      throw new UnauthorizedError(detail);
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
  // K>2 targets: the class treated as the favorable outcome for fairness
  // auditing and report framing. Required when fairness is used on a
  // multiclass target.
  favorable_class?: string | number;
  sensitive_feature?: string;
  categorical_encoding?: 'ordinal' | 'onehot';
  // Search strategy: sampler + optional early-stopping pruner (ASHA/Hyperband).
  sampler?: 'tpe' | 'random' | 'grid';
  pruner?: 'none' | 'asha' | 'hyperband';
  // Performance knobs (Settings → Optimizer Performance).
  max_steps?: number;
  convergence_interval?: number;
  max_vmap?: number;
  // PennyLane simulator device; backend falls back to default.qubit.
  dev_type?: 'default.qubit' | 'lightning.qubit';
  // Fairness-aware search. 'constrained' requires sampler 'tpe';
  // 'multi_objective' coerces the pruner to 'none'; both require sensitive_feature.
  fairness_mode?: 'off' | 'constrained' | 'multi_objective';
  fairness_metric?:
    | 'equal_opportunity_difference'
    | 'disparate_impact'
    | 'demographic_parity_difference';
  fairness_threshold?: number;
}

// Pareto-front entry of a multi-objective run; values = [f1, disparity].
export interface ParetoTrial {
  trial: number;
  values: number[];
  params: Record<string, any>;
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
  // null for FAILED/PRUNED/RUNNING trials; user_attrs.error records failures.
  value: number | null;
  // Multi-objective runs: [f1, disparity] for COMPLETE trials.
  values?: number[] | null;
  params: Record<string, any>;
  state: string;
  user_attrs?: Record<string, any>;
  // Live pruning telemetry: intermediate reports made so far + latest value.
  n_reports?: number;
  last_intermediate_value?: number | null;
}

export interface OptimizationTrials {
  trials: OptimizationTrial[];
  best_trial: { value: number; params: Record<string, any> } | null;
  pareto_trials?: ParetoTrial[] | null;
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
  // Multiclass: which class's SHAP values to plot (default class 0).
  class_index?: number;
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
  task_type?: 'binary' | 'multiclass';
  n_classes?: number;
  class_labels?: string[] | null;
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
  // Binary: positive-class AUC. Multiclass: macro-averaged one-vs-rest AUC.
  roc_auc: number | null;
  average_precision: number | null;
  task_type?: 'binary' | 'multiclass';
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

// Binary tasks return single curves; multiclass tasks return one-vs-rest
// per-class curve sets instead.
export interface RocCurvePoints {
  fpr: number[];
  tpr: number[];
  auc: number | null;
}

export interface PrCurvePoints {
  precision: number[];
  recall: number[];
  average_precision: number | null;
}

export interface CurvesData {
  roc:
    | RocCurvePoints
    | { per_class: Array<{ label: string } & RocCurvePoints>; macro_auc: number | null }
    | null;
  pr: PrCurvePoints | { per_class: Array<{ label: string } & PrCurvePoints> } | null;
  task_type?: 'binary' | 'multiclass';
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

export interface ShapDataRequest {
  optimization_id: string;
  trial_number?: number;
  subset_size?: number;
  sample_index?: number;
  // Multiclass: which class's SHAP slice to return (default class 0).
  class_index?: number;
}

export interface ShapData {
  optimization_id: string;
  feature_names: string[];
  /** samples × features SHAP values (one class slice for multiclass) */
  values: number[][];
  /** raw feature values, row-aligned with `values` */
  data: number[][];
  base_value: number;
  n_samples: number;
  // Which class slice `values`/`base_value` refer to (-1 = single-output).
  class_index?: number;
  n_classes?: number;
  class_labels?: string[] | null;
  status: string;
}

export async function getShapData(body: ShapDataRequest): Promise<ShapData> {
  return request<ShapData>('/api/v1/analysis/shap/data', {
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
  analysis_snapshot_id: string;
  analysis_revision: number;
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

export interface AnalysisConfig {
  trial_number: number | null;
  use_proba: boolean;
  subset_size: number;
  class_index: number;
  sample_index: number;
}

export interface AnalysisSnapshotPayload {
  feature_importance: FeatureImportance[] | null;
  plots: Record<string, string>;
  study_plots: Record<string, PlotlyFigureJSON | null> | null;
  metrics: Record<string, any> | null;
  confusion_matrix_plot: string | null;
  roc_auc: number | null;
  average_precision: number | null;
  fairness: FairnessResponse | null;
  curves_data: CurvesData | null;
  confusion_data: ConfusionMatrixData | null;
  importance_data: FeatureImportanceData | null;
  shap_data: ShapData | null;
  warnings: Record<string, string>;
}

export interface AnalysisSnapshotSummary {
  id: string;
  optimization_id: string;
  revision: number;
  config: AnalysisConfig;
  completed_at: string;
}

export interface AnalysisSnapshot extends AnalysisSnapshotSummary {
  payload: AnalysisSnapshotPayload;
}

export interface AnalysisJob {
  id: string;
  snapshot_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  current_section?: string | null;
  error?: string | null;
  revision?: number;
}

export async function startAnalysisJob(body: {
  optimization_id: string;
  trial_number?: number;
  use_proba: boolean;
  subset_size: number;
  class_index: number;
  sample_index: number;
}): Promise<AnalysisJob> {
  return request<AnalysisJob>('/api/v1/analysis/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function getAnalysisJob(id: string): Promise<AnalysisJob> {
  return request<AnalysisJob>(`/api/v1/analysis/jobs/${id}`);
}

export async function getAnalysisSnapshot(id: string): Promise<AnalysisSnapshot> {
  return request<AnalysisSnapshot>(`/api/v1/analysis/snapshots/${id}`);
}

export async function listAnalysisSnapshots(
  optimizationId: string
): Promise<AnalysisSnapshotSummary[]> {
  const result = await request<{ snapshots: AnalysisSnapshotSummary[] }>(
    `/api/v1/analysis/snapshots?optimization_id=${encodeURIComponent(optimizationId)}`
  );
  return result.snapshots;
}

export interface PersistedReport {
  id: string;
  snapshot_id: string;
  snapshot_revision: number;
  status: 'running' | 'completed' | 'failed';
  markdown: string | null;
  provider: string;
  model_name: string;
  created_at: string;
}

export async function listSnapshotReports(snapshotId: string): Promise<PersistedReport[]> {
  const result = await request<{ reports: PersistedReport[] }>(
    `/api/v1/analysis/snapshots/${snapshotId}/reports`
  );
  return result.reports;
}

export async function updateSnapshotFairness(
  snapshotId: string,
  body: { sensitive_feature?: string; mitigate?: boolean; constraint?: string }
): Promise<{ fairness: FairnessResponse; snapshot_id: string; revision: number }> {
  return request(`/api/v1/analysis/snapshots/${snapshotId}/fairness`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
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

export async function getHealth(): Promise<{ status: string }> {
  return request<{ status: string }>('/api/v1/health');
}

// ---------------------------------------------------------------------------
// Auth (server-side Auth0 flow on the backend; see /auth routes)
// ---------------------------------------------------------------------------

export interface AuthUser {
  sub: string;
  name?: string;
  nickname?: string;
  email?: string;
  picture?: string;
  [claim: string]: unknown;
}

export interface AuthProfile {
  user: AuthUser | null;
  auth_enabled: boolean;
}

/** Resolves to { user: null } when logged out or when auth is disabled. */
export async function getAuthProfile(): Promise<AuthProfile> {
  try {
    return await request<AuthProfile>('/auth/profile');
  } catch (error) {
    if (error instanceof UnauthorizedError) {
      return { user: null, auth_enabled: true };
    }
    throw error;
  }
}

/** Full-page navigation targets — the backend drives the Auth0 redirects. */
export function loginUrl(screenHint?: 'signup'): string {
  const returnTo = typeof window !== 'undefined' ? window.location.href : '/';
  const params = new URLSearchParams({ returnTo });
  if (screenHint) params.set('screen_hint', screenHint);
  return `${API_BASE_URL}/auth/login?${params}`;
}

export function logoutUrl(): string {
  // Origin only — Auth0 matches return_to against Allowed Logout URLs exactly.
  const returnTo = typeof window !== 'undefined' ? window.location.origin : '/';
  return `${API_BASE_URL}/auth/logout?${new URLSearchParams({ returnTo })}`;
}
