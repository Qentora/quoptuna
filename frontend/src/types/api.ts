export interface Dataset {
  id: string;
  name: string;
  description?: string;
  source: 'upload' | 'uci';
  rows: number;
  columns: number;
  features: string[];
  target?: string;
  createdAt: string;
  filePath?: string;
}

export interface ModelConfig {
  name: string;
  type: 'quantum' | 'classical';
  category: string;
  hyperparameters: Record<string, any>;
}

export interface OptimizationConfig {
  studyName: string;
  nTrials: number;
  timeout?: number;
  sampler: 'TPE' | 'Random' | 'GridSearch';
  direction: 'minimize' | 'maximize';
}

export interface OptimizationTrial {
  number: number;
  params: Record<string, any>;
  value: number;
  state: 'running' | 'complete' | 'pruned' | 'failed';
  datetime_start: string;
  datetime_complete?: string;
}

export interface OptimizationResult {
  id: string;
  studyName: string;
  bestValue: number;
  bestParams: Record<string, any>;
  bestTrial: number;
  trials: OptimizationTrial[];
  status: 'running' | 'completed' | 'failed';
}

export interface SHAPAnalysis {
  id: string;
  plots: {
    bar?: string;
    beeswarm?: string;
    violin?: string;
    heatmap?: string;
    waterfall?: string;
  };
  values: number[][];
  featureNames: string[];
  report?: string;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  error?: string;
}
