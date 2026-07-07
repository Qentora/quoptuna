export interface PlotlyFigureJSON {
  data: Array<Record<string, any>>;
  layout: Record<string, any>;
}

export interface FairnessMetrics {
  by_group: Record<string, Record<string, number>>;
  overall: Record<string, number>;
  disparities: Record<string, number>;
}

export interface FairnessResult {
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

export interface WorkflowData {
  dataset: {
    id: string;
    name: string;
    source: 'upload' | 'uci';
    rows: number;
    columns: string[];
  } | null;
  features: {
    selectedFeatures: string[];
    targetColumn: string | null;
    labelMapping: { neg: string | number | null; pos: string | number | null };
    // Protected attribute used for fairness auditing (may be a column that is
    // not among the selected model features).
    sensitiveFeature: string | null;
    // How categorical feature columns are encoded server-side.
    categoricalEncoding: 'ordinal' | 'onehot';
  };
  configuration: {
    studyName: string;
    numTrials: number;
  };
  optimization: {
    executionId: string | null;
    status: 'idle' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'interrupted';
    bestValue: number | null;
    bestParams: Record<string, any> | null;
    trials: Array<{
      trial: number;
      // null for FAILED trials (user_attrs.error records why).
      value: number | null;
      params: Record<string, any>;
      state?: string;
      user_attrs?: Record<string, any>;
    }>;
    selectedTrial: number | null;
  };
  analysis: {
    featureImportance: Array<{ feature: string; importance: number }> | null;
    // SHAP plots plus reserved keys: rocCurve, prCurve.
    plots: Record<string, string>;
    // Interactive Optuna study figures as Plotly JSON, keyed by plot name.
    studyPlots: Record<string, PlotlyFigureJSON | null> | null;
    metrics: Record<string, any> | null;
    confusionMatrixPlot: string | null;
    rocAuc: number | null;
    averagePrecision: number | null;
    fairness: FairnessResult | null;
  };
  report: {
    markdown: string | null;
  };
}

export const initialWorkflowData: WorkflowData = {
  dataset: null,
  features: {
    selectedFeatures: [],
    targetColumn: null,
    labelMapping: { neg: null, pos: null },
    sensitiveFeature: null,
    categoricalEncoding: 'ordinal',
  },
  configuration: {
    studyName: 'my-optimization-study',
    numTrials: 50,
  },
  optimization: {
    executionId: null,
    status: 'idle',
    bestValue: null,
    bestParams: null,
    trials: [],
    selectedTrial: null,
  },
  analysis: {
    featureImportance: null,
    plots: {},
    studyPlots: null,
    metrics: null,
    confusionMatrixPlot: null,
    rocAuc: null,
    averagePrecision: null,
    fairness: null,
  },
  report: { markdown: null },
};

// Heuristic to label a trial's model_type as quantum or classical for display.
const CLASSICAL_HINTS = [
  'svc',
  'svm',
  'mlp',
  'perceptron',
  'forest',
  'boosting',
  'adaboost',
  'logistic',
  'tree',
  'knn',
  'gaussian',
  'naive',
  'bayes',
];

export function isClassicalModel(modelType: string | undefined): boolean {
  if (!modelType) return false;
  const lower = modelType.toLowerCase();
  return CLASSICAL_HINTS.some((hint) => lower.includes(hint));
}
