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
    trials: Array<{ trial: number; value: number; params: Record<string, any>; state?: string }>;
    selectedTrial: number | null;
  };
  analysis: {
    featureImportance: Array<{ feature: string; importance: number }> | null;
    // SHAP plots plus reserved keys: rocCurve, prCurve, optimizationHistory, paramImportances.
    plots: Record<string, string>;
    metrics: Record<string, any> | null;
    confusionMatrixPlot: string | null;
    rocAuc: number | null;
    averagePrecision: number | null;
  };
  report: {
    markdown: string | null;
  };
}

export const initialWorkflowData: WorkflowData = {
  dataset: null,
  features: { selectedFeatures: [], targetColumn: null, labelMapping: { neg: null, pos: null } },
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
    metrics: null,
    confusionMatrixPlot: null,
    rocAuc: null,
    averagePrecision: null,
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
