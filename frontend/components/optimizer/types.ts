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
  // Multiclass runs audit "favorable class vs rest"; mitigation is
  // binary-only, so it stays null for multiclass.
  task_type?: 'binary' | 'multiclass';
  favorable_class?: string | null;
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
    // K>2 targets: the class treated as the favorable outcome for fairness
    // auditing and report framing (favorable vs rest). Unused for binary.
    favorableClass: string | null;
    // Protected attribute used for fairness auditing (may be a column that is
    // not among the selected model features).
    sensitiveFeature: string | null;
    // How categorical feature columns are encoded server-side.
    categoricalEncoding: 'ordinal' | 'onehot';
  };
  configuration: {
    studyName: string;
    numTrials: number;
    sampler: 'tpe' | 'random' | 'grid';
    pruner: 'none' | 'asha' | 'hyperband';
    // Fairness-aware search: constrained (TPE feasibility constraint) or
    // multi-objective (F1 vs disparity Pareto front). Requires a protected
    // attribute selected in the Features step.
    fairnessMode: 'off' | 'constrained' | 'multi_objective';
    fairnessMetric:
      | 'equal_opportunity_difference'
      | 'disparate_impact'
      | 'demographic_parity_difference';
    // Difference metrics: feasible when disparity <= threshold (default 0.1).
    // Disparate impact: feasible when ratio >= threshold (default 0.8).
    fairnessThreshold: number | null;
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
      // Multi-objective runs: [f1, disparity]; null otherwise.
      values?: number[] | null;
      params: Record<string, any>;
      state?: string;
      user_attrs?: Record<string, any>;
    }>;
    selectedTrial: number | null;
    // Pareto front of a multi-objective run (values = [f1, disparity]).
    paretoTrials: Array<{
      trial: number;
      values: number[];
      params: Record<string, any>;
    }> | null;
  };
  analysis: {
    snapshotId: string | null;
    snapshotRevision: number | null;
    status: 'idle' | 'pending' | 'running' | 'completed' | 'failed';
    config: {
      trialNumber: number | null;
      useProba: boolean;
      subsetSize: number;
      classIndex: number;
      sampleIndex: number;
    } | null;
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
    favorableClass: null,
    sensitiveFeature: null,
    categoricalEncoding: 'ordinal',
  },
  configuration: {
    studyName: 'my-optimization-study',
    numTrials: 50,
    sampler: 'tpe',
    pruner: 'asha',
    fairnessMode: 'off',
    fairnessMetric: 'equal_opportunity_difference',
    fairnessThreshold: null,
  },
  optimization: {
    executionId: null,
    status: 'idle',
    bestValue: null,
    bestParams: null,
    trials: [],
    selectedTrial: null,
    paretoTrials: null,
  },
  analysis: {
    snapshotId: null,
    snapshotRevision: null,
    status: 'idle',
    config: null,
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

// Classical model names, mirroring the backend registry in
// src/quoptuna/backend/models.py — everything else there is a quantum model.
// An explicit list (not substring hints) because e.g. TreeTensorClassifier is
// quantum despite containing "tree".
const CLASSICAL_MODELS = new Set(
  ['SVC', 'SVClinear', 'MLPClassifier', 'Perceptron', 'ConvolutionalNeuralNetwork'].map((m) =>
    m.toLowerCase()
  )
);

export function isClassicalModel(modelType: string | undefined): boolean {
  if (!modelType) return false;
  return CLASSICAL_MODELS.has(modelType.toLowerCase());
}
