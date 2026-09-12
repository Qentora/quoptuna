/**
 * Report-generation settings: which evidence reaches the agents, and the agent
 * prompts themselves.
 *
 * Stored in localStorage (non-secret) alongside `appSettings`, and edited on the
 * Settings page. Prompt fields are empty by default, which means "use the
 * backend default" — the backend serves those defaults from
 * `GET /api/v1/analysis/report-prompts` so the shipped prompt and the one the
 * user sees are always the same text.
 *
 * `REPORT_SETTING_DOCS` carries the inline guidance rendered next to each field.
 * It intentionally duplicates the backend's `REPORT_PROMPT_SETTINGS` wording for
 * the offline/first-paint case; the Settings page prefers the served copy when
 * it is available so there is one source of truth at runtime.
 */

const STORAGE_KEY = 'quoptuna.reportSettings.v1';

export const DEFAULT_MAX_TRIAL_ROWS = 40;
export const MAX_TRIAL_ROWS_LIMIT = 500;

export interface ReportSettings {
  /** Empty string = use the backend's default analyst prompt. */
  analystInstructions: string;
  /** Empty string = use the backend's default reviewer prompt. */
  reviewerInstructions: string;
  enableReview: boolean;
  includeFairness: boolean;
  includeFairnessSearch: boolean;
  includePareto: boolean;
  includeShapDetail: boolean;
  includeTrialHistory: boolean;
  includeStudyPlots: boolean;
  attachFigures: boolean;
  maxTrialRows: number;
}

export const DEFAULT_REPORT_SETTINGS: ReportSettings = {
  analystInstructions: '',
  reviewerInstructions: '',
  enableReview: true,
  includeFairness: true,
  includeFairnessSearch: true,
  includePareto: true,
  includeShapDetail: true,
  includeTrialHistory: true,
  includeStudyPlots: true,
  attachFigures: true,
  maxTrialRows: DEFAULT_MAX_TRIAL_ROWS,
};

/** Toggle keys, in the order the Settings page renders them. */
export const REPORT_TOGGLES = [
  'enableReview',
  'includeFairness',
  'includeFairnessSearch',
  'includePareto',
  'includeShapDetail',
  'includeTrialHistory',
  'includeStudyPlots',
  'attachFigures',
] as const;

export type ReportToggle = (typeof REPORT_TOGGLES)[number];

export interface SettingDoc {
  label: string;
  /** What the setting does. */
  description: string;
  /** When to turn it on, and what goes wrong if you don't. */
  when: string;
}

export const REPORT_SETTING_DOCS: Record<ReportToggle | 'maxTrialRows', SettingDoc> = {
  enableReview: {
    label: 'Run the reviewer pass',
    description:
      'Sends the draft through a second agent that re-checks every number against the evidence bundle before it is saved. Roughly doubles token cost and latency.',
    when: 'Keep on for anything you will publish, cite, or hand to a reviewer — it is the step that catches invented numbers and broken tables. Turn off only for quick iteration while tuning your own prompt.',
  },
  includeFairness: {
    label: 'Include the fairness audit',
    description:
      'Adds per-group metrics, the disparity summary, the group plots and any ThresholdOptimizer mitigation, and enables the Fairness Audit section.',
    when: 'Turn on whenever a protected attribute was selected — it is the only way the report can discuss group harms, and without it the agent is explicitly told not to speculate about them. Has no effect on runs without an audit.',
  },
  includeFairnessSearch: {
    label: 'Include fairness-aware search details',
    description:
      'Adds the search-time fairness configuration: mode (constrained or multi-objective), disparity metric, feasibility threshold and direction, per-trial disparities and feasible-trial counts.',
    when: 'Turn on whenever the run used a fairness mode other than “off”. Without it the report presents a fairness-constrained result as an unconstrained one, misstating both the objective and the accuracy trade-off. Irrelevant when fairness mode was “off”.',
  },
  includePareto: {
    label: 'Include the Pareto front table',
    description:
      'Adds every Pareto-optimal trial (F1 vs disparity, with hyperparameters) plus the knee point, and asks for the trade-off table and its interpretation.',
    when: 'Turn on for every multi-objective run: the front is the result, and the single reported model is only one point on it. Required if the report must justify why that point was chosen. Adds nothing to single-objective runs.',
  },
  includeShapDetail: {
    label: 'Include per-feature SHAP statistics',
    description:
      'Adds mean |SHAP|, signed mean SHAP and feature value ranges per feature on top of the plots.',
    when: 'Keep on for explainability or audit deliverables — it lets the agent quantify feature effects and their direction instead of describing the picture. Turn off for a short summary, or on very wide feature sets.',
  },
  includeTrialHistory: {
    label: 'Include the per-trial history',
    description:
      'Adds the trial table (state, objective values, hyperparameters, training time) up to the row cap, plus per-model-family aggregates.',
    when: 'Keep on to justify the chosen configuration and discuss search efficiency or pruning. Turn off for large budgets when only the outcome matters — or lower the row cap instead.',
  },
  includeStudyPlots: {
    label: 'Include study diagnostics',
    description:
      'Adds numeric summaries extracted from the Optuna study figures — hyperparameter importances and the best-value-so-far trace — and lists those figures in the manifest.',
    when: 'Turn on when the report must comment on convergence or on which hyperparameters mattered. Turn off when the budget was too small for importances to be meaningful (under ~10 completed trials).',
  },
  attachFigures: {
    label: 'Send the rendered figures to the model',
    description:
      'Uploads the SHAP, curve, confusion-matrix and fairness images alongside the numeric evidence so the agent can describe them.',
    when: 'Keep on for any report that references figures. Turn off with a text-only or cheaper model, or when the provider rejects image input — the numeric evidence still supports every table, and figure references are dropped rather than guessed.',
  },
  maxTrialRows: {
    label: 'Trial-table row cap',
    description:
      'Maximum per-trial rows sent to the agent. Trials are ranked by objective value, so the cap keeps the best ones; state counts and aggregates still cover every trial.',
    when: 'Lower it (10–20) for large budgets or small context windows; raise it when the report must enumerate the whole search.',
  },
};

export const PROMPT_DOCS: Record<'analystInstructions' | 'reviewerInstructions', SettingDoc> = {
  analystInstructions: {
    label: 'Analyst prompt',
    description:
      'System prompt for the drafting agent. It defines the report skeleton, the interpretation standard, and the markdown contract.',
    when: 'Edit when you need a different document shape — a journal results section, a model card, a regulator-facing summary. Keep the markdown contract and the grounding rule; removing them is what produces unstructured, hallucinated reports. Leave empty to use the QuOptuna default.',
  },
  reviewerInstructions: {
    label: 'Reviewer prompt',
    description:
      'System prompt for the second-pass editor that re-checks every number against the evidence bundle and repairs the markdown.',
    when: 'Edit to add house style or a domain checklist. Weakening the grounding check is the fastest way to let fabricated metrics through. Leave empty for the default.',
  },
};

function sanitizeRows(value: unknown): number {
  const n = Math.floor(Number(value));
  if (!Number.isFinite(n) || n < 0) return DEFAULT_MAX_TRIAL_ROWS;
  return Math.min(n, MAX_TRIAL_ROWS_LIMIT);
}

export function loadReportSettings(): ReportSettings {
  if (typeof window === 'undefined') return { ...DEFAULT_REPORT_SETTINGS };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_REPORT_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<ReportSettings>;
    return {
      ...DEFAULT_REPORT_SETTINGS,
      ...parsed,
      maxTrialRows: sanitizeRows(parsed.maxTrialRows),
    };
  } catch {
    return { ...DEFAULT_REPORT_SETTINGS };
  }
}

export function saveReportSettings(settings: ReportSettings): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ ...settings, maxTrialRows: sanitizeRows(settings.maxTrialRows) })
  );
}

/** Shape the stored settings into the report request body the API expects. */
export function reportSettingsPayload(settings: ReportSettings) {
  return {
    analyst_instructions: settings.analystInstructions.trim() || undefined,
    reviewer_instructions: settings.reviewerInstructions.trim() || undefined,
    enable_review: settings.enableReview,
    include_fairness: settings.includeFairness,
    include_fairness_search: settings.includeFairnessSearch,
    include_pareto: settings.includePareto,
    include_shap_detail: settings.includeShapDetail,
    include_trial_history: settings.includeTrialHistory,
    include_study_plots: settings.includeStudyPlots,
    attach_figures: settings.attachFigures,
    max_trial_rows: sanitizeRows(settings.maxTrialRows),
  };
}
