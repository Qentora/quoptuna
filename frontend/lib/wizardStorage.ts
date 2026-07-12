/**
 * localStorage persistence for the optimizer wizard, so a browser refresh (or
 * navigating away) never loses the in-progress workflow. Runs themselves are
 * durably persisted by the backend; this only covers the client-side step
 * state between saves.
 */

import { type WorkflowData, initialWorkflowData } from '@/components/optimizer/types';

const STORAGE_KEY = 'quoptuna.wizard.v1';

export interface WizardState {
  currentStep: number;
  completedSteps: number[];
  workflowData: WorkflowData;
}

/**
 * Persist only the server-side snapshot identity and lightweight workflow
 * state. Analysis values and artifacts are restored from the backend.
 */
function stripHeavyData(state: WizardState): WizardState {
  return {
    ...state,
    workflowData: {
      ...state.workflowData,
      analysis: {
        ...state.workflowData.analysis,
        plots: {},
        studyPlots: null,
        confusionMatrixPlot: null,
        fairness: null,
        featureImportance: null,
        metrics: null,
      },
    },
  };
}

export function saveWizardState(state: WizardState): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stripHeavyData(state)));
  } catch {
    // Quota exceeded (analysis plots are megabytes of base64). Losing the
    // WHOLE state silently is far worse than losing the plots — retry with
    // the heavy payloads stripped.
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(stripHeavyData(state)));
    } catch {
      // Storage genuinely unavailable — persistence is best-effort.
    }
  }
}

export function loadWizardState(): WizardState | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as WizardState;
    if (!parsed || typeof parsed.currentStep !== 'number' || !parsed.workflowData) return null;
    // States saved before new configuration fields existed (e.g. sampler/
    // pruner) rehydrate with defaults filled in.
    parsed.workflowData.configuration = {
      ...initialWorkflowData.configuration,
      ...parsed.workflowData.configuration,
    };
    parsed.workflowData.optimization = {
      ...initialWorkflowData.optimization,
      ...parsed.workflowData.optimization,
    };
    parsed.workflowData.analysis = {
      ...initialWorkflowData.analysis,
      ...parsed.workflowData.analysis,
    };
    return parsed;
  } catch {
    window.localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

export function clearWizardState(): void {
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(STORAGE_KEY);
}
