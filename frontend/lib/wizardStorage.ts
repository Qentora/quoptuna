/**
 * localStorage persistence for the optimizer wizard, so a browser refresh (or
 * navigating away) never loses the in-progress workflow. Runs themselves are
 * durably persisted by the backend; this only covers the client-side step
 * state between saves.
 */

import type { WorkflowData } from '@/components/optimizer/types';

const STORAGE_KEY = 'quoptuna.wizard.v1';

export interface WizardState {
  currentStep: number;
  completedSteps: number[];
  workflowData: WorkflowData;
}

export function saveWizardState(state: WizardState): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // Storage full or unavailable — persistence is best-effort.
  }
}

export function loadWizardState(): WizardState | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as WizardState;
    if (!parsed || typeof parsed.currentStep !== 'number' || !parsed.workflowData) return null;
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
