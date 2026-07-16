/**
 * Plain (non-secret) app-wide settings stored in localStorage.
 *
 * Currently holds the Optuna storage database name, which applies to all
 * optimization runs and is configured once on the Settings page rather than
 * per-run in the wizard.
 */

const STORAGE_KEY = 'quoptuna.appSettings.v1';

export const DEFAULT_DATABASE_NAME = 'results';

// Optimizer performance defaults. Deliberately faster than the model-class
// defaults (max_vmap 1 / max_steps 10000 / convergence_interval 200): on the
// small tabular datasets QuOptuna targets, vectorized circuits and a tighter
// step budget cut trial wall-clock severalfold.
export const DEFAULT_MAX_VMAP = 32;
export const DEFAULT_MAX_STEPS = 2000;
export const DEFAULT_CONVERGENCE_INTERVAL = 100;

// PennyLane simulator for quantum models. lightning.qubit (C++ state vector)
// is usually faster; the backend falls back to default.qubit if unavailable.
export const DEV_TYPE_OPTIONS = ['default.qubit', 'lightning.qubit'] as const;
export type DevType = (typeof DEV_TYPE_OPTIONS)[number];
export const DEFAULT_DEV_TYPE: DevType = 'default.qubit';

// train() requires batch_size % max_vmap == 0; batch size is 32 in the
// default search space, so valid widths are its divisors.
export const MAX_VMAP_OPTIONS = [1, 2, 4, 8, 16, 32] as const;

export interface AppSettings {
  databaseName: string;
  maxVmap: number;
  maxSteps: number;
  convergenceInterval: number;
  devType: DevType;
}

const DEFAULTS: AppSettings = {
  databaseName: DEFAULT_DATABASE_NAME,
  maxVmap: DEFAULT_MAX_VMAP,
  maxSteps: DEFAULT_MAX_STEPS,
  convergenceInterval: DEFAULT_CONVERGENCE_INTERVAL,
  devType: DEFAULT_DEV_TYPE,
};

export function loadAppSettings(): AppSettings {
  if (typeof window === 'undefined') return { ...DEFAULTS };
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULTS };
    return { ...DEFAULTS, ...(JSON.parse(raw) as Partial<AppSettings>) };
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveAppSettings(settings: AppSettings): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

/** Database name for new runs; trailing ".db" is stripped to avoid name.db.db files. */
export function getDatabaseName(): string {
  const name = loadAppSettings().databaseName.trim() || DEFAULT_DATABASE_NAME;
  return name.replace(/\.db$/i, '') || DEFAULT_DATABASE_NAME;
}

function sanitizeInt(value: unknown, fallback: number): number {
  const n = Math.floor(Number(value));
  return Number.isFinite(n) && n >= 1 ? n : fallback;
}

/** Optimizer performance knobs for new runs, sanitized to safe values. */
export function getOptimizerSettings(): {
  maxVmap: number;
  maxSteps: number;
  convergenceInterval: number;
  devType: DevType;
} {
  const s = loadAppSettings();
  const rawVmap = sanitizeInt(s.maxVmap, DEFAULT_MAX_VMAP);
  // Snap to the largest valid divisor of the batch size (32) not above the
  // stored value, so a hand-edited localStorage value can't crash train().
  const maxVmap = [...MAX_VMAP_OPTIONS].reverse().find((v) => v <= rawVmap) ?? 1;
  const devType = DEV_TYPE_OPTIONS.includes(s.devType) ? s.devType : DEFAULT_DEV_TYPE;
  return {
    maxVmap,
    maxSteps: sanitizeInt(s.maxSteps, DEFAULT_MAX_STEPS),
    convergenceInterval: sanitizeInt(s.convergenceInterval, DEFAULT_CONVERGENCE_INTERVAL),
    devType,
  };
}
