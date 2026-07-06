/**
 * Plain (non-secret) app-wide settings stored in localStorage.
 *
 * Currently holds the Optuna storage database name, which applies to all
 * optimization runs and is configured once on the Settings page rather than
 * per-run in the wizard.
 */

const STORAGE_KEY = 'quoptuna.appSettings.v1';

export const DEFAULT_DATABASE_NAME = 'results';

export interface AppSettings {
  databaseName: string;
}

const DEFAULTS: AppSettings = { databaseName: DEFAULT_DATABASE_NAME };

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
