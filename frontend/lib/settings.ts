/**
 * Lightweight localStorage-backed store for LLM provider API keys.
 * Shared by the Settings page and the Optimizer report step.
 */

export type ApiKeyProvider = 'openai' | 'anthropic' | 'google';

const STORAGE_KEY = 'quoptuna.apiKeys';

export type ApiKeys = Record<ApiKeyProvider, string>;

const EMPTY_KEYS: ApiKeys = { openai: '', anthropic: '', google: '' };

export function loadApiKeys(): ApiKeys {
  if (typeof window === 'undefined') {
    return { ...EMPTY_KEYS };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...EMPTY_KEYS };
    return { ...EMPTY_KEYS, ...(JSON.parse(raw) as Partial<ApiKeys>) };
  } catch {
    return { ...EMPTY_KEYS };
  }
}

export function saveApiKeys(keys: ApiKeys): void {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(keys));
}
