/**
 * Encrypted-at-rest store for LLM provider API keys.
 *
 * Shared by the Settings page and the Optimizer report step. Keys are
 * AES-GCM encrypted before being written to localStorage; the wrapping key is
 * a non-extractable CryptoKey kept in IndexedDB, so the secrets never touch
 * localStorage (or anywhere else) as clear text. This is best-effort browser
 * hardening, not a substitute for server-side secret management.
 */

export type ApiKeyProvider = 'openai' | 'anthropic' | 'google';

export type ApiKeys = Record<ApiKeyProvider, string>;

const STORAGE_KEY = 'quoptuna.apiKeys';
const DB_NAME = 'quoptuna.secrets';
const DB_STORE = 'crypto';
const CRYPTO_KEY_ID = 'apiKeysKey';
const IV_LENGTH = 12;

const EMPTY_KEYS: ApiKeys = { openai: '', anthropic: '', google: '' };

function isBrowserCryptoAvailable(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.indexedDB !== 'undefined' &&
    typeof window.crypto !== 'undefined' &&
    typeof window.crypto.subtle !== 'undefined'
  );
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(DB_STORE)) {
        request.result.createObjectStore(DB_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

function idbGet(id: string): Promise<unknown> {
  return openDb().then(
    (db) =>
      new Promise<unknown>((resolve, reject) => {
        const tx = db.transaction(DB_STORE, 'readonly');
        const request = tx.objectStore(DB_STORE).get(id);
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
        tx.oncomplete = () => db.close();
      })
  );
}

function idbSet(id: string, value: unknown): Promise<void> {
  return openDb().then(
    (db) =>
      new Promise<void>((resolve, reject) => {
        const tx = db.transaction(DB_STORE, 'readwrite');
        tx.objectStore(DB_STORE).put(value, id);
        tx.oncomplete = () => {
          db.close();
          resolve();
        };
        tx.onerror = () => reject(tx.error);
      })
  );
}

let cachedKey: CryptoKey | null = null;

async function getCryptoKey(): Promise<CryptoKey> {
  if (cachedKey) return cachedKey;
  const existing = await idbGet(CRYPTO_KEY_ID);
  if (existing instanceof CryptoKey) {
    cachedKey = existing;
    return existing;
  }
  const key = await window.crypto.subtle.generateKey({ name: 'AES-GCM', length: 256 }, false, [
    'encrypt',
    'decrypt',
  ]);
  await idbSet(CRYPTO_KEY_ID, key);
  cachedKey = key;
  return key;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

export async function loadApiKeys(): Promise<ApiKeys> {
  if (!isBrowserCryptoAvailable()) {
    return { ...EMPTY_KEYS };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...EMPTY_KEYS };
    const key = await getCryptoKey();
    const payload = base64ToBytes(raw);
    const iv = payload.slice(0, IV_LENGTH);
    const ciphertext = payload.slice(IV_LENGTH);
    const plaintext = await window.crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, ciphertext);
    const parsed = JSON.parse(new TextDecoder().decode(plaintext)) as Partial<ApiKeys>;
    return { ...EMPTY_KEYS, ...parsed };
  } catch {
    return { ...EMPTY_KEYS };
  }
}

export async function saveApiKeys(keys: ApiKeys): Promise<void> {
  if (!isBrowserCryptoAvailable()) return;
  const key = await getCryptoKey();
  const iv = window.crypto.getRandomValues(new Uint8Array(IV_LENGTH));
  const plaintext = new TextEncoder().encode(JSON.stringify(keys));
  const ciphertext = new Uint8Array(
    await window.crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, plaintext)
  );
  const payload = new Uint8Array(iv.length + ciphertext.length);
  payload.set(iv);
  payload.set(ciphertext, iv.length);
  window.localStorage.setItem(STORAGE_KEY, bytesToBase64(payload));
}
