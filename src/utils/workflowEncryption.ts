const ENCRYPTION_FORMAT = 'comfyui-mobile-encrypted-workflow';
const ENCRYPTION_VERSION = 1;
const PBKDF2_ITERATIONS = 250_000;

export interface EncryptedWorkflowEnvelope {
  encrypted: true;
  format: typeof ENCRYPTION_FORMAT;
  version: typeof ENCRYPTION_VERSION;
  kdf: 'PBKDF2-SHA256';
  cipher: 'AES-256-GCM';
  iterations: number;
  salt: string;
  iv: string;
  data: string;
}

let memorySecret: string | null = null;

function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function base64ToBytes(value: string): Uint8Array {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function getTokenFromUrl(): string | null {
  try {
    const token = new URLSearchParams(window.location.search).get('token');
    return token && token.trim() ? token.trim() : null;
  } catch {
    return null;
  }
}

async function getServerWorkflowKey(): Promise<string | null> {
  try {
    const response = await fetch('/workflow-key', {
      cache: 'no-store',
      credentials: 'same-origin',
    });
    if (!response.ok) return null;
    const data = await response.json();
    const secret = typeof data?.key === 'string' ? data.key.trim() : '';
    return secret || null;
  } catch {
    return null;
  }
}

function getOrCreateBrowserWorkflowKey(): string | null {
  try {
    const storageKey = 'comfyui-mobile-workflow-encryption-key';
    const existing = window.localStorage.getItem(storageKey)?.trim();
    if (existing) return existing;

    const bytes = crypto.getRandomValues(new Uint8Array(32));
    const generated = bytesToBase64(bytes);
    window.localStorage.setItem(storageKey, generated);
    return generated;
  } catch {
    return null;
  }
}

async function getSecret(): Promise<string> {
  if (memorySecret) return memorySecret;

  const urlToken = getTokenFromUrl();
  if (urlToken) {
    memorySecret = urlToken;
    return memorySecret;
  }

  const serverKey = await getServerWorkflowKey();
  if (serverKey) {
    memorySecret = serverKey;
    return memorySecret;
  }

  const browserKey = getOrCreateBrowserWorkflowKey();
  if (browserKey) {
    memorySecret = browserKey;
    return memorySecret;
  }

  throw new Error('Workflow encryption key unavailable');
}

function asBufferSource(bytes: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer;
}

async function deriveKey(secret: string, salt: Uint8Array): Promise<CryptoKey> {
  const material = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    'PBKDF2',
    false,
    ['deriveKey'],
  );
  return crypto.subtle.deriveKey(
    { name: 'PBKDF2', salt: asBufferSource(salt), iterations: PBKDF2_ITERATIONS, hash: 'SHA-256' },
    material,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt', 'decrypt'],
  );
}

export function isEncryptedWorkflow(value: unknown): value is EncryptedWorkflowEnvelope {
  if (!value || typeof value !== 'object') return false;
  const record = value as Record<string, unknown>;
  return record.encrypted === true && record.format === ENCRYPTION_FORMAT && record.version === ENCRYPTION_VERSION;
}

export async function encryptWorkflowForStorage(workflow: unknown): Promise<EncryptedWorkflowEnvelope> {
  if (!crypto?.subtle) throw new Error('WebCrypto is required for workflow encryption');
  const salt = crypto.getRandomValues(new Uint8Array(16));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(await getSecret(), salt);
  const plaintext = new TextEncoder().encode(JSON.stringify(workflow));
  const encrypted = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv: asBufferSource(iv) }, key, plaintext));
  return {
    encrypted: true,
    format: ENCRYPTION_FORMAT,
    version: ENCRYPTION_VERSION,
    kdf: 'PBKDF2-SHA256',
    cipher: 'AES-256-GCM',
    iterations: PBKDF2_ITERATIONS,
    salt: bytesToBase64(salt),
    iv: bytesToBase64(iv),
    data: bytesToBase64(encrypted),
  };
}

export async function decryptWorkflowFromStorage<T = unknown>(stored: unknown): Promise<T> {
  if (!isEncryptedWorkflow(stored)) return stored as T;
  if (!crypto?.subtle) throw new Error('WebCrypto is required for workflow decryption');
  const salt = base64ToBytes(stored.salt);
  const iv = base64ToBytes(stored.iv);
  const ciphertext = base64ToBytes(stored.data);
  const key = await deriveKey(await getSecret(), salt);
  try {
    const plaintext = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: asBufferSource(iv) }, key, asBufferSource(ciphertext));
    return JSON.parse(new TextDecoder().decode(plaintext)) as T;
  } catch {
    throw new Error('Could not decrypt workflow. Check that you entered the same login token / unlock key used when it was saved.');
  }
}

export function clearWorkflowEncryptionKey(): void {
  memorySecret = null;
}
