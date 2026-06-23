import type { QueueInfo, History, HistoryOutputImage } from '../types';
import type { QueueWorkflowDiff } from '@/utils/workflowDiff';

function stringInput(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null;
}

function numberInput(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

type ApiPromptNode = { class_type?: unknown; inputs?: Record<string, unknown> };

type NativeMlxQueueCandidate = {
  imagePath: string;
  prompt: string;
  negativePrompt?: string;
  steps: number;
  seed?: number;
  width?: number;
  height?: number;
  guidance?: number;
};

function nativeApiUrl(path: string): string {
  if (typeof window !== 'undefined' && window.location?.origin) {
    return new URL(path, window.location.origin).toString();
  }
  return path;
}

function nativeApiPath(...parts: string[]): string {
  // Do not write literal "/api/generate" or "/api/history" here. The Z-Image
  // wrapper rewrites those literals inside served JS to "/comfy/api/..." for
  // ComfyUI compatibility, which bypasses the warmed native BigLove Klein route.
  return ['', 'api', ...parts].join('/');
}

function asPromptNode(value: unknown): ApiPromptNode | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const node = value as ApiPromptNode;
  return node.inputs && typeof node.inputs === 'object' ? node : null;
}

function getLinkedNode(nodesById: Map<string, ApiPromptNode>, value: unknown): ApiPromptNode | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const nodeId = String(value[0]);
  return nodesById.get(nodeId) ?? null;
}

function resolveSamplerLoadImage(sampler: ApiPromptNode | undefined, nodesById: Map<string, ApiPromptNode>): ApiPromptNode | undefined {
  if (!sampler?.inputs) return undefined;
  const seen = new Set<ApiPromptNode>();
  const visit = (node: ApiPromptNode | null): ApiPromptNode | undefined => {
    if (!node || seen.has(node)) return undefined;
    seen.add(node);
    if (node.class_type === 'LoadImage') return node;
    const inputs = node.inputs ?? {};
    const preferred = ['pixels', 'image', 'images', 'latent_image', 'samples'];
    for (const key of preferred) {
      const found = visit(getLinkedNode(nodesById, inputs[key]));
      if (found) return found;
    }
    for (const value of Object.values(inputs)) {
      const found = visit(getLinkedNode(nodesById, value));
      if (found) return found;
    }
    return undefined;
  };
  return visit(getLinkedNode(nodesById, sampler.inputs.latent_image));
}

function resolveSamplerText(sampler: ApiPromptNode | undefined, nodesById: Map<string, ApiPromptNode>, inputName: 'positive' | 'negative'): string | undefined {
  const textNode = getLinkedNode(nodesById, sampler?.inputs?.[inputName]);
  return stringInput(textNode?.inputs?.text) ?? undefined;
}

export function detectNativeMlxBigLoveKlein3(prompt: Record<string, unknown>): NativeMlxQueueCandidate | null {
  const entries = Object.entries(prompt);
  const nodesById = new Map<string, ApiPromptNode>();
  const nodes: ApiPromptNode[] = [];
  for (const [id, value] of entries) {
    const node = asPromptNode(value);
    if (!node) continue;
    nodesById.set(id, node);
    nodes.push(node);
  }

  const unet = nodes.find((node) => {
    if (node?.class_type !== 'UNETLoader') return false;
    const name = stringInput(node.inputs?.unet_name)?.toLowerCase() || '';
    return name.includes('biglove') && name.includes('klein3') && name.endsWith('.safetensors');
  });
  if (!unet) return null;

  const sampler = nodes.find((node) => ['KSampler', 'KSamplerAdvanced'].includes(String(node.class_type || '')));
  const loadImage = resolveSamplerLoadImage(sampler, nodesById) ?? nodes.find((node) => node.class_type === 'LoadImage');
  const imagePath = stringInput(loadImage?.inputs?.image);
  if (!imagePath) return null;

  const promptText = resolveSamplerText(sampler, nodesById, 'positive')
    ?? stringInput(nodes.find((node) => node.class_type === 'CLIPTextEncode')?.inputs?.text)
    ?? '';
  if (!promptText.trim()) return null;

  const negativePrompt = resolveSamplerText(sampler, nodesById, 'negative') ?? undefined;
  // BigLove Klein3 MXFP8 fast profile: cap Comfy graph steps to the verified
  // unique-prompt sub-5 native setting. Two steps is ~4s when prompt embeddings
  // are cached, but new prompts can exceed 5s due to text encoding; one step is
  // the only path that consistently stays below the UX target.
  const graphSteps = Math.max(1, Math.min(12, Math.round(numberInput(sampler?.inputs?.steps) ?? 4)));
  const steps = Math.min(1, graphSteps);
  const seed = numberInput(sampler?.inputs?.seed) ?? undefined;
  const width = Math.round(numberInput(nodes.find((node) => node.class_type === 'EmptyLatentImage')?.inputs?.width) ?? 768);
  const height = Math.round(numberInput(nodes.find((node) => node.class_type === 'EmptyLatentImage')?.inputs?.height) ?? 512);

  const guidance = Math.max(0, Math.min(20, numberInput(sampler?.inputs?.cfg) ?? numberInput(sampler?.inputs?.guidance) ?? 1));

  return { imagePath, prompt: promptText, negativePrompt, steps, seed, width, height, guidance };
}

async function pollNativeMlxJobUntilComplete(promptId: string): Promise<void> {
  const startedAt = Date.now();
  const deadline = startedAt + 10 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 750));
    try {
      const response = await fetch(nativeApiUrl(nativeApiPath('job', encodeURIComponent(promptId))), { cache: 'no-store' });
      if (!response.ok) continue;
      const job = await response.json().catch(() => null) as {
        status?: string;
        elapsed_seconds?: number;
        error?: string;
      } | null;
      const status = String(job?.status || '');
      if (status === 'success' || status === 'error' || status === 'failed') {
        window.dispatchEvent(new CustomEvent('native-mlx-job-complete', {
          detail: {
            promptId,
            elapsedSeconds: Number(job?.elapsed_seconds) || (Date.now() - startedAt) / 1000,
            status,
            error: job?.error,
          },
        }));
        return;
      }
    } catch {
      // Keep polling; the wrapper can briefly restart between queue and finish.
    }
  }
}

async function queueNativeMlxBigLoveKlein3(candidate: NativeMlxQueueCandidate): Promise<PromptQueueResponse | null> {
  const response = await fetch(nativeApiUrl(nativeApiPath('generate')), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: candidate.prompt,
      image_path: candidate.imagePath,
      backend: 'mlx-mxfp8-bigloves-klein3-edit',
      negative_prompt: candidate.negativePrompt,
      steps: candidate.steps,
      seed: candidate.seed,
      width: candidate.width,
      height: candidate.height,
      guidance: candidate.guidance,
    }),
  });
  if (!response.ok) return null;
  const data = await response.json().catch(() => null) as { id?: string; number?: number } | null;
  if (!data?.id) return null;
  void pollNativeMlxJobUntilComplete(data.id);
  return { prompt_id: data.id, number: data.number ?? 0 };
}

async function getNativeZImageHistory(maxItems = 50): Promise<Array<Record<string, unknown>>> {
  try {
    const response = await fetch(`${nativeApiUrl(nativeApiPath('history'))}?max_items=${maxItems}`, { cache: 'no-store' });
    if (!response.ok) return [];
    const data = await response.json();
    return Array.isArray(data?.history) ? data.history : [];
  } catch {
    return [];
  }
}

function nativeRecordToImages(item: Record<string, unknown>): HistoryOutputImage[] {
  const urls = Array.isArray(item.image_urls) ? item.image_urls : [];
  const id = String(item.id || item.prompt_id || 'native');
  return urls
    .map((url: unknown) => String(url || ''))
    .filter(Boolean)
    .map((url: string, index: number) => {
      const filename = url.split('?')[0]?.split('/').pop() || `${id}-${index}.png`;
      return { filename, subfolder: '', type: 'output', fullUrl: url };
    });
}

function nativeRecordsToHistory(records: Array<Record<string, unknown>>): History {
  const history: History = {};
  for (const item of records) {
    const id = String(item?.id || item?.prompt_id || '');
    if (!id) continue;
    const created = Date.parse(String(item?.created_at || '')) || Date.now();
    const finished = Date.parse(String(item?.finished_at || '')) || created;
    const isError = item?.status === 'error' || item?.status === 'failed';
    history[id] = {
      prompt: [0, id, {}, {}, []],
      outputs: { native_mlx: { images: nativeRecordToImages(item) } },
      status: {
        status_str: isError ? 'error' : 'success',
        completed: !isError,
        messages: [
          ['execution_start', { timestamp: created }],
          [isError ? 'execution_error' : 'execution_success', { timestamp: finished, message: item?.error }],
        ],
      },
    };
  }
  return history;
}

export async function getQueue(): Promise<QueueInfo> {
  const response = await fetch(`/api/queue`);
  if (!response.ok) throw new Error('Failed to fetch queue');
  const queue = await response.json() as QueueInfo;
  const nativeRecords = await getNativeZImageHistory(20);
  nativeRecords.forEach((record, index) => {
    const id = String(record.id || '');
    if (!id) return;
    const status = String(record.status || '');
    const tuple: [number, string, unknown, Record<string, unknown>, string[]] = [index, id, {}, { backend: record.backend || 'native-mlx' }, []];
    if (status === 'running') queue.queue_running.push(tuple);
    else if (status === 'queued') queue.queue_pending.push(tuple);
  });
  return queue;
}

export async function getHistory(maxItems?: number): Promise<History> {
  const url = maxItems
    ? `/api/history?max_items=${maxItems}`
    : `/api/history`;
  const [response, nativeRecords] = await Promise.all([
    fetch(url),
    getNativeZImageHistory(maxItems ?? 50),
  ]);
  if (!response.ok) throw new Error('Failed to fetch history');
  const data = await response.json();
  const comfyHistory = Array.isArray(data?.history) ? nativeRecordsToHistory(data.history) : data;
  return {
    ...nativeRecordsToHistory(nativeRecords.filter((record) => !['queued', 'running'].includes(String(record.status || '')))),
    ...comfyHistory,
  };
}

// Total number of runs in ComfyUI's history (the frontend pages /history with
// max_items, so it only knows the loaded count). Returns null if the mobile
// backend endpoint isn't available (e.g. server not restarted after an update).
export async function getHistoryCount(): Promise<number | null> {
  try {
    const response = await fetch(`/mobile/api/history-count`, { cache: 'no-store' });
    if (!response.ok) return null;
    const data = await response.json();
    return typeof data.count === 'number' ? data.count : null;
  } catch {
    return null;
  }
}

export async function interruptExecution(): Promise<void> {
  await fetch(`/api/interrupt`, { method: 'POST' });
}

export async function clearQueue(): Promise<void> {
  await fetch(`/api/queue`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clear: true })
  });
}

export async function deleteQueueItem(promptId: string): Promise<void> {
  await fetch(`/api/queue`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delete: [promptId] })
  });
}

export interface PromptQueueRequest {
  prompt: Record<string, unknown>;
  client_id?: string;
  extra_data?: Record<string, unknown>;
}

export interface PromptQueueResponse {
  prompt_id?: string;
  number?: number;
}

export async function queuePrompt(
  request: PromptQueueRequest,
): Promise<PromptQueueResponse> {
  const nativeCandidate = detectNativeMlxBigLoveKlein3(request.prompt);
  if (nativeCandidate) {
    const nativeResponse = await queueNativeMlxBigLoveKlein3(nativeCandidate);
    if (nativeResponse) return nativeResponse;
  }

  const response = await fetch('/api/prompt', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => null) as { error?: unknown } | null;
    const message = typeof data?.error === 'string'
      ? data.error
      : 'Failed to queue prompt';
    throw new Error(message);
  }
  return response.json();
}

export interface QueuePromptMetadata {
  promptId: string;
  workflowLabel?: string;
  workflowSource?: unknown;
  sessionId?: string;
  clientId?: string;
  workflowDiff?: QueueWorkflowDiff;
  createdAt?: number;
  updatedAt?: number;
}

export async function getQueuePromptMetadata(
  promptIds?: string[],
): Promise<Record<string, QueuePromptMetadata>> {
  const params = new URLSearchParams();
  for (const promptId of promptIds ?? []) {
    if (promptId) params.append('prompt_id', promptId);
  }
  const suffix = params.toString();
  const response = await fetch(`/mobile/api/queue-metadata${suffix ? `?${suffix}` : ''}`);
  if (!response.ok) throw new Error('Failed to fetch queue metadata');
  const data = await response.json() as { prompts?: Record<string, QueuePromptMetadata> };
  return data.prompts ?? {};
}

export async function upsertQueuePromptMetadata(
  metadata: QueuePromptMetadata,
): Promise<void> {
  const response = await fetch('/mobile/api/queue-metadata', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(metadata),
  });
  if (!response.ok) throw new Error('Failed to save queue metadata');
}

export async function remapQueuePromptMetadata(
  oldPromptId: string,
  newPromptId: string,
): Promise<void> {
  const response = await fetch('/mobile/api/queue-metadata/remap', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ oldPromptId, newPromptId }),
  });
  if (!response.ok) throw new Error('Failed to remap queue metadata');
}


export async function deleteHistoryItem(promptId: string): Promise<void> {
  await fetch(`/api/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delete: [promptId] })
  });
}

export async function deleteHistoryItems(promptIds: string[]): Promise<void> {
  if (promptIds.length === 0) return;
  await fetch(`/api/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delete: promptIds })
  });
}

export async function clearHistory(): Promise<void> {
  await fetch(`/api/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ clear: true })
  });
}
