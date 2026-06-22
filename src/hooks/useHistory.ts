import { create } from 'zustand';
import * as api from '@/api/client';
import type { HistoryOutputImage, Workflow } from '@/api/types';
import { useWorkflowStore, getWorkflowSignature } from '@/hooks/useWorkflow';
import { useQueueStore } from '@/hooks/useQueue';
import { decryptWorkflowFromStorage } from '@/utils/workflowEncryption';

export interface HistoryEntry {
  prompt_id: string;
  timestamp: number;
  durationSeconds?: number;
  success?: boolean;
  errorMessage?: string | null;
  outputs: {
    images: HistoryOutputImage[];
  };
  prompt: Record<string, unknown>;
  workflow?: Workflow;
}

interface HistoryState {
  history: HistoryEntry[];
  isLoading: boolean;

  // Actions
  fetchHistory: () => Promise<void>;
  deleteItem: (promptId: string) => Promise<void>;
  clearHistory: () => Promise<void>;
  clearEmptyItems: () => Promise<void>;
  addHistoryEntry: (entry: HistoryEntry) => void;
}

export const useHistoryStore = create<HistoryState>((set, get) => ({
  history: [],
  isLoading: false,

  addHistoryEntry: (entry) => {
    set((state) => {
      const existingIndex = state.history.findIndex(h => h.prompt_id === entry.prompt_id);
      if (existingIndex !== -1) {
        const existing = state.history[existingIndex];
        const seen = new Set(existing.outputs.images.map((img) => `${img.type}/${img.subfolder}/${img.filename}`));
        const mergedImages = [...existing.outputs.images];
        for (const img of entry.outputs.images) {
          const key = `${img.type}/${img.subfolder}/${img.filename}`;
          if (!seen.has(key)) {
            seen.add(key);
            mergedImages.push(img);
          }
        }
        const merged = {
          ...existing,
          ...entry,
          outputs: { images: mergedImages },
          prompt: Object.keys(entry.prompt ?? {}).length ? entry.prompt : existing.prompt,
          workflow: entry.workflow ?? existing.workflow,
          durationSeconds: entry.durationSeconds ?? existing.durationSeconds,
        };
        const nextHistory = [...state.history];
        nextHistory.splice(existingIndex, 1);
        return { history: [merged, ...nextHistory] };
      }
      // Add to top
      return { history: [entry, ...state.history] };
    });
    const queueStore = useQueueStore.getState();
    if (queueStore.queueItemExpanded[entry.prompt_id] === undefined) {
      queueStore.setQueueItemExpanded(entry.prompt_id, true);
    }
    if (entry.workflow && entry.durationSeconds) {
      const signature = getWorkflowSignature(entry.workflow);
      useWorkflowStore.getState().updateWorkflowDuration(signature, entry.durationSeconds * 1000);
    }
  },

  fetchHistory: async () => {
    set({ isLoading: true });
    try {
      const data = await api.getHistory(50); // Get last 50 items
      const asText = (value: unknown): string | null => {
        if (typeof value === "string") {
          const trimmed = value.trim();
          return trimmed.length > 0 ? trimmed : null;
        }
        if (typeof value === 'number' && Number.isFinite(value)) {
          return value.toString();
        }
        return null;
      };
      const getExecutionErrorMessage = (msgData: Record<string, unknown>): string | null => {
        const direct = asText(msgData.exception_message) ??
          asText(msgData.error) ??
          asText(msgData.message) ??
          asText(msgData.exception_type);
        if (direct) return direct;
        const details = asText((msgData as { details?: unknown }).details);
        if (details) return details;
        const traceback = asText(msgData.traceback);
        const node = asText(msgData.node_id) || asText(msgData.node);
        if (traceback && node) return `${node}: ${traceback}`;
        if (traceback) return traceback;
        if (node) return `${node}: execution error`;
        return null;
      };

      const entries: HistoryEntry[] = await Promise.all(Object.entries(data).map(async ([prompt_id, item]) => {
        // Collect all images from all output nodes
        const images: HistoryOutputImage[] = [];
        for (const output of Object.values(item.outputs)) {
          if (output.images) {
            images.push(...output.images);
          }
          if (output.gifs) {
            images.push(...output.gifs);
          }
          if (output.videos) {
            images.push(...output.videos);
          }
        }

        // Extract timestamp and duration from status messages if available
        let timestamp = Date.now();
        let startTime: number | null = null;
        let endTime: number | null = null;
        let failed = false;
        let errorMessage: string | null = null;
        if (item.status?.messages) {
          for (const [msgType, msgData] of item.status.messages) {
            if (msgType === 'execution_start' && msgData.timestamp) {
              timestamp = msgData.timestamp as number;
              startTime = msgData.timestamp as number;
            }
            if ((msgType === 'execution_end' || msgType === 'execution_success') && msgData.timestamp) {
              endTime = msgData.timestamp as number;
            }
            if (msgType === 'execution_error') {
              failed = true;
              if (typeof msgData === 'object' && msgData !== null && !Array.isArray(msgData)) {
                const nextError = getExecutionErrorMessage(msgData as Record<string, unknown>);
                if (nextError) errorMessage = nextError;
              } else {
                const nextError = asText(msgData as unknown);
                if (nextError) errorMessage = nextError;
              }
            }
          }
        }

        if (startTime === null && timestamp) {
          startTime = timestamp;
        }

        const durationSeconds = (startTime !== null && endTime !== null && endTime >= startTime)
          ? (endTime - startTime) / 1000
          : undefined;
        const statusStr = item.status?.status_str?.toLowerCase() || '';
        const success = !failed && item.status?.completed !== false && !statusStr.includes('error');
        const storedWorkflow = (item.prompt?.[3] as { extra_pnginfo?: { workflow?: unknown } } | undefined)?.extra_pnginfo?.workflow;
        let workflow: Workflow | undefined;
        if (storedWorkflow) {
          try {
            workflow = await decryptWorkflowFromStorage<Workflow>(storedWorkflow);
          } catch (err) {
            console.warn('Could not decrypt embedded workflow metadata for history item:', prompt_id, err);
          }
        }

        return {
          prompt_id,
          timestamp,
          durationSeconds,
          success,
          errorMessage,
          outputs: { images },
          prompt: item.prompt[2] as Record<string, unknown>,
          workflow
        };
      }));

      // Sort by timestamp, newest first
      entries.sort((a, b) => b.timestamp - a.timestamp);

      set({ history: entries });
      const queueStore = useQueueStore.getState();
      for (const entry of entries) {
        if (queueStore.queueItemExpanded[entry.prompt_id] === undefined) {
          queueStore.setQueueItemExpanded(entry.prompt_id, true);
        }
        if (entry.workflow && entry.durationSeconds) {
          const signature = getWorkflowSignature(entry.workflow);
          useWorkflowStore.getState().updateWorkflowDuration(signature, entry.durationSeconds * 1000);
        }
      }
    } catch (err) {
      console.error('Failed to fetch history:', err);
    } finally {
      set({ isLoading: false });
    }
  },

  deleteItem: async (promptId) => {
    try {
      await api.deleteHistoryItem(promptId);
      set((state) => ({
        history: state.history.filter((item) => item.prompt_id !== promptId)
      }));
    } catch (err) {
      console.error('Failed to delete history item:', err);
    }
  },

  clearHistory: async () => {
    try {
      await api.clearHistory();
    } catch (err) {
      console.error('Failed to clear history:', err);
      try {
        const promptIds = get().history.map((item) => item.prompt_id);
        await api.deleteHistoryItems(promptIds);
      } catch (deleteErr) {
        console.error('Failed to delete history items:', deleteErr);
      }
    } finally {
      set({ history: [] });
    }
  },
  clearEmptyItems: async () => {
    const promptIds = get().history
      .filter((item) => item.outputs.images.length === 0)
      .map((item) => item.prompt_id);
    if (promptIds.length === 0) return;
    try {
      await api.deleteHistoryItems(promptIds);
      set((state) => ({
        history: state.history.filter((item) => !promptIds.includes(item.prompt_id))
      }));
    } catch (err) {
      console.error('Failed to delete empty history items:', err);
    }
  }
}));
