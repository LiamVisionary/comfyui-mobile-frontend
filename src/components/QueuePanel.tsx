import { useEffect, useMemo, useRef, useState } from 'react';
import { useShallow } from 'zustand/shallow';
import { useQueueStore } from '@/hooks/useQueue';
import { useHistoryStore } from '@/hooks/useHistory';
import { getWorkflowSignature, useWorkflowStore } from '@/hooks/useWorkflow';
import { useNavigationStore } from '@/hooks/useNavigation';
import { useOverallProgress } from '@/hooks/useOverallProgress';
import type { HistoryOutputImage, NodeTypes, Workflow, WorkflowNode } from '@/api/types';
import { getFileWorkflow, getUserImages, saveWorkflowFavorite, type FileItem } from '@/api/client';
import { buildViewerImages } from '@/utils/viewerImages';
import type { HistoryEntryData, QueueItemData, UnifiedItem, ViewerImage } from './QueuePanel/types';
import { getNodeWidgetIndexMap, getWidgetValue } from '@/utils/workflowInputs';
import { QueueImageMenu } from './QueuePanel/QueueImageMenu';
import { QueueToast } from './QueuePanel/QueueToast';
import { getBatchSources } from './QueuePanel/queueUtils';
import { downloadBatch, downloadImage, saveImageToFiles } from '@/utils/downloads';
import { copyTextToClipboard } from '@/utils/clipboard';
import { QueueList } from './QueuePanel/QueueList';
import { useQueueMenuDismiss } from '@/hooks/useQueueMenuDismiss';
import { resolveExecutingNodeLabel } from '@/utils/executionLabels';
import { buildWorkflowFavoriteRecord } from '@/utils/workflowFavorites';

interface QueuePanelProps {
  visible: boolean;
  onImageClick?: (images: Array<ViewerImage>, index: number, enableFollowQueue?: boolean) => void;
}

function cleanStaticFilenamePrefix(prefix: string): string {
  return prefix
    .replace(/%date:[^%]+%/g, '')
    .replace(/%[^%]+%/g, '')
    .replace(/[\\/]+$/g, '')
    .trim();
}

function workflowOutputPrefixes(workflow: Workflow | null, nodeTypes: NodeTypes | null): string[] {
  if (!workflow || !nodeTypes) return [];
  const prefixes = new Set<string>();
  const addPrefix = (node: WorkflowNode) => {
    const widgetIndex = getNodeWidgetIndexMap(workflow, node)?.filename_prefix;
    const value = getWidgetValue(node, 'filename_prefix', widgetIndex);
    if (typeof value !== 'string') return;
    const cleaned = cleanStaticFilenamePrefix(value);
    if (cleaned.length >= 3) prefixes.add(cleaned);
  };

  for (const node of workflow.nodes ?? []) {
    const nodeType = nodeTypes[node.type];
    const hasPrefixInput = Boolean(nodeType?.input?.required?.filename_prefix || nodeType?.input?.optional?.filename_prefix);
    if (hasPrefixInput || /save.*image|image.*save/i.test(node.type)) {
      addPrefix(node);
    }
  }

  for (const subgraph of workflow.definitions?.subgraphs ?? []) {
    for (const node of subgraph.nodes ?? []) {
      const nodeType = nodeTypes[node.type];
      const hasPrefixInput = Boolean(nodeType?.input?.required?.filename_prefix || nodeType?.input?.optional?.filename_prefix);
      if (hasPrefixInput || /save.*image|image.*save/i.test(node.type)) {
        addPrefix(node);
      }
    }
  }

  return [...prefixes];
}

function fileMatchesWorkflowPrefix(file: FileItem, prefixes: string[]): boolean {
  if (!prefixes.length || (file.type !== 'image' && file.type !== 'video')) return false;
  const path = file.id.replace(/^(output|input|temp)\//, '');
  return prefixes.some((prefix) => {
    const normalized = prefix.replace(/^\/+|\/+$/g, '');
    const basename = normalized.split('/').pop() || normalized;
    return path.startsWith(normalized) || file.name.startsWith(basename);
  });
}

function outputFileToHistoryEntry(file: FileItem, workflow?: Workflow): HistoryEntryData | null {
  if (file.type !== 'image' && file.type !== 'video') return null;
  const path = file.id.replace(/^(output|input|temp)\//, '');
  const slash = path.lastIndexOf('/');
  const subfolder = slash >= 0 ? path.slice(0, slash) : '';
  const filename = slash >= 0 ? path.slice(slash + 1) : path;
  const output: HistoryOutputImage = { filename, subfolder, type: 'output' };
  return {
    prompt_id: `file-output:${path}`,
    timestamp: file.date ?? Date.now(),
    success: true,
    outputs: { images: [output] },
    prompt: {},
    workflow,
  };
}

function historyImageKey(image: HistoryOutputImage): string {
  const path = image.subfolder ? `${image.subfolder}/${image.filename}` : image.filename;
  return `${image.type}/${path}`;
}

export function QueuePanel({ visible, onImageClick }: QueuePanelProps) {
  const running = useQueueStore((s) => s.running);
  const pending = useQueueStore((s) => s.pending);
  const fetchQueue = useQueueStore((s) => s.fetchQueue);
  const deleteQueueItem = useQueueStore((s) => s.deleteItem);
  const clearQueue = useQueueStore((s) => s.clearQueue);
  const interrupt = useQueueStore((s) => s.interrupt);
  const loadWorkflow = useWorkflowStore((s) => s.loadWorkflow);
  const setCurrentPanel = useNavigationStore((s) => s.setCurrentPanel);
  const workflow = useWorkflowStore((s) => s.workflow);
  const nodeTypes = useWorkflowStore((s) => s.nodeTypes);
  const executingNodeId = useWorkflowStore((s) => s.executingNodeId);
  const executingNodePath = useWorkflowStore((s) => s.executingNodePath);
  const workflowDurationStats = useWorkflowStore((s) => s.workflowDurationStats);
  const promptOutputs = useWorkflowStore((s) => s.promptOutputs);

  const history = useHistoryStore((s) => s.history);
  const fetchHistory = useHistoryStore((s) => s.fetchHistory);
  const deleteHistoryItem = useHistoryStore((s) => s.deleteItem);
  const { isExecuting, progress, executingPromptId } = useWorkflowStore(
    useShallow((s) => ({
      isExecuting: s.isExecuting,
      progress: s.progress,
      executingPromptId: s.executingPromptId,
    }))
  );
  const effectiveExecutingId = executingPromptId || (running.length === 1 ? running[0].prompt_id : null);
  const executingNodeLabel = useMemo(() => {
    return resolveExecutingNodeLabel(
      executingNodePath,
      executingNodeId,
      workflow,
      nodeTypes,
    );
  }, [workflow, executingNodeId, executingNodePath, nodeTypes]);
  const overallProgress = useOverallProgress({
    workflow,
    runKey: executingPromptId || effectiveExecutingId,
    isRunning: isExecuting || Boolean(effectiveExecutingId),
    workflowDurationStats,
  });
  const [menuState, setMenuState] = useState<{
    open: boolean;
    top: number;
    right: number;
    imageSrc: string;
    workflow?: Workflow;
    prompt?: Record<string, unknown>;
    file?: FileItem;
    promptId?: string;
    hasVideoOutputs?: boolean;
    hasImageOutputs?: boolean;
  } | null>(null);
  const [downloaded, setDownloaded] = useState<Record<string, boolean>>({});
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [workflowOutputHistory, setWorkflowOutputHistory] = useState<HistoryEntryData[]>([]);
  const [visibleCount, setVisibleCount] = useState(0);
  const outputPrefixes = useMemo(() => workflowOutputPrefixes(workflow, nodeTypes), [workflow, nodeTypes]);
  const workflowSignature = useMemo(() => workflow ? getWorkflowSignature(workflow) : null, [workflow]);
  const listRef = useRef<HTMLDivElement>(null);
  const totalCountRef = useRef(0);
  const wasOpenRef = useRef(false);
  const hasMountedRef = useRef(false);
  const previousEffectiveExecutingIdRef = useRef<string | null>(null);
  const pendingAutoScrollDoneIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (visible) {
      if (!wasOpenRef.current) {
        if (hasMountedRef.current) {
          if (listRef.current) {
            listRef.current.scrollTop = 0;
          }
        }
      }
      void fetchQueue().then(() => {
        setHasLoadedOnce(true);
      });
      // Let the panel paint before expensive history refresh work starts.
      window.setTimeout(() => {
        void fetchHistory();
      }, 250);
    }
    wasOpenRef.current = visible;
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;
    }
  }, [visible, fetchQueue, fetchHistory]);

  useEffect(() => {
    if (!visible || !workflow) return;
    let cancelled = false;
    const loadWorkflowOutputs = async () => {
      try {
        const files = await getUserImages('output', 1000, 0, 'modified', true, null, false);
        const mediaFiles = files
          .filter((file) => file.type === 'image' || file.type === 'video')
          .sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
          .slice(0, 120);
        const entries: HistoryEntryData[] = [];
        for (const file of mediaFiles) {
          if (fileMatchesWorkflowPrefix(file, outputPrefixes)) {
            const entry = outputFileToHistoryEntry(file, workflow);
            if (entry) entries.push(entry);
            continue;
          }
          if (!workflowSignature) continue;
          const path = file.id.replace(/^(output|input|temp)\//, '');
          try {
            const fileWorkflow = await getFileWorkflow(path, 'output');
            if (getWorkflowSignature(fileWorkflow) === workflowSignature) {
              const entry = outputFileToHistoryEntry(file, fileWorkflow);
              if (entry) entries.push(entry);
            }
          } catch {
            // Many historical ComfyUI files have no embedded workflow metadata; filename-prefix matching above handles those.
          }
        }
        entries.sort((a, b) => b.timestamp - a.timestamp);
        if (!cancelled) setWorkflowOutputHistory(entries.slice(0, 50));
      } catch (err) {
        console.warn('Failed to fetch workflow output files for queue panel:', err);
        if (!cancelled) setWorkflowOutputHistory([]);
      }
    };
    const timeoutId = window.setTimeout(loadWorkflowOutputs, 800);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [visible, workflow, outputPrefixes, workflowSignature]);

  useEffect(() => {
    if (!isExecuting && visible) {
      fetchHistory();
    }
  }, [isExecuting, visible, fetchHistory]);

  // Queue view is embedded; no modal scroll locking.

  useQueueMenuDismiss(Boolean(menuState?.open), () => setMenuState(null), 'queue-image-menu');

  const handleCopyWorkflow = async (workflow: Workflow | undefined) => {
    if (!workflow) return;
    const text = JSON.stringify(workflow, null, 2);
    const copied = await copyTextToClipboard(text);
    setToastMessage(copied ? 'Copied to clipboard' : 'Failed to copy');
    setTimeout(() => setToastMessage(null), 2000);
  };

  const handleSaveToAlbum = async (src: string) => {
    await downloadImage(src, 'image.png', (downloadedSrc) => {
      setDownloaded((prev) => ({ ...prev, [downloadedSrc]: true }));
    });
  };

  const handleSaveToFiles = async (src: string) => {
    await saveImageToFiles(src, 'image.png', (downloadedSrc) => {
      setDownloaded((prev) => ({ ...prev, [downloadedSrc]: true }));
    });
  };

  const unifiedList = useMemo(() => {
    const items: Record<string, UnifiedItem> = {};

    history.forEach(item => {
      items[item.prompt_id] = { id: item.prompt_id, status: 'done', data: item, timestamp: item.timestamp };
    });

    const historyOutputKeys = new Set<string>();
    history.forEach(item => {
      item.outputs.images.forEach((image) => historyOutputKeys.add(historyImageKey(image)));
    });

    workflowOutputHistory.forEach(item => {
      const firstImage = item.outputs.images[0];
      if (firstImage && historyOutputKeys.has(historyImageKey(firstImage))) return;
      if (!items[item.prompt_id]) {
        items[item.prompt_id] = { id: item.prompt_id, status: 'done', data: item, timestamp: item.timestamp };
      }
    });

    running.forEach(item => {
      if (!items[item.prompt_id]) {
        items[item.prompt_id] = { id: item.prompt_id, status: 'running', data: item };
      }
    });

    pending.forEach(item => {
      if (!items[item.prompt_id]) {
        items[item.prompt_id] = { id: item.prompt_id, status: 'pending', data: item };
      }
    });

    if (executingPromptId && items[executingPromptId]) {
      items[executingPromptId].status = 'running';
    }

    const list = Object.values(items);
    list.sort((a, b) => {
      const statusOrder = { 'running': 0, 'pending': 1, 'done': 2 };
      if(statusOrder[a.status] !== statusOrder[b.status]) {
        return statusOrder[a.status] - statusOrder[b.status];
      }
      if (a.status === 'pending') {
        const aNumber = (a.data as QueueItemData).number;
        const bNumber = (b.data as QueueItemData).number;
        return bNumber - aNumber; // Highest number (newest) first
      }
      if (a.status === 'done') {
        return (b.timestamp || 0) - (a.timestamp || 0); // Newest timestamp first
      }
      return 0;
    });

    return list;
  }, [pending, running, history, workflowOutputHistory, executingPromptId]);

  const initialVisibleCount = useMemo(() => {
    if (unifiedList.length === 0) return 0;
    return Math.min(unifiedList.length, 30);
  }, [unifiedList]);

  const viewerImages = useMemo(() => {
    const doneItems = unifiedList.filter((item) => item.status === 'done').map((item) => item.data);
    return buildViewerImages(doneItems, { alt: 'Generation' });
  }, [unifiedList]);

  const firstDoneItemId = useMemo(() => {
    const firstDone = unifiedList.find((item) => item.status === 'done');
    return firstDone?.id ?? null;
  }, [unifiedList]);

  useEffect(() => {
    const previousId = previousEffectiveExecutingIdRef.current;
    if (previousId && !effectiveExecutingId) {
      pendingAutoScrollDoneIdRef.current = previousId;
    }
    previousEffectiveExecutingIdRef.current = effectiveExecutingId;
  }, [effectiveExecutingId]);

  useEffect(() => {
    if (!visible || !firstDoneItemId) return;
    if (pendingAutoScrollDoneIdRef.current !== firstDoneItemId) return;
    pendingAutoScrollDoneIdRef.current = null;
    setVisibleCount((prev) => Math.max(prev, initialVisibleCount));
    window.setTimeout(() => {
      const safeId = firstDoneItemId.replace(/\\/g, '\\\\').replace(/\"/g, '\\\"');
      const card = listRef.current?.querySelector(`[data-queue-card-id="${safeId}"]`);
      card?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
  }, [visible, firstDoneItemId, initialVisibleCount]);

  useEffect(() => {
    if (!visible) return;
    totalCountRef.current = unifiedList.length;
    queueMicrotask(() => {
      setVisibleCount((prev) => Math.max(prev, initialVisibleCount));
    });
  }, [visible, unifiedList.length, initialVisibleCount]);

  useEffect(() => {
    if (!visible) return;
    const el = listRef.current;
    if (!el) return;
    if (visibleCount >= unifiedList.length) return;
    if (el.scrollHeight <= el.clientHeight + 20) {
      queueMicrotask(() => {
        setVisibleCount((prev) => Math.min(unifiedList.length, prev + 10));
      });
    }
  }, [visible, visibleCount, unifiedList.length]);

  const handleDeleteItem = (item: UnifiedItem) => {
    if (item.status === 'pending') deleteQueueItem(item.id);
    if (item.status === 'done') deleteHistoryItem(item.id);
  };

  const handleCancelItem = (item: UnifiedItem) => {
    if (item.status === 'running') {
      interrupt().then(() => fetchQueue());
      return;
    }
    if (item.status === 'pending') {
      deleteQueueItem(item.id).then(() => fetchQueue());
    }
  };

  const handlePurgeQueue = async () => {
    if (running.length > 0) {
      await interrupt();
    }
    if (pending.length > 0) {
      await clearQueue();
    }
    await fetchQueue();
    setToastMessage('Queue purged');
    setTimeout(() => setToastMessage(null), 2000);
  };

  const handleOpenMenu = (payload: { top: number; right: number; imageSrc: string; workflow?: Workflow; prompt?: Record<string, unknown>; file?: FileItem; promptId?: string; hasVideoOutputs?: boolean; hasImageOutputs?: boolean }) => {
    const { top, right, imageSrc, workflow, prompt, file, promptId, hasVideoOutputs, hasImageOutputs } = payload;
    setMenuState({
      open: true,
      top,
      right,
      imageSrc,
      workflow,
      prompt,
      file,
      promptId,
      hasVideoOutputs,
      hasImageOutputs
    });
  };

  const handleListScroll = () => {
    const el = listRef.current;
    if (!el) return;
    const remaining = el.scrollHeight - el.scrollTop - el.clientHeight;
    if (remaining < 400) {
      setVisibleCount((prev) => Math.min(totalCountRef.current, prev + 10));
    }
  };

  const handleMenuLoadWorkflow = (workflow: Workflow, promptId: string) => {
    loadWorkflow(
      workflow,
      `history-${promptId}.json`,
      { source: { type: 'history', promptId } }
    );
    setCurrentPanel('workflow');
  };

  const handleFavoriteWorkflow = async (payload: { workflow?: Workflow; prompt?: Record<string, unknown>; file?: FileItem; src?: string; promptId?: string }) => {
    if (!payload.workflow || !payload.file) {
      setToastMessage('No workflow metadata available for this image');
      setTimeout(() => setToastMessage(null), 2400);
      return;
    }
    try {
      const record = await buildWorkflowFavoriteRecord({
        workflow: payload.workflow,
        prompt: payload.prompt,
        file: payload.file,
        src: payload.src,
        promptId: payload.promptId,
      });
      await saveWorkflowFavorite(record);
      setToastMessage('Favorite workflow shortcut saved');
    } catch (err) {
      console.error('Failed to favorite workflow:', err);
      setToastMessage('Failed to save favorite shortcut');
    }
    setTimeout(() => setToastMessage(null), 2400);
  };

  const handleBatchSaveToAlbum = async (sources: string[]) => {
    await downloadBatch(sources, (downloadedSrc) => {
      setDownloaded((prev) => ({ ...prev, [downloadedSrc]: true }));
    }, 'album');
  };

  const handleBatchSaveToFiles = async (sources: string[]) => {
    await downloadBatch(sources, (downloadedSrc) => {
      setDownloaded((prev) => ({ ...prev, [downloadedSrc]: true }));
    }, 'files');
  };

  return (
    <div
      id="queue-panel-wrapper"
      className="absolute inset-x-0 top-[69px] bottom-0"
      style={{ display: visible ? 'block' : 'none' }}
    >
      <div className="flex flex-col bg-gray-100 min-h-full">
        <QueueList
          listRef={listRef}
          unifiedList={unifiedList}
          visibleCount={visibleCount}
          hasLoadedOnce={hasLoadedOnce}
          effectiveExecutingId={effectiveExecutingId}
          progress={progress}
          overallProgress={overallProgress}
          executingNodeLabel={executingNodeLabel}
          onDeleteItem={handleDeleteItem}
          onCancelItem={handleCancelItem}
          onStop={interrupt}
          onPurgeQueue={handlePurgeQueue}
          canPurgeQueue={pending.length + running.length > 0}
          onImageClick={onImageClick}
          viewerImages={viewerImages}
          promptOutputs={promptOutputs}
          onOpenMenu={handleOpenMenu}
          downloaded={downloaded}
          firstDoneItemId={firstDoneItemId}
          onScroll={handleListScroll}
        />

        <QueueImageMenu
          menuState={menuState}
          unifiedList={unifiedList}
          onClose={() => setMenuState(null)}
          onLoadWorkflow={handleMenuLoadWorkflow}
          onCopyWorkflow={handleCopyWorkflow}
          onFavoriteWorkflow={handleFavoriteWorkflow}
          onSaveToAlbum={(src) => handleSaveToAlbum(src)}
          onSaveToFiles={(src) => handleSaveToFiles(src)}
          onBatchSaveToAlbum={handleBatchSaveToAlbum}
          onBatchSaveToFiles={handleBatchSaveToFiles}
          onDeleteHistory={(promptId) => deleteHistoryItem(promptId)}
          getBatchSources={getBatchSources}
        />

        <QueueToast message={toastMessage} />
      </div>
    </div>
  );
}
