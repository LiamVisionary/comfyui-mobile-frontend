import type { HistoryOutputImage, Workflow } from '@/api/types';
import type { UnifiedItem, ViewerImage } from './types';
import { QueueCard } from './QueueCard';
import { InboxIcon } from '@/components/icons';
import { LoadingSpinner } from '@/components/LoadingSpinner';

interface QueueListProps {
  listRef: React.RefObject<HTMLDivElement | null>;
  unifiedList: UnifiedItem[];
  visibleCount: number;
  hasLoadedOnce: boolean;
  effectiveExecutingId: string | null;
  progress: number;
  overallProgress?: number | null;
  executingNodeLabel?: string | null;
  onDeleteItem: (item: UnifiedItem) => void;
  onCancelItem: (item: UnifiedItem) => void;
  onStop: () => void;
  onPurgeQueue: () => void;
  canPurgeQueue: boolean;
  onImageClick?: (images: Array<ViewerImage>, index: number, enableFollowQueue?: boolean) => void;
  viewerImages: Array<ViewerImage>;
  promptOutputs: Record<string, HistoryOutputImage[]>;
  onOpenMenu: (payload: { top: number; right: number; imageSrc: string; workflow?: Workflow; promptId?: string; hasVideoOutputs?: boolean; hasImageOutputs?: boolean }) => void;
  downloaded: Record<string, boolean>;
  firstDoneItemId: string | null;
  onScroll: () => void;
}

export function QueueList({
  listRef,
  unifiedList,
  visibleCount,
  hasLoadedOnce,
  effectiveExecutingId,
  progress,
  overallProgress,
  executingNodeLabel,
  onDeleteItem,
  onCancelItem,
  onStop,
  onPurgeQueue,
  canPurgeQueue,
  onImageClick,
  viewerImages,
  promptOutputs,
  onOpenMenu,
  downloaded,
  firstDoneItemId,
  onScroll
}: QueueListProps) {
  return (
    <div
      ref={listRef}
      className="flex-1 overflow-y-auto p-4 space-y-4 overscroll-contain scroll-container"
      data-queue-list="true"
      onScroll={onScroll}
    >
      {unifiedList.length > 0 && (
        <div className="sticky top-0 z-10 -mx-4 -mt-4 mb-2 border-b border-gray-200 bg-gray-100/95 px-4 py-3 backdrop-blur">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 className="text-base font-bold text-gray-900">Queue + recent outputs</h2>
              <p className="text-xs text-gray-500">
                Active jobs and generated images in one newest-first row list. Tap a thumbnail to view it larger, then close to return here.
              </p>
            </div>
            <button
              type="button"
              onClick={onPurgeQueue}
              disabled={!canPurgeQueue}
              className="shrink-0 rounded-lg bg-red-600 px-3 py-2 text-xs font-bold text-white shadow-sm disabled:bg-gray-300 disabled:text-gray-500"
              aria-label="Cancel the running item and clear all queued items"
            >
              Purge queue
            </button>
          </div>
        </div>
      )}

      {unifiedList.length === 0 && !hasLoadedOnce && (
        <div className="flex items-center justify-center min-h-[calc(100vh-180px)] text-gray-400">
          <div className="text-center">
            <LoadingSpinner size="lg" color="gray" className="mx-auto mb-4" />
            <p className="text-lg">Loading...</p>
          </div>
        </div>
      )}
      {unifiedList.length === 0 && hasLoadedOnce && (
        <div className="flex items-center justify-center min-h-[calc(100vh-180px)] text-gray-500">
          <div className="text-center p-8">
            <div className="flex items-center justify-center mb-4">
              <InboxIcon className="w-10 h-10 text-gray-300" />
            </div>
            <p className="text-lg font-medium">Queue is empty</p>
            <p className="text-sm mt-2">
              Run a workflow to see items here
            </p>
          </div>
        </div>
      )}

      {unifiedList.slice(0, visibleCount).map((item) => (
        <QueueCard
          key={item.id}
          item={item}
          isActuallyRunning={item.id === effectiveExecutingId}
          progress={progress}
          overallProgress={overallProgress}
          executingNodeLabel={executingNodeLabel}
          onDelete={() => onDeleteItem(item)}
          onCancel={() => onCancelItem(item)}
          onStop={onStop}
          onImageClick={onImageClick}
          viewerImages={viewerImages}
          runningImages={promptOutputs[item.id] ?? []}
          onOpenMenu={onOpenMenu}
          downloaded={downloaded}
          isTopDoneItem={item.id === firstDoneItemId}
        />
      ))}
      <div className="h-20" />
    </div>
  );
}
