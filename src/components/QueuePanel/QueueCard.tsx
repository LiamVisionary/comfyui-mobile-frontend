import { useEffect, useMemo, useRef, useState } from 'react';
import { getImageUrl, getPreviewUrl, getThumbnailUrl } from '@/api/client';
import type { Workflow } from '@/api/types';
import { useQueueStore } from '@/hooks/useQueue';
import { extractMetadata } from '@/utils/metadata';
import { CaretDownIcon, CaretRightIcon, CheckIcon, CloudDownloadIcon, HourglassIcon, XMarkIcon, XSmallIcon } from '@/components/icons';
import type { HistoryOutputImage } from '@/api/types';
import { isHistoryEntryData, type UnifiedItem, type ViewerImage } from './types';
import { getMediaType, isVideoFilename } from '@/utils/media';
import { ContextMenuButton } from '@/components/buttons/ContextMenuButton';
import { ProgressiveImage } from '@/components/ProgressiveImage';

interface QueueCardProps {
  item: UnifiedItem;
  isActuallyRunning: boolean;
  progress: number;
  overallProgress?: number | null;
  executingNodeLabel?: string | null;
  onDelete: () => void;
  onCancel: () => void;
  onStop: () => void;
  onImageClick?: (images: Array<ViewerImage>, index: number, enableFollowQueue?: boolean) => void;
  viewerImages: Array<ViewerImage>;
  runningImages: HistoryOutputImage[];
  onOpenMenu: (payload: { top: number; right: number; imageSrc: string; workflow?: Workflow; prompt?: Record<string, unknown>; file?: { id: string; name: string; type: 'image' | 'video'; fullUrl?: string }; promptId?: string; hasVideoOutputs?: boolean; hasImageOutputs?: boolean }) => void;
  downloaded: Record<string, boolean>;
  isTopDoneItem: boolean;
}

export function QueueCard({
  item,
  isActuallyRunning,
  progress,
  overallProgress,
  executingNodeLabel,
  onDelete,
  onCancel,
  onStop,
  onImageClick,
  viewerImages,
  runningImages,
  onOpenMenu,
  downloaded,
  isTopDoneItem
}: QueueCardProps) {
  const previewVisibility = useQueueStore((s) => s.previewVisibility);
  const previewVisibilityDefault = useQueueStore((s) => s.previewVisibilityDefault);
  const showQueueMetadata = useQueueStore((s) => s.showQueueMetadata);
  const queueItemExpanded = useQueueStore((s) => s.queueItemExpanded[item.id]);
  const setQueueItemExpanded = useQueueStore((s) => s.setQueueItemExpanded);
  const queueItemHideImages = useQueueStore((s) => s.queueItemHideImages[item.id]);
  const videoRefs = useRef(new Map<string, HTMLVideoElement>());
  const playedVideoSources = useRef(new Set<string>());
  const [endedVideoSources, setEndedVideoSources] = useState<Set<string>>(new Set());
  const isPending = item.status === 'pending' && !isActuallyRunning;
  const isRunning = item.status === 'running' || isActuallyRunning;
  const isDone = item.status === 'done';

  const hasStoredExpanded = queueItemExpanded !== undefined;
  const expanded = hasStoredExpanded ? queueItemExpanded : false;
  const prevIsDoneRef = useRef(isDone);

  useEffect(() => {
    if (!hasStoredExpanded) {
      setQueueItemExpanded(item.id, false);
    }
  }, [hasStoredExpanded, item.id, setQueueItemExpanded]);

  useEffect(() => {
    if (!prevIsDoneRef.current && isDone) {
      setQueueItemExpanded(item.id, true);
    }
    prevIsDoneRef.current = isDone;
  }, [isDone, item.id, setQueueItemExpanded]);

  useEffect(() => {
    playedVideoSources.current.clear();
    setEndedVideoSources(new Set());
  }, [item.id]);

  const handleToggle = () => {
    setQueueItemExpanded(item.id, !expanded);
  };

  const historyData = isDone && isHistoryEntryData(item.data) ? item.data : null;
  const sourceImages = useMemo(() => (
    historyData ? historyData.outputs.images : (isRunning ? runningImages : [])
  ), [historyData, isRunning, runningImages]);
  const previewsVisible = Boolean(
    item.data.prompt_id
      ? previewVisibility[item.data.prompt_id] ?? previewVisibilityDefault
      : previewVisibilityDefault
  );
  const { savedImages, displayImages } = useMemo(() => {
    const saved = sourceImages.filter((img: HistoryOutputImage) => img.type === 'output');
    const showPreviews = previewsVisible || saved.length === 0;
    return {
      savedImages: saved,
      displayImages: showPreviews
        ? sourceImages
        : sourceImages.filter((img: HistoryOutputImage) => img.type === 'output')
    };
  }, [previewsVisible, sourceImages]);
  const hasVideoOutputs = useMemo(() => (
    sourceImages.some((img: HistoryOutputImage) => isVideoFilename(img.filename))
  ), [sourceImages]);
  const hasImageOutputs = useMemo(() => (
    sourceImages.some((img: HistoryOutputImage) => !isVideoFilename(img.filename))
  ), [sourceImages]);
  const visibleImages = useMemo(() => {
    if (!queueItemHideImages || !hasVideoOutputs) return displayImages;
    return displayImages.filter((img: HistoryOutputImage) => isVideoFilename(img.filename));
  }, [displayImages, hasVideoOutputs, queueItemHideImages]);
  const placeholderClass = 'aspect-square w-full bg-gray-100 flex flex-col items-center justify-center text-gray-400';
  const durationSeconds = historyData?.durationSeconds;
  const success = historyData ? historyData.success !== false : true;
  const isFailedDoneItem = isDone && historyData?.success === false;
  const donePlaceholderMessage = isFailedDoneItem
    ? historyData?.errorMessage || 'Execution failed'
    : 'No images saved';
  const donePlaceholderClass = isFailedDoneItem
    ? 'text-sm text-red-600 px-4 text-center'
    : 'text-sm';
  const durationLabel = formatDuration(durationSeconds);
  const displayNodeProgress = overallProgress === 100 ? 100 : progress;
  const rowThumb = visibleImages[0];
  const rowThumbSrc = rowThumb
    ? (isVideoFilename(rowThumb.filename) ? getImageUrl(rowThumb.filename, rowThumb.subfolder, rowThumb.type) : getThumbnailUrl(rowThumb.filename, rowThumb.subfolder, rowThumb.type))
    : '';
  const savedCount = savedImages.length;
  const visibleCount = visibleImages.length;
  const rowTitle = isRunning
    ? 'Generating now'
    : isPending
      ? 'Waiting in queue'
      : success
        ? `${savedCount || visibleCount || 0} saved ${savedCount === 1 ? 'output' : 'outputs'}`
        : 'Failed generation';
  const rowSubtitle = isDone
    ? `${new Date(item.timestamp || 0).toLocaleString([], { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })}${durationLabel ? ` · ${durationLabel}` : ''}`
    : isPending
      ? `Queue #${(item.data as { number?: number }).number ?? '—'}`
      : (executingNodeLabel ? `Executing: ${executingNodeLabel}` : 'Running workflow');

  const metadata = useMemo(() => {
    if (!showQueueMetadata || !item.data.prompt) return null;
    return extractMetadata(item.data.prompt);
  }, [showQueueMetadata, item.data.prompt]);

  const queueViewerImages = useMemo(() => {
    if (!isRunning) return viewerImages;
    return visibleImages.map((img: HistoryOutputImage) => ({
      src: getImageUrl(img.filename, img.subfolder, img.type),
      alt: 'Generation',
      mediaType: getMediaType(img.filename)
    }));
  }, [isRunning, viewerImages, visibleImages]);

  useEffect(() => {
    if (!expanded) return;
    visibleImages.forEach((img: HistoryOutputImage) => {
      if (!isVideoFilename(img.filename)) return;
      const src = getImageUrl(img.filename, img.subfolder, img.type);
      if (playedVideoSources.current.has(src)) return;
      playedVideoSources.current.add(src);
      const videoEl = videoRefs.current.get(src);
      if (!videoEl) return;
      videoEl.currentTime = 0;
      setEndedVideoSources((prev) => {
        const next = new Set(prev);
        next.delete(src);
        return next;
      });
      const playPromise = videoEl.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        playPromise.catch(() => {});
      }
    });
  }, [expanded, visibleImages]);

  const handleToggleButtonClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    handleToggle();
  };

  const handleStopClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    onStop();
  };

  const handleCancelClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    onCancel();
  };

  const handleDeleteClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    onDelete();
  };

  const handleOpenMenuClick = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
    const right = Math.max(8, window.innerWidth - rect.right);
    const firstSaved = savedImages[0];
    const firstSrc = firstSaved ? getImageUrl(firstSaved.filename, firstSaved.subfolder, firstSaved.type) : '';
    const firstPath = firstSaved
      ? (firstSaved.subfolder ? `${firstSaved.subfolder}/${firstSaved.filename}` : firstSaved.filename)
      : '';
    onOpenMenu({
      top: rect.bottom + 6,
      right,
      imageSrc: firstSrc,
      workflow: historyData?.workflow,
      prompt: historyData?.prompt,
      file: firstSaved ? {
        id: `${firstSaved.type}/${firstPath}`,
        name: firstSaved.filename,
        type: isVideoFilename(firstSaved.filename) ? 'video' : 'image',
        fullUrl: firstSrc,
      } : undefined,
      promptId: item.data.prompt_id || item.id,
      hasVideoOutputs,
      hasImageOutputs
    });
  };

  const handleVideoEnded = (src: string) => () => {
    setEndedVideoSources((prev) => new Set(prev).add(src));
  };

  const handleVideoPlay = (src: string) => () => {
    setEndedVideoSources((prev) => {
      const next = new Set(prev);
      next.delete(src);
      return next;
    });
  };

  const handleMediaClick = (src: string, index: number, isTop: boolean) => () => {
    const resolvedIndex = queueViewerImages.findIndex((entry: ViewerImage) => entry.src === src);
    onImageClick?.(queueViewerImages, resolvedIndex >= 0 ? resolvedIndex : index, isTop);
  };

  const handleReplayClick = (src: string) => (event: React.MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    const videoEl = videoRefs.current.get(src);
    if (!videoEl) return;
    videoEl.currentTime = 0;
    const playPromise = videoEl.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {});
    }
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden transition-all duration-300" data-queue-card-id={item.id} data-queue-status={item.status}>
      <div onClick={handleToggle} className={`px-3 py-2 border-b flex justify-between items-center cursor-pointer select-none ${isRunning ? 'bg-blue-50 border-blue-100' : 'bg-gray-50 border-gray-100'}`}>
        <div className="flex items-center gap-1 min-w-0">
          <button
            onClick={handleToggleButtonClick}
            className="w-8 h-8 -ml-2 flex items-center justify-center text-gray-400 hover:text-gray-600 shrink-0"
          >
            {!expanded ? (
              <CaretRightIcon className="w-6 h-6" />
            ) : (
              <CaretDownIcon className="w-6 h-6" />
            )}
          </button>
          {isRunning && <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />}
          {isPending && <span className="text-xs font-bold text-gray-400">PENDING</span>}
          {isRunning && <span className="text-xs font-bold text-blue-600">GENERATING</span>}
          {isDone && <span className="text-xs font-bold text-gray-500">{new Date(item.timestamp || 0).toLocaleTimeString()}</span>}
          {isRunning && !expanded && isActuallyRunning && overallProgress != null && overallProgress < 100 && (
            <span className="ml-1 text-xs font-semibold text-blue-600">{Math.min(100, Math.max(0, overallProgress))}%</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isRunning ? (
            <button
              onClick={handleStopClick}
              className="px-2 py-1 text-xs font-semibold text-white bg-red-600 rounded hover:bg-red-700"
            >
              Cancel
            </button>
          ) : isDone ? (
            <ContextMenuButton
              onClick={handleOpenMenuClick}
              ariaLabel="Image options"
              buttonSize={7}
              iconSize={4}
            />
          ) : (
            <button
              onClick={handleDeleteClick}
              className="text-gray-400 hover:text-red-500 px-2"
              title={isPending ? 'Remove from Queue' : 'Delete from History'}
              aria-label={isPending ? 'Remove from Queue' : 'Delete from History'}
            >
              <XMarkIcon className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3 px-3 py-3 bg-white">
        <button
          type="button"
          className="relative h-20 w-20 shrink-0 overflow-hidden rounded-lg bg-gray-100 text-gray-400"
          onClick={rowThumbSrc ? handleMediaClick(rowThumbSrc, 0, isTopDoneItem) : handleToggleButtonClick}
          aria-label={rowThumbSrc ? 'Open output in large viewer' : 'Show queue item details'}
        >
          {rowThumbSrc ? (
            isVideoFilename(rowThumb?.filename || '') ? (
              <video src={rowThumbSrc} className="h-full w-full object-cover" muted playsInline preload="metadata" />
            ) : (
              <img src={rowThumbSrc} alt="Generation thumbnail" className="h-full w-full object-cover" loading={isTopDoneItem || isRunning ? 'eager' : 'lazy'} decoding="async" />
            )
          ) : (
            <div className="flex h-full w-full items-center justify-center">
              {isRunning ? <div className="h-7 w-7 animate-spin rounded-full border-4 border-blue-200 border-t-blue-500" /> : <HourglassIcon className="h-7 w-7 text-gray-300" />}
            </div>
          )}
          {visibleCount > 1 && (
            <span className="absolute bottom-1 right-1 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-bold text-white">{visibleCount}</span>
          )}
        </button>
        <button type="button" onClick={handleToggleButtonClick} className="min-w-0 flex-1 text-left">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 shrink-0 rounded-full ${isRunning ? 'bg-blue-500 animate-pulse' : isPending ? 'bg-amber-400' : success ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="truncate text-sm font-bold text-gray-900">{rowTitle}</span>
          </div>
          <p className="mt-1 truncate text-xs text-gray-500">{rowSubtitle}</p>
          {isRunning && overallProgress != null && overallProgress < 100 && (
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-gray-200">
              <div className="h-full bg-blue-500" style={{ width: `${Math.min(100, Math.max(0, overallProgress))}%` }} />
            </div>
          )}
          {(isRunning || isPending) && (
            <button
              type="button"
              onClick={handleCancelClick}
              className="mt-2 rounded-md bg-red-50 px-3 py-1.5 text-xs font-bold text-red-700"
            >
              {isRunning ? 'Cancel current run' : 'Remove from queue'}
            </button>
          )}
          {isDone && visibleCount > 0 && (
            <p className="mt-1 text-[11px] font-medium text-gray-400">Tap thumbnail for large view · tap row for details</p>
          )}
        </button>
      </div>

      {expanded && (
        <div className="relative w-full animate-in fade-in duration-200">
          {visibleImages.length > 0 ? (
            <div className="flex flex-col">
              {(() => {
                let saveIndex = 0;
                let previewIndex = 0;
                return visibleImages.map((img: HistoryOutputImage, i: number) => {
                  const isPreview = img.type !== 'output';
                  const labelIndex = isPreview ? ++previewIndex : ++saveIndex;
                  const labelText = `${isPreview ? 'Preview' : 'Save'} #${labelIndex}`;
                  const fullSrc = getImageUrl(img.filename, img.subfolder, img.type);
                  const src = isPreview || isVideoFilename(img.filename)
                    ? fullSrc
                    : getPreviewUrl(img.filename, img.subfolder, img.type);
                  const isDownloaded = downloaded[fullSrc];
                  const showLabel = visibleImages.length > 1;

                  return (
                    <div key={i} className="relative">
                      {durationLabel && (
                        <div
                          className={`absolute top-2 left-2 flex items-center gap-1 px-2 py-1 rounded text-xs font-semibold ${
                            success ? 'bg-green-600/90 text-white' : 'bg-red-600/90 text-white'
                          }`}
                        >
                          {success ? (
                            <CheckIcon className="w-3.5 h-3.5" />
                          ) : (
                            <XSmallIcon className="w-3.5 h-3.5" />
                          )}
                          <span>{durationLabel}</span>
                        </div>
                      )}
                      {isVideoFilename(img.filename) ? (
                        <>
                          <video
                            src={src}
                            className="w-full h-auto block"
                            muted
                            playsInline
                            preload="metadata"
                            ref={(el) => {
                              if (el) {
                                videoRefs.current.set(src, el);
                              } else {
                                videoRefs.current.delete(src);
                              }
                            }}
                            onEnded={handleVideoEnded(src)}
                            onPlay={handleVideoPlay(src)}
                            onClick={handleMediaClick(src, i, isTopDoneItem)}
                          />
                          {endedVideoSources.has(src) && (
                            <button
                              type="button"
                              className="absolute inset-0 flex items-center justify-center bg-black/35 text-white"
                              onClick={handleReplayClick(src)}
                              aria-label="Replay video"
                            >
                              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-black/60 text-2xl">
                                ↻
                              </span>
                            </button>
                          )}
                        </>
                      ) : (
                        <ProgressiveImage
                          fullSrc={fullSrc}
                          previewSrc={src}
                          alt="Generation"
                          className="w-full h-auto block"
                          loading={isTopDoneItem || isRunning ? 'eager' : 'lazy'}
                          decoding="async"
                          fetchPriority={isTopDoneItem || isRunning ? 'high' : 'auto'}
                          loadFull={false}
                          onClick={handleMediaClick(fullSrc, i, isTopDoneItem)}
                        />
                      )}
                      {isDownloaded && (
                        <div className="absolute bottom-2 right-2 w-7 h-7 rounded-full bg-black/60 text-white flex items-center justify-center">
                          <CloudDownloadIcon className="w-4 h-4" />
                        </div>
                      )}
                      {showLabel && (
                        <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/60 text-white text-xs font-medium rounded backdrop-blur-sm shadow-sm pointer-events-none">
                          {labelText}
                        </div>
                      )}
                      {metadata && (
                        <div className={`absolute right-2 flex flex-col-reverse items-end gap-1 pointer-events-none ${isDownloaded ? 'bottom-10' : 'bottom-2'}`}>
                          {metadata.model && <div className="px-1.5 py-0.5 bg-black/50 text-white text-[10px] rounded backdrop-blur-sm">model: {metadata.model}</div>}
                          {metadata.sampler && <div className="px-1.5 py-0.5 bg-black/50 text-white text-[10px] rounded backdrop-blur-sm">sampler: {metadata.sampler}</div>}
                          {metadata.steps && <div className="px-1.5 py-0.5 bg-black/50 text-white text-[10px] rounded backdrop-blur-sm">steps: {metadata.steps}</div>}
                          {metadata.cfg && <div className="px-1.5 py-0.5 bg-black/50 text-white text-[10px] rounded backdrop-blur-sm">cfg: {metadata.cfg}</div>}
                        </div>
                      )}
                    </div>
                  );
                });
              })()}
            </div>
          ) : isDone ? (
            <div className={placeholderClass}>
              <span className={donePlaceholderClass}>{donePlaceholderMessage}</span>
            </div>
          ) : isRunning ? (
            <div className={placeholderClass} style={{ minHeight: '300px' }}>
              {isActuallyRunning && overallProgress != null && overallProgress < 100 ? (
                <div className="w-full max-w-xs px-4">
                  <p className="font-semibold text-base text-gray-900 mb-3">
                    Executing: {executingNodeLabel || 'Running'}
                  </p>
                  <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                    <span>Progress</span>
                    <span>{displayNodeProgress}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                    <div
                      className="h-full bg-green-500 transition-none"
                      style={{ width: `${Math.min(100, Math.max(0, displayNodeProgress))}%` }}
                    />
                  </div>
                  <div className="mt-3 flex items-center justify-between text-xs text-gray-500 mb-1">
                    <span>Overall</span>
                    <span>{overallProgress}%</span>
                  </div>
                  <div className="h-2 rounded-full bg-gray-200 overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-none"
                      style={{ width: `${Math.min(100, Math.max(0, overallProgress))}%` }}
                    />
                  </div>
                </div>
              ) : (
                <>
                  <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin mb-4" />
                  <p className="font-medium text-lg text-blue-600">Generating...</p>
                </>
              )}
            </div>
          ) : (
            <div className={placeholderClass} style={{ minHeight: '100px' }}>
              <HourglassIcon className="w-8 h-8 text-gray-300" />
              <span className="text-xs mt-2 opacity-40">Waiting to start...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function formatDuration(seconds?: number): string {
  const safeSeconds = seconds === undefined || Number.isNaN(seconds) ? 0 : seconds;
  if (safeSeconds < 10) return `${safeSeconds.toFixed(1)}s`;
  return `${Math.round(safeSeconds)}s`;
}
