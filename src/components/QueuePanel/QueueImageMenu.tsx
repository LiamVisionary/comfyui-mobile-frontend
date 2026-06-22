import { createPortal } from 'react-dom';
import type { FileItem } from '@/api/client';
import type { Workflow } from '@/api/types';
import type { UnifiedItem } from './types';
import { CopyIcon, DownloadDeviceIcon, EyeIcon, EyeOffIcon, FolderIcon, TrashIcon, WorkflowIcon, BookmarkOutlineIcon } from '@/components/icons';
import { useQueueStore } from '@/hooks/useQueue';
import { ContextMenuBuilder } from '@/components/menus/ContextMenuBuilder';

interface QueueImageMenuProps {
  menuState: {
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
  } | null;
  unifiedList: UnifiedItem[];
  onClose: () => void;
  onLoadWorkflow: (workflow: Workflow, promptId: string) => void;
  onCopyWorkflow: (workflow?: Workflow) => void;
  onFavoriteWorkflow: (payload: { workflow?: Workflow; prompt?: Record<string, unknown>; file?: FileItem; src?: string; promptId?: string }) => void;
  onSaveToAlbum: (src: string) => Promise<void>;
  onSaveToFiles: (src: string) => Promise<void>;
  onBatchSaveToAlbum: (sources: string[]) => Promise<void>;
  onBatchSaveToFiles: (sources: string[]) => Promise<void>;
  onDeleteHistory: (promptId: string) => void;
  getBatchSources: (promptId: string, list: UnifiedItem[]) => string[];
}

export function QueueImageMenu({
  menuState,
  unifiedList,
  onClose,
  onLoadWorkflow,
  onCopyWorkflow,
  onFavoriteWorkflow,
  onSaveToAlbum,
  onSaveToFiles,
  onBatchSaveToAlbum,
  onBatchSaveToFiles,
  onDeleteHistory,
  getBatchSources
}: QueueImageMenuProps) {
  const promptId = menuState?.promptId ?? '';
  const queueItemHideImages = useQueueStore((s) => s.queueItemHideImages[promptId]);
  const toggleQueueItemHideImages = useQueueStore((s) => s.toggleQueueItemHideImages);

  const handleLoadWorkflowClick = () => {
    if (menuState?.workflow && menuState.promptId) {
      onLoadWorkflow(menuState.workflow, menuState.promptId);
    }
    onClose();
  };

  const handleCopyWorkflowClick = () => {
    onCopyWorkflow(menuState?.workflow);
    onClose();
  };

  const handleFavoriteWorkflowClick = () => {
    onFavoriteWorkflow({
      workflow: menuState?.workflow,
      prompt: menuState?.prompt,
      file: menuState?.file,
      src: menuState?.imageSrc,
      promptId: menuState?.promptId,
    });
    onClose();
  };

  const getCurrentBatchSources = () => menuState?.promptId
    ? getBatchSources(menuState.promptId, unifiedList)
    : [];

  const handleSaveToAlbumClick = async () => {
    const batchSources = getCurrentBatchSources();
    if (batchSources.length > 1) {
      await onBatchSaveToAlbum(batchSources);
    } else if (menuState) {
      await onSaveToAlbum(menuState.imageSrc);
    }
    onClose();
  };

  const handleSaveToFilesClick = async () => {
    const batchSources = getCurrentBatchSources();
    if (batchSources.length > 1) {
      await onBatchSaveToFiles(batchSources);
    } else if (menuState) {
      await onSaveToFiles(menuState.imageSrc);
    }
    onClose();
  };

  const handleToggleHideImagesClick = () => {
    toggleQueueItemHideImages(promptId);
    onClose();
  };

  const handleDeleteClick = () => {
    if (menuState?.promptId) {
      onDeleteHistory(menuState.promptId);
    }
    onClose();
  };

  if (!menuState?.open) return null;

  return createPortal(
    <div
      id="queue-image-menu"
      className="fixed z-[1200] w-44"
      style={{ top: menuState.top, right: menuState.right }}
    >
      <ContextMenuBuilder
        items={[
          {
            key: 'load-workflow',
            label: 'Load workflow',
            icon: <WorkflowIcon className="w-4 h-4" />,
            onClick: handleLoadWorkflowClick,
            disabled: !menuState.workflow
          },
          {
            key: 'copy-workflow',
            label: 'Copy workflow',
            icon: <CopyIcon className="w-4 h-4" />,
            onClick: handleCopyWorkflowClick,
            disabled: !menuState.workflow
          },
          {
            key: 'favorite-workflow',
            label: 'Favorite workflow shortcut',
            icon: <BookmarkOutlineIcon className="w-4 h-4" />,
            onClick: handleFavoriteWorkflowClick,
            disabled: !menuState.workflow || !menuState.file
          },
          {
            key: 'save-album',
            label: menuState.promptId && getBatchSources(menuState.promptId, unifiedList).length > 1
              ? 'Save batch to album'
              : 'Save image to album',
            icon: <DownloadDeviceIcon className="w-4 h-4" />,
            onClick: handleSaveToAlbumClick
          },
          {
            key: 'save-files',
            label: menuState.promptId && getBatchSources(menuState.promptId, unifiedList).length > 1
              ? 'Save batch to Files'
              : 'Save to Files',
            icon: <FolderIcon className="w-4 h-4" />,
            onClick: handleSaveToFilesClick
          },
          {
            key: 'toggle-images',
            label: queueItemHideImages ? 'Show images' : 'Hide images',
            icon: queueItemHideImages
              ? <EyeIcon className="w-4 h-4" />
              : <EyeOffIcon className="w-4 h-4" />,
            onClick: handleToggleHideImagesClick,
            hidden: !(menuState.hasVideoOutputs && menuState.hasImageOutputs && menuState.promptId)
          },
          {
            key: 'delete',
            label: 'Delete',
            icon: <TrashIcon className="w-4 h-4" />,
            onClick: handleDeleteClick,
            color: 'danger'
          }
        ]}
      />
    </div>,
    document.body
  );
}
