import { DeleteButton } from "@/components/buttons/DeleteButton";
import { LoadWorkflowButton } from "@/components/buttons/LoadWorkflowButton";
import { UseInWorkflowButton } from "@/components/buttons/UseInWorkflowButton";
import { MetadataButton } from "@/components/buttons/MetadataButton";
import { DownloadDeviceIcon, FolderIcon, BookmarkOutlineIcon } from "@/components/icons";

interface MediaViewerActionsProps {
  isVideo: boolean;
  canLoadWorkflow: boolean;
  showMetadataToggle?: boolean;
  canToggleMetadata: boolean;
  onDelete: () => void;
  onLoadWorkflow: () => void;
  onUseInWorkflow: () => void;
  onFavoriteWorkflow?: () => void;
  onToggleMetadata: () => void;
  onSaveToAlbum: () => void;
  onSaveToFiles: () => void;
}

export function MediaViewerActions({
  isVideo,
  canLoadWorkflow,
  showMetadataToggle,
  canToggleMetadata,
  onDelete,
  onLoadWorkflow,
  onUseInWorkflow,
  onFavoriteWorkflow,
  onToggleMetadata,
  onSaveToAlbum,
  onSaveToFiles,
}: MediaViewerActionsProps) {
  return (
    <div
      className="absolute inset-x-0 px-3 pb-2 pt-2 flex items-center justify-between"
      style={{ bottom: "calc(var(--bottom-bar-offset, 0px) + 4px)" }}
    >
      <DeleteButton onClick={onDelete} />
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onSaveToAlbum}
          aria-label="Save image to album"
          className="pointer-events-auto w-9 h-9 rounded-full bg-black/40 text-white flex items-center justify-center hover:bg-black/60 transition-colors"
        >
          <DownloadDeviceIcon className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={onSaveToFiles}
          aria-label="Save to Files"
          className="pointer-events-auto w-9 h-9 rounded-full bg-black/40 text-white flex items-center justify-center hover:bg-black/60 transition-colors"
        >
          <FolderIcon className="w-5 h-5" />
        </button>
        {onFavoriteWorkflow && (
          <button
            type="button"
            onClick={onFavoriteWorkflow}
            aria-label="Favorite workflow shortcut"
            className="pointer-events-auto w-9 h-9 rounded-full bg-black/40 text-white flex items-center justify-center hover:bg-black/60 transition-colors"
          >
            <BookmarkOutlineIcon className="w-5 h-5" />
          </button>
        )}
        {canLoadWorkflow && <LoadWorkflowButton onClick={onLoadWorkflow} />}
        {!isVideo && (
          <>
          <UseInWorkflowButton onClick={onUseInWorkflow} />
          {showMetadataToggle && (
            <MetadataButton
              onClick={onToggleMetadata}
              disabled={!canToggleMetadata}
            />
          )}
          </>
        )}
      </div>
    </div>
  );
}
