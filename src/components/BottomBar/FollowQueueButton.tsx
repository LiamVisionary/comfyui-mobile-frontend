import { useCallback, useMemo } from "react";
import { ProgressRing, QueueStackIcon } from "@/components/icons";

interface FollowQueueButtonProps {
  currentPanel: 'workflow' | 'queue' | 'outputs';
  viewerOpen: boolean;
  followQueue: boolean;
  queueSize: number;
  overallProgress: number | null;
  onOpenFollowQueue?: () => void;
  onCloseFollowQueue?: () => void;
}

export function FollowQueueButton({
  currentPanel,
  viewerOpen,
  followQueue,
  queueSize,
  overallProgress,
  onOpenFollowQueue,
  onCloseFollowQueue,
}: FollowQueueButtonProps) {
  const handleClick = useCallback(() => {
    if (viewerOpen) {
      onCloseFollowQueue?.();
    } else {
      onOpenFollowQueue?.();
    }
  }, [viewerOpen, onCloseFollowQueue, onOpenFollowQueue]);

  const ariaLabel = useMemo(() => {
    if (viewerOpen) return "Close image viewer and return to queue and recent outputs";
    return "Open queue and recent outputs";
  }, [viewerOpen]);

  const buttonClassName = useMemo(() => {
    if (currentPanel === 'queue' && !viewerOpen) return "bg-blue-500 text-white";
    if (!viewerOpen) return "bg-gray-100 text-gray-700 hover:bg-gray-200";
    return followQueue
      ? "bg-green-500 text-white"
      : "bg-gray-100 text-gray-700 hover:bg-gray-200";
  }, [currentPanel, viewerOpen, followQueue]);

  return (
    <button
      onClick={handleClick}
      className={`relative w-12 h-12 rounded-xl flex items-center justify-center text-2xl transition-colors ${buttonClassName}`}
      aria-label={ariaLabel}
    >
      <span className="absolute inset-0 flex items-center justify-center">
        <QueueStackIcon
          className="w-6 h-6"
          showSlash={viewerOpen && !followQueue}
        />
      </span>
      {queueSize > 0 && (
        <div
          className="queue-badge absolute top-0 right-0 translate-x-[18px] -translate-y-[18px] w-6 h-6 rounded-full bg-blue-500 text-white
                     flex items-center justify-center font-bold text-xs border-2 border-white relative z-20"
        >
          {overallProgress !== null && (
            <ProgressRing
              className="absolute z-10 pointer-events-none"
              width="24"
              height="24"
              style={{
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%) rotate(-90deg)",
              }}
              progress={overallProgress}
            />
          )}
          {queueSize}
        </div>
      )}
    </button>
  );
}
