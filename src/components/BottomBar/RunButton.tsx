import { useState } from 'react';
import { useWorkflowStore } from '@/hooks/useWorkflow';
import { useQueueStore } from '@/hooks/useQueue';

const MIN_VISIBLE_STARTING_MS = 450;

export function RunButton() {
  const workflow = useWorkflowStore((s) => s.workflow);
  const runCount = useWorkflowStore((s) => s.runCount);
  const isLoading = useWorkflowStore((s) => s.isLoading);
  const isExecuting = useWorkflowStore((s) => s.isExecuting);
  const queueWorkflow = useWorkflowStore((s) => s.queueWorkflow);
  const runningCount = useQueueStore((s) => s.running.length);
  const pendingCount = useQueueStore((s) => s.pending.length);
  const [isStarting, setIsStarting] = useState(false);
  const canRun = workflow !== null && !isStarting;
  const isBusy = isStarting || isLoading || isExecuting || runningCount > 0;

  const handleRun = async () => {
    if (!canRun) return;

    setIsStarting(true);
    const startedAt = Date.now();

    if ('vibrate' in navigator) {
      navigator.vibrate(20);
    }

    try {
      await queueWorkflow(runCount);
    } finally {
      const elapsed = Date.now() - startedAt;
      const remaining = Math.max(0, MIN_VISIBLE_STARTING_MS - elapsed);
      window.setTimeout(() => setIsStarting(false), remaining);
    }
  };

  const label = isStarting || isLoading
    ? 'Starting…'
    : isExecuting || runningCount > 0
      ? 'Running…'
      : pendingCount > 0
        ? 'Queued'
        : 'Run';

  return (
    <button
      onClick={handleRun}
      disabled={!canRun}
      aria-busy={isBusy}
      className={
        `flex-1 py-3 px-6 rounded-xl font-semibold text-lg min-h-[48px] transition-all flex items-center justify-center gap-2 `
        + (canRun
          ? 'bg-blue-500 text-white active:bg-blue-600'
          : 'bg-gray-300 text-gray-500 cursor-not-allowed')
      }
    >
      {isBusy && (
        <span
          className="inline-block h-4 w-4 rounded-full border-2 border-current border-t-transparent animate-spin"
          aria-hidden="true"
        />
      )}
      <span>{label}</span>
    </button>
  );
}
