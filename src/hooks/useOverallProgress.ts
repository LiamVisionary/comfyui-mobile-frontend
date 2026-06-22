import { useEffect, useMemo, useRef, useState } from 'react';
import type { Workflow } from '@/api/types';
import { useQueueStore } from '@/hooks/useQueue';
import { useWorkflowStore } from '@/hooks/useWorkflow';

interface OverallProgressInput {
  workflow: Workflow | null;
  runKey: string | null;
  isRunning: boolean;
  workflowDurationStats: Record<string, { avgMs: number; count: number }>;
}

const isExecutableWorkflowNode = (node: Workflow['nodes'][number]): boolean => {
  // ComfyUI mode 4 means bypassed. Notes/reroutes/groups are not prompt nodes.
  if (node.mode === 4) return false;
  if (node.type === 'Note' || node.type === 'Reroute' || node.type === 'PrimitiveNode') return false;
  return true;
};

export function useOverallProgress({
  workflow,
  runKey,
  isRunning,
  workflowDurationStats,
}: OverallProgressInput): number | null {
  const running = useQueueStore((state) => state.running);
  const currentNodePath = useWorkflowStore((state) => state.executingNodePath);
  const currentNodeProgress = useWorkflowStore((state) => state.progress);
  const completedNodeIds = useWorkflowStore((state) => state.executionCompletedNodeIds);
  const [heldComplete, setHeldComplete] = useState(false);
  const holdTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastRunKeyRef = useRef<string | null>(null);

  const promptNodeIds = useMemo(() => {
    if (!runKey) return [] as string[];
    const runningItem = running.find((item) => item.prompt_id === runKey) ?? running[0];
    if (!runningItem?.prompt || typeof runningItem.prompt !== 'object') return [] as string[];
    return Object.keys(runningItem.prompt);
  }, [runKey, running]);

  const fallbackTotalNodes = useMemo(() => {
    if (!workflow) return 0;
    return workflow.nodes.filter(isExecutableWorkflowNode).length;
  }, [workflow]);

  const realPercent = useMemo(() => {
    if (!workflow || !runKey || !isRunning) return null;

    const totalNodes = promptNodeIds.length || fallbackTotalNodes;
    if (totalNodes <= 0) return null;

    let completed = 0;
    const counted = new Set<string>();
    const denominatorIds = promptNodeIds.length ? new Set(promptNodeIds) : null;

    for (const nodeId of Object.keys(completedNodeIds)) {
      if (denominatorIds && !denominatorIds.has(nodeId)) continue;
      if (counted.has(nodeId)) continue;
      counted.add(nodeId);
      completed += 1;
    }

    const currentId = currentNodePath?.trim() ?? '';
    const currentBelongsToPrompt =
      currentId.length > 0 && (!denominatorIds || denominatorIds.has(currentId));
    const currentFraction =
      currentBelongsToPrompt && !counted.has(currentId)
        ? Math.min(0.99, Math.max(0, currentNodeProgress / 100))
        : 0;

    const percent = ((completed + currentFraction) / totalNodes) * 100;
    const rounded = Math.min(100, Math.max(0, Math.round(percent)));

    // While ComfyUI still reports an active run, never let the derived node
    // counter reach 100. Some workflows have redacted/expanded/cached prompt
    // node sets that make the UI-side denominator too small, so the overlay can
    // hit 100 after text encode even though sampler/save nodes are still running.
    // The real completion signal is `executing: { node: null }`, which flips
    // isRunning false and allows the overlay to close.
    return isRunning ? Math.min(99, rounded) : rounded;
  }, [workflow, runKey, isRunning, promptNodeIds, fallbackTotalNodes, completedNodeIds, currentNodePath, currentNodeProgress]);

  useEffect(() => {
    // `workflowDurationStats` is intentionally accepted for call-site stability,
    // but overall progress is now derived only from real ComfyUI node events.
    void workflowDurationStats;
  }, [workflowDurationStats]);

  useEffect(() => {
    if (holdTimeoutRef.current) {
      clearTimeout(holdTimeoutRef.current);
      holdTimeoutRef.current = null;
    }

    if (runKey) {
      lastRunKeyRef.current = runKey;
      setHeldComplete(false);
      return;
    }

    if (lastRunKeyRef.current) {
      lastRunKeyRef.current = null;
      setHeldComplete(true);
      holdTimeoutRef.current = setTimeout(() => setHeldComplete(false), 250);
      return;
    }

    setHeldComplete(false);

    return () => {
      if (holdTimeoutRef.current) {
        clearTimeout(holdTimeoutRef.current);
        holdTimeoutRef.current = null;
      }
    };
  }, [runKey]);

  if (realPercent !== null) return realPercent;
  if (heldComplete) return 100;
  return null;
}
