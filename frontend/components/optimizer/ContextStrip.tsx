'use client';

import { StatusDot } from '@/components/ui/status-dot';
import { loadApiKeys } from '@/lib/settings';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import type { WorkflowData } from './types';

function Chip({
  label,
  value,
  tone,
}: {
  label: string;
  value: React.ReactNode;
  tone?: 'emerald';
}) {
  return (
    <span
      className={cn(
        'inline-flex max-w-[12rem] items-baseline gap-1 truncate rounded-full border border-border bg-muted px-2 py-0.5',
        tone === 'emerald' &&
          'border-accent-emerald bg-accent-emerald/30 text-accent-emerald-foreground'
      )}
    >
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate font-medium">{value}</span>
    </span>
  );
}

/**
 * Compact horizontal study context shown in the optimizer header band, replacing the old
 * right-rail StudySummary. Reuses the same workflow data + API-key readiness.
 */
export function ContextStrip({ workflowData }: { workflowData: WorkflowData }) {
  const { dataset, features, configuration, optimization } = workflowData;
  const [hasKey, setHasKey] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    loadApiKeys()
      .then((keys) => {
        if (active) setHasKey(Object.values(keys).some((v) => v.trim().length > 0));
      })
      .catch(() => active && setHasKey(false));
    return () => {
      active = false;
    };
  }, []);

  const bestF1 =
    optimization.status === 'completed' && optimization.bestValue !== null
      ? optimization.bestValue.toFixed(4)
      : null;

  return (
    <div className="flex min-w-0 flex-wrap items-center gap-1.5 text-xs">
      <Chip label="Dataset" value={dataset?.name ?? '—'} />
      <Chip label="Features" value={features.selectedFeatures.length || '—'} />
      <Chip label="Target" value={features.targetColumn ?? '—'} />
      <Chip label="Trials" value={configuration.numTrials} />
      {bestF1 && <Chip label="Best F1" value={bestF1} tone="emerald" />}
      {hasKey === null ? (
        <span className="inline-flex rounded-full border border-border bg-muted px-2 py-0.5">
          <StatusDot status="idle" label="Keys…" />
        </span>
      ) : hasKey ? (
        <span className="inline-flex rounded-full border border-border bg-muted px-2 py-0.5">
          <StatusDot status="online" label="API ready" />
        </span>
      ) : (
        <Link
          href="/settings"
          className="rounded-full border border-accent-amber bg-accent-amber/30 px-2 py-0.5 font-medium text-accent-amber-foreground hover:underline"
        >
          Add API key
        </Link>
      )}
    </div>
  );
}
