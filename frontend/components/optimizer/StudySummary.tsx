'use client';

import { StatusDot } from '@/components/ui/status-dot';
import { loadApiKeys } from '@/lib/settings';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import type { WorkflowData } from './types';

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="truncate text-right text-sm font-medium">{value}</span>
    </div>
  );
}

export function StudySummary({
  workflowData,
  className,
}: {
  workflowData: WorkflowData;
  className?: string;
}) {
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
    <aside className={cn('rounded-lg border border-border bg-card p-4 text-sm', className)}>
      <h3 className="mb-3 font-semibold tracking-tight">Study Summary</h3>
      <div className="divide-y divide-border">
        <Row label="Dataset" value={dataset?.name ?? <Muted>Not selected</Muted>} />
        <Row
          label="Features"
          value={
            features.selectedFeatures.length > 0 ? (
              features.selectedFeatures.length
            ) : (
              <Muted>—</Muted>
            )
          }
        />
        <Row label="Target" value={features.targetColumn ?? <Muted>—</Muted>} />
        <Row label="Trials" value={configuration.numTrials} />
        {bestF1 && (
          <Row
            label="Best F1"
            value={<span className="text-accent-emerald-foreground">{bestF1}</span>}
          />
        )}
      </div>

      <div className="mt-3 border-t border-border pt-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs text-muted-foreground">API key</span>
          {hasKey === null ? (
            <StatusDot status="idle" label="Checking…" className="text-xs" />
          ) : hasKey ? (
            <StatusDot status="online" label="Ready" className="text-xs" />
          ) : (
            <Link href="/settings" className="text-xs font-medium text-brand hover:underline">
              Add in Settings
            </Link>
          )}
        </div>
      </div>
    </aside>
  );
}

function Muted({ children }: { children: React.ReactNode }) {
  return <span className="text-muted-foreground">{children}</span>;
}
