'use client';

import { initialWorkflowData } from '@/components/optimizer/types';
import type { WorkflowData } from '@/components/optimizer/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Metric } from '@/components/ui/metric';
import { PageShell } from '@/components/ui/page-shell';
import { StatusDot } from '@/components/ui/status-dot';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
import {
  type PastRun,
  type RunStatus,
  deleteOptimization,
  getOptimizationDetail,
  listOptimizations,
} from '@/lib/api';
import { saveWizardState } from '@/lib/wizardStorage';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle2, CircleSlash, Loader2, PauseCircle, Trash2, XCircle } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import toast from 'react-hot-toast';

const ACTIVE_STATUSES: RunStatus[] = ['running', 'pending'];

const STATUS_BADGE: Record<RunStatus, { label: string; variant: string }> = {
  completed: { label: 'Completed', variant: 'emerald' },
  running: { label: 'Running', variant: 'quantum' },
  pending: { label: 'Pending', variant: 'secondary' },
  failed: { label: 'Failed', variant: 'destructive' },
  cancelled: { label: 'Cancelled', variant: 'secondary' },
  interrupted: { label: 'Interrupted', variant: 'classical' },
};

function formatDate(value: string | null): string {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString();
}

export default function RunsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [openingId, setOpeningId] = useState<string | null>(null);

  const { data: runs = [], isLoading } = useQuery({
    queryKey: ['optimization-runs'],
    queryFn: listOptimizations,
    // Keep the "current runs" section live while anything is in progress.
    refetchInterval: (query) =>
      (query.state.data ?? []).some((r) => ACTIVE_STATUSES.includes(r.status)) ? 3000 : false,
  });

  const activeRuns = runs.filter((r) => ACTIVE_STATUSES.includes(r.status));
  const pastRuns = runs.filter((r) => !ACTIVE_STATUSES.includes(r.status));
  const count = (status: RunStatus) => runs.filter((r) => r.status === status).length;

  /** Rehydrate the wizard from a persisted run and jump into it. */
  const openRun = async (run: PastRun) => {
    setOpeningId(run.id);
    try {
      const detail = await getOptimizationDetail(run.id);
      const req = detail.request ?? {};
      const workflowData: WorkflowData = {
        ...initialWorkflowData,
        dataset: detail.dataset
          ? {
              id: detail.dataset.id,
              name: detail.dataset.name,
              source: detail.dataset.source,
              rows: detail.dataset.rows ?? 0,
              columns: detail.dataset.columns ?? [],
            }
          : req.dataset_id
            ? {
                id: req.dataset_id,
                name: run.dataset_name ?? req.dataset_id,
                source: req.dataset_source ?? 'upload',
                rows: 0,
                columns: [],
              }
            : null,
        features: {
          selectedFeatures: req.selected_features ?? [],
          targetColumn: req.target_column ?? null,
          labelMapping: {
            neg: req.label_mapping?.neg ?? null,
            pos: req.label_mapping?.pos ?? null,
          },
          sensitiveFeature: req.sensitive_feature ?? null,
        },
        configuration: {
          studyName: req.study_name ?? run.study_name ?? 'my-optimization-study',
          numTrials: req.num_trials ?? 50,
        },
        optimization: {
          executionId: detail.id,
          status: detail.status,
          bestValue: detail.best_value,
          bestParams: detail.best_params,
          trials: [],
          selectedTrial: null,
        },
        analysis: { ...initialWorkflowData.analysis },
        report: { markdown: null },
      };

      const completed = detail.status === 'completed';
      saveWizardState({
        currentStep: completed ? 5 : 4,
        completedSteps: completed ? [1, 2, 3, 4] : [1, 2, 3],
        workflowData,
      });
      router.push('/optimizer');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Could not open run');
      setOpeningId(null);
    }
  };

  const removeRun = async (run: PastRun) => {
    try {
      await deleteOptimization(run.id);
      toast.success(
        ACTIVE_STATUSES.includes(run.status) ? 'Optimization cancelled' : 'Run deleted'
      );
      await queryClient.invalidateQueries({ queryKey: ['optimization-runs'] });
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Could not delete run');
    }
  };

  return (
    <PageShell title="Runs" contentClassName="mx-auto max-w-6xl">
      {/* Study status summary */}
      <div className="mb-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          title="Completed"
          value={count('completed')}
          icon={<CheckCircle2 className="h-5 w-5 text-accent-emerald-foreground" />}
          tone="emerald"
        />
        <StatCard
          title="In Progress"
          value={activeRuns.length}
          icon={<Loader2 className="h-5 w-5 text-brand" />}
          tone="brand"
        />
        <StatCard
          title="Interrupted"
          value={count('interrupted') + count('cancelled')}
          icon={<PauseCircle className="h-5 w-5 text-accent-orange-foreground" />}
          tone="amber"
        />
        <StatCard
          title="Failed"
          value={count('failed')}
          icon={<XCircle className="h-5 w-5 text-destructive" />}
        />
      </div>

      {/* Current runs */}
      <section className="mb-6">
        <h2 className="mb-3 text-base font-semibold tracking-tight">Current Runs</h2>
        {activeRuns.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="p-4 text-sm text-muted-foreground">
              No optimization is currently running.
            </CardContent>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            {activeRuns.map((run) => (
              <Card key={run.id}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex min-w-0 items-center gap-2">
                      <StatusDot status="busy" />
                      <p className="truncate font-medium">{run.study_name ?? run.id}</p>
                    </div>
                    <Badge variant={STATUS_BADGE[run.status].variant as any}>
                      {STATUS_BADGE[run.status].label}
                    </Badge>
                  </div>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {run.dataset_name ?? 'Unknown dataset'} · trial {run.current_trial ?? 0} of{' '}
                    {run.total_trials ?? '—'}
                  </p>
                  <div className="mt-3 flex items-center gap-2">
                    <Button
                      type="button"
                      size="sm"
                      onClick={() => void openRun(run)}
                      disabled={openingId === run.id}
                    >
                      {openingId === run.id ? 'Opening…' : 'Open'}
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void removeRun(run)}
                    >
                      <CircleSlash className="h-4 w-4" />
                      Cancel
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </section>

      {/* Past runs */}
      <section>
        <h2 className="mb-3 text-base font-semibold tracking-tight">Past Runs &amp; Studies</h2>
        {isLoading ? (
          <Card className="border-dashed">
            <CardContent className="p-4 text-sm text-muted-foreground">Loading runs…</CardContent>
          </Card>
        ) : pastRuns.length === 0 ? (
          <Card className="border-dashed">
            <CardContent className="p-6 text-center text-sm text-muted-foreground">
              No past runs yet. Finished optimizations will appear here and can be reopened at any
              time — even after a backend restart.
            </CardContent>
          </Card>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <tr>
                  <Th>Study</Th>
                  <Th>Dataset</Th>
                  <Th>Status</Th>
                  <Th className="text-right">Best F1</Th>
                  <Th className="text-right">Trials</Th>
                  <Th>Started</Th>
                  <Th>Completed</Th>
                  <Th className="w-32 text-right">Actions</Th>
                </tr>
              </TableHead>
              <TableBody>
                {pastRuns.map((run) => (
                  <tr key={run.id} className="transition-colors hover:bg-muted">
                    <Td className="font-medium">{run.study_name ?? run.id}</Td>
                    <Td className="text-xs text-muted-foreground">{run.dataset_name ?? '—'}</Td>
                    <Td>
                      <Badge variant={STATUS_BADGE[run.status].variant as any}>
                        {STATUS_BADGE[run.status].label}
                      </Badge>
                    </Td>
                    <Td className="text-right font-semibold tabular-nums">
                      {run.best_value != null ? run.best_value.toFixed(4) : '—'}
                    </Td>
                    <Td className="text-right text-xs text-muted-foreground tabular-nums">
                      {run.current_trial ?? 0}/{run.total_trials ?? '—'}
                    </Td>
                    <Td className="text-xs text-muted-foreground">{formatDate(run.started_at)}</Td>
                    <Td className="text-xs text-muted-foreground">
                      {formatDate(run.completed_at)}
                    </Td>
                    <Td className="text-right">
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          type="button"
                          size="sm"
                          variant="secondary"
                          onClick={() => void openRun(run)}
                          disabled={openingId === run.id}
                        >
                          {openingId === run.id ? 'Opening…' : 'Open'}
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="ghost"
                          aria-label="Delete run"
                          onClick={() => void removeRun(run)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </Td>
                  </tr>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </section>
    </PageShell>
  );
}

function StatCard({
  title,
  value,
  icon,
  tone = 'default',
}: {
  title: string;
  value: number;
  icon: React.ReactNode;
  tone?: 'default' | 'emerald' | 'brand' | 'amber';
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-medium text-muted-foreground">{title}</h3>
          <span className="flex h-8 w-8 items-center justify-center rounded-md bg-muted">
            {icon}
          </span>
        </div>
        <Metric value={value} tone={tone} className="block text-2xl" />
      </CardContent>
    </Card>
  );
}
