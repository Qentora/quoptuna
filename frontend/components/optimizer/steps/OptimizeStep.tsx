'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { Metric } from '@/components/ui/metric';
import { Progress } from '@/components/ui/progress';
import { StatusDot } from '@/components/ui/status-dot';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { fetchOptimizationTrials, pollOptimization, startOptimization } from '@/lib/api';
import { getDatabaseName } from '@/lib/appSettings';
import { cn } from '@/lib/utils';
import { Atom, Check, Cpu, PlayCircle, RotateCcw, Trophy } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts';
import { ErrorBanner } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';
import { isClassicalModel } from '../types';

export function OptimizeStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTrial, setCurrentTrial] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [liveTrials, setLiveTrials] = useState<any[]>([]);
  const [bestValue, setBestValue] = useState<number | null>(null);

  const { dataset, features, configuration, optimization } = workflowData;
  const hasResults = optimization.status === 'completed';
  const wasStopped =
    optimization.status === 'interrupted' ||
    optimization.status === 'cancelled' ||
    optimization.status === 'failed';

  useEffect(() => {
    setFooter({ canContinue: hasResults, nextBusy: isRunning, backDisabled: isRunning });
  }, [hasResults, isRunning, setFooter]);

  // Poll a job to completion, mirroring progress into local state and the
  // final outcome into workflowData. Shared by fresh starts and resumes.
  const track = async (id: string) => {
    setIsRunning(true);
    setError(null);
    try {
      const finalStatus = await pollOptimization(id, (status, trialsData) => {
        // The live trial list from the Optuna DB is fresher than the job's
        // current_trial counter; prefer it when available.
        const trialCount = trialsData?.trials?.length ?? status.current_trial;
        setCurrentTrial(trialCount);
        setProgress(status.total_trials ? (trialCount / status.total_trials) * 100 : 0);
        if (trialsData?.trials) {
          setLiveTrials(trialsData.trials);
          if (trialsData.best_trial) setBestValue(trialsData.best_trial.value);
        }
      });

      if (finalStatus.status !== 'completed') {
        setError(finalStatus.error || `Optimization ${finalStatus.status}`);
        setWorkflowData((prev) => ({
          ...prev,
          optimization: { ...prev.optimization, executionId: id, status: finalStatus.status },
        }));
        setIsRunning(false);
        return;
      }

      // Prefer the real per-trial history from the Optuna DB over the job's
      // stored trials: older backends synthesized completion trials from
      // best_params, which mislabels every row as the winning model family.
      const liveTrialsData = await fetchOptimizationTrials(id).catch(() => null);
      const trials = liveTrialsData?.trials?.length
        ? liveTrialsData.trials
        : (finalStatus.trials ?? []);
      const succeeded = trials.filter((t) => t.value !== null);
      const bestTrialNumber =
        succeeded.length > 0
          ? succeeded.reduce((best, t) => ((t.value ?? 0) > (best.value ?? 0) ? t : best)).trial
          : null;

      setWorkflowData((prev) => ({
        ...prev,
        optimization: {
          executionId: id,
          status: 'completed',
          bestValue: finalStatus.best_value,
          bestParams: finalStatus.best_params,
          trials,
          selectedTrial: prev.optimization.selectedTrial ?? bestTrialNumber,
        },
      }));
      setIsRunning(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Optimization failed');
      setIsRunning(false);
    }
  };

  const run = async () => {
    if (!dataset) {
      setError('No dataset selected');
      return;
    }
    setIsRunning(true);
    setProgress(0);
    setCurrentTrial(0);
    setError(null);

    const labelMapping =
      features.labelMapping.neg !== null && features.labelMapping.pos !== null
        ? { neg: features.labelMapping.neg, pos: features.labelMapping.pos }
        : undefined;

    try {
      const { id } = await startOptimization({
        dataset_id: dataset.id,
        dataset_source: dataset.source,
        selected_features: features.selectedFeatures,
        target_column: features.targetColumn as string,
        study_name: configuration.studyName,
        database_name: getDatabaseName(),
        num_trials: configuration.numTrials,
        label_mapping: labelMapping,
        sensitive_feature: features.sensitiveFeature ?? undefined,
        categorical_encoding: features.categoricalEncoding,
      });

      // Persist the execution id immediately so a refresh mid-run can resume.
      setWorkflowData((prev) => ({
        ...prev,
        optimization: {
          ...prev.optimization,
          executionId: id,
          status: 'running',
          selectedTrial: null,
        },
      }));

      await track(id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Optimization failed');
      setIsRunning(false);
    }
  };

  // On mount: resume polling for a still-running job (e.g. after a page
  // refresh), and refetch trials for a rehydrated completed run.
  const resumed = useRef(false);
  // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount.
  useEffect(() => {
    if (resumed.current) return;
    resumed.current = true;
    const id = optimization.executionId;
    if (!id) return;
    if (optimization.status === 'running' || optimization.status === 'pending') {
      void track(id);
    } else if (optimization.status === 'completed' && optimization.trials.length === 0) {
      fetchOptimizationTrials(id)
        .then((trialsData) => {
          const trials = trialsData.trials ?? [];
          const succeeded = trials.filter((t) => t.value !== null);
          const bestTrialNumber =
            succeeded.length > 0
              ? succeeded.reduce((best, t) => ((t.value ?? 0) > (best.value ?? 0) ? t : best)).trial
              : null;
          setWorkflowData((prev) => ({
            ...prev,
            optimization: {
              ...prev.optimization,
              trials,
              bestValue: trialsData.best_trial?.value ?? prev.optimization.bestValue,
              bestParams: trialsData.best_trial?.params ?? prev.optimization.bestParams,
              selectedTrial: prev.optimization.selectedTrial ?? bestTrialNumber,
            },
          }));
        })
        .catch(() => undefined);
    }
  }, []);

  const allTrials = hasResults ? optimization.trials : liveTrials;
  // Failed trials (value null) sort to the bottom and never win "best".
  const sorted = [...allTrials].sort((a, b) => (b.value ?? -1) - (a.value ?? -1));
  const bestQuantum = sorted.find(
    (t) => t.value !== null && !isClassicalModel(t.params?.model_type)
  );
  const bestClassical = sorted.find(
    (t) => t.value !== null && isClassicalModel(t.params?.model_type)
  );

  const selectTrial = (trialNumber: number) =>
    setWorkflowData((prev) => ({
      ...prev,
      optimization: { ...prev.optimization, selectedTrial: trialNumber },
    }));

  const displayBest = hasResults ? (optimization.bestValue ?? bestValue) : bestValue;

  return (
    <div className="flex flex-col gap-4">
      <StepHeader
        step={4}
        title="Run Optimization"
        subtitle="Execute the study and pick a trial to analyze"
      />

      <ErrorBanner message={error} />

      {!hasResults && !isRunning && (
        <Card>
          <CardContent className="flex flex-col items-center gap-3 text-center">
            <Button type="button" size="lg" onClick={run}>
              {wasStopped ? <RotateCcw /> : <PlayCircle />}
              {wasStopped ? 'Restart Optimization' : 'Start Optimization'}
            </Button>
            <p className="text-xs text-muted-foreground">
              {wasStopped
                ? `The previous run was ${optimization.status}. Restarting launches a new run with the same configuration.`
                : `Runs ${configuration.numTrials} trials across quantum and classical models. This may take several minutes.`}
            </p>
          </CardContent>
        </Card>
      )}

      {(isRunning || hasResults || liveTrials.length > 0) && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Card size="sm">
            <CardContent className="flex flex-col gap-1">
              <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
                Status
              </span>
              <span className="flex items-center gap-2 text-sm font-medium">
                <StatusDot status={isRunning ? 'busy' : hasResults ? 'online' : 'offline'} />
                {isRunning ? 'Running' : hasResults ? 'Completed' : optimization.status}
              </span>
            </CardContent>
          </Card>
          <Card size="sm">
            <CardContent className="flex flex-col gap-1">
              <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
                Trials
              </span>
              <span className="text-sm font-medium tabular-nums">
                {isRunning ? `${currentTrial} / ${configuration.numTrials}` : sorted.length}
              </span>
            </CardContent>
          </Card>
          <Card size="sm">
            <CardContent className="flex flex-col gap-1">
              <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
                Progress
              </span>
              <div className="flex items-center gap-2">
                <Metric
                  value={`${Math.round(hasResults ? 100 : progress)}%`}
                  tone="brand"
                  className="text-sm"
                />
                <Progress value={hasResults ? 100 : progress} className="flex-1" />
              </div>
            </CardContent>
          </Card>
          <Card size="sm">
            <CardContent className="flex flex-col gap-1">
              <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
                Best F1
              </span>
              {displayBest != null ? (
                <Metric value={displayBest.toFixed(4)} tone="emerald" className="text-sm" />
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {(hasResults || liveTrials.length > 0) && (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <BestTrialCard
            label="Best Quantum"
            trial={bestQuantum}
            accent="quantum"
            selectedTrial={optimization.selectedTrial}
            onSelect={hasResults ? selectTrial : undefined}
          />
          <BestTrialCard
            label="Best Classical"
            trial={bestClassical}
            accent="classical"
            selectedTrial={optimization.selectedTrial}
            onSelect={hasResults ? selectTrial : undefined}
          />
        </div>
      )}

      <div className="grid grid-cols-1 items-start gap-3 xl:grid-cols-2">
        <OptimizationHistoryCard trials={allTrials} />

        {allTrials.length > 0 && (
          <Card size="sm" className="gap-0 py-0">
            <div className="flex items-center justify-between gap-3 border-b border-border bg-muted px-3 py-2">
              <h4 className="text-xs font-semibold text-foreground">
                All trials ({sorted.length})
              </h4>
              {hasResults && (
                <span className="text-xs text-muted-foreground">
                  {optimization.selectedTrial !== null ? (
                    <>
                      Analyzing{' '}
                      <span className="font-medium text-brand">#{optimization.selectedTrial}</span>{' '}
                      — click any row to change
                    </>
                  ) : (
                    'Click a row to choose a trial to analyze'
                  )}
                </span>
              )}
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10" />
                  <TableHead className="w-12 text-right">Rank</TableHead>
                  <TableHead className="text-right">F1 score</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead className="text-right">Trial</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {sorted.map((trial, index) => {
                  const failed = trial.value === null || trial.state === 'FAIL';
                  const selectable = hasResults && !failed;
                  const selected = optimization.selectedTrial === trial.trial;
                  const classical = isClassicalModel(trial.params?.model_type);
                  const isBest = index === 0 && !failed;
                  return (
                    <TableRow
                      key={trial.trial}
                      onClick={selectable ? () => selectTrial(trial.trial) : undefined}
                      onKeyDown={
                        selectable
                          ? (e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault();
                                selectTrial(trial.trial);
                              }
                            }
                          : undefined
                      }
                      tabIndex={selectable ? 0 : undefined}
                      aria-selected={selected}
                      className={cn(
                        'focus:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand',
                        selectable && 'cursor-pointer',
                        failed && 'opacity-60',
                        selected && 'bg-brand/10 ring-1 ring-inset ring-brand/40 hover:bg-brand/10'
                      )}
                    >
                      <TableCell className="text-center">
                        {selected ? (
                          <Check size={16} className="mx-auto text-brand" />
                        ) : isBest ? (
                          <Trophy size={14} className="mx-auto text-accent-amber-foreground" />
                        ) : null}
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                        {index + 1}
                      </TableCell>
                      <TableCell
                        className={cn(
                          'text-right font-semibold tabular-nums',
                          selected && 'text-brand'
                        )}
                      >
                        {failed ? (
                          <span
                            className="font-medium text-destructive"
                            title={String(trial.user_attrs?.error ?? 'Trial failed')}
                          >
                            failed
                          </span>
                        ) : (
                          trial.value?.toFixed(4)
                        )}
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {trial.params?.model_type ?? 'N/A'}
                        {failed && trial.user_attrs?.error != null && (
                          <span className="block max-w-[280px] truncate text-[11px] text-destructive/80">
                            {String(trial.user_attrs.error)}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-xs">
                        <Badge variant={classical ? 'classical' : 'quantum'}>
                          {classical ? 'Classical' : 'Quantum'}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right text-xs text-muted-foreground tabular-nums">
                        #{trial.trial}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </Card>
        )}
      </div>
    </div>
  );
}

interface HistoryTrial {
  trial: number;
  value: number | null;
  params?: Record<string, unknown>;
}

function OptimizationHistoryCard({ trials }: { trials: HistoryTrial[] }) {
  const valid = trials
    .filter((t) => t.value !== null)
    .sort((a, b) => a.trial - b.trial)
    .map((t) => ({
      trial: t.trial,
      value: t.value as number,
      model: (t.params?.model_type as string | undefined) ?? 'N/A',
    }));
  if (valid.length === 0) return null;
  // One row per trial with the value under its family's key so each family
  // renders as its own connected line with dot markers.
  const chartData = valid.map((t) => ({
    trial: t.trial,
    model: t.model,
    quantum: isClassicalModel(t.model) ? null : t.value,
    classical: isClassicalModel(t.model) ? t.value : null,
  }));
  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle className="text-sm">Optimization history</CardTitle>
        <CardDescription className="text-xs">
          F1 score per trial, colored by model family
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            quantum: { label: 'Quantum', color: 'var(--chart-1)' },
            classical: { label: 'Classical', color: 'var(--chart-2)' },
          }}
          className="aspect-[16/10] min-h-64 w-full"
        >
          <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="trial"
              type="number"
              name="Trial"
              domain={[0, 'dataMax']}
              allowDecimals={false}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 12 }}
            />
            <YAxis
              type="number"
              name="F1"
              domain={[0, 1]}
              width={34}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 12 }}
            />
            <ChartTooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={
                <ChartTooltipContent
                  labelFormatter={(_, payload) => {
                    const p = payload?.[0]?.payload as { trial: number; model: string } | undefined;
                    return p ? `Trial #${p.trial} · ${p.model}` : '';
                  }}
                />
              }
            />
            <Line
              dataKey="quantum"
              type="monotone"
              stroke="var(--color-quantum)"
              strokeWidth={2}
              connectNulls
              dot={{ r: 4, fill: 'var(--color-quantum)', strokeWidth: 0 }}
              activeDot={{ r: 6 }}
            />
            <Line
              dataKey="classical"
              type="monotone"
              stroke="var(--color-classical)"
              strokeWidth={2}
              connectNulls
              dot={{ r: 4, fill: 'var(--color-classical)', strokeWidth: 0 }}
              activeDot={{ r: 6 }}
            />
            <ChartLegend content={<ChartLegendContent />} />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

function BestTrialCard({
  label,
  trial,
  accent,
  selectedTrial,
  onSelect,
}: {
  label: string;
  trial: { trial: number; value: number; params?: Record<string, unknown> } | undefined;
  accent: 'quantum' | 'classical';
  selectedTrial: number | null;
  onSelect?: (trialNumber: number) => void;
}) {
  const accentClasses =
    accent === 'quantum'
      ? 'ring-accent-purple bg-accent-purple/20'
      : 'ring-accent-orange bg-accent-orange/20';
  const isSelected = !!trial && selectedTrial === trial.trial;
  const Icon = accent === 'quantum' ? Atom : Cpu;
  const modelType = (trial?.params?.model_type as string | undefined) ?? 'N/A';

  return (
    <Card size="sm" className={cn(accentClasses, isSelected && 'ring-2 ring-inset ring-brand/50')}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-sm">
          <span className="flex items-center gap-2">
            <Icon
              size={16}
              className={
                accent === 'quantum'
                  ? 'text-accent-purple-foreground'
                  : 'text-accent-orange-foreground'
              }
            />
            {label}
          </span>
          {isSelected && (
            <Badge variant="emerald">
              <Check size={12} /> Analyzing
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {trial ? (
          <>
            <div className="flex items-baseline gap-2">
              <Metric
                value={trial.value.toFixed(4)}
                tone={accent === 'quantum' ? 'brand' : 'amber'}
                className="text-3xl"
              />
              <span className="text-xs text-muted-foreground">F1</span>
            </div>
            <p className="mt-1 truncate text-xs text-muted-foreground">
              {modelType} · trial #{trial.trial}
            </p>
            {onSelect && !isSelected && (
              <Button
                type="button"
                size="sm"
                variant="secondary"
                className="mt-3"
                onClick={() => onSelect(trial.trial)}
              >
                Analyze this
              </Button>
            )}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">No trial yet</p>
        )}
      </CardContent>
    </Card>
  );
}
