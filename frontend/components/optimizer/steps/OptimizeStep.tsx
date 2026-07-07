'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Metric } from '@/components/ui/metric';
import { StatusDot } from '@/components/ui/status-dot';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
import { fetchOptimizationTrials, pollOptimization, startOptimization } from '@/lib/api';
import { getDatabaseName } from '@/lib/appSettings';
import { cn } from '@/lib/utils';
import * as Progress from '@radix-ui/react-progress';
import { Atom, Check, Cpu, PlayCircle, RotateCcw, Trophy } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
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

      const trials = finalStatus.trials ?? [];
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

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="shrink-0 space-y-4">
        <StepHeader
          step={4}
          title="Run Optimization"
          subtitle="Execute the study and pick a trial to analyze"
        />

        <ErrorBanner message={error} />

        {!hasResults && !isRunning && (
          <section className="rounded-lg border border-border bg-card p-6 text-center">
            <Button type="button" size="lg" onClick={run}>
              {wasStopped ? <RotateCcw className="h-5 w-5" /> : <PlayCircle className="h-5 w-5" />}
              {wasStopped ? 'Restart Optimization' : 'Start Optimization'}
            </Button>
            <p className="mt-3 text-sm text-muted-foreground">
              {wasStopped
                ? `The previous run was ${optimization.status}. Restarting launches a new run with the same configuration.`
                : `Runs ${configuration.numTrials} trials across quantum and classical models. This may take several minutes.`}
            </p>
          </section>
        )}

        {isRunning && (
          <section className="rounded-lg border border-border bg-card p-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <StatusDot status="busy" />
                <div>
                  <p className="font-medium">Optimization in progress</p>
                  <p className="text-sm text-muted-foreground">
                    Trial {currentTrial} of {configuration.numTrials}
                  </p>
                </div>
              </div>
              <Metric value={`${Math.round(progress)}%`} tone="brand" className="text-2xl" />
            </div>
            <Progress.Root
              className="h-2.5 w-full overflow-hidden rounded-full bg-muted"
              value={progress}
            >
              <Progress.Indicator
                className="h-full bg-brand transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </Progress.Root>
            {bestValue !== null && (
              <div className="mt-4 flex items-center justify-between rounded-md border border-accent-emerald bg-accent-emerald/40 p-3">
                <span className="text-sm font-medium text-accent-emerald-foreground">
                  Current best F1
                </span>
                <Metric value={bestValue.toFixed(4)} tone="emerald" className="text-lg" />
              </div>
            )}
          </section>
        )}

        {(hasResults || liveTrials.length > 0) && (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
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
      </div>

      {allTrials.length > 0 && (
        <TableContainer className="flex min-h-0 flex-1 flex-col">
          <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border bg-muted px-4 py-3">
            <h4 className="text-sm font-semibold text-foreground">All trials ({sorted.length})</h4>
            {hasResults && (
              <span className="text-xs text-muted-foreground">
                {optimization.selectedTrial !== null ? (
                  <>
                    Analyzing{' '}
                    <span className="font-medium text-brand">#{optimization.selectedTrial}</span> —
                    click any row to change
                  </>
                ) : (
                  'Click a row to choose a trial to analyze'
                )}
              </span>
            )}
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            <Table stickyHeader>
              <TableHead>
                <tr>
                  <Th className="w-10" />
                  <Th className="w-12 text-right">Rank</Th>
                  <Th className="text-right">F1 score</Th>
                  <Th>Model</Th>
                  <Th>Type</Th>
                  <Th className="text-right">Trial</Th>
                </tr>
              </TableHead>
              <TableBody>
                {sorted.map((trial, index) => {
                  const failed = trial.value === null || trial.state === 'FAIL';
                  const selectable = hasResults && !failed;
                  const selected = optimization.selectedTrial === trial.trial;
                  const classical = isClassicalModel(trial.params?.model_type);
                  const isBest = index === 0 && !failed;
                  return (
                    <tr
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
                        'transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-brand',
                        selectable && 'cursor-pointer',
                        failed && 'opacity-60',
                        selected ? 'bg-brand/10 ring-1 ring-inset ring-brand/40' : 'hover:bg-muted'
                      )}
                    >
                      <Td className="text-center">
                        {selected ? (
                          <Check size={16} className="mx-auto text-brand" />
                        ) : isBest ? (
                          <Trophy size={14} className="mx-auto text-accent-amber-foreground" />
                        ) : null}
                      </Td>
                      <Td className="text-right text-xs text-muted-foreground tabular-nums">
                        {index + 1}
                      </Td>
                      <Td
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
                      </Td>
                      <Td className="text-xs text-muted-foreground">
                        {trial.params?.model_type ?? 'N/A'}
                        {failed && trial.user_attrs?.error != null && (
                          <span className="block max-w-[280px] truncate text-[11px] text-destructive/80">
                            {String(trial.user_attrs.error)}
                          </span>
                        )}
                      </Td>
                      <Td className="text-xs">
                        <Badge variant={classical ? 'classical' : 'quantum'}>
                          {classical ? 'Classical' : 'Quantum'}
                        </Badge>
                      </Td>
                      <Td className="text-right text-xs text-muted-foreground tabular-nums">
                        #{trial.trial}
                      </Td>
                    </tr>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </TableContainer>
      )}
    </div>
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
      ? 'border-accent-purple bg-accent-purple/20'
      : 'border-accent-orange bg-accent-orange/20';
  const isSelected = !!trial && selectedTrial === trial.trial;
  const Icon = accent === 'quantum' ? Atom : Cpu;
  const modelType = (trial?.params?.model_type as string | undefined) ?? 'N/A';

  return (
    <div
      className={cn(
        'flex flex-col rounded-lg border p-4 transition-shadow',
        accentClasses,
        isSelected && 'ring-2 ring-inset ring-brand/50'
      )}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon
            size={16}
            className={
              accent === 'quantum'
                ? 'text-accent-purple-foreground'
                : 'text-accent-orange-foreground'
            }
          />
          <span className="text-sm font-semibold">{label}</span>
        </div>
        {isSelected && (
          <Badge variant="emerald">
            <Check size={12} /> Analyzing
          </Badge>
        )}
      </div>

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
              className="mt-3 self-start"
              onClick={() => onSelect(trial.trial)}
            >
              Analyze this
            </Button>
          )}
        </>
      ) : (
        <p className="mt-1 text-sm text-muted-foreground">No trial yet</p>
      )}
    </div>
  );
}
