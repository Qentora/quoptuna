'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Metric } from '@/components/ui/metric';
import { StatusDot } from '@/components/ui/status-dot';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
import { pollOptimization, startOptimization } from '@/lib/api';
import { cn } from '@/lib/utils';
import * as Progress from '@radix-ui/react-progress';
import { Loader2, PlayCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
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

  useEffect(() => {
    setFooter({ canContinue: hasResults, nextBusy: isRunning, backDisabled: isRunning });
  }, [hasResults, isRunning, setFooter]);

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
        database_name: configuration.databaseName,
        num_trials: configuration.numTrials,
        label_mapping: labelMapping,
      });

      const finalStatus = await pollOptimization(id, (status, trialsData) => {
        setCurrentTrial(status.current_trial);
        setProgress(status.total_trials ? (status.current_trial / status.total_trials) * 100 : 0);
        if (trialsData?.trials) {
          setLiveTrials(trialsData.trials);
          if (trialsData.best_trial) setBestValue(trialsData.best_trial.value);
        }
      });

      if (finalStatus.status === 'failed') {
        setError(finalStatus.error || 'Optimization failed');
        setIsRunning(false);
        return;
      }

      const trials = finalStatus.trials ?? [];
      const bestTrialNumber =
        trials.length > 0
          ? trials.reduce((best, t) => (t.value > best.value ? t : best)).trial
          : null;

      setWorkflowData((prev) => ({
        ...prev,
        optimization: {
          executionId: id,
          status: 'completed',
          bestValue: finalStatus.best_value,
          bestParams: finalStatus.best_params,
          trials,
          selectedTrial: bestTrialNumber,
        },
      }));
      setIsRunning(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Optimization failed');
      setIsRunning(false);
    }
  };

  const allTrials = hasResults ? optimization.trials : liveTrials;
  const sorted = [...allTrials].sort((a, b) => b.value - a.value);
  const bestQuantum = sorted.find((t) => !isClassicalModel(t.params?.model_type));
  const bestClassical = sorted.find((t) => isClassicalModel(t.params?.model_type));

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
              <PlayCircle className="h-5 w-5" />
              Start Optimization
            </Button>
            <p className="mt-3 text-sm text-muted-foreground">
              Runs {configuration.numTrials} trials across quantum and classical models. This may
              take several minutes.
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
            <BestTrialCard label="Best Quantum" trial={bestQuantum} accent="quantum" />
            <BestTrialCard label="Best Classical" trial={bestClassical} accent="classical" />
          </div>
        )}
      </div>

      {allTrials.length > 0 && (
        <TableContainer className="flex min-h-0 flex-1 flex-col">
          <div className="shrink-0 border-b border-border bg-muted px-4 py-3">
            <h4 className="text-sm font-semibold text-foreground">
              {hasResults ? 'Best trial auto-selected — pick another to analyze instead' : 'Trials'}
            </h4>
          </div>
          <div className="min-h-0 flex-1 overflow-auto">
            <Table stickyHeader>
              <TableHead>
                <tr>
                  <Th>Trial</Th>
                  <Th>F1</Th>
                  <Th>Model</Th>
                  <Th>Type</Th>
                  <Th />
                </tr>
              </TableHead>
              <TableBody>
                {sorted.map((trial) => {
                  const selected = optimization.selectedTrial === trial.trial;
                  const classical = isClassicalModel(trial.params?.model_type);
                  return (
                    <tr
                      key={trial.trial}
                      className={cn(
                        selected ? 'bg-brand/10 ring-1 ring-inset ring-brand/40' : 'hover:bg-muted'
                      )}
                    >
                      <Td>#{trial.trial}</Td>
                      <Td className="font-medium">{trial.value.toFixed(4)}</Td>
                      <Td className="text-xs text-muted-foreground">
                        {trial.params?.model_type ?? 'N/A'}
                      </Td>
                      <Td className="text-xs">
                        <Badge variant={classical ? 'classical' : 'quantum'}>
                          {classical ? 'Classical' : 'Quantum'}
                        </Badge>
                      </Td>
                      <Td className="text-right">
                        {hasResults && (
                          <Button
                            type="button"
                            size="sm"
                            variant={selected ? 'default' : 'secondary'}
                            onClick={() => selectTrial(trial.trial)}
                          >
                            {selected ? 'Selected' : 'Select'}
                          </Button>
                        )}
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
}: {
  label: string;
  trial: any;
  accent: 'quantum' | 'classical';
}) {
  const classes =
    accent === 'quantum'
      ? 'border-accent-purple bg-accent-purple/30 text-accent-purple-foreground'
      : 'border-accent-orange bg-accent-orange/30 text-accent-orange-foreground';
  return (
    <div className={cn('rounded-lg border p-4', classes)}>
      <p className="font-medium">{label}</p>
      {trial ? (
        <>
          <Metric
            value={trial.value.toFixed(4)}
            tone={accent === 'quantum' ? 'brand' : 'amber'}
            className="mt-1 block text-2xl"
          />
          <p className="text-xs text-muted-foreground">
            F1 · {trial.params?.model_type ?? 'N/A'} · #{trial.trial}
          </p>
        </>
      ) : (
        <p className="mt-1 text-sm text-muted-foreground">No trial yet</p>
      )}
    </div>
  );
}
