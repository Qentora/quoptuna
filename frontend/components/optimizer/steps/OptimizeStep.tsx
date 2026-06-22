'use client';

import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
import { pollOptimization, startOptimization } from '@/lib/api';
import { cn } from '@/lib/utils';
import * as Progress from '@radix-ui/react-progress';
import { Loader2, PlayCircle } from 'lucide-react';
import { useState } from 'react';
import { ErrorBanner, NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';
import { isClassicalModel } from '../types';

export function OptimizeStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTrial, setCurrentTrial] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [liveTrials, setLiveTrials] = useState<any[]>([]);
  const [bestValue, setBestValue] = useState<number | null>(null);

  const { dataset, features, configuration, optimization } = workflowData;
  const hasResults = optimization.status === 'completed';

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
    <div className="space-y-6">
      <StepHeader
        step={4}
        title="Run Optimization"
        subtitle="Execute the study and pick a trial to analyze"
      />

      <ErrorBanner message={error} />

      <Alert variant="info">
        <span className="text-sm">
          Study: <strong>{configuration.studyName}</strong> | Trials:{' '}
          <strong>{configuration.numTrials}</strong> | Features:{' '}
          <strong>{features.selectedFeatures.length}</strong>
        </span>
      </Alert>

      {!hasResults && !isRunning && (
        <Card className="bg-muted p-6 text-center">
          <Button type="button" size="lg" onClick={run}>
            <PlayCircle className="h-5 w-5" />
            Start Optimization
          </Button>
          <p className="mt-3 text-sm text-muted-foreground">
            Runs {configuration.numTrials} trials across quantum and classical models. This may take
            several minutes.
          </p>
        </Card>
      )}

      {isRunning && (
        <Card className="p-6">
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Loader2 className="h-6 w-6 animate-spin text-foreground" />
              <div>
                <p className="font-medium">Optimization in Progress</p>
                <p className="text-sm text-muted-foreground">
                  Trial {currentTrial} of {configuration.numTrials}
                </p>
              </div>
            </div>
            <span className="text-2xl font-bold">{Math.round(progress)}%</span>
          </div>
          <Progress.Root
            className="h-3 w-full overflow-hidden rounded-full bg-muted"
            value={progress}
          >
            <Progress.Indicator
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </Progress.Root>
          {bestValue !== null && (
            <div className="mt-4 flex items-center justify-between rounded-md border border-accent-emerald bg-accent-emerald/40 p-3">
              <span className="text-sm font-medium text-accent-emerald-foreground">
                Current Best F1:
              </span>
              <span className="text-lg font-bold text-accent-emerald-foreground">
                {bestValue.toFixed(4)}
              </span>
            </div>
          )}
        </Card>
      )}

      {(hasResults || liveTrials.length > 0) && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <BestTrialCard label="Best Quantum" trial={bestQuantum} accent="quantum" />
          <BestTrialCard label="Best Classical" trial={bestClassical} accent="classical" />
        </div>
      )}

      {allTrials.length > 0 && (
        <TableContainer>
          <div className="border-b border-border bg-muted px-4 py-2">
            <h4 className="text-sm font-semibold text-foreground">
              Trials (select one to analyze)
            </h4>
          </div>
          <div className="max-h-72 overflow-y-auto">
            <Table>
              <TableHead className="sticky top-0">
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
                    <tr key={trial.trial} className={selected ? 'bg-accent' : 'hover:bg-muted'}>
                      <Td>#{trial.trial}</Td>
                      <Td className="font-medium">{trial.value.toFixed(4)}</Td>
                      <Td className="text-xs text-muted-foreground">
                        {trial.params?.model_type ?? 'N/A'}
                      </Td>
                      <Td className="text-xs">
                        <span
                          className={cn(
                            'rounded-full px-2 py-0.5 text-xs font-semibold',
                            classical
                              ? 'bg-accent-orange text-accent-orange-foreground'
                              : 'bg-accent-purple text-accent-purple-foreground'
                          )}
                        >
                          {classical ? 'Classical' : 'Quantum'}
                        </span>
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

      <NavButtons
        onBack={onBack}
        onNext={onNext}
        backDisabled={isRunning}
        nextDisabled={!hasResults || isRunning}
      />
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
          <p className="mt-1 text-2xl font-bold text-foreground">{trial.value.toFixed(4)}</p>
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
