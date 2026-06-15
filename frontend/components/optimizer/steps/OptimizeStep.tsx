'use client';

import { pollOptimization, startOptimization } from '@/lib/api';
import * as Progress from '@radix-ui/react-progress';
import { BarChart3, Loader2, PlayCircle } from 'lucide-react';
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
        title="Run Optimization"
        subtitle="Execute the study and pick a trial to analyze"
      />

      <ErrorBanner message={error} />

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
        <BarChart3 className="w-5 h-5 text-blue-600 mt-0.5" />
        <p className="text-sm text-blue-700">
          Study: <strong>{configuration.studyName}</strong> | Trials:{' '}
          <strong>{configuration.numTrials}</strong> | Features:{' '}
          <strong>{features.selectedFeatures.length}</strong>
        </p>
      </div>

      {!hasResults && !isRunning && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
          <button
            type="button"
            onClick={run}
            className="px-8 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-semibold inline-flex items-center gap-2"
          >
            <PlayCircle className="w-5 h-5" />
            Start Optimization
          </button>
          <p className="text-sm text-blue-700 mt-3">
            Runs {configuration.numTrials} trials across quantum and classical models. This may take
            several minutes.
          </p>
        </div>
      )}

      {isRunning && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
              <div>
                <p className="font-medium text-gray-900">Optimization in Progress</p>
                <p className="text-sm text-gray-500">
                  Trial {currentTrial} of {configuration.numTrials}
                </p>
              </div>
            </div>
            <span className="text-2xl font-bold text-blue-600">{Math.round(progress)}%</span>
          </div>
          <Progress.Root
            className="w-full bg-gray-200 rounded-full h-3 overflow-hidden"
            value={progress}
          >
            <Progress.Indicator
              className="bg-blue-600 h-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </Progress.Root>
          {bestValue !== null && (
            <div className="mt-4 flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-md">
              <span className="text-sm font-medium text-green-900">Current Best F1:</span>
              <span className="text-lg font-bold text-green-700">{bestValue.toFixed(4)}</span>
            </div>
          )}
        </div>
      )}

      {(hasResults || liveTrials.length > 0) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <BestTrialCard label="Best Quantum" trial={bestQuantum} accent="purple" />
          <BestTrialCard label="Best Classical" trial={bestClassical} accent="orange" />
        </div>
      )}

      {allTrials.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg">
          <div className="px-4 py-2 border-b border-gray-200 bg-gray-50">
            <h4 className="text-sm font-semibold text-gray-700">Trials (select one to analyze)</h4>
          </div>
          <div className="max-h-72 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="bg-gray-100 sticky top-0">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-gray-700">Trial</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-700">F1</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-700">Model</th>
                  <th className="px-4 py-2 text-left font-medium text-gray-700">Type</th>
                  <th className="px-4 py-2" />
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {sorted.map((trial) => {
                  const selected = optimization.selectedTrial === trial.trial;
                  return (
                    <tr key={trial.trial} className={selected ? 'bg-blue-50' : 'hover:bg-gray-50'}>
                      <td className="px-4 py-2 text-gray-900">#{trial.trial}</td>
                      <td className="px-4 py-2 font-medium text-gray-900">
                        {trial.value.toFixed(4)}
                      </td>
                      <td className="px-4 py-2 text-gray-700 text-xs">
                        {trial.params?.model_type ?? 'N/A'}
                      </td>
                      <td className="px-4 py-2 text-xs">
                        {isClassicalModel(trial.params?.model_type) ? 'Classical' : 'Quantum'}
                      </td>
                      <td className="px-4 py-2 text-right">
                        {hasResults && (
                          <button
                            type="button"
                            onClick={() => selectTrial(trial.trial)}
                            className={`px-3 py-1 rounded text-xs ${
                              selected
                                ? 'bg-blue-600 text-white'
                                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                            }`}
                          >
                            {selected ? 'Selected' : 'Select'}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
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
  accent: 'purple' | 'orange';
}) {
  const color = accent === 'purple' ? 'text-purple-700' : 'text-orange-700';
  const border = accent === 'purple' ? 'border-purple-200' : 'border-orange-200';
  const bg = accent === 'purple' ? 'bg-purple-50' : 'bg-orange-50';
  return (
    <div className={`rounded-lg border ${border} ${bg} p-4`}>
      <p className={`font-medium ${color}`}>{label}</p>
      {trial ? (
        <>
          <p className="text-2xl font-bold text-gray-900 mt-1">{trial.value.toFixed(4)}</p>
          <p className="text-xs text-gray-600">
            F1 · {trial.params?.model_type ?? 'N/A'} · #{trial.trial}
          </p>
        </>
      ) : (
        <p className="text-sm text-gray-500 mt-1">No trial yet</p>
      )}
    </div>
  );
}
