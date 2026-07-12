'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Metric } from '@/components/ui/metric';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { useDatasetPreview } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import { useEffect } from 'react';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

const PRESETS = [
  { label: 'Quick', trials: 25, hint: '~3–8 min' },
  { label: 'Standard', trials: 50, hint: '~5–15 min' },
  { label: 'Thorough', trials: 150, hint: '~20–45 min' },
];

export function ConfigureStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const { configuration } = workflowData;
  const { sensitiveFeature, favorableClass, targetColumn } = workflowData.features;
  const fairnessOn = configuration.fairnessMode !== 'off';

  // Multiclass targets need a favorable class before fairness-aware search
  // can run (the backend rejects fairness_mode != off without one).
  const preview = useDatasetPreview(workflowData.dataset?.id ?? null);
  const targetValues = targetColumn
    ? (preview.data?.target_values_by_column[targetColumn] ?? [])
    : [];
  const isMulticlass = targetValues.length > 2;
  const fairnessBlockedByFavorable = isMulticlass && !favorableClass;

  const hasStudyName = configuration.studyName.trim().length > 0;

  useEffect(() => {
    setFooter({ canContinue: hasStudyName });
  }, [hasStudyName, setFooter]);

  // A fairness mode left on after the favorable class was cleared would be
  // stuck (select disabled) and rejected by the backend — reset it.
  useEffect(() => {
    if (fairnessOn && fairnessBlockedByFavorable) {
      setWorkflowData((prev) => ({
        ...prev,
        configuration: { ...prev.configuration, fairnessMode: 'off' },
      }));
    }
  }, [fairnessOn, fairnessBlockedByFavorable, setWorkflowData]);

  const update = (field: keyof typeof configuration, value: string | number | null) =>
    setWorkflowData((prev) => ({
      ...prev,
      configuration: { ...prev.configuration, [field]: value },
    }));

  // The backend rejects incompatible combinations; keep them impossible here:
  // constraints are TPE-only, multi-objective studies cannot prune.
  const setFairnessMode = (mode: string) =>
    setWorkflowData((prev) => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        fairnessMode: mode as typeof prev.configuration.fairnessMode,
        sampler: mode === 'constrained' ? 'tpe' : prev.configuration.sampler,
        pruner: mode === 'multi_objective' ? 'none' : prev.configuration.pruner,
      },
    }));

  const defaultThreshold = configuration.fairnessMetric === 'disparate_impact' ? 0.8 : 0.1;

  return (
    <div className="space-y-4">
      <StepHeader
        step={3}
        title="Optimization Configuration"
        subtitle="Set up the hyperparameter optimization study"
      />

      <div className="flex flex-col gap-4">
        {/* Study name */}
        <Card>
          <CardHeader>
            <CardTitle>Study</CardTitle>
          </CardHeader>
          <CardContent>
            <Field className="max-w-xl">
              <FieldLabel htmlFor="study-name">Study Name</FieldLabel>
              <Input
                id="study-name"
                type="text"
                value={configuration.studyName}
                onChange={(e) => update('studyName', e.target.value)}
              />
              <FieldDescription>
                Unique name for this optimization study — it identifies the run everywhere (runs
                list, replay)
              </FieldDescription>
            </Field>
          </CardContent>
        </Card>

        {/* Trial budget */}
        <Card>
          <CardHeader className="flex flex-row items-center justify-between gap-3">
            <CardTitle>Trial budget</CardTitle>
            <Metric value={configuration.numTrials} tone="brand" className="text-sm" />
          </CardHeader>
          <CardContent className="grid items-start gap-x-6 gap-y-4 lg:grid-cols-2">
            <div className="grid grid-cols-3 gap-3">
              {PRESETS.map((preset) => {
                const active = configuration.numTrials === preset.trials;
                return (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => update('numTrials', preset.trials)}
                    className={cn(
                      'rounded-lg border p-3 text-left transition-colors',
                      active
                        ? 'border-brand bg-brand/10 text-brand'
                        : 'border-border hover:border-foreground'
                    )}
                  >
                    <span className="block text-sm font-semibold">{preset.label}</span>
                    <span className="block text-xs text-muted-foreground">
                      {preset.trials} trials
                    </span>
                    <span className="mt-1 block text-xs text-muted-foreground">{preset.hint}</span>
                  </button>
                );
              })}
            </div>

            <Field className="lg:pt-1">
              <FieldLabel htmlFor="trials">Fine-tune</FieldLabel>
              <Slider
                id="trials"
                min={1}
                max={300}
                step={1}
                value={[configuration.numTrials]}
                onValueChange={([v]) => update('numTrials', v)}
              />
              <FieldDescription>More trials = better results but longer run time.</FieldDescription>
            </Field>
          </CardContent>
        </Card>

        {/* Search strategy */}
        <Card>
          <CardHeader>
            <CardTitle>Search strategy</CardTitle>
          </CardHeader>
          <CardContent className="grid items-start gap-x-6 gap-y-4 lg:grid-cols-2">
            <Field>
              <FieldLabel htmlFor="sampler">Sampler</FieldLabel>
              <Select value={configuration.sampler} onValueChange={(v) => update('sampler', v)}>
                <SelectTrigger
                  id="sampler"
                  className="w-full max-w-xs"
                  disabled={configuration.fairnessMode === 'constrained'}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tpe">TPE (Bayesian, recommended)</SelectItem>
                  <SelectItem value="random">Random search</SelectItem>
                  <SelectItem value="grid">Grid search</SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                {configuration.fairnessMode === 'constrained'
                  ? 'Locked to TPE — constrained mode expresses the fairness threshold as a TPE feasibility constraint; the other samplers silently ignore constraints.'
                  : 'How hyperparameter configurations are proposed. Grid search may finish before the trial budget if it exhausts all combinations.'}
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="pruner">Pruner</FieldLabel>
              <Select value={configuration.pruner} onValueChange={(v) => update('pruner', v)}>
                <SelectTrigger
                  id="pruner"
                  className="w-full max-w-xs"
                  disabled={configuration.fairnessMode === 'multi_objective'}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None (train every trial fully)</SelectItem>
                  <SelectItem value="asha">ASHA (successive halving)</SelectItem>
                  <SelectItem value="hyperband">Hyperband</SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                {configuration.fairnessMode === 'multi_objective'
                  ? 'Disabled for the Pareto search — pruning ranks trials by a single intermediate metric, but here a trial weak on F1 may still be Pareto-optimal on fairness, and Optuna rejects intermediate reports on multi-objective studies.'
                  : 'Early-stops unpromising quantum model trainings to save compute. Kernel and classical models always train to completion.'}
              </FieldDescription>
            </Field>
          </CardContent>
        </Card>

        {/* Fairness-aware search */}
        <Card>
          <CardHeader>
            <CardTitle>Fairness-aware search</CardTitle>
          </CardHeader>
          <CardContent className="grid items-start gap-x-6 gap-y-4 lg:grid-cols-3">
            <Field>
              <FieldLabel htmlFor="fairness-mode">Mode</FieldLabel>
              <Select
                value={configuration.fairnessMode}
                onValueChange={setFairnessMode}
                disabled={!sensitiveFeature || fairnessBlockedByFavorable}
              >
                <SelectTrigger id="fairness-mode" className="w-full max-w-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="off">Off</SelectItem>
                  <SelectItem value="constrained">Constrained (threshold)</SelectItem>
                  <SelectItem value="multi_objective">Multi-objective (Pareto)</SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                {!sensitiveFeature
                  ? 'Select a protected attribute in the Features step to enable.'
                  : fairnessBlockedByFavorable
                    ? undefined
                    : configuration.fairnessMode === 'constrained'
                      ? 'Deprioritizes trials whose disparity exceeds the threshold (forces the TPE sampler).'
                      : configuration.fairnessMode === 'multi_objective'
                        ? 'Searches the F1-vs-disparity Pareto front (disables pruning).'
                        : `Feeds fairness on "${sensitiveFeature}" back into the search.`}
              </FieldDescription>
              {sensitiveFeature && fairnessBlockedByFavorable && (
                <p className="text-xs text-accent-amber-foreground">
                  Multiclass target: select a favorable class in the Features step before enabling
                  fairness-aware search (the disparity is computed on favorable vs rest).
                </p>
              )}
            </Field>

            <Field>
              <FieldLabel htmlFor="fairness-metric">Disparity metric</FieldLabel>
              <Select
                value={configuration.fairnessMetric}
                onValueChange={(v) => update('fairnessMetric', v)}
                disabled={!fairnessOn}
              >
                <SelectTrigger id="fairness-metric" className="w-full max-w-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="equal_opportunity_difference">
                    Equal opportunity difference
                  </SelectItem>
                  <SelectItem value="disparate_impact">Disparate impact</SelectItem>
                  <SelectItem value="demographic_parity_difference">
                    Demographic parity difference
                  </SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                {fairnessOn
                  ? 'What the search minimizes (multi-objective) or constrains (constrained mode). For multiclass targets the disparity is computed on the favorable class vs the rest.'
                  : 'Enable a fairness mode to configure the disparity metric.'}
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="fairness-threshold">Threshold</FieldLabel>
              <Input
                id="fairness-threshold"
                type="number"
                min={0}
                max={1}
                step={0.05}
                disabled={configuration.fairnessMode !== 'constrained'}
                placeholder={String(defaultThreshold)}
                value={configuration.fairnessThreshold ?? ''}
                onChange={(e) =>
                  update('fairnessThreshold', e.target.value === '' ? null : Number(e.target.value))
                }
                className="max-w-xs"
              />
              <FieldDescription>
                {configuration.fairnessMode !== 'constrained'
                  ? 'Used by constrained mode only — multi-objective explores the whole trade-off instead of enforcing a cutoff.'
                  : configuration.fairnessMetric === 'disparate_impact'
                    ? 'Feasible when the DI ratio is at least this (default 0.8, four-fifths rule).'
                    : 'Feasible when the disparity is at most this (default 0.1).'}
              </FieldDescription>
            </Field>
          </CardContent>
        </Card>
      </div>

      <p className="text-xs text-muted-foreground">
        Results are stored in the shared Optuna database configured in{' '}
        <Link href="/settings" className="underline hover:text-foreground">
          Settings
        </Link>
        .
      </p>
    </div>
  );
}
