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

  const hasStudyName = configuration.studyName.trim().length > 0;

  useEffect(() => {
    setFooter({ canContinue: hasStudyName });
  }, [hasStudyName, setFooter]);

  const update = (field: keyof typeof configuration, value: string | number) =>
    setWorkflowData((prev) => ({
      ...prev,
      configuration: { ...prev.configuration, [field]: value },
    }));

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
                <SelectTrigger id="sampler" className="w-full max-w-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="tpe">TPE (Bayesian, recommended)</SelectItem>
                  <SelectItem value="random">Random search</SelectItem>
                  <SelectItem value="grid">Grid search</SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                How hyperparameter configurations are proposed. Grid search may finish before the
                trial budget if it exhausts all combinations.
              </FieldDescription>
            </Field>

            <Field>
              <FieldLabel htmlFor="pruner">Pruner</FieldLabel>
              <Select value={configuration.pruner} onValueChange={(v) => update('pruner', v)}>
                <SelectTrigger id="pruner" className="w-full max-w-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="none">None (train every trial fully)</SelectItem>
                  <SelectItem value="asha">ASHA (successive halving)</SelectItem>
                  <SelectItem value="hyperband">Hyperband</SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                Early-stops unpromising quantum model trainings to save compute. Kernel and
                classical models always train to completion.
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
