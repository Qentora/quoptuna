'use client';

import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Metric } from '@/components/ui/metric';
import { cn } from '@/lib/utils';
import * as Slider from '@radix-ui/react-slider';
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
    <div className="space-y-6">
      <StepHeader
        step={3}
        title="Optimization Configuration"
        subtitle="Set up the hyperparameter optimization study"
      />

      {/* Study name */}
      <section className="rounded-lg border border-border bg-card">
        <div className="border-b border-border bg-muted px-4 py-3">
          <h4 className="text-sm font-semibold">Study</h4>
        </div>
        <div className="p-4">
          <Field
            label="Study Name"
            htmlFor="study-name"
            helper="Unique name for this optimization study — it identifies the run everywhere (runs list, replay)"
          >
            <Input
              id="study-name"
              type="text"
              value={configuration.studyName}
              onChange={(e) => update('studyName', e.target.value)}
            />
          </Field>
        </div>
      </section>

      {/* Trial budget */}
      <section className="rounded-lg border border-border bg-card">
        <div className="flex items-center justify-between gap-3 border-b border-border bg-muted px-4 py-3">
          <h4 className="text-sm font-semibold">Trial budget</h4>
          <Metric value={configuration.numTrials} tone="brand" className="text-sm" />
        </div>
        <div className="space-y-4 p-4">
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

          <div>
            <label
              className="mb-2 block text-xs font-medium text-muted-foreground"
              htmlFor="trials"
            >
              Fine-tune
            </label>
            <Slider.Root
              id="trials"
              className="relative flex h-5 w-full touch-none select-none items-center"
              min={1}
              max={300}
              step={1}
              value={[configuration.numTrials]}
              onValueChange={([v]) => update('numTrials', v)}
            >
              <Slider.Track className="relative h-1.5 grow rounded-full bg-muted">
                <Slider.Range className="absolute h-full rounded-full bg-brand" />
              </Slider.Track>
              <Slider.Thumb className="block h-4 w-4 rounded-full border-2 border-brand bg-background focus:outline-none focus:ring-2 focus:ring-brand" />
            </Slider.Root>
            <p className="mt-2 text-sm text-muted-foreground">
              More trials = better results but longer run time.
            </p>
          </div>
        </div>
      </section>

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
