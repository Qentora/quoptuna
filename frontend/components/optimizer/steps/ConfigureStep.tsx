'use client';

import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Metric } from '@/components/ui/metric';
import { cn } from '@/lib/utils';
import * as Slider from '@radix-ui/react-slider';
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

  useEffect(() => {
    setFooter({ canContinue: true });
  }, [setFooter]);

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

      <div className="space-y-5">
        <Field
          label="Study Name"
          htmlFor="study-name"
          helper="Unique name for this optimization study"
        >
          <Input
            id="study-name"
            type="text"
            value={configuration.studyName}
            onChange={(e) => update('studyName', e.target.value)}
          />
        </Field>

        <div>
          <label className="mb-2 block text-sm font-medium" htmlFor="trials">
            Trial budget
          </label>
          <div className="mb-3 grid grid-cols-3 gap-2">
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
                      ? 'border-brand bg-brand/10 text-brand shadow-glow-brand'
                      : 'border-border hover:border-foreground'
                  )}
                >
                  <span className="block text-sm font-semibold">{preset.label}</span>
                  <span className="block text-xs text-muted-foreground">
                    {preset.trials} trials
                  </span>
                  <span className="mt-1 block text-[11px] text-muted-foreground">
                    {preset.hint}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Fine-tune</span>
            <Metric value={configuration.numTrials} tone="brand" glow className="text-sm" />
          </div>
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
          <p className="mt-1 text-sm text-muted-foreground">
            More trials = better results but longer run time.
          </p>
        </div>

        <details className="rounded-lg border border-border p-3">
          <summary className="cursor-pointer text-sm font-medium">Advanced</summary>
          <div className="mt-3">
            <Field
              label="Database Name"
              htmlFor="db-name"
              helper={
                <>
                  SQLite database (stored under{' '}
                  <code className="font-mono">db/&lt;name&gt;.db</code>)
                </>
              }
            >
              <Input
                id="db-name"
                type="text"
                value={configuration.databaseName}
                onChange={(e) => update('databaseName', e.target.value)}
              />
            </Field>
          </div>
        </details>
      </div>
    </div>
  );
}
