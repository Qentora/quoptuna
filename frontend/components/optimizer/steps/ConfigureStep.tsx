'use client';

import { Alert } from '@/components/ui/alert';
import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import * as Slider from '@radix-ui/react-slider';
import { FileText } from 'lucide-react';
import { NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

export function ConfigureStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const { configuration, features, dataset } = workflowData;

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

      {dataset && (
        <Alert variant="info" icon={false}>
          <div className="flex items-start gap-3">
            <FileText className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="font-medium">{dataset.name}</p>
              <p className="mt-1 text-sm opacity-80">
                Features: {features.selectedFeatures.join(', ')} | Target: {features.targetColumn}
              </p>
            </div>
          </div>
        </Alert>
      )}

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

        <Field
          label="Database Name"
          htmlFor="db-name"
          helper={
            <>
              SQLite database (stored under <code className="font-mono">db/&lt;name&gt;.db</code>)
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

        <div>
          <div className="mb-2 flex items-center justify-between">
            <label className="text-sm font-medium" htmlFor="trials">
              Number of Trials
            </label>
            <span className="text-sm font-semibold">{configuration.numTrials}</span>
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
              <Slider.Range className="absolute h-full rounded-full bg-primary" />
            </Slider.Track>
            <Slider.Thumb className="block h-4 w-4 rounded-full border-2 border-primary bg-background focus:outline-none focus:ring-2 focus:ring-ring" />
          </Slider.Root>
          <p className="mt-1 text-sm text-muted-foreground">
            Recommended: 50-200 trials (more trials = better results but longer run time)
          </p>
        </div>
      </div>

      <NavButtons onBack={onBack} onNext={onNext} />
    </div>
  );
}
