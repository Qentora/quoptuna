'use client';

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
        title="Optimization Configuration"
        subtitle="Set up the hyperparameter optimization study"
      />

      {dataset && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
          <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
          <div>
            <p className="font-medium text-blue-900">{dataset.name}</p>
            <p className="text-sm text-blue-700 mt-1">
              Features: {features.selectedFeatures.join(', ')} | Target: {features.targetColumn}
            </p>
          </div>
        </div>
      )}

      <div className="space-y-5">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="study-name">
            Study Name
          </label>
          <input
            id="study-name"
            type="text"
            value={configuration.studyName}
            onChange={(e) => update('studyName', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-sm text-gray-500 mt-1">Unique name for this optimization study</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2" htmlFor="db-name">
            Database Name
          </label>
          <input
            id="db-name"
            type="text"
            value={configuration.databaseName}
            onChange={(e) => update('databaseName', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500"
          />
          <p className="text-sm text-gray-500 mt-1">
            SQLite database (stored under <code>db/&lt;name&gt;.db</code>)
          </p>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-medium text-gray-700" htmlFor="trials">
              Number of Trials
            </label>
            <span className="text-sm font-semibold text-blue-600">{configuration.numTrials}</span>
          </div>
          <Slider.Root
            id="trials"
            className="relative flex items-center select-none touch-none w-full h-5"
            min={1}
            max={300}
            step={1}
            value={[configuration.numTrials]}
            onValueChange={([v]) => update('numTrials', v)}
          >
            <Slider.Track className="bg-gray-200 relative grow rounded-full h-1.5">
              <Slider.Range className="absolute bg-blue-600 rounded-full h-full" />
            </Slider.Track>
            <Slider.Thumb className="block w-4 h-4 bg-white border-2 border-blue-600 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-300" />
          </Slider.Root>
          <p className="text-sm text-gray-500 mt-1">
            Recommended: 50-200 trials (more trials = better results but longer run time)
          </p>
        </div>
      </div>

      <NavButtons onBack={onBack} onNext={onNext} />
    </div>
  );
}
