'use client';

import { cn } from '@/lib/utils';
import { Check } from 'lucide-react';

export interface StepDef {
  id: number;
  title: string;
}

interface StepProgressProps {
  steps: StepDef[];
  currentStep: number;
  completedSteps: number[];
  onStepClick: (id: number) => void;
  className?: string;
}

const CHEVRON =
  '[clip-path:polygon(0_0,calc(100%-12px)_0,100%_50%,calc(100%-12px)_100%,0_100%,12px_50%)]';
const CHEVRON_FIRST =
  '[clip-path:polygon(0_0,calc(100%-12px)_0,100%_50%,calc(100%-12px)_100%,0_100%)]';
const CHEVRON_LAST = '[clip-path:polygon(0_0,100%_0,100%_100%,0_100%,12px_50%)]';

/**
 * Horizontal chevron stepper (brand-filled up to the current step). Completed steps are
 * clickable; upcoming steps are locked. Collapses to a compact "Step N of M · Title" indicator
 * with a thin progress bar below `md`.
 */
export function StepProgress({
  steps,
  currentStep,
  completedSteps,
  onStepClick,
  className,
}: StepProgressProps) {
  const total = steps.length;
  const currentTitle = steps.find((s) => s.id === currentStep)?.title ?? '';

  return (
    <div className={className}>
      {/* Full chevron stepper (md+) */}
      <div className="hidden items-stretch md:flex">
        {steps.map((step, index) => {
          const done = completedSteps.includes(step.id);
          const current = currentStep === step.id;
          const enabled = completedSteps.includes(step.id - 1) || step.id === 1;
          const filled = done || current;
          const shape =
            index === 0 ? CHEVRON_FIRST : index === steps.length - 1 ? CHEVRON_LAST : CHEVRON;
          return (
            <button
              key={step.id}
              type="button"
              onClick={() => enabled && onStepClick(step.id)}
              disabled={!enabled}
              aria-current={current ? 'step' : undefined}
              className={cn(
                'flex h-8 flex-1 items-center justify-center gap-1.5 px-3 text-xs font-medium transition-colors',
                index > 0 && '-ml-2',
                shape,
                filled ? 'bg-brand text-brand-foreground' : 'bg-muted text-muted-foreground',
                enabled && !current && 'cursor-pointer hover:opacity-90',
                !enabled && 'cursor-not-allowed'
              )}
            >
              {done && <Check size={13} className="shrink-0" />}
              <span className="truncate">{step.title}</span>
            </button>
          );
        })}
      </div>

      {/* Compact indicator (below md) */}
      <div className="md:hidden">
        <p className="text-xs font-medium text-muted-foreground">
          Step {currentStep} of {total} · <span className="text-foreground">{currentTitle}</span>
        </p>
        <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-brand transition-all"
            style={{ width: `${(currentStep / total) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
