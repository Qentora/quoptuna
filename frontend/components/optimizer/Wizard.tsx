'use client';

import { Button } from '@/components/ui/button';
import { PageHeader } from '@/components/ui/page-header';
import { cn } from '@/lib/utils';
import { clearWizardState, loadWizardState, saveWizardState } from '@/lib/wizardStorage';
import {
  BarChart3,
  Database,
  FileText,
  History,
  type LucideIcon,
  PlayCircle,
  RotateCcw,
  Settings2,
  SlidersHorizontal,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { ContextStrip } from './ContextStrip';
import { StepProgress } from './StepProgress';
import { AnalyzeStep } from './steps/AnalyzeStep';
import { ConfigureStep } from './steps/ConfigureStep';
import { DatasetStep } from './steps/DatasetStep';
import { FeaturesStep } from './steps/FeaturesStep';
import { OptimizeStep } from './steps/OptimizeStep';
import { ReportStep } from './steps/ReportStep';
import type { WorkflowData } from './types';
import { initialWorkflowData } from './types';

const steps: Array<{ id: number; title: string; description: string; icon: LucideIcon }> = [
  { id: 1, title: 'Dataset', description: 'Upload or select your dataset', icon: Database },
  { id: 2, title: 'Features', description: 'Select features and target', icon: SlidersHorizontal },
  { id: 3, title: 'Configure', description: 'Setup optimization parameters', icon: Settings2 },
  { id: 4, title: 'Optimize', description: 'Run hyperparameter optimization', icon: PlayCircle },
  { id: 5, title: 'Analyze', description: 'SHAP analysis and visualizations', icon: BarChart3 },
  { id: 6, title: 'Report', description: 'Generate AI summary', icon: FileText },
];

const stepIcons: Record<number, LucideIcon> = Object.fromEntries(steps.map((s) => [s.id, s.icon]));

export interface StepFooterState {
  canContinue: boolean;
  nextLabel?: string;
  hideNext?: boolean;
  nextBusy?: boolean;
  backDisabled?: boolean;
}

export interface StepProps {
  onNext: () => void;
  onBack?: () => void;
  workflowData: WorkflowData;
  setWorkflowData: React.Dispatch<React.SetStateAction<WorkflowData>>;
  setFooter: (state: StepFooterState) => void;
}

export function Wizard() {
  // Restore any saved state synchronously on first render (the page is
  // client-only via ssr:false, so localStorage is available here).
  const restored = useRef(loadWizardState());
  const [currentStep, setCurrentStep] = useState(restored.current?.currentStep ?? 1);
  const [completedSteps, setCompletedSteps] = useState<number[]>(
    restored.current?.completedSteps ?? []
  );
  const [workflowData, setWorkflowData] = useState<WorkflowData>(
    restored.current
      ? { ...initialWorkflowData, ...restored.current.workflowData }
      : initialWorkflowData
  );
  const [footer, setFooter] = useState<StepFooterState>({ canContinue: false });

  // Autosave (debounced) so a refresh or navigation never loses progress.
  useEffect(() => {
    const timer = setTimeout(
      () => saveWizardState({ currentStep, completedSteps, workflowData }),
      300
    );
    return () => clearTimeout(timer);
  }, [currentStep, completedSteps, workflowData]);

  const handleReset = () => {
    clearWizardState();
    setCurrentStep(1);
    setCompletedSteps([]);
    setWorkflowData(initialWorkflowData);
  };

  // Reset footer on step change so a stale canContinue can't leak across steps.
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-run on step change is the intent.
  useEffect(() => {
    setFooter({ canContinue: false });
  }, [currentStep]);

  const handleStepClick = (stepId: number) => {
    if (completedSteps.includes(stepId - 1) || stepId === 1) {
      setCurrentStep(stepId);
    }
  };

  const handleNextStep = () => {
    if (currentStep < steps.length) {
      setCompletedSteps((prev) => Array.from(new Set([...prev, currentStep])));
      setCurrentStep((s) => s + 1);
    }
  };

  const handlePreviousStep = () => {
    if (currentStep > 1) setCurrentStep((s) => s - 1);
  };

  const stepProps: StepProps = {
    onNext: handleNextStep,
    onBack: handlePreviousStep,
    workflowData,
    setWorkflowData,
    setFooter,
  };

  // Steps that manage their own height (panels fill the frame and scroll internally): Dataset (1),
  // Features (2), Optimize (4), Analyze (5), Report (6). Step 3 (Configure) is a short form that
  // keeps the container's normal vertical scroll.
  const fillsFrame = currentStep !== 3;

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-background">
      {/* Header: title + context strip + chevron stepper (pinned) */}
      <header className="shrink-0 border-b border-border bg-card px-6 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <PageHeader title="Optimizer" />
          <div className="flex items-center gap-2">
            <ContextStrip workflowData={workflowData} />
            <Button type="button" variant="outline" size="sm" asChild>
              <Link href="/runs">
                <History className="h-4 w-4" />
                Runs
              </Link>
            </Button>
            <Button type="button" variant="outline" size="sm" onClick={handleReset}>
              <RotateCcw className="h-4 w-4" />
              Start new run
            </Button>
          </div>
        </div>
        <StepProgress
          steps={steps}
          currentStep={currentStep}
          completedSteps={completedSteps}
          onStepClick={handleStepClick}
          className="mt-4"
        />
      </header>

      {/* Body: content region (scrolls for most steps; DatasetStep fills the frame) */}
      <div className="min-h-0 flex-1 px-6 py-4">
        <div
          className={cn(
            'h-full min-h-0 rounded-lg border border-border bg-card',
            fillsFrame ? 'flex flex-col overflow-hidden' : 'overflow-y-auto'
          )}
        >
          <div
            className={cn(
              'mx-auto w-full max-w-4xl p-6',
              fillsFrame && 'flex min-h-0 flex-1 flex-col'
            )}
          >
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
                className={fillsFrame ? 'flex min-h-0 w-full flex-1 flex-col' : undefined}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.18 }}
              >
                {currentStep === 1 && <DatasetStep {...stepProps} />}
                {currentStep === 2 && <FeaturesStep {...stepProps} />}
                {currentStep === 3 && <ConfigureStep {...stepProps} />}
                {currentStep === 4 && <OptimizeStep {...stepProps} />}
                {currentStep === 5 && <AnalyzeStep {...stepProps} />}
                {currentStep === 6 && <ReportStep {...stepProps} />}
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>

      {/* Single Back/Next footer (pinned) */}
      <footer className="shrink-0 border-t border-border bg-card/95 px-6 py-3 backdrop-blur supports-[backdrop-filter]:bg-card/80">
        <div className="flex items-center justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={handlePreviousStep}
            disabled={currentStep === 1 || footer.backDisabled}
          >
            Previous
          </Button>
          {!footer.hideNext && (
            <Button
              type="button"
              onClick={handleNextStep}
              disabled={!footer.canContinue || footer.nextBusy}
            >
              {footer.nextLabel ?? 'Next Step'}
            </Button>
          )}
        </div>
      </footer>
    </div>
  );
}

export function StepHeader({
  title,
  subtitle,
  step,
}: {
  title: string;
  subtitle: string;
  step?: number;
}) {
  const Icon = step ? stepIcons[step] : BarChart3;
  return (
    <div>
      <h2 className="flex items-center gap-2 text-xl font-bold tracking-tight">
        {Icon && <Icon className="h-6 w-6 text-muted-foreground" />}
        {title}
      </h2>
      <p className="mt-1 text-sm text-muted-foreground">{subtitle}</p>
    </div>
  );
}
