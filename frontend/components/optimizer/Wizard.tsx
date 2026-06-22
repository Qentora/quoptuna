'use client';

import { Card } from '@/components/ui/card';
import { PageHeader } from '@/components/ui/page-header';
import { cn } from '@/lib/utils';
import {
  BarChart3,
  Check,
  Database,
  FileText,
  type LucideIcon,
  PlayCircle,
  Settings2,
  SlidersHorizontal,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useState } from 'react';
import { StudySummary } from './StudySummary';
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

export interface StepProps {
  onNext: () => void;
  onBack?: () => void;
  workflowData: WorkflowData;
  setWorkflowData: React.Dispatch<React.SetStateAction<WorkflowData>>;
}

export function Wizard() {
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);
  const [workflowData, setWorkflowData] = useState<WorkflowData>(initialWorkflowData);

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
  };

  return (
    <div className="flex h-full flex-col bg-background">
      <div className="border-b border-border bg-card p-6">
        <PageHeader title="Optimizer" />
      </div>

      <div className="grid flex-1 grid-cols-1 gap-6 overflow-hidden p-6 md:grid-cols-[200px_minmax(0,1fr)] lg:grid-cols-[200px_minmax(0,1fr)_240px]">
        {/* Left: vertical stepper */}
        <nav className="hidden md:block">
          <ol className="space-y-1">
            {steps.map((step) => {
              const enabled = completedSteps.includes(step.id - 1) || step.id === 1;
              const done = completedSteps.includes(step.id);
              const current = currentStep === step.id;
              const Icon = step.icon;
              return (
                <li key={step.id}>
                  <button
                    type="button"
                    onClick={() => handleStepClick(step.id)}
                    disabled={!enabled}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-lg px-2 py-2 text-left transition-colors',
                      current
                        ? 'bg-brand/10 text-brand'
                        : enabled
                          ? 'text-muted-foreground hover:bg-accent/60 hover:text-foreground'
                          : 'cursor-not-allowed text-muted-foreground/50'
                    )}
                  >
                    <span
                      className={cn(
                        'flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border transition-colors',
                        done
                          ? 'border-transparent bg-accent-emerald text-accent-emerald-foreground'
                          : current
                            ? 'border-transparent bg-brand text-brand-foreground shadow-glow-brand'
                            : 'border-border bg-card'
                      )}
                    >
                      {done ? <Check size={16} /> : <Icon size={16} />}
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-medium">{step.title}</span>
                      <span className="block text-[11px] text-muted-foreground">
                        Step {step.id} of {steps.length}
                      </span>
                    </span>
                  </button>
                </li>
              );
            })}
          </ol>
        </nav>

        {/* Center: step content (scrollable, sticky footer lives inside steps) */}
        <div className="overflow-y-auto">
          <Card className="p-8">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStep}
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
          </Card>
        </div>

        {/* Right: study summary rail */}
        <div className="hidden lg:block">
          <StudySummary workflowData={workflowData} className="sticky top-0" />
        </div>
      </div>
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
