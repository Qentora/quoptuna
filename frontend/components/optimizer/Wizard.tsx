'use client';

import { Card } from '@/components/ui/card';
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
      <div className="border-b border-border bg-card px-8 py-6">
        <h1 className="text-2xl font-bold tracking-tight">QuOptuna Optimizer</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Quantum-enhanced machine learning with automated hyperparameter optimization
        </p>
      </div>

      <div className="border-b border-border bg-card px-8 py-4">
        <div className="mx-auto flex max-w-5xl items-center justify-between">
          {steps.map((step, index) => {
            const enabled = completedSteps.includes(step.id - 1) || step.id === 1;
            const done = completedSteps.includes(step.id);
            const current = currentStep === step.id;
            const Icon = step.icon;
            return (
              <div key={step.id} className="flex flex-1 items-center">
                <div className="flex flex-1 flex-col items-center">
                  <button
                    type="button"
                    onClick={() => handleStepClick(step.id)}
                    disabled={!enabled}
                    className={cn(
                      'flex h-10 w-10 items-center justify-center rounded-full border transition-colors',
                      done
                        ? 'border-transparent bg-accent-emerald text-accent-emerald-foreground'
                        : current
                          ? 'border-transparent bg-primary text-primary-foreground'
                          : 'border-border bg-card text-muted-foreground',
                      enabled && !current ? 'cursor-pointer hover:border-foreground' : '',
                      !enabled && 'cursor-not-allowed'
                    )}
                  >
                    {done ? <Check className="h-5 w-5" /> : <Icon className="h-5 w-5" />}
                  </button>
                  <div className="mt-2 text-center">
                    <p
                      className={cn(
                        'text-sm font-medium',
                        current
                          ? 'text-foreground'
                          : done
                            ? 'text-accent-emerald-foreground'
                            : 'text-muted-foreground'
                      )}
                    >
                      {step.title}
                    </p>
                    <p className="mt-1 max-w-[120px] text-xs text-muted-foreground">
                      {step.description}
                    </p>
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={cn(
                      'mx-2 h-0.5 flex-1',
                      done ? 'bg-accent-emerald-foreground' : 'bg-border'
                    )}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <Card className="mx-auto max-w-5xl p-8">
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
