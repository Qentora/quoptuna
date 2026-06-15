'use client';

import { BarChart3, CheckCircle2, Circle } from 'lucide-react';
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

const steps = [
  { id: 1, title: 'Dataset', description: 'Upload or select your dataset' },
  { id: 2, title: 'Features', description: 'Select features and target' },
  { id: 3, title: 'Configure', description: 'Setup optimization parameters' },
  { id: 4, title: 'Optimize', description: 'Run hyperparameter optimization' },
  { id: 5, title: 'Analyze', description: 'SHAP analysis and visualizations' },
  { id: 6, title: 'Report', description: 'Generate AI summary' },
];

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
    <div className="h-full flex flex-col bg-gray-50">
      <div className="bg-white border-b border-gray-200 px-8 py-6">
        <h1 className="text-3xl font-bold text-gray-900">QuOptuna Optimizer</h1>
        <p className="text-gray-600 mt-2">
          Quantum-Enhanced Machine Learning with Automated Hyperparameter Optimization
        </p>
      </div>

      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <div className="flex items-center justify-between max-w-5xl mx-auto">
          {steps.map((step, index) => {
            const enabled = completedSteps.includes(step.id - 1) || step.id === 1;
            return (
              <div key={step.id} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <button
                    type="button"
                    onClick={() => handleStepClick(step.id)}
                    disabled={!enabled}
                    className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all ${
                      completedSteps.includes(step.id)
                        ? 'bg-green-500 border-green-500 text-white'
                        : currentStep === step.id
                          ? 'bg-blue-500 border-blue-500 text-white'
                          : 'bg-white border-gray-300 text-gray-400'
                    } ${enabled && currentStep !== step.id ? 'hover:border-blue-400 cursor-pointer' : 'cursor-not-allowed'}`}
                  >
                    {completedSteps.includes(step.id) ? (
                      <CheckCircle2 className="w-5 h-5" />
                    ) : (
                      <Circle className="w-5 h-5" />
                    )}
                  </button>
                  <div className="text-center mt-2">
                    <p
                      className={`text-sm font-medium ${
                        currentStep === step.id
                          ? 'text-blue-600'
                          : completedSteps.includes(step.id)
                            ? 'text-green-600'
                            : 'text-gray-400'
                      }`}
                    >
                      {step.title}
                    </p>
                    <p className="text-xs text-gray-500 mt-1 max-w-[120px]">{step.description}</p>
                  </div>
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-0.5 flex-1 mx-2 ${
                      completedSteps.includes(step.id) ? 'bg-green-500' : 'bg-gray-300'
                    }`}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-5xl mx-auto bg-white rounded-lg shadow-sm border border-gray-200 p-8">
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
        </div>
      </div>
    </div>
  );
}

export function StepHeader({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <div>
      <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
        <BarChart3 className="w-6 h-6 text-blue-600" />
        {title}
      </h2>
      <p className="text-gray-600 mt-2">{subtitle}</p>
    </div>
  );
}
