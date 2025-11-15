import { useState, useRef } from 'react';
import { CheckCircle2, Circle, X } from 'lucide-react';
import { fetchUCIDataset, uploadDataset } from '../lib/api';

type Step = {
  id: number;
  title: string;
  description: string;
};

const steps: Step[] = [
  { id: 1, title: 'Dataset', description: 'Upload or select your dataset' },
  { id: 2, title: 'Features', description: 'Select features and target' },
  { id: 3, title: 'Configure', description: 'Setup optimization parameters' },
  { id: 4, title: 'Optimize', description: 'Run hyperparameter optimization' },
  { id: 5, title: 'Analyze', description: 'SHAP analysis and visualizations' },
  { id: 6, title: 'Report', description: 'Generate AI summary' },
];

export function Optimizer() {
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  const handleStepClick = (stepId: number) => {
    if (completedSteps.includes(stepId - 1) || stepId === 1) {
      setCurrentStep(stepId);
    }
  };

  const handleNextStep = () => {
    if (currentStep < steps.length) {
      setCompletedSteps([...completedSteps, currentStep]);
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePreviousStep = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-6">
        <h1 className="text-3xl font-bold text-gray-900">QuOptuna Optimizer</h1>
        <p className="text-gray-600 mt-2">
          Quantum-Enhanced Machine Learning with Automated Hyperparameter Optimization
        </p>
      </div>

      {/* Progress Steps */}
      <div className="bg-white border-b border-gray-200 px-8 py-4">
        <div className="flex items-center justify-between max-w-5xl mx-auto">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center flex-1">
              <div className="flex flex-col items-center flex-1">
                <button
                  onClick={() => handleStepClick(step.id)}
                  disabled={!completedSteps.includes(step.id - 1) && step.id !== 1}
                  className={`w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all ${
                    completedSteps.includes(step.id)
                      ? 'bg-green-500 border-green-500 text-white'
                      : currentStep === step.id
                      ? 'bg-blue-500 border-blue-500 text-white'
                      : 'bg-white border-gray-300 text-gray-400'
                  } ${
                    (completedSteps.includes(step.id - 1) || step.id === 1) && currentStep !== step.id
                      ? 'hover:border-blue-400 cursor-pointer'
                      : 'cursor-not-allowed'
                  }`}
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
          ))}
        </div>
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-5xl mx-auto bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          {currentStep === 1 && <DatasetStep onNext={handleNextStep} />}
          {currentStep === 2 && <FeaturesStep onNext={handleNextStep} onBack={handlePreviousStep} />}
          {currentStep === 3 && <ConfigureStep onNext={handleNextStep} onBack={handlePreviousStep} />}
          {currentStep === 4 && <OptimizeStep onNext={handleNextStep} onBack={handlePreviousStep} />}
          {currentStep === 5 && <AnalyzeStep onNext={handleNextStep} onBack={handlePreviousStep} />}
          {currentStep === 6 && <ReportStep onBack={handlePreviousStep} />}
        </div>
      </div>
    </div>
  );
}

// Step Components
function DatasetStep({ onNext }: { onNext: () => void }) {
  const [showUCIModal, setShowUCIModal] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState<{ name: string; source: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);

    try {
      const result = await uploadDataset(file);
      setSelectedDataset({ name: result.filename, source: 'upload' });
      setIsLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload dataset');
      setIsLoading(false);
    }
  };

  const handleUCISelect = async (datasetId: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await fetchUCIDataset(datasetId);
      setSelectedDataset({ name: result.name, source: 'uci' });
      setShowUCIModal(false);
      setIsLoading(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch UCI dataset');
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Dataset Selection</h2>
        <p className="text-gray-600 mt-2">Upload your own dataset or select from UCI ML Repository</p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md">
          {error}
        </div>
      )}

      {selectedDataset && (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-md">
          Selected: {selectedDataset.name} (from {selectedDataset.source === 'upload' ? 'upload' : 'UCI repository'})
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Upload Dataset */}
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
        >
          <div className="text-gray-400 mb-4">
            <svg className="mx-auto h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Dataset</h3>
          <p className="text-sm text-gray-500">Click to browse or drag and drop CSV file</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* UCI Repository */}
        <div
          onClick={() => setShowUCIModal(true)}
          className="border-2 border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors cursor-pointer"
        >
          <div className="text-gray-400 mb-4">
            <svg className="mx-auto h-12 w-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">UCI Repository</h3>
          <p className="text-sm text-gray-500">Select from pre-loaded datasets</p>
        </div>
      </div>

      <div className="flex justify-end pt-4">
        <button
          onClick={onNext}
          disabled={!selectedDataset || isLoading}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Loading...' : 'Next Step'}
        </button>
      </div>

      {/* UCI Modal */}
      {showUCIModal && (
        <UCIDatasetModal
          onClose={() => setShowUCIModal(false)}
          onSelect={handleUCISelect}
          isLoading={isLoading}
        />
      )}
    </div>
  );
}

function FeaturesStep({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Feature Selection</h2>
        <p className="text-gray-600 mt-2">Select input features and target column</p>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <p className="text-sm text-gray-500 text-center">Feature selection UI will be implemented here</p>
      </div>

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function ConfigureStep({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Optimization Configuration</h2>
        <p className="text-gray-600 mt-2">Set up hyperparameter optimization parameters</p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Study Name</label>
          <input
            type="text"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="my-optimization-study"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Database Name</label>
          <input
            type="text"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="results.db"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Number of Trials</label>
          <input
            type="number"
            min="1"
            max="1000"
            defaultValue="100"
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <p className="text-sm text-gray-500 mt-1">Recommended: 50-200 trials</p>
        </div>
      </div>

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function OptimizeStep({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Run Optimization</h2>
        <p className="text-gray-600 mt-2">Execute hyperparameter optimization and view results</p>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
        <button className="px-8 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors font-semibold">
          Start Optimization
        </button>
        <p className="text-sm text-blue-700 mt-3">Click to begin hyperparameter optimization</p>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <p className="text-sm text-gray-500 text-center">Best trials and performance metrics will appear here</p>
      </div>

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function AnalyzeStep({ onNext, onBack }: { onNext: () => void; onBack: () => void }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">SHAP Analysis & Visualizations</h2>
        <p className="text-gray-600 mt-2">Understand feature importance and model behavior</p>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <p className="text-sm text-gray-500 text-center">SHAP plots (bar, beeswarm, violin, heatmap, waterfall) will appear here</p>
      </div>

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function ReportStep({ onBack }: { onBack: () => void }) {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Generate Summary Report</h2>
        <p className="text-gray-600 mt-2">AI-powered analysis report with insights and recommendations</p>
      </div>

      <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
        <button className="px-8 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors font-semibold">
          Generate AI Report
        </button>
        <p className="text-sm text-green-700 mt-3">Create comprehensive analysis using your preferred LLM</p>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <p className="text-sm text-gray-500 text-center">Generated report will appear here</p>
      </div>

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors"
        >
          Previous
        </button>
        <button
          className="px-6 py-2 bg-gray-300 text-gray-500 rounded-md cursor-not-allowed"
          disabled
        >
          Complete
        </button>
      </div>
    </div>
  );
}

// UCI Dataset Modal Component
function UCIDatasetModal({
  onClose,
  onSelect,
  isLoading,
}: {
  onClose: () => void;
  onSelect: (datasetId: number) => void;
  isLoading: boolean;
}) {
  const popularDatasets = [
    { id: 53, name: 'Iris', description: '150 samples, 4 features - Classic classification dataset' },
    { id: 109, name: 'Wine Quality', description: '1599 samples, 11 features - Red wine quality dataset' },
    { id: 17, name: 'Breast Cancer Wisconsin', description: '569 samples, 30 features - Diagnostic dataset' },
    { id: 186, name: 'Wine', description: '178 samples, 13 features - Wine recognition dataset' },
    { id: 267, name: 'Banknote Authentication', description: '1372 samples, 4 features - Classification task' },
  ];

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <h3 className="text-xl font-bold text-gray-900">Select UCI Dataset</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            disabled={isLoading}
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <p className="text-gray-600 mb-4">Choose from popular UCI Machine Learning Repository datasets:</p>
          <div className="space-y-3">
            {popularDatasets.map((dataset) => (
              <button
                key={dataset.id}
                onClick={() => onSelect(dataset.id)}
                disabled={isLoading}
                className="w-full text-left p-4 border border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="flex items-start justify-between">
                  <div>
                    <h4 className="font-semibold text-gray-900">{dataset.name}</h4>
                    <p className="text-sm text-gray-500 mt-1">{dataset.description}</p>
                    <p className="text-xs text-gray-400 mt-1">Dataset ID: {dataset.id}</p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-200 bg-gray-50">
          <p className="text-sm text-gray-500">
            Need a different dataset? You can manually enter the dataset ID in the workflow builder.
          </p>
        </div>
      </div>
    </div>
  );
}
