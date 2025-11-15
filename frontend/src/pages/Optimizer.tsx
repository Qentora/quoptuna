import { useState, useRef } from 'react';
import { CheckCircle2, Circle, X, Loader2, FileText, PlayCircle, BarChart3 } from 'lucide-react';
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

// Shared workflow data type
export interface WorkflowData {
  dataset: {
    id: string;
    name: string;
    source: 'upload' | 'uci';
    rows: number;
    columns: string[];
    preview?: Array<Record<string, any>>;
  } | null;
  features: {
    selectedFeatures: string[];
    targetColumn: string | null;
  };
  configuration: {
    studyName: string;
    databaseName: string;
    numTrials: number;
  };
  optimization: {
    executionId: string | null;
    status: 'idle' | 'running' | 'completed' | 'failed';
    results: any | null;
  };
  analysis: {
    shapData: any | null;
  };
  report: {
    summary: string | null;
  };
}

export function Optimizer() {
  const [currentStep, setCurrentStep] = useState(1);
  const [completedSteps, setCompletedSteps] = useState<number[]>([]);

  // Shared workflow data
  const [workflowData, setWorkflowData] = useState<WorkflowData>({
    dataset: null,
    features: {
      selectedFeatures: [],
      targetColumn: null,
    },
    configuration: {
      studyName: 'my-optimization-study',
      databaseName: 'results.db',
      numTrials: 100,
    },
    optimization: {
      executionId: null,
      status: 'idle',
      results: null,
    },
    analysis: {
      shapData: null,
    },
    report: {
      summary: null,
    },
  });

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
          {currentStep === 1 && (
            <DatasetStep
              onNext={handleNextStep}
              workflowData={workflowData}
              setWorkflowData={setWorkflowData}
            />
          )}
          {currentStep === 2 && (
            <FeaturesStep
              onNext={handleNextStep}
              onBack={handlePreviousStep}
              workflowData={workflowData}
              setWorkflowData={setWorkflowData}
            />
          )}
          {currentStep === 3 && (
            <ConfigureStep
              onNext={handleNextStep}
              onBack={handlePreviousStep}
              workflowData={workflowData}
              setWorkflowData={setWorkflowData}
            />
          )}
          {currentStep === 4 && (
            <OptimizeStep
              onNext={handleNextStep}
              onBack={handlePreviousStep}
              workflowData={workflowData}
              setWorkflowData={setWorkflowData}
            />
          )}
          {currentStep === 5 && (
            <AnalyzeStep
              onNext={handleNextStep}
              onBack={handlePreviousStep}
              workflowData={workflowData}
              setWorkflowData={setWorkflowData}
            />
          )}
          {currentStep === 6 && (
            <ReportStep
              onNext={() => {}}
              onBack={handlePreviousStep}
              workflowData={workflowData}
              setWorkflowData={setWorkflowData}
            />
          )}
        </div>
      </div>
    </div>
  );
}

// Step Components
interface StepProps {
  onNext: () => void;
  onBack?: () => void;
  workflowData: WorkflowData;
  setWorkflowData: React.Dispatch<React.SetStateAction<WorkflowData>>;
}

function DatasetStep({ onNext, workflowData, setWorkflowData }: StepProps) {
  const [showUCIModal, setShowUCIModal] = useState(false);
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
      setWorkflowData((prev) => ({
        ...prev,
        dataset: {
          id: result.id,
          name: result.filename,
          source: 'upload',
          rows: result.rows,
          columns: result.columns,
        },
      }));
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
      setWorkflowData((prev) => ({
        ...prev,
        dataset: {
          id: result.dataset_id,
          name: result.name,
          source: 'uci',
          rows: result.rows,
          columns: result.columns,
        },
      }));
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

      {workflowData.dataset && (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-md">
          <p className="font-medium">
            Selected: {workflowData.dataset.name}
          </p>
          <p className="text-sm mt-1">
            Source: {workflowData.dataset.source === 'upload' ? 'Uploaded File' : 'UCI Repository'} |
            Rows: {workflowData.dataset.rows} |
            Columns: {workflowData.dataset.columns.length}
          </p>
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
          disabled={!workflowData.dataset || isLoading}
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

function FeaturesStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const toggleFeature = (column: string) => {
    setWorkflowData((prev) => {
      const currentFeatures = prev.features.selectedFeatures;
      const newFeatures = currentFeatures.includes(column)
        ? currentFeatures.filter((f) => f !== column)
        : [...currentFeatures, column];

      return {
        ...prev,
        features: {
          ...prev.features,
          selectedFeatures: newFeatures,
        },
      };
    });
  };

  const setTarget = (column: string) => {
    setWorkflowData((prev) => ({
      ...prev,
      features: {
        ...prev.features,
        targetColumn: prev.features.targetColumn === column ? null : column,
      },
    }));
  };

  const selectAllFeatures = () => {
    if (!workflowData.dataset) return;
    const allColumns = workflowData.dataset.columns.filter(
      (col) => col !== workflowData.features.targetColumn
    );
    setWorkflowData((prev) => ({
      ...prev,
      features: {
        ...prev.features,
        selectedFeatures: allColumns,
      },
    }));
  };

  const clearAllFeatures = () => {
    setWorkflowData((prev) => ({
      ...prev,
      features: {
        ...prev.features,
        selectedFeatures: [],
      },
    }));
  };

  const canProceed =
    workflowData.features.selectedFeatures.length > 0 &&
    workflowData.features.targetColumn !== null;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Feature Selection</h2>
        <p className="text-gray-600 mt-2">Select input features and target column for optimization</p>
      </div>

      {/* No Dataset Warning */}
      {!workflowData.dataset && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-md">
          <p className="font-medium">No dataset selected</p>
          <p className="text-sm mt-1">Please go back to Step 1 and select a dataset first.</p>
        </div>
      )}

      {/* Dataset Info */}
      {workflowData.dataset && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <p className="font-medium text-blue-900">{workflowData.dataset.name}</p>
              <p className="text-sm text-blue-700">
                {workflowData.dataset.rows} rows × {workflowData.dataset.columns.length} columns
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Column Selection */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Columns</h3>
          <div className="flex gap-2">
            <button
              onClick={selectAllFeatures}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              Select All as Features
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={clearAllFeatures}
              className="text-sm text-blue-600 hover:text-blue-700"
            >
              Clear Selection
            </button>
          </div>
        </div>

        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Column Name</th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-900">Feature</th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-900">Target</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {(workflowData.dataset?.columns || []).map((column) => (
                <tr key={column} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">{column}</td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="checkbox"
                      checked={workflowData.features.selectedFeatures.includes(column)}
                      onChange={() => toggleFeature(column)}
                      disabled={workflowData.features.targetColumn === column}
                      className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500 disabled:opacity-50"
                    />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="radio"
                      name="target"
                      checked={workflowData.features.targetColumn === column}
                      onChange={() => setTarget(column)}
                      className="w-4 h-4 text-blue-600 focus:ring-blue-500"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Selection Summary */}
        <div className="bg-gray-50 rounded-lg p-4 space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Selected Features:</span>
            <span className="font-medium text-gray-900">
              {workflowData.features.selectedFeatures.length} column(s)
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Target Column:</span>
            <span className="font-medium text-gray-900">
              {workflowData.features.targetColumn || 'Not selected'}
            </span>
          </div>
        </div>

        {!canProceed && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-md text-sm">
            Please select at least one feature and one target column to continue.
          </div>
        )}
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
          disabled={!canProceed}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function ConfigureStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const updateConfig = (field: keyof WorkflowData['configuration'], value: string | number) => {
    setWorkflowData((prev) => ({
      ...prev,
      configuration: {
        ...prev.configuration,
        [field]: value,
      },
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Optimization Configuration</h2>
        <p className="text-gray-600 mt-2">Set up hyperparameter optimization parameters</p>
      </div>

      {/* Dataset Summary */}
      {workflowData.dataset && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
            <div className="flex-1">
              <p className="font-medium text-blue-900">{workflowData.dataset.name}</p>
              <p className="text-sm text-blue-700 mt-1">
                Features: {workflowData.features.selectedFeatures.join(', ')} |
                Target: {workflowData.features.targetColumn}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Study Name</label>
          <input
            type="text"
            value={workflowData.configuration.studyName}
            onChange={(e) => updateConfig('studyName', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="my-optimization-study"
          />
          <p className="text-sm text-gray-500 mt-1">Unique name for this optimization study</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Database Name</label>
          <input
            type="text"
            value={workflowData.configuration.databaseName}
            onChange={(e) => updateConfig('databaseName', e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="results.db"
          />
          <p className="text-sm text-gray-500 mt-1">SQLite database to store optimization results</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Number of Trials</label>
          <input
            type="number"
            min="1"
            max="1000"
            value={workflowData.configuration.numTrials}
            onChange={(e) => updateConfig('numTrials', parseInt(e.target.value))}
            className="w-full px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
          <p className="text-sm text-gray-500 mt-1">Recommended: 50-200 trials (more trials = better results but longer time)</p>
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

function OptimizeStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentTrial, setCurrentTrial] = useState(0);

  const startOptimization = async () => {
    setIsRunning(true);
    setProgress(0);
    setCurrentTrial(0);

    // Simulate optimization progress
    const totalTrials = workflowData.configuration.numTrials;
    for (let i = 1; i <= totalTrials; i++) {
      await new Promise((resolve) => setTimeout(resolve, 50)); // Simulate trial execution
      setCurrentTrial(i);
      setProgress((i / totalTrials) * 100);
    }

    // Simulate results
    const mockResults = {
      bestValue: 0.9234,
      bestParams: {
        n_estimators: 150,
        max_depth: 8,
        learning_rate: 0.05,
        min_samples_split: 5,
      },
      trials: Array.from({ length: Math.min(10, totalTrials) }, (_, i) => ({
        trial: i + 1,
        value: 0.85 + Math.random() * 0.1,
        params: {
          n_estimators: Math.floor(50 + Math.random() * 200),
          max_depth: Math.floor(3 + Math.random() * 10),
        },
      })),
    };

    setWorkflowData((prev) => ({
      ...prev,
      optimization: {
        executionId: `exec_${Date.now()}`,
        status: 'completed',
        results: mockResults,
      },
    }));

    setIsRunning(false);
  };

  const hasResults = workflowData.optimization.status === 'completed';

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Run Optimization</h2>
        <p className="text-gray-600 mt-2">Execute hyperparameter optimization and view results</p>
      </div>

      {/* Configuration Summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium text-blue-900">Optimization Configuration</p>
            <p className="text-sm text-blue-700 mt-1">
              Study: {workflowData.configuration.studyName} |
              Trials: {workflowData.configuration.numTrials} |
              Features: {workflowData.features.selectedFeatures.length}
            </p>
          </div>
        </div>
      </div>

      {/* Start Button */}
      {!hasResults && !isRunning && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
          <button
            onClick={startOptimization}
            className="px-8 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors font-semibold inline-flex items-center gap-2"
          >
            <PlayCircle className="w-5 h-5" />
            Start Optimization
          </button>
          <p className="text-sm text-blue-700 mt-3">
            This will run {workflowData.configuration.numTrials} trials to find the best hyperparameters
          </p>
        </div>
      )}

      {/* Progress */}
      {isRunning && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
              <div>
                <p className="font-medium text-gray-900">Optimization in Progress</p>
                <p className="text-sm text-gray-500">
                  Trial {currentTrial} of {workflowData.configuration.numTrials}
                </p>
              </div>
            </div>
            <span className="text-2xl font-bold text-blue-600">{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className="bg-blue-600 h-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Results */}
      {hasResults && workflowData.optimization.results && (
        <div className="space-y-4">
          {/* Best Result */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <div className="flex items-start gap-3 mb-4">
              <BarChart3 className="w-6 h-6 text-green-600" />
              <div>
                <h3 className="text-lg font-semibold text-green-900">Best Trial Results</h3>
                <p className="text-sm text-green-700">Optimal hyperparameters found</p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-green-700 mb-1">Best Score</p>
                <p className="text-3xl font-bold text-green-900">
                  {workflowData.optimization.results.bestValue.toFixed(4)}
                </p>
              </div>
              <div className="bg-white rounded-lg p-4">
                <p className="text-sm font-medium text-gray-700 mb-2">Best Parameters</p>
                <div className="space-y-1 text-sm text-gray-600">
                  {Object.entries(workflowData.optimization.results.bestParams).map(([key, value]) => (
                    <div key={key} className="flex justify-between">
                      <span className="font-mono">{key}:</span>
                      <span className="font-semibold">{String(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Trial History */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Trial History</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">Trial</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">Score</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">n_estimators</th>
                    <th className="px-4 py-2 text-left text-sm font-semibold text-gray-900">max_depth</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {workflowData.optimization.results.trials.map((trial: any) => (
                    <tr key={trial.trial} className="hover:bg-gray-50">
                      <td className="px-4 py-2 text-sm text-gray-900">#{trial.trial}</td>
                      <td className="px-4 py-2 text-sm font-medium text-gray-900">
                        {trial.value.toFixed(4)}
                      </td>
                      <td className="px-4 py-2 text-sm text-gray-600">{trial.params.n_estimators}</td>
                      <td className="px-4 py-2 text-sm text-gray-600">{trial.params.max_depth}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          disabled={isRunning}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          disabled={!hasResults || isRunning}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function AnalyzeStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedPlot, setSelectedPlot] = useState<string>('summary');

  const generateSHAP = async () => {
    setIsGenerating(true);

    // Simulate SHAP generation
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const mockSHAPData = {
      featureImportance: workflowData.features.selectedFeatures.map((feature) => ({
        feature,
        importance: Math.random() * 0.5 + 0.1,
      })).sort((a, b) => b.importance - a.importance),
      generated: true,
    };

    setWorkflowData((prev) => ({
      ...prev,
      analysis: {
        shapData: mockSHAPData,
      },
    }));

    setIsGenerating(false);
  };

  const hasSHAP = workflowData.analysis.shapData?.generated;

  const plotTypes = [
    { id: 'summary', name: 'Summary Plot', description: 'Overall feature importance' },
    { id: 'bar', name: 'Bar Plot', description: 'Mean absolute SHAP values' },
    { id: 'beeswarm', name: 'Beeswarm Plot', description: 'Distribution of SHAP values' },
    { id: 'waterfall', name: 'Waterfall Plot', description: 'Individual prediction explanation' },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">SHAP Analysis & Visualizations</h2>
        <p className="text-gray-600 mt-2">Understand feature importance and model behavior</p>
      </div>

      {/* Optimization Summary */}
      {workflowData.optimization.results && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <BarChart3 className="w-5 h-5 text-green-600 mt-0.5" />
            <div>
              <p className="font-medium text-green-900">
                Best Score: {workflowData.optimization.results.bestValue.toFixed(4)}
              </p>
              <p className="text-sm text-green-700">
                Analyzing {workflowData.features.selectedFeatures.length} features
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Generate SHAP Button */}
      {!hasSHAP && !isGenerating && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 text-center">
          <button
            onClick={generateSHAP}
            className="px-8 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors font-semibold inline-flex items-center gap-2"
          >
            <BarChart3 className="w-5 h-5" />
            Generate SHAP Analysis
          </button>
          <p className="text-sm text-blue-700 mt-3">
            This will generate SHAP (SHapley Additive exPlanations) visualizations
          </p>
        </div>
      )}

      {/* Generating */}
      {isGenerating && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 text-center">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
          <p className="font-medium text-gray-900">Generating SHAP Analysis...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a moment</p>
        </div>
      )}

      {/* SHAP Results */}
      {hasSHAP && workflowData.analysis.shapData && (
        <div className="space-y-4">
          {/* Plot Type Selector */}
          <div className="flex gap-2 overflow-x-auto">
            {plotTypes.map((plot) => (
              <button
                key={plot.id}
                onClick={() => setSelectedPlot(plot.id)}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap ${
                  selectedPlot === plot.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {plot.name}
              </button>
            ))}
          </div>

          {/* Feature Importance Chart */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              {plotTypes.find((p) => p.id === selectedPlot)?.name}
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              {plotTypes.find((p) => p.id === selectedPlot)?.description}
            </p>

            {/* Bar Chart Visualization */}
            <div className="space-y-3">
              {workflowData.analysis.shapData.featureImportance.map((item: any, index: number) => (
                <div key={item.feature} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium text-gray-700">{item.feature}</span>
                    <span className="text-gray-500">{item.importance.toFixed(3)}</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${(item.importance / workflowData.analysis.shapData.featureImportance[0].importance) * 100}%`,
                        backgroundColor: `hsl(${220 - index * 20}, 70%, 50%)`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Insights */}
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
            <h4 className="font-medium text-purple-900 mb-2">Key Insights</h4>
            <ul className="text-sm text-purple-700 space-y-1">
              <li>• {workflowData.analysis.shapData.featureImportance[0].feature} has the highest impact on predictions</li>
              <li>• Top 3 features account for majority of model behavior</li>
              <li>• All selected features contribute to the final predictions</li>
            </ul>
          </div>
        </div>
      )}

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          disabled={isGenerating}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          onClick={onNext}
          disabled={!hasSHAP || isGenerating}
          className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          Next Step
        </button>
      </div>
    </div>
  );
}

function ReportStep({ onBack, workflowData, setWorkflowData }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);

  const generateReport = async () => {
    setIsGenerating(true);

    // Simulate AI report generation
    await new Promise((resolve) => setTimeout(resolve, 3000));

    const report = `# Optimization Analysis Report

## Dataset Overview
- **Dataset**: ${workflowData.dataset?.name || 'N/A'}
- **Source**: ${workflowData.dataset?.source === 'uci' ? 'UCI ML Repository' : 'Uploaded File'}
- **Samples**: ${workflowData.dataset?.rows || 0} rows
- **Features Used**: ${workflowData.features.selectedFeatures.join(', ')}
- **Target Variable**: ${workflowData.features.targetColumn}

## Optimization Results
The hyperparameter optimization process completed successfully with ${workflowData.configuration.numTrials} trials.

### Best Model Performance
- **Score**: ${workflowData.optimization.results?.bestValue.toFixed(4) || 'N/A'}
- **Optimal Hyperparameters**:
${Object.entries(workflowData.optimization.results?.bestParams || {}).map(([k, v]) => `  - ${k}: ${v}`).join('\n')}

## Feature Importance Analysis
Based on SHAP analysis, the following features have the highest impact on model predictions:

${workflowData.analysis.shapData?.featureImportance.slice(0, 5).map((item: any, i: number) =>
`${i + 1}. **${item.feature}**: ${item.importance.toFixed(3)}`
).join('\n')}

## Key Insights

1. **Model Performance**: The optimized model achieved a score of ${workflowData.optimization.results?.bestValue.toFixed(4)}, indicating strong predictive performance on the selected dataset.

2. **Feature Contributions**: The feature "${workflowData.analysis.shapData?.featureImportance[0]?.feature}" has the most significant impact on predictions, suggesting it should be carefully monitored in production.

3. **Hyperparameter Configuration**: The optimal configuration suggests using ${workflowData.optimization.results?.bestParams.n_estimators || 'N/A'} estimators with a maximum depth of ${workflowData.optimization.results?.bestParams.max_depth || 'N/A'}, balancing model complexity and performance.

## Recommendations

1. **Production Deployment**: Consider deploying this model with the identified optimal hyperparameters for best results.

2. **Monitoring**: Implement continuous monitoring for the top contributing features to detect potential data drift.

3. **Further Optimization**: For even better results, consider:
   - Increasing the number of trials to ${workflowData.configuration.numTrials * 2}
   - Exploring additional feature engineering techniques
   - Testing different ML algorithms

4. **Validation**: Perform cross-validation on an independent test set to ensure generalization.

---
*Report generated on ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}*
`;

    setWorkflowData((prev) => ({
      ...prev,
      report: {
        summary: report,
      },
    }));

    setIsGenerating(false);
  };

  const hasReport = workflowData.report.summary !== null;

  const downloadReport = () => {
    const blob = new Blob([workflowData.report.summary || ''], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimization-report-${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-gray-900">Generate Summary Report</h2>
        <p className="text-gray-600 mt-2">AI-powered analysis report with insights and recommendations</p>
      </div>

      {/* Workflow Summary */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <BarChart3 className="w-5 h-5 text-purple-600 mt-0.5" />
          <div className="flex-1">
            <p className="font-medium text-purple-900">Workflow Complete</p>
            <div className="grid grid-cols-2 gap-2 mt-2 text-sm text-purple-700">
              <div>Dataset: {workflowData.dataset?.name}</div>
              <div>Features: {workflowData.features.selectedFeatures.length}</div>
              <div>Trials: {workflowData.configuration.numTrials}</div>
              <div>Best Score: {workflowData.optimization.results?.bestValue.toFixed(4)}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Generate Button */}
      {!hasReport && !isGenerating && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
          <button
            onClick={generateReport}
            className="px-8 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors font-semibold inline-flex items-center gap-2"
          >
            <FileText className="w-5 h-5" />
            Generate AI Report
          </button>
          <p className="text-sm text-green-700 mt-3">
            Create a comprehensive markdown report with insights and recommendations
          </p>
        </div>
      )}

      {/* Generating */}
      {isGenerating && (
        <div className="bg-white border border-gray-200 rounded-lg p-6 text-center">
          <Loader2 className="w-12 h-12 text-green-600 animate-spin mx-auto mb-4" />
          <p className="font-medium text-gray-900">Generating AI Report...</p>
          <p className="text-sm text-gray-500 mt-2">Analyzing results and creating insights</p>
        </div>
      )}

      {/* Report Display */}
      {hasReport && workflowData.report.summary && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Generated Report</h3>
            <button
              onClick={downloadReport}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors text-sm inline-flex items-center gap-2"
            >
              <FileText className="w-4 h-4" />
              Download Report
            </button>
          </div>

          <div className="bg-white border border-gray-200 rounded-lg p-6 max-h-[500px] overflow-y-auto">
            <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">
              {workflowData.report.summary}
            </pre>
          </div>

          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="text-sm text-green-700">
              ✓ Report generated successfully! You can download it or continue with further analysis.
            </p>
          </div>
        </div>
      )}

      <div className="flex justify-between pt-4">
        <button
          onClick={onBack}
          disabled={isGenerating}
          className="px-6 py-2 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Previous
        </button>
        <button
          disabled={!hasReport}
          className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {hasReport ? 'Workflow Complete ✓' : 'Complete'}
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
