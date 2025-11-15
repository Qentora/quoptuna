import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import type { WorkflowNode } from '../../types/workflow';
import { uploadDataset } from '../../lib/api';

interface NodeConfigPanelProps {
  node: WorkflowNode | null;
  onClose: () => void;
  onSave: (nodeId: string, config: any) => void;
}

export function NodeConfigPanel({ node, onClose, onSave }: NodeConfigPanelProps) {
  const [config, setConfig] = useState<any>({});
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    if (node?.data.config) {
      setConfig(node.data.config);
    } else {
      setConfig({});
    }
  }, [node]);

  if (!node) return null;

  const handleSave = () => {
    onSave(node.id, config);
    onClose();
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setIsUploading(true);
      const response = await uploadDataset(file);
      setConfig({
        ...config,
        file_path: response.file_path,
        filename: response.filename,
        rows: response.rows,
        columns: response.columns,
      });
    } catch (error) {
      alert(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsUploading(false);
    }
  };

  const renderConfigForm = () => {
    switch (node.data.type) {
      case 'data-upload':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Upload CSV File</label>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                disabled={isUploading}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              {isUploading && <p className="text-sm text-gray-500 mt-2">Uploading...</p>}
              {config.filename && (
                <div className="mt-2 p-2 bg-green-50 rounded text-sm">
                  <p><strong>File:</strong> {config.filename}</p>
                  <p><strong>Rows:</strong> {config.rows}</p>
                  <p><strong>Columns:</strong> {config.columns?.join(', ')}</p>
                </div>
              )}
            </div>
          </div>
        );

      case 'data-uci':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Dataset ID</label>
              <input
                type="text"
                value={config.dataset_id || ''}
                onChange={(e) => setConfig({ ...config, dataset_id: e.target.value })}
                placeholder="e.g., 53 for Iris"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                Popular: 53 (Iris), 109 (Wine), 17 (Breast Cancer)
              </p>
            </div>
          </div>
        );

      case 'feature-selection':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Feature Columns (X)</label>
              <input
                type="text"
                value={config.x_columns?.join(', ') || ''}
                onChange={(e) =>
                  setConfig({
                    ...config,
                    x_columns: e.target.value.split(',').map((s) => s.trim()),
                  })
                }
                placeholder="column1, column2, column3"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">Comma-separated column names</p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Target Column (y)</label>
              <input
                type="text"
                value={config.y_column || ''}
                onChange={(e) => setConfig({ ...config, y_column: e.target.value })}
                placeholder="target"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        );

      case 'quantum-model':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Quantum Model</label>
              <select
                value={config.model_name || ''}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a model...</option>
                <option value="DataReuploading">Data Reuploading</option>
                <option value="QuantumKitchen">Quantum Kitchen</option>
                <option value="SeparableTwoDesign">Separable Two Design</option>
                <option value="BasicEntanglerLayers">Basic Entangler Layers</option>
                <option value="StronglyEntanglingLayers">Strongly Entangling Layers</option>
                <option value="QuantumMetricLearning">Quantum Metric Learning</option>
                <option value="SimplifiedTwoDesign">Simplified Two Design</option>
                <option value="QCNN">QCNN</option>
                <option value="TreeTensorNetwork">Tree Tensor Network</option>
                <option value="MERA">MERA</option>
              </select>
            </div>
          </div>
        );

      case 'classical-model':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Classical Model</label>
              <select
                value={config.model_name || ''}
                onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a model...</option>
                <option value="RandomForest">Random Forest</option>
                <option value="LogisticRegression">Logistic Regression</option>
                <option value="SVC">Support Vector Classifier</option>
                <option value="GradientBoosting">Gradient Boosting</option>
                <option value="AdaBoost">AdaBoost</option>
                <option value="KNN">K-Nearest Neighbors</option>
                <option value="DecisionTree">Decision Tree</option>
                <option value="NaiveBayes">Naive Bayes</option>
              </select>
            </div>
          </div>
        );

      case 'optuna-config':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Study Name</label>
              <input
                type="text"
                value={config.study_name || ''}
                onChange={(e) => setConfig({ ...config, study_name: e.target.value })}
                placeholder="workflow_study"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Number of Trials</label>
              <input
                type="number"
                value={config.n_trials || 100}
                onChange={(e) =>
                  setConfig({ ...config, n_trials: parseInt(e.target.value) })
                }
                min="1"
                max="1000"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-xs text-gray-500 mt-1">
                More trials = better results but longer execution time
              </p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-2">Database Name</label>
              <input
                type="text"
                value={config.db_name || ''}
                onChange={(e) => setConfig({ ...config, db_name: e.target.value })}
                placeholder="workflow_optimization.db"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        );

      case 'shap-analysis':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Plot Types</label>
              <div className="space-y-2">
                {['bar', 'beeswarm', 'violin'].map((plotType) => (
                  <label key={plotType} className="flex items-center">
                    <input
                      type="checkbox"
                      checked={config.plot_types?.includes(plotType) || false}
                      onChange={(e) => {
                        const currentTypes = config.plot_types || [];
                        const newTypes = e.target.checked
                          ? [...currentTypes, plotType]
                          : currentTypes.filter((t: string) => t !== plotType);
                        setConfig({ ...config, plot_types: newTypes });
                      }}
                      className="mr-2"
                    />
                    <span className="capitalize">{plotType}</span>
                  </label>
                ))}
              </div>
            </div>
          </div>
        );

      case 'export-model':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Export Path</label>
              <input
                type="text"
                value={config.export_path || ''}
                onChange={(e) => setConfig({ ...config, export_path: e.target.value })}
                placeholder="./models/my_model.pkl"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        );

      case 'generate-report':
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">LLM Provider</label>
              <select
                value={config.llm_provider || 'openai'}
                onChange={(e) => setConfig({ ...config, llm_provider: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="google">Google</option>
              </select>
              <p className="text-xs text-gray-500 mt-1">
                Requires API key to be configured in backend
              </p>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-gray-500 text-sm">
            This node type does not require configuration.
          </div>
        );
    }
  };

  return (
    <div className="fixed right-0 top-0 bottom-0 w-96 bg-white border-l border-gray-300 shadow-lg z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold">Configure Node</h2>
        <button
          onClick={onClose}
          className="p-1 hover:bg-gray-100 rounded transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mb-4">
          <h3 className="font-medium text-gray-900">{node.data.label}</h3>
          <p className="text-sm text-gray-500">Type: {node.data.type}</p>
        </div>

        {renderConfigForm()}
      </div>

      {/* Footer */}
      <div className="border-t border-gray-200 p-4 flex gap-2">
        <button
          onClick={handleSave}
          className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Save Configuration
        </button>
        <button
          onClick={onClose}
          className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300 transition-colors"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
