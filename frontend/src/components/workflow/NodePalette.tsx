import { Database, Settings, Brain, BarChart3, FileOutput, Gauge } from 'lucide-react';
import type { NodeType } from '../../types/workflow';

interface NodeCategory {
  title: string;
  icon: React.ReactNode;
  nodes: {
    type: NodeType;
    label: string;
    description: string;
  }[];
}

const nodeCategories: NodeCategory[] = [
  {
    title: 'Data',
    icon: <Database className="w-4 h-4" />,
    nodes: [
      { type: 'data-upload', label: 'Upload CSV', description: 'Upload dataset from file' },
      { type: 'data-uci', label: 'UCI Dataset', description: 'Fetch from UCI repository' },
      { type: 'data-preview', label: 'Data Preview', description: 'View dataset statistics' },
      { type: 'feature-selection', label: 'Select Features', description: 'Choose X and y columns' },
    ],
  },
  {
    title: 'Preprocessing',
    icon: <Settings className="w-4 h-4" />,
    nodes: [
      { type: 'train-test-split', label: 'Train/Test Split', description: 'Split data into train and test sets' },
      { type: 'scaler', label: 'Standard Scaler', description: 'Normalize features' },
      { type: 'label-encoding', label: 'Label Encoding', description: 'Encode target labels' },
    ],
  },
  {
    title: 'Models',
    icon: <Brain className="w-4 h-4" />,
    nodes: [
      { type: 'quantum-model', label: 'Quantum Model', description: '18 quantum ML models' },
      { type: 'classical-model', label: 'Classical Model', description: '8 classical ML models' },
    ],
  },
  {
    title: 'Optimization',
    icon: <Gauge className="w-4 h-4" />,
    nodes: [
      { type: 'optuna-config', label: 'Optuna Config', description: 'Configure optimization study' },
      { type: 'optimization', label: 'Run Optimization', description: 'Execute hyperparameter tuning' },
    ],
  },
  {
    title: 'Analysis',
    icon: <BarChart3 className="w-4 h-4" />,
    nodes: [
      { type: 'shap-analysis', label: 'SHAP Analysis', description: 'Explainability analysis' },
      { type: 'confusion-matrix', label: 'Confusion Matrix', description: 'Classification performance' },
      { type: 'feature-importance', label: 'Feature Importance', description: 'Feature ranking' },
    ],
  },
  {
    title: 'Output',
    icon: <FileOutput className="w-4 h-4" />,
    nodes: [
      { type: 'export-model', label: 'Export Model', description: 'Save trained model' },
      { type: 'generate-report', label: 'Generate Report', description: 'AI-powered report' },
    ],
  },
];

interface NodePaletteProps {
  onNodeSelect: (type: NodeType, label: string) => void;
}

export function NodePalette({ onNodeSelect }: NodePaletteProps) {
  const onDragStart = (event: React.DragEvent, nodeType: NodeType, label: string) => {
    event.dataTransfer.setData('application/reactflow', JSON.stringify({ type: nodeType, label }));
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="w-64 bg-card border-r border-border p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">Node Palette</h2>

      {nodeCategories.map((category) => (
        <div key={category.title} className="mb-6">
          <div className="flex items-center gap-2 mb-2 text-sm font-semibold text-muted-foreground">
            {category.icon}
            <span>{category.title}</span>
          </div>

          <div className="space-y-2">
            {category.nodes.map((node) => (
              <div
                key={node.type}
                draggable
                onDragStart={(e) => onDragStart(e, node.type, node.label)}
                onClick={() => onNodeSelect(node.type, node.label)}
                className="bg-secondary hover:bg-accent p-3 rounded-md cursor-move transition-colors group"
              >
                <div className="font-medium text-sm">{node.label}</div>
                <div className="text-xs text-muted-foreground mt-1">{node.description}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
