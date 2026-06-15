import { Brain, Cpu } from 'lucide-react';

const quantumModels = [
  'Data Reuploading',
  'Circuit Centric',
  'Dressed Quantum Circuit',
  'Quantum Kitchen Sinks',
  'IQP Variational',
  'IQP Kernel',
  'Projected Quantum Kernel',
  'Quantum Metric Learning',
  'Vanilla QNN',
  'Quantum Boltzmann Machine',
  'Tree Tensor Network',
  'WeiNet',
  'Quanvolutional Neural Network',
  'Separable',
  'Convolutional Neural Network',
];

const classicalModels = [
  'Support Vector Classifier',
  'Multi-layer Perceptron',
  'Perceptron',
  'Random Forest',
  'Gradient Boosting',
  'AdaBoost',
  'Logistic Regression',
  'Decision Tree',
];

export function Models() {
  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-8">Model Library</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div>
          <div className="flex items-center gap-3 mb-4">
            <Brain className="w-6 h-6 text-purple-500" />
            <h2 className="text-2xl font-semibold">Quantum Models</h2>
            <span className="bg-purple-100 text-purple-800 text-xs font-semibold px-2.5 py-0.5 rounded">
              {quantumModels.length}
            </span>
          </div>
          <div className="bg-card rounded-lg border border-border divide-y divide-border">
            {quantumModels.map((model) => (
              <div key={model} className="p-4 hover:bg-accent transition-colors">
                <p className="font-medium">{model}</p>
                <p className="text-sm text-muted-foreground">PennyLane-based quantum circuit</p>
              </div>
            ))}
          </div>
        </div>

        <div>
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="w-6 h-6 text-blue-500" />
            <h2 className="text-2xl font-semibold">Classical Models</h2>
            <span className="bg-blue-100 text-blue-800 text-xs font-semibold px-2.5 py-0.5 rounded">
              {classicalModels.length}
            </span>
          </div>
          <div className="bg-card rounded-lg border border-border divide-y divide-border">
            {classicalModels.map((model) => (
              <div key={model} className="p-4 hover:bg-accent transition-colors">
                <p className="font-medium">{model}</p>
                <p className="text-sm text-muted-foreground">Scikit-learn implementation</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
