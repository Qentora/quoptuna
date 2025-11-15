import { memo } from 'react';
import { Handle, Position } from 'reactflow';
import type { NodeProps } from 'reactflow';
import { Database, Upload, Settings, Brain, BarChart3, FileOutput, Gauge, CheckCircle2, Loader2, XCircle, Circle, Play } from 'lucide-react';
import type { NodeData } from '../../types/workflow';

const getNodeIcon = (type: string) => {
  if (type.startsWith('data-')) return Database;
  if (type.startsWith('quantum-') || type.startsWith('classical-')) return Brain;
  if (type.includes('scaler') || type.includes('split') || type.includes('encoding')) return Settings;
  if (type.includes('optuna') || type.includes('optimization')) return Gauge;
  if (type.includes('shap') || type.includes('confusion') || type.includes('importance')) return BarChart3;
  if (type.includes('export') || type.includes('report')) return FileOutput;
  return Upload;
};

const getStatusIcon = (status?: string) => {
  switch (status) {
    case 'running':
      return <Loader2 className="w-4 h-4 animate-spin text-blue-500" />;
    case 'complete':
      return <CheckCircle2 className="w-4 h-4 text-green-500" />;
    case 'error':
      return <XCircle className="w-4 h-4 text-red-500" />;
    default:
      return <Circle className="w-4 h-4 text-gray-400" />;
  }
};

export const CustomNode = memo(({ data, selected, id }: NodeProps<NodeData>) => {
  const Icon = getNodeIcon(data.type);

  const handleRunFromNode = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (data.onRunFromNode) {
      data.onRunFromNode(id);
    }
  };

  return (
    <div
      className={`px-4 py-3 shadow-md rounded-lg border-2 bg-white min-w-[200px] ${
        selected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-300'
      }`}
    >
      <Handle type="target" position={Position.Top} className="w-3 h-3 !bg-blue-500" />

      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 flex-1">
          <Icon className="w-5 h-5 text-gray-600" />
          <div className="font-medium text-sm">{data.label}</div>
        </div>
        <div className="flex items-center gap-1">
          {data.status !== 'running' && (
            <button
              onClick={handleRunFromNode}
              className="p-1 hover:bg-blue-50 rounded transition-colors"
              title="Run workflow from this node"
            >
              <Play className="w-4 h-4 text-blue-600" />
            </button>
          )}
          {getStatusIcon(data.status)}
        </div>
      </div>

      {data.config && Object.keys(data.config).length > 0 && (
        <div className="mt-2 pt-2 border-t border-gray-200">
          <div className="text-xs text-gray-500 space-y-1">
            {Object.entries(data.config).slice(0, 2).map(([key, value]) => (
              <div key={key}>
                <span className="font-medium">{key}:</span> {String(value)}
              </div>
            ))}
          </div>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} className="w-3 h-3 !bg-green-500" />
    </div>
  );
});

CustomNode.displayName = 'CustomNode';
