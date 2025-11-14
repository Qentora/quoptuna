import type { Node, Edge } from 'reactflow';

export type NodeType =
  | 'data-upload'
  | 'data-uci'
  | 'data-preview'
  | 'feature-selection'
  | 'train-test-split'
  | 'scaler'
  | 'label-encoding'
  | 'quantum-model'
  | 'classical-model'
  | 'optuna-config'
  | 'optimization'
  | 'shap-analysis'
  | 'confusion-matrix'
  | 'feature-importance'
  | 'export-model'
  | 'generate-report';

export interface NodeData {
  label: string;
  type: NodeType;
  config?: Record<string, any>;
  status?: 'idle' | 'running' | 'complete' | 'error';
  result?: any;
}

export interface WorkflowNode extends Node {
  data: NodeData;
}

export interface Workflow {
  id: string;
  name: string;
  description?: string;
  nodes: WorkflowNode[];
  edges: Edge[];
  createdAt: string;
  updatedAt: string;
  status: 'draft' | 'running' | 'completed' | 'failed';
}

export interface WorkflowExecution {
  id: string;
  workflowId: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  startedAt: string;
  completedAt?: string;
  results: Record<string, any>;
  errors?: string[];
}
