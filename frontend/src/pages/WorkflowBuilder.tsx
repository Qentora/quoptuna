import { useCallback, useRef } from 'react';
import ReactFlow, { Background, Controls, MiniMap, ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import { NodePalette } from '../components/workflow/NodePalette';
import { CustomNode } from '../components/workflow/CustomNode';
import { useWorkflowStore } from '../stores/workflow';
import type { NodeType, WorkflowNode } from '../types/workflow';
import { Play, Save, Trash2 } from 'lucide-react';

const nodeTypes = {
  custom: CustomNode,
};

function WorkflowBuilderContent() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    clearWorkflow,
    saveWorkflow,
  } = useWorkflowStore();

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const data = JSON.parse(event.dataTransfer.getData('application/reactflow'));
      const { type, label } = data as { type: NodeType; label: string };

      const position = {
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      };

      const newNode: WorkflowNode = {
        id: `${type}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          label,
          type,
          status: 'idle',
        },
      };

      addNode(newNode);
    },
    [addNode]
  );

  const handleNodeSelect = (type: NodeType, label: string) => {
    const newNode: WorkflowNode = {
      id: `${type}-${Date.now()}`,
      type: 'custom',
      position: {
        x: Math.random() * 400 + 100,
        y: Math.random() * 400 + 100,
      },
      data: {
        label,
        type,
        status: 'idle',
      },
    };

    addNode(newNode);
  };

  const handleSave = () => {
    const name = prompt('Enter workflow name:');
    if (name) {
      saveWorkflow(name);
      alert('Workflow saved successfully!');
    }
  };

  const handleRun = () => {
    // TODO: Implement workflow execution
    alert('Workflow execution will be implemented in the backend integration phase!');
  };

  return (
    <div className="flex h-screen bg-background">
      <NodePalette onNodeSelect={handleNodeSelect} />

      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-card border-b border-border p-4 flex items-center justify-between">
          <h1 className="text-xl font-bold">Workflow Builder</h1>

          <div className="flex gap-2">
            <button
              onClick={handleRun}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              <Play className="w-4 h-4" />
              Run
            </button>
            <button
              onClick={handleSave}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              <Save className="w-4 h-4" />
              Save
            </button>
            <button
              onClick={clearWorkflow}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div ref={reactFlowWrapper} className="flex-1">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
            className="bg-gray-50"
          >
            <Background />
            <Controls />
            <MiniMap />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}

export function WorkflowBuilder() {
  return (
    <ReactFlowProvider>
      <WorkflowBuilderContent />
    </ReactFlowProvider>
  );
}
