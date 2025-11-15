import { useCallback, useRef, useState } from 'react';
import ReactFlow, { Background, Controls, MiniMap, ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import { NodePalette } from '../components/workflow/NodePalette';
import { CustomNode } from '../components/workflow/CustomNode';
import { NodeConfigPanel } from '../components/workflow/NodeConfigPanel';
import { useWorkflowStore } from '../stores/workflow';
import type { NodeType, WorkflowNode } from '../types/workflow';
import { Play, Save, Trash2, Loader2, CheckCircle2, XCircle } from 'lucide-react';
import { executeWorkflow, pollExecutionStatus } from '../lib/api';

const nodeTypes = {
  custom: CustomNode,
};

function WorkflowBuilderContent() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStatus, setExecutionStatus] = useState<string | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);

  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    clearWorkflow,
    saveWorkflow,
    updateNode,
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

  const handleNodeClick = useCallback((event: React.MouseEvent, node: any) => {
    setSelectedNode(node as WorkflowNode);
  }, []);

  const handleConfigSave = (nodeId: string, config: any) => {
    updateNode(nodeId, { config });
  };

  const handleRun = async () => {
    if (nodes.length === 0) {
      alert('Please add at least one node to the workflow');
      return;
    }

    try {
      setIsExecuting(true);
      setExecutionStatus('Starting workflow execution...');
      setExecutionError(null);

      // Set all nodes to idle status
      nodes.forEach((node) => {
        updateNode(node.id, { status: 'idle' });
      });

      // Execute workflow
      const response = await executeWorkflow({
        name: 'Temporary Workflow',
        nodes: nodes,
        edges: edges,
      });

      setExecutionStatus(`Execution started (ID: ${response.execution_id})`);

      // Poll for completion
      const result = await pollExecutionStatus(
        response.execution_id,
        (status) => {
          setExecutionStatus(
            `Status: ${status.status} - ${
              status.status === 'running' ? 'Processing...' : status.message || ''
            }`
          );

          // Update node statuses based on execution
          if (status.status === 'running') {
            nodes.forEach((node) => {
              updateNode(node.id, { status: 'running' });
            });
          }
        }
      );

      if (result.status === 'completed') {
        // Mark all nodes as complete
        nodes.forEach((node) => {
          updateNode(node.id, { status: 'complete' });
        });

        setExecutionStatus('Workflow completed successfully!');
        alert(
          `Workflow completed!\n\nExecution ID: ${result.id}\nResults: ${
            result.result?.node_results
              ? Object.keys(result.result.node_results).length + ' nodes executed'
              : 'See console for details'
          }`
        );
        console.log('Execution result:', result);
      } else {
        // Mark nodes as error
        nodes.forEach((node) => {
          updateNode(node.id, { status: 'error' });
        });

        setExecutionError(result.error || 'Workflow execution failed');
        alert(`Workflow failed: ${result.error}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setExecutionError(errorMessage);
      alert(`Failed to execute workflow: ${errorMessage}`);

      // Mark all nodes as error
      nodes.forEach((node) => {
        updateNode(node.id, { status: 'error' });
      });
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      <NodePalette onNodeSelect={handleNodeSelect} />

      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-card border-b border-border p-4">
          <div className="flex items-center justify-between mb-2">
            <h1 className="text-xl font-bold">Workflow Builder</h1>

            <div className="flex gap-2">
              <button
                onClick={handleRun}
                disabled={isExecuting}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isExecuting ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                {isExecuting ? 'Running...' : 'Run'}
              </button>
              <button
                onClick={handleSave}
                disabled={isExecuting}
                className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors disabled:opacity-50"
              >
                <Save className="w-4 h-4" />
                Save
              </button>
              <button
                onClick={clearWorkflow}
                disabled={isExecuting}
                className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors disabled:opacity-50"
              >
                <Trash2 className="w-4 h-4" />
                Clear
              </button>
            </div>
          </div>

          {/* Status display */}
          {(executionStatus || executionError) && (
            <div className="flex items-center gap-2 text-sm">
              {executionError ? (
                <>
                  <XCircle className="w-4 h-4 text-red-500" />
                  <span className="text-red-600">{executionError}</span>
                </>
              ) : executionStatus?.includes('completed') ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-green-500" />
                  <span className="text-green-600">{executionStatus}</span>
                </>
              ) : (
                <>
                  <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                  <span className="text-blue-600">{executionStatus}</span>
                </>
              )}
            </div>
          )}
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
            onNodeClick={handleNodeClick}
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

      {/* Configuration Panel */}
      <NodeConfigPanel
        node={selectedNode}
        onClose={() => setSelectedNode(null)}
        onSave={handleConfigSave}
      />
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
