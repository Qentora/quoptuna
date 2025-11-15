import { useCallback, useRef, useState } from 'react';
import ReactFlow, { Background, Controls, MiniMap, ReactFlowProvider } from 'reactflow';
import 'reactflow/dist/style.css';
import toast from 'react-hot-toast';
import { NodePalette } from '../components/workflow/NodePalette';
import { CustomNode } from '../components/workflow/CustomNode';
import { NodeConfigPanel } from '../components/workflow/NodeConfigPanel';
import { ErrorModal } from '../components/ui/ErrorModal';
import { NodeResultModal } from '../components/ui/NodeResultModal';
import { useWorkflowStore } from '../stores/workflow';
import type { NodeType, WorkflowNode } from '../types/workflow';
import { Play, Save, Trash2, Loader2, CheckCircle2, XCircle, StopCircle } from 'lucide-react';
import { executeWorkflow, pollExecutionStatus } from '../lib/api';

const nodeTypes = {
  custom: CustomNode,
};

function WorkflowBuilderContent() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [currentExecutionId, setCurrentExecutionId] = useState<string | null>(null);
  const [executionStatus, setExecutionStatus] = useState<string | null>(null);
  const [executionError, setExecutionError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<WorkflowNode | null>(null);
  const [stopRequested, setStopRequested] = useState(false);

  // Modal states
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorModalTitle, setErrorModalTitle] = useState('');
  const [errorModalMessage, setErrorModalMessage] = useState('');
  const [errorModalDetails, setErrorModalDetails] = useState<any>(null);

  // Node result modal state
  const [showResultModal, setShowResultModal] = useState(false);
  const [resultModalNodeId, setResultModalNodeId] = useState('');
  const [resultModalNodeLabel, setResultModalNodeLabel] = useState('');
  const [resultModalData, setResultModalData] = useState<any>(null);

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
      toast.success(`Workflow "${name}" has been saved successfully!`);
    }
  };

  const handleNodeClick = useCallback((event: React.MouseEvent, node: any) => {
    setSelectedNode(node as WorkflowNode);
  }, []);

  const handleConfigSave = (nodeId: string, config: any) => {
    updateNode(nodeId, { config });
  };

  const handleStop = () => {
    setStopRequested(true);
    setExecutionStatus('Stopping execution...');
    // Reset after a short delay
    setTimeout(() => {
      setIsExecuting(false);
      setCurrentExecutionId(null);
      setStopRequested(false);
      setExecutionStatus('Execution stopped by user');
      nodes.forEach((node) => {
        updateNode(node.id, { status: 'idle' });
      });
    }, 500);
  };

  const handleRunFromNode = useCallback((nodeId: string) => {
    // For now, just run the entire workflow
    // TODO: Implement running from specific node
    console.log(`Running workflow from node: ${nodeId}`);
    handleRun();
  }, [nodes, edges]);

  const handleViewResult = useCallback((nodeId: string, result: any) => {
    const node = nodes.find((n) => n.id === nodeId);
    if (node) {
      setResultModalNodeId(nodeId);
      setResultModalNodeLabel(node.data.label);
      setResultModalData(result);
      setShowResultModal(true);
    }
  }, [nodes]);

  const handleRun = async () => {
    if (nodes.length === 0) {
      setErrorModalTitle('No Nodes');
      setErrorModalMessage('Please add at least one node to the workflow before running.');
      setErrorModalDetails(null);
      setShowErrorModal(true);
      return;
    }

    try {
      setIsExecuting(true);
      setStopRequested(false);
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

      setCurrentExecutionId(response.execution_id);
      setExecutionStatus(`Execution started (ID: ${response.execution_id})`);

      // Poll for completion
      const result = await pollExecutionStatus(
        response.execution_id,
        (status) => {
          if (stopRequested) {
            return; // Stop polling if stop was requested
          }

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
          updateNode(node.id, { status: 'complete', result: result.result?.node_results?.[node.id] });
        });

        setExecutionStatus('Workflow completed successfully!');

        // Show success toast
        const nodesExecuted = result.result?.node_results
          ? Object.keys(result.result.node_results).length
          : 0;
        toast.success(`Workflow executed successfully! ${nodesExecuted} nodes completed.`, {
          duration: 4000,
        });

        console.log('Execution result:', result);
      } else {
        // Mark nodes as error
        nodes.forEach((node) => {
          updateNode(node.id, { status: 'error' });
        });

        setExecutionError(result.error || 'Workflow execution failed');

        // Show error modal
        setErrorModalTitle('Workflow Failed');
        setErrorModalMessage('The workflow execution encountered an error.');
        setErrorModalDetails(result.error);
        setShowErrorModal(true);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setExecutionError(errorMessage);

      // Show error modal
      setErrorModalTitle('Execution Error');
      setErrorModalMessage('Failed to execute workflow.');
      setErrorModalDetails(errorMessage);
      setShowErrorModal(true);

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
              {isExecuting && (
                <button
                  onClick={handleStop}
                  className="flex items-center gap-2 px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700 transition-colors"
                >
                  <StopCircle className="w-4 h-4" />
                  Stop
                </button>
              )}
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
            nodes={nodes.map(node => ({
              ...node,
              data: {
                ...node.data,
                onRunFromNode: handleRunFromNode,
                onViewResult: handleViewResult,
              },
            }))}
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

      {/* Error Modal */}
      <ErrorModal
        isOpen={showErrorModal}
        onClose={() => setShowErrorModal(false)}
        title={errorModalTitle}
        message={errorModalMessage}
        details={errorModalDetails}
      />

      {/* Node Result Modal */}
      <NodeResultModal
        isOpen={showResultModal}
        onClose={() => setShowResultModal(false)}
        nodeId={resultModalNodeId}
        nodeLabel={resultModalNodeLabel}
        result={resultModalData}
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
