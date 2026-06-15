'use client';

import {
  useDatasetPreview,
  useLoadUCIDataset,
  useUCIDatasets,
  useUploadDataset,
} from '@/lib/hooks';
import * as Dialog from '@radix-ui/react-dialog';
import { Database, Loader2, Upload, X } from 'lucide-react';
import { useRef, useState } from 'react';
import { ErrorBanner, NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

export function DatasetStep({ onNext, workflowData, setWorkflowData }: StepProps) {
  const [showUCIModal, setShowUCIModal] = useState(false);
  const [customId, setCustomId] = useState('');
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uciDatasets = useUCIDatasets();
  const upload = useUploadDataset();
  const loadUCI = useLoadUCIDataset();
  const preview = useDatasetPreview(workflowData.dataset?.id ?? null);

  const isLoading = upload.isPending || loadUCI.isPending;

  const applyDataset = (d: {
    id: string;
    name: string;
    source: 'upload' | 'uci';
    rows: number;
    columns: string[];
  }) => {
    setWorkflowData((prev) => ({
      ...prev,
      dataset: d,
      features: {
        selectedFeatures: [],
        targetColumn: null,
        labelMapping: { neg: null, pos: null },
      },
    }));
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setError(null);
    try {
      const result = await upload.mutateAsync(file);
      applyDataset({
        id: result.id,
        name: result.filename,
        source: 'upload',
        rows: result.rows,
        columns: result.columns,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload dataset');
    }
  };

  const handleUCISelect = async (datasetId: number) => {
    setError(null);
    try {
      const result = await loadUCI.mutateAsync(datasetId);
      applyDataset({
        id: result.id,
        name: result.name,
        source: 'uci',
        rows: result.rows,
        columns: result.columns,
      });
      setShowUCIModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch UCI dataset');
    }
  };

  return (
    <div className="space-y-6">
      <StepHeader
        title="Dataset Selection"
        subtitle="Upload your own dataset or select from the UCI ML Repository"
      />

      <ErrorBanner message={error} />

      {workflowData.dataset && (
        <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-md">
          <p className="font-medium">Selected: {workflowData.dataset.name}</p>
          <p className="text-sm mt-1">
            Source: {workflowData.dataset.source === 'upload' ? 'Uploaded File' : 'UCI Repository'}{' '}
            | Rows: {workflowData.dataset.rows} | Columns: {workflowData.dataset.columns.length}
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors"
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Dataset</h3>
          <p className="text-sm text-gray-500">Click to browse for a CSV file</p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
          />
        </button>

        <button
          type="button"
          onClick={() => setShowUCIModal(true)}
          className="border-2 border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 transition-colors"
        >
          <Database className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">UCI Repository</h3>
          <p className="text-sm text-gray-500">Select from popular datasets</p>
        </button>
      </div>

      {/* Preview table */}
      {workflowData.dataset && (
        <div className="border border-gray-200 rounded-lg">
          <div className="px-4 py-2 border-b border-gray-200 bg-gray-50 flex items-center justify-between">
            <h4 className="text-sm font-semibold text-gray-700">Data Preview</h4>
            {preview.isFetching && <Loader2 className="w-4 h-4 animate-spin text-gray-400" />}
          </div>
          {preview.isError && (
            <p className="px-4 py-3 text-sm text-red-600">Could not load preview.</p>
          )}
          {preview.data && (
            <div className="overflow-x-auto max-h-72">
              <table className="w-full text-sm">
                <thead className="bg-gray-100 sticky top-0">
                  <tr>
                    {preview.data.columns.map((col) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-gray-700">
                        {col}
                        <span className="block text-[10px] font-normal text-gray-400">
                          {preview.data?.dtypes[col]}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {preview.data.head.map((row, i) => (
                    <tr key={i}>
                      {preview.data?.columns.map((col) => (
                        <td key={col} className="px-3 py-1.5 text-gray-600 whitespace-nowrap">
                          {String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <NavButtons onNext={onNext} nextDisabled={!workflowData.dataset || isLoading} />

      <Dialog.Root open={showUCIModal} onOpenChange={setShowUCIModal}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 bg-black/50 z-40" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 w-full max-w-2xl -translate-x-1/2 -translate-y-1/2 rounded-lg bg-white shadow-xl flex flex-col max-h-[80vh]">
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <Dialog.Title className="text-xl font-bold text-gray-900">
                Select UCI Dataset
              </Dialog.Title>
              <Dialog.Close className="text-gray-400 hover:text-gray-600">
                <X className="w-6 h-6" />
              </Dialog.Close>
            </div>
            <div className="flex-1 overflow-y-auto p-6 space-y-3">
              {uciDatasets.isLoading && (
                <div className="flex items-center gap-2 text-gray-500">
                  <Loader2 className="w-4 h-4 animate-spin" /> Loading datasets...
                </div>
              )}
              {uciDatasets.data?.map((dataset) => (
                <button
                  key={dataset.id}
                  type="button"
                  onClick={() => handleUCISelect(dataset.id)}
                  disabled={isLoading}
                  className="w-full text-left p-4 border border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors disabled:opacity-50"
                >
                  <h4 className="font-semibold text-gray-900">{dataset.name}</h4>
                  {dataset.description && (
                    <p className="text-sm text-gray-500 mt-1">{dataset.description}</p>
                  )}
                  <p className="text-xs text-gray-400 mt-1">Dataset ID: {dataset.id}</p>
                </button>
              ))}
            </div>
            <div className="p-6 border-t border-gray-200 bg-gray-50 flex items-center gap-2">
              <input
                type="number"
                value={customId}
                onChange={(e) => setCustomId(e.target.value)}
                placeholder="Enter a UCI dataset ID"
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
              />
              <button
                type="button"
                disabled={!customId || isLoading}
                onClick={() => handleUCISelect(Number(customId))}
                className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm hover:bg-blue-700 disabled:bg-gray-300"
              >
                {isLoading ? 'Loading...' : 'Load by ID'}
              </button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
