'use client';

import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
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
        step={1}
        title="Dataset Selection"
        subtitle="Upload your own dataset or select from the UCI ML Repository"
      />

      <ErrorBanner message={error} />

      {workflowData.dataset && (
        <Alert variant="success">
          <p className="font-medium">Selected: {workflowData.dataset.name}</p>
          <p className="mt-1 text-sm">
            Source: {workflowData.dataset.source === 'upload' ? 'Uploaded File' : 'UCI Repository'}{' '}
            | Rows: {workflowData.dataset.rows} | Columns: {workflowData.dataset.columns.length}
          </p>
        </Alert>
      )}

      <div className="grid grid-cols-2 gap-6">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="rounded-lg border-2 border-dashed border-border p-8 text-center transition-colors hover:border-foreground"
        >
          <Upload className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />
          <h3 className="mb-2 text-base font-semibold">Upload Dataset</h3>
          <p className="text-sm text-muted-foreground">Click to browse for a CSV file</p>
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
          className="rounded-lg border-2 border-border p-8 text-center transition-colors hover:border-foreground"
        >
          <Database className="mx-auto mb-4 h-12 w-12 text-muted-foreground" />
          <h3 className="mb-2 text-base font-semibold">UCI Repository</h3>
          <p className="text-sm text-muted-foreground">Select from popular datasets</p>
        </button>
      </div>

      {workflowData.dataset && (
        <TableContainer>
          <div className="flex items-center justify-between border-b border-border bg-muted px-4 py-2">
            <h4 className="text-sm font-semibold text-foreground">Data Preview</h4>
            {preview.isFetching && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
            )}
          </div>
          {preview.isError && (
            <p className="px-4 py-3 text-sm text-destructive">Could not load preview.</p>
          )}
          {preview.data && (
            <div className="max-h-72 overflow-x-auto">
              <Table>
                <TableHead className="sticky top-0">
                  <tr>
                    {preview.data.columns.map((col) => (
                      <Th key={col}>
                        {col}
                        <span className="block text-[10px] font-normal text-muted-foreground">
                          {preview.data?.dtypes[col]}
                        </span>
                      </Th>
                    ))}
                  </tr>
                </TableHead>
                <TableBody>
                  {preview.data.head.map((row, i) => (
                    <tr key={i}>
                      {preview.data?.columns.map((col) => (
                        <Td key={col} className="whitespace-nowrap text-muted-foreground">
                          {String(row[col])}
                        </Td>
                      ))}
                    </tr>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </TableContainer>
      )}

      <NavButtons onNext={onNext} nextDisabled={!workflowData.dataset || isLoading} />

      <Dialog.Root open={showUCIModal} onOpenChange={setShowUCIModal}>
        <Dialog.Portal>
          <Dialog.Overlay className="fixed inset-0 z-40 bg-black/50 data-[state=open]:animate-overlayShow" />
          <Dialog.Content className="fixed left-1/2 top-1/2 z-50 flex max-h-[80vh] w-full max-w-2xl -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border border-border bg-popover text-popover-foreground shadow-lg data-[state=open]:animate-contentShow">
            <div className="flex items-center justify-between border-b border-border p-6">
              <Dialog.Title className="text-lg font-bold">Select UCI Dataset</Dialog.Title>
              <Dialog.Close className="text-muted-foreground transition-colors hover:text-foreground">
                <X className="h-6 w-6" />
              </Dialog.Close>
            </div>
            <div className="flex-1 space-y-3 overflow-y-auto p-6">
              {uciDatasets.isLoading && (
                <div className="flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" /> Loading datasets...
                </div>
              )}
              {uciDatasets.data?.map((dataset) => (
                <button
                  key={dataset.id}
                  type="button"
                  onClick={() => handleUCISelect(dataset.id)}
                  disabled={isLoading}
                  className="w-full rounded-lg border border-border p-4 text-left transition-colors hover:border-foreground hover:bg-accent disabled:opacity-50"
                >
                  <h4 className="font-semibold">{dataset.name}</h4>
                  {dataset.description && (
                    <p className="mt-1 text-sm text-muted-foreground">{dataset.description}</p>
                  )}
                  <p className="mt-1 text-xs text-muted-foreground">Dataset ID: {dataset.id}</p>
                </button>
              ))}
            </div>
            <div className="flex items-center gap-2 border-t border-border bg-muted p-6">
              <Input
                type="number"
                value={customId}
                onChange={(e) => setCustomId(e.target.value)}
                placeholder="Enter a UCI dataset ID"
              />
              <Button
                type="button"
                disabled={!customId || isLoading}
                onClick={() => handleUCISelect(Number(customId))}
              >
                {isLoading ? 'Loading...' : 'Load by ID'}
              </Button>
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
    </div>
  );
}
