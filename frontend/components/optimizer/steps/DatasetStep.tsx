'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
import {
  useDatasetPreview,
  useLoadUCIDataset,
  useUCIDatasets,
  useUploadDataset,
} from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Database, Loader2, Search, Upload } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { ErrorBanner, NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

export function DatasetStep({ onNext, workflowData, setWorkflowData }: StepProps) {
  const [customId, setCustomId] = useState('');
  const [search, setSearch] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const uciDatasets = useUCIDatasets();
  const upload = useUploadDataset();
  const loadUCI = useLoadUCIDataset();
  const preview = useDatasetPreview(workflowData.dataset?.id ?? null);

  const isLoading = upload.isPending || loadUCI.isPending;

  const filteredDatasets = useMemo(() => {
    const list = uciDatasets.data ?? [];
    const q = search.trim().toLowerCase();
    if (!q) return list;
    return list.filter(
      (d) => d.name.toLowerCase().includes(q) || (d.description ?? '').toLowerCase().includes(q)
    );
  }, [uciDatasets.data, search]);

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

  const uploadFile = async (file: File) => {
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

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void uploadFile(file);
  };

  const handleDrop = (event: React.DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void uploadFile(file);
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
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch UCI dataset');
    }
  };

  const selected = workflowData.dataset;

  return (
    <div className="space-y-6">
      <StepHeader
        step={1}
        title="Dataset Selection"
        subtitle="Upload your own dataset or select from the UCI ML Repository"
      />

      <ErrorBanner message={error} />

      {selected && (
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <Database className="h-4 w-4 text-accent-emerald-foreground" />
            <p className="font-semibold">{selected.name}</p>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Badge variant="emerald">
              {selected.source === 'upload' ? 'Uploaded file' : 'UCI Repository'}
            </Badge>
            <Badge variant="secondary">{selected.rows} rows</Badge>
            <Badge variant="secondary">{selected.columns.length} columns</Badge>
          </div>
        </div>
      )}

      {/* Drag-and-drop upload zone */}
      <button
        type="button"
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={cn(
          'block w-full rounded-lg border-2 border-dashed p-8 text-center transition-colors',
          dragOver ? 'border-brand bg-brand/5' : 'border-border hover:border-foreground'
        )}
      >
        {isLoading ? (
          <Loader2 className="mx-auto mb-3 h-10 w-10 animate-spin text-muted-foreground" />
        ) : (
          <Upload className="mx-auto mb-3 h-10 w-10 text-muted-foreground" />
        )}
        <h3 className="text-base font-semibold">Drag &amp; drop a CSV file</h3>
        <p className="mt-1 text-sm text-muted-foreground">or click to browse</p>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={handleFileUpload}
          className="hidden"
        />
      </button>

      {/* Inline searchable UCI list */}
      <div className="rounded-lg border border-border">
        <div className="flex items-center justify-between gap-3 border-b border-border bg-muted px-4 py-3">
          <h4 className="text-sm font-semibold">UCI ML Repository</h4>
          <div className="relative w-56">
            <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search datasets"
              className="pl-8"
            />
          </div>
        </div>
        <div className="max-h-64 space-y-2 overflow-y-auto p-3">
          {uciDatasets.isLoading && (
            <div className="flex items-center gap-2 px-1 py-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading datasets…
            </div>
          )}
          {!uciDatasets.isLoading && filteredDatasets.length === 0 && (
            <p className="px-1 py-2 text-sm text-muted-foreground">No matching datasets.</p>
          )}
          {filteredDatasets.map((dataset) => (
            <button
              key={dataset.id}
              type="button"
              onClick={() => handleUCISelect(dataset.id)}
              disabled={isLoading}
              className="w-full rounded-lg border border-border p-3 text-left transition-colors hover:border-foreground hover:bg-accent disabled:opacity-50"
            >
              <div className="flex items-center justify-between gap-2">
                <h4 className="font-semibold">{dataset.name}</h4>
                <span className="shrink-0 text-xs text-muted-foreground">ID: {dataset.id}</span>
              </div>
              {dataset.description && (
                <p className="mt-1 line-clamp-2 text-sm text-muted-foreground">
                  {dataset.description}
                </p>
              )}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 border-t border-border bg-muted p-3">
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
            {isLoading ? 'Loading…' : 'Load by ID'}
          </Button>
        </div>
      </div>

      {selected && (
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
            <div className="max-h-72 overflow-y-auto overflow-x-auto">
              <Table stickyHeader>
                <TableHead>
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
                    <tr key={i} className="even:bg-muted/40">
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

      <NavButtons onNext={onNext} nextDisabled={!selected || isLoading} />
    </div>
  );
}
