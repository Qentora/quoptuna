'use client';

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
import { Loader2, Search, Upload } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { ErrorBanner } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

export function DatasetStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
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

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
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

  useEffect(() => {
    setFooter({ canContinue: !!selected && !isLoading });
  }, [selected, isLoading, setFooter]);

  return (
    <div className="flex h-full min-h-0 flex-col gap-6">
      <div className="shrink-0 space-y-6">
        <StepHeader
          step={1}
          title="Dataset Selection"
          subtitle="Upload your own dataset or select from the UCI ML Repository"
        />

        <ErrorBanner message={error} />
      </div>

      {/* Dataset sources: compact upload bar + full-width UCI repository.
          The whole area is a drop target for CSV files. */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        className={cn(
          'flex shrink-0 flex-col gap-3 rounded-lg transition-shadow',
          dragOver && 'ring-2 ring-brand/50'
        )}
      >
        {/* Top bar: compact upload button + hint */}
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant="secondary"
            disabled={isLoading}
            onClick={() => fileInputRef.current?.click()}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Upload className="h-4 w-4" />
            )}
            Upload CSV
          </Button>
          <p className="text-sm text-muted-foreground">
            {dragOver
              ? 'Drop the CSV to upload'
              : 'Upload your own CSV or pick from the UCI repository below — drag & drop anywhere here.'}
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>

        {/* Full-width UCI repository with a 2–3 column list */}
        <div className="flex h-64 flex-col rounded-lg border border-border bg-card">
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
          <div className="grid min-h-0 flex-1 grid-cols-1 gap-2 overflow-y-auto p-3 sm:grid-cols-2 xl:grid-cols-3">
            {uciDatasets.isLoading && (
              <div className="col-span-full flex items-center gap-2 px-1 py-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading datasets…
              </div>
            )}
            {!uciDatasets.isLoading && filteredDatasets.length === 0 && (
              <p className="col-span-full px-1 py-2 text-sm text-muted-foreground">
                No matching datasets.
              </p>
            )}
            {filteredDatasets.map((dataset) => (
              <button
                key={dataset.id}
                type="button"
                onClick={() => handleUCISelect(dataset.id)}
                disabled={isLoading}
                className="rounded-lg border border-border p-3 text-left transition-colors hover:border-foreground hover:bg-accent disabled:opacity-50"
              >
                <div className="flex items-center justify-between gap-2">
                  <h4 className="truncate font-semibold">{dataset.name}</h4>
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
      </div>

      {/* Preview region fills the remaining frame height */}
      <div className="flex min-h-0 flex-1 flex-col">
        {selected ? (
          <TableContainer className="flex min-h-0 flex-1 flex-col">
            <div className="flex shrink-0 items-center justify-between border-b border-border bg-muted px-4 py-2">
              <h4 className="text-sm font-semibold text-foreground">
                Data Preview · {selected.name}
              </h4>
              {preview.isFetching && (
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              )}
            </div>
            {preview.isError && (
              <p className="px-4 py-3 text-sm text-destructive">Could not load preview.</p>
            )}
            {preview.data && (
              <div className="min-h-0 flex-1 overflow-auto">
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
        ) : (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
            Select or upload a dataset to preview its rows.
          </div>
        )}
      </div>
    </div>
  );
}
