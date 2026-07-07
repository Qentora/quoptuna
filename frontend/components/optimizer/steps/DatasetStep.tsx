'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
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

const PREVIEW_ROWS = 15;

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
        sensitiveFeature: null,
        categoricalEncoding: 'ordinal',
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
    <div className="space-y-4">
      <StepHeader
        step={1}
        title="Dataset Selection"
        subtitle="Upload your own dataset or select from the UCI ML Repository"
      />

      <ErrorBanner message={error} />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(280px,1fr)_2fr]">
        {/* Left panel: dataset selection (upload + UCI list). Whole card is a drop target. */}
        <Card
          size="sm"
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={cn('transition-shadow', dragOver && 'ring-2 ring-brand/50')}
        >
          <CardHeader className="border-b pb-3">
            <CardTitle>Choose a dataset</CardTitle>
            <CardAction>
              <Button
                type="button"
                variant="secondary"
                size="sm"
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
            </CardAction>
            <p className="text-xs text-muted-foreground">
              {dragOver ? 'Drop the CSV to upload' : 'Drag & drop a CSV anywhere on this panel.'}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <h4 className="text-xs font-semibold">UCI ML Repository</h4>
              <div className="relative w-44">
                <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search datasets"
                  className="h-8 pl-8 text-xs"
                />
              </div>
            </div>

            {uciDatasets.isLoading && (
              <div className="space-y-2">
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
                <Skeleton className="h-12 w-full" />
              </div>
            )}
            {!uciDatasets.isLoading && filteredDatasets.length === 0 && (
              <p className="py-2 text-xs text-muted-foreground">No matching datasets.</p>
            )}

            <div className="space-y-1.5">
              {filteredDatasets.map((dataset) => {
                const isSelected = selected?.source === 'uci' && selected.name === dataset.name;
                return (
                  <button
                    key={dataset.id}
                    type="button"
                    onClick={() => handleUCISelect(dataset.id)}
                    disabled={isLoading}
                    className={cn(
                      'w-full rounded-md border border-border px-2.5 py-2 text-left transition-colors hover:border-foreground hover:bg-accent disabled:opacity-50',
                      isSelected && 'border-brand bg-accent'
                    )}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold">{dataset.name}</span>
                      <span className="shrink-0 text-[10px] text-muted-foreground">
                        ID: {dataset.id}
                      </span>
                    </div>
                    {dataset.description && (
                      <p className="mt-0.5 line-clamp-1 text-[11px] text-muted-foreground">
                        {dataset.description}
                      </p>
                    )}
                  </button>
                );
              })}
            </div>

            <div className="flex items-center gap-2 border-t pt-3">
              <Input
                type="number"
                value={customId}
                onChange={(e) => setCustomId(e.target.value)}
                placeholder="UCI dataset ID"
                className="h-8 text-xs"
              />
              <Button
                type="button"
                size="sm"
                disabled={!customId || isLoading}
                onClick={() => handleUCISelect(Number(customId))}
              >
                {isLoading ? 'Loading…' : 'Load by ID'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Right panel: live preview + quick facts */}
        <Card size="sm">
          <CardHeader className="border-b pb-3">
            <CardTitle className="truncate">
              {selected ? `Data Preview · ${selected.name}` : 'Data Preview'}
            </CardTitle>
            <CardAction className="flex items-center gap-2">
              {preview.isFetching && (
                <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
              )}
              {selected && (
                <>
                  <Badge variant="outline">{selected.rows.toLocaleString()} rows</Badge>
                  <Badge variant="outline">{selected.columns.length} cols</Badge>
                  <Badge variant={selected.source === 'uci' ? 'secondary' : 'brand'}>
                    {selected.source === 'uci' ? 'UCI' : 'Upload'}
                  </Badge>
                </>
              )}
            </CardAction>
            {selected && (
              <p className="text-xs text-muted-foreground">
                First {PREVIEW_ROWS} rows with column dtypes.
              </p>
            )}
          </CardHeader>
          <CardContent>
            {!selected && (
              <div className="flex items-center justify-center rounded-md border border-dashed border-border py-12 text-xs text-muted-foreground">
                Select or upload a dataset to preview its rows.
              </div>
            )}
            {selected && preview.isError && (
              <p className="py-2 text-xs text-destructive">Could not load preview.</p>
            )}
            {selected && preview.isLoading && (
              <div className="space-y-2">
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            )}
            {selected && preview.data && (
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    {preview.data.columns.map((col) => (
                      <TableHead key={col} className="h-auto py-1.5">
                        {col}
                        <span className="block text-[10px] font-normal text-muted-foreground">
                          {preview.data?.dtypes[col]}
                        </span>
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {preview.data.head.slice(0, PREVIEW_ROWS).map((row, i) => (
                    <TableRow key={i} className="even:bg-muted/40">
                      {preview.data?.columns.map((col) => (
                        <TableCell key={col} className="py-1.5 text-muted-foreground">
                          {String(row[col])}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
