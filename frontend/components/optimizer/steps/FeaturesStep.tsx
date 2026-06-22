'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { useDatasetPreview } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Check, Crosshair, Search, Sparkles, X } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

const isNumericDtype = (dtype: string | undefined) =>
  !!dtype && /int|float|double|number/i.test(dtype);

export function FeaturesStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const preview = useDatasetPreview(workflowData.dataset?.id ?? null);
  const { selectedFeatures, targetColumn, labelMapping } = workflowData.features;
  const [search, setSearch] = useState('');

  const columns = workflowData.dataset?.columns ?? [];
  const dtypes = preview.data?.dtypes ?? {};

  const filteredColumns = useMemo(() => {
    const q = search.trim().toLowerCase();
    return q ? columns.filter((c) => c.toLowerCase().includes(q)) : columns;
  }, [columns, search]);

  const toggleFeature = (column: string) => {
    if (column === targetColumn) return;
    setWorkflowData((prev) => {
      const current = prev.features.selectedFeatures;
      const next = current.includes(column)
        ? current.filter((f) => f !== column)
        : [...current, column];
      return { ...prev, features: { ...prev.features, selectedFeatures: next } };
    });
  };

  const setTarget = (column: string) => {
    setWorkflowData((prev) => ({
      ...prev,
      features: {
        ...prev.features,
        targetColumn: prev.features.targetColumn === column ? null : column,
        selectedFeatures: prev.features.selectedFeatures.filter((f) => f !== column),
        labelMapping: { neg: null, pos: null },
      },
    }));
  };

  const setMapping = (side: 'neg' | 'pos', value: string) => {
    setWorkflowData((prev) => ({
      ...prev,
      features: {
        ...prev.features,
        labelMapping: { ...prev.features.labelMapping, [side]: value },
      },
    }));
  };

  const smartSelect = () => {
    const numeric = columns.filter((c) => c !== targetColumn && isNumericDtype(dtypes[c]));
    setWorkflowData((prev) => ({
      ...prev,
      features: { ...prev.features, selectedFeatures: numeric },
    }));
  };

  const clearAll = () =>
    setWorkflowData((prev) => ({ ...prev, features: { ...prev.features, selectedFeatures: [] } }));

  const targetValues = targetColumn
    ? (preview.data?.target_values_by_column[targetColumn] ?? [])
    : [];
  const needsMapping = targetValues.length === 2;
  const mappingComplete =
    !needsMapping ||
    (labelMapping.neg !== null &&
      labelMapping.pos !== null &&
      labelMapping.neg !== labelMapping.pos);

  const canProceed = selectedFeatures.length > 0 && targetColumn !== null && mappingComplete;

  const missing: string[] = [];
  if (selectedFeatures.length === 0) missing.push('Select at least one feature');
  if (!targetColumn) missing.push('Choose a target column');
  if (needsMapping && !mappingComplete) missing.push('Complete the label mapping');

  useEffect(() => {
    setFooter({ canContinue: canProceed });
  }, [canProceed, setFooter]);

  return (
    <div className="space-y-6">
      <StepHeader
        step={2}
        title="Feature Selection"
        subtitle="Select input features, the target column, and map labels for binary classification"
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Left panel: searchable columns */}
        <div className="flex flex-col rounded-lg border border-border">
          <div className="flex items-center justify-between gap-2 border-b border-border bg-muted px-3 py-2">
            <h3 className="text-sm font-semibold">Columns ({columns.length})</h3>
            <button
              type="button"
              onClick={smartSelect}
              className="inline-flex items-center gap-1 text-xs font-medium text-brand hover:underline"
            >
              <Sparkles size={14} /> Smart select
            </button>
          </div>
          <div className="border-b border-border p-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search columns"
                className="pl-8"
              />
            </div>
          </div>
          <div className="max-h-80 overflow-y-auto p-2">
            {filteredColumns.map((column) => {
              const isFeature = selectedFeatures.includes(column);
              const isTarget = targetColumn === column;
              return (
                <div
                  key={column}
                  className={cn(
                    'mb-1 flex items-center justify-between gap-2 rounded-md px-2 py-1.5 transition-colors',
                    isTarget ? 'bg-brand/10' : 'hover:bg-muted'
                  )}
                >
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="truncate text-sm">{column}</span>
                    {dtypes[column] && (
                      <Badge variant="outline" className="shrink-0">
                        {dtypes[column]}
                      </Badge>
                    )}
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    <button
                      type="button"
                      onClick={() => toggleFeature(column)}
                      disabled={isTarget}
                      className={cn(
                        'rounded-md px-2 py-1 text-xs font-medium transition-colors',
                        isFeature
                          ? 'bg-brand/15 text-brand'
                          : 'text-muted-foreground hover:bg-accent disabled:opacity-40'
                      )}
                    >
                      {isFeature ? 'Feature ✓' : 'Feature'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setTarget(column)}
                      title="Set as target"
                      className={cn(
                        'rounded-md p-1 transition-colors',
                        isTarget
                          ? 'bg-brand text-brand-foreground'
                          : 'text-muted-foreground hover:bg-accent'
                      )}
                    >
                      <Crosshair size={14} />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right panel: selection */}
        <div className="space-y-4">
          <div className="rounded-lg border border-border p-3">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold">
                Selected features ({selectedFeatures.length})
              </h3>
              {selectedFeatures.length > 0 && (
                <button
                  type="button"
                  onClick={clearAll}
                  className="text-xs font-medium text-muted-foreground hover:underline"
                >
                  Clear
                </button>
              )}
            </div>
            {selectedFeatures.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No features yet — pick columns on the left or use Smart select.
              </p>
            ) : (
              <div className="flex flex-wrap gap-1.5">
                {selectedFeatures.map((f) => (
                  <span
                    key={f}
                    className="inline-flex items-center gap-1 rounded-full bg-brand/15 px-2.5 py-0.5 text-xs font-medium text-brand"
                  >
                    {f}
                    <button
                      type="button"
                      onClick={() => toggleFeature(f)}
                      aria-label={`Remove ${f}`}
                    >
                      <X size={12} />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="rounded-lg border border-border p-3">
            <h3 className="mb-2 text-sm font-semibold">Target column</h3>
            {targetColumn ? (
              <Badge variant="emerald" size="md">
                {targetColumn}
              </Badge>
            ) : (
              <p className="text-sm text-muted-foreground">
                Use the crosshair on a column to set the target.
              </p>
            )}

            {targetColumn && needsMapping && (
              <div className="mt-3 rounded-md border border-border bg-muted p-3">
                <p className="text-sm font-medium">Label mapping (binary)</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Quantum models require labels encoded as -1 / 1.
                </p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <label className="text-xs" htmlFor="map-neg">
                    <span className="mb-1 block">Maps to -1</span>
                    <Select
                      id="map-neg"
                      value={labelMapping.neg === null ? '' : String(labelMapping.neg)}
                      onChange={(e) => setMapping('neg', e.target.value)}
                    >
                      <option value="">Select…</option>
                      {targetValues.map((v) => (
                        <option key={String(v)} value={String(v)}>
                          {String(v)}
                        </option>
                      ))}
                    </Select>
                  </label>
                  <label className="text-xs" htmlFor="map-pos">
                    <span className="mb-1 block">Maps to 1</span>
                    <Select
                      id="map-pos"
                      value={labelMapping.pos === null ? '' : String(labelMapping.pos)}
                      onChange={(e) => setMapping('pos', e.target.value)}
                    >
                      <option value="">Select…</option>
                      {targetValues.map((v) => (
                        <option key={String(v)} value={String(v)}>
                          {String(v)}
                        </option>
                      ))}
                    </Select>
                  </label>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sticky validation bar */}
      <div
        className={cn(
          'flex items-center gap-2 rounded-lg border p-3 text-sm',
          canProceed
            ? 'border-accent-emerald bg-accent-emerald/30 text-accent-emerald-foreground'
            : 'border-accent-amber bg-accent-amber/30 text-accent-amber-foreground'
        )}
      >
        {canProceed ? (
          <>
            <Check size={16} />
            <span className="font-medium">Ready to continue</span>
          </>
        ) : (
          <span>{missing.join(' · ')}</span>
        )}
      </div>
    </div>
  );
}
