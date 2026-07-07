'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Checkbox } from '@/components/ui/checkbox';
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { useDatasetPreview } from '@/lib/hooks';
import { cn } from '@/lib/utils';
import { Check, Search, Sparkles } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

const isNumericDtype = (dtype: string | undefined) =>
  !!dtype && /int|float|double|number/i.test(dtype);

// Mirror the backend caps in prepare.py.
const MAX_ONEHOT_CATEGORIES = 10;
const MAX_ORDINAL_CATEGORIES = 100;

const NONE_VALUE = '__none__';

export function FeaturesStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const preview = useDatasetPreview(workflowData.dataset?.id ?? null);
  const { selectedFeatures, targetColumn, labelMapping, sensitiveFeature, categoricalEncoding } =
    workflowData.features;
  const [search, setSearch] = useState('');

  const columns = workflowData.dataset?.columns ?? [];
  const dtypes = preview.data?.dtypes ?? {};
  const missingCounts = preview.data?.missing ?? {};
  const uniqueCounts = preview.data?.unique_counts ?? {};

  // Categorical columns get auto-encoded by the backend; block ones whose
  // cardinality exceeds the cap of the chosen encoding method.
  const isCategorical = (column: string) => !isNumericDtype(dtypes[column]);
  const encodingCap =
    categoricalEncoding === 'onehot' ? MAX_ONEHOT_CATEGORIES : MAX_ORDINAL_CATEGORIES;
  const tooManyCategories = (column: string) =>
    isCategorical(column) && (uniqueCounts[column] ?? 0) > encodingCap;
  const hasCategoricalSelected = selectedFeatures.some(isCategorical);
  const categoricalSelectedCount = selectedFeatures.filter(isCategorical).length;

  const setEncoding = (value: 'ordinal' | 'onehot') =>
    setWorkflowData((prev) => ({
      ...prev,
      features: { ...prev.features, categoricalEncoding: value },
    }));

  const filteredColumns = useMemo(() => {
    const q = search.trim().toLowerCase();
    return q ? columns.filter((c) => c.toLowerCase().includes(q)) : columns;
  }, [columns, search]);

  const toggleFeature = (column: string) => {
    if (column === targetColumn || tooManyCategories(column)) return;
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
        targetColumn: column === NONE_VALUE ? null : column,
        selectedFeatures: prev.features.selectedFeatures.filter((f) => f !== column),
        labelMapping: { neg: null, pos: null },
      },
    }));
  };

  const setSensitiveFeature = (value: string) => {
    setWorkflowData((prev) => ({
      ...prev,
      features: { ...prev.features, sensitiveFeature: value === NONE_VALUE ? null : value },
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

  // Preselect all usable columns once when the preview loads and there is no
  // prior selection (preserves persisted workflowData state on revisit).
  const didDefaultSelect = useRef(false);
  useEffect(() => {
    if (didDefaultSelect.current || !preview.data || columns.length === 0) return;
    didDefaultSelect.current = true;
    if (selectedFeatures.length > 0 || targetColumn !== null) return;
    const usable = columns.filter((c) => !tooManyCategories(c));
    if (usable.length === 0) return;
    setWorkflowData((prev) => ({
      ...prev,
      features: { ...prev.features, selectedFeatures: usable },
    }));
  });

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
    <div className="flex flex-col gap-4">
      <StepHeader
        step={2}
        title="Feature Selection"
        subtitle="Select input features, the target column, and map labels for binary classification"
      />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1fr_320px]">
        {/* Left: feature checkbox grid */}
        <Card size="sm">
          <CardHeader className="border-b">
            <CardTitle className="flex items-center justify-between gap-3">
              <span>Features ({columns.length} columns)</span>
              <span className="flex items-center gap-1">
                <Button type="button" variant="ghost" size="sm" onClick={smartSelect}>
                  <Sparkles size={14} /> Smart select
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={clearAll}
                  disabled={selectedFeatures.length === 0}
                >
                  Clear
                </Button>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="relative max-w-xs">
              <Search className="absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search columns"
                className="pl-8"
              />
            </div>

            {filteredColumns.length === 0 ? (
              <p className="text-sm text-muted-foreground">No matching columns.</p>
            ) : (
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 md:grid-cols-3 xl:grid-cols-4">
                {filteredColumns.map((column) => {
                  const isFeature = selectedFeatures.includes(column);
                  const isTarget = targetColumn === column;
                  const blocked = tooManyCategories(column);
                  const disabled = isTarget || blocked;
                  return (
                    <label
                      key={column}
                      htmlFor={`feature-${column}`}
                      className={cn(
                        'flex min-w-0 items-center gap-2 rounded-md px-2 py-1.5 transition-colors',
                        disabled ? 'opacity-50' : 'cursor-pointer hover:bg-muted'
                      )}
                      title={
                        blocked
                          ? `${uniqueCounts[column]} categories exceed the ${categoricalEncoding} limit of ${encodingCap}`
                          : isTarget
                            ? 'Target column'
                            : dtypes[column]
                      }
                    >
                      <Checkbox
                        id={`feature-${column}`}
                        checked={isFeature}
                        disabled={disabled}
                        onCheckedChange={() => toggleFeature(column)}
                        aria-label={`Toggle ${column} as feature`}
                      />
                      <span className="truncate text-xs">{column}</span>
                      {isTarget ? (
                        <Badge variant="emerald" className="shrink-0">
                          target
                        </Badge>
                      ) : blocked ? (
                        <Badge variant="destructive" className="shrink-0">
                          {uniqueCounts[column]} cats
                        </Badge>
                      ) : isCategorical(column) && isFeature ? (
                        <Badge variant="amber" className="shrink-0">
                          {categoricalEncoding === 'onehot' ? 'one-hot' : 'ordinal'}
                        </Badge>
                      ) : null}
                      {(missingCounts[column] ?? 0) > 0 && (
                        <Badge
                          variant="secondary"
                          className="shrink-0"
                          title={`${missingCounts[column]} missing values — imputed automatically`}
                        >
                          {missingCounts[column]} NA
                        </Badge>
                      )}
                    </label>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right rail */}
        <Card size="sm" className="lg:sticky lg:top-4 lg:self-start">
          <CardHeader className="border-b">
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Field>
              <FieldLabel htmlFor="target-column">Target column</FieldLabel>
              <Select value={targetColumn ?? NONE_VALUE} onValueChange={setTarget}>
                <SelectTrigger id="target-column" className="w-full">
                  <SelectValue placeholder="Select target…" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE_VALUE}>None — choose a target</SelectItem>
                  {columns.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <FieldDescription>The column the model learns to predict.</FieldDescription>
            </Field>

            {targetColumn && needsMapping && (
              <div className="rounded-md border border-border bg-muted p-3">
                <p className="text-xs font-medium">Label mapping (binary)</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Quantum models require labels encoded as -1 / 1.
                </p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <Field>
                    <FieldLabel htmlFor="map-neg">Maps to -1</FieldLabel>
                    <Select
                      value={labelMapping.neg === null ? '' : String(labelMapping.neg)}
                      onValueChange={(v) => setMapping('neg', v)}
                    >
                      <SelectTrigger id="map-neg" className="w-full">
                        <SelectValue placeholder="Select…" />
                      </SelectTrigger>
                      <SelectContent>
                        {targetValues.map((v) => (
                          <SelectItem key={String(v)} value={String(v)}>
                            {String(v)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                  <Field>
                    <FieldLabel htmlFor="map-pos">Maps to 1</FieldLabel>
                    <Select
                      value={labelMapping.pos === null ? '' : String(labelMapping.pos)}
                      onValueChange={(v) => setMapping('pos', v)}
                    >
                      <SelectTrigger id="map-pos" className="w-full">
                        <SelectValue placeholder="Select…" />
                      </SelectTrigger>
                      <SelectContent>
                        {targetValues.map((v) => (
                          <SelectItem key={String(v)} value={String(v)}>
                            {String(v)}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                </div>
              </div>
            )}

            <Separator />

            <Field>
              <FieldLabel htmlFor="categorical-encoding">Categorical encoding</FieldLabel>
              <Select
                value={categoricalEncoding}
                onValueChange={(v) => setEncoding(v as 'ordinal' | 'onehot')}
              >
                <SelectTrigger id="categorical-encoding" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ordinal">Ordinal — 1 column per feature</SelectItem>
                  <SelectItem value="onehot">One-hot — 1 column per category</SelectItem>
                </SelectContent>
              </Select>
              <FieldDescription>
                How categorical columns become numbers. Ordinal keeps one column per feature (faster
                for quantum models); one-hot adds a column per category.
                {!hasCategoricalSelected && ' No categorical features are currently selected.'}
              </FieldDescription>
            </Field>

            <Separator />

            <Field>
              <FieldLabel htmlFor="protected-attribute">Protected attribute (optional)</FieldLabel>
              <Select value={sensitiveFeature ?? NONE_VALUE} onValueChange={setSensitiveFeature}>
                <SelectTrigger id="protected-attribute" className="w-full">
                  <SelectValue placeholder="None — skip fairness audit" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NONE_VALUE}>None — skip fairness audit</SelectItem>
                  {columns
                    .filter((c) => c !== targetColumn)
                    .map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
              <FieldDescription>
                A categorical column (e.g. sex, race, age group) used to audit fairness across
                groups. Never used for training.
              </FieldDescription>
            </Field>

            <Separator />

            {/* Selection summary */}
            <div className="space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Selected features</span>
                <Badge variant={selectedFeatures.length > 0 ? 'brand' : 'outline'}>
                  {selectedFeatures.length}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Categorical (auto-encoded)</span>
                <Badge variant={categoricalSelectedCount > 0 ? 'amber' : 'outline'}>
                  {categoricalSelectedCount}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Target</span>
                {targetColumn ? (
                  <Badge variant="emerald">{targetColumn}</Badge>
                ) : (
                  <span className="text-muted-foreground">—</span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Validation bar */}
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
