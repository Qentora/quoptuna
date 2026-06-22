'use client';

import { Alert } from '@/components/ui/alert';
import { Select } from '@/components/ui/select';
import { Table, TableBody, TableContainer, TableHead, Td, Th } from '@/components/ui/table';
import { useDatasetPreview } from '@/lib/hooks';
import { FileText } from 'lucide-react';
import { NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

export function FeaturesStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const preview = useDatasetPreview(workflowData.dataset?.id ?? null);
  const { selectedFeatures, targetColumn, labelMapping } = workflowData.features;

  const toggleFeature = (column: string) => {
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

  const selectAll = () => {
    const cols = (workflowData.dataset?.columns ?? []).filter((c) => c !== targetColumn);
    setWorkflowData((prev) => ({
      ...prev,
      features: { ...prev.features, selectedFeatures: cols },
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

  return (
    <div className="space-y-6">
      <StepHeader
        step={2}
        title="Feature Selection"
        subtitle="Select input features, the target column, and map labels for binary classification"
      />

      {workflowData.dataset && (
        <Alert variant="info" icon={false}>
          <div className="flex items-start gap-3">
            <FileText className="mt-0.5 h-5 w-5 shrink-0" />
            <div>
              <p className="font-medium">{workflowData.dataset.name}</p>
              <p className="text-sm opacity-80">
                {workflowData.dataset.rows} rows × {workflowData.dataset.columns.length} columns
              </p>
            </div>
          </div>
        </Alert>
      )}

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold">Columns</h3>
          <div className="flex gap-2 text-sm">
            <button
              type="button"
              onClick={selectAll}
              className="font-medium text-foreground hover:underline"
            >
              Select All as Features
            </button>
            <span className="text-border">|</span>
            <button
              type="button"
              onClick={clearAll}
              className="font-medium text-foreground hover:underline"
            >
              Clear Selection
            </button>
          </div>
        </div>

        <TableContainer>
          <Table>
            <TableHead>
              <tr>
                <Th className="text-foreground">Column</Th>
                <Th className="text-center text-foreground">Feature</Th>
                <Th className="text-center text-foreground">Target</Th>
              </tr>
            </TableHead>
            <TableBody>
              {(workflowData.dataset?.columns ?? []).map((column) => (
                <tr key={column} className="hover:bg-muted">
                  <Td>{column}</Td>
                  <Td className="text-center">
                    <input
                      type="checkbox"
                      checked={selectedFeatures.includes(column)}
                      onChange={() => toggleFeature(column)}
                      disabled={targetColumn === column}
                      className="h-4 w-4 rounded accent-primary disabled:opacity-50"
                    />
                  </Td>
                  <Td className="text-center">
                    <input
                      type="radio"
                      name="target"
                      checked={targetColumn === column}
                      onChange={() => setTarget(column)}
                      className="h-4 w-4 accent-primary"
                    />
                  </Td>
                </tr>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {targetColumn && needsMapping && (
          <Alert variant="info" icon={false} className="block">
            <p className="font-medium">Label Mapping (binary classification)</p>
            <p className="mt-1 text-sm opacity-80">
              Quantum models require labels encoded as -1 / 1. Map each target value below.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-4">
              <label className="text-sm" htmlFor="map-neg">
                <span className="mb-1 block">Maps to -1 (negative)</span>
                <Select
                  id="map-neg"
                  value={labelMapping.neg === null ? '' : String(labelMapping.neg)}
                  onChange={(e) => setMapping('neg', e.target.value)}
                >
                  <option value="">Select...</option>
                  {targetValues.map((v) => (
                    <option key={String(v)} value={String(v)}>
                      {String(v)}
                    </option>
                  ))}
                </Select>
              </label>
              <label className="text-sm" htmlFor="map-pos">
                <span className="mb-1 block">Maps to 1 (positive)</span>
                <Select
                  id="map-pos"
                  value={labelMapping.pos === null ? '' : String(labelMapping.pos)}
                  onChange={(e) => setMapping('pos', e.target.value)}
                >
                  <option value="">Select...</option>
                  {targetValues.map((v) => (
                    <option key={String(v)} value={String(v)}>
                      {String(v)}
                    </option>
                  ))}
                </Select>
              </label>
            </div>
          </Alert>
        )}

        <div className="space-y-2 rounded-lg bg-muted p-4 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Selected Features:</span>
            <span className="font-medium">{selectedFeatures.length} column(s)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Target Column:</span>
            <span className="font-medium">{targetColumn || 'Not selected'}</span>
          </div>
        </div>

        {!canProceed && (
          <Alert variant="warning">
            Select at least one feature, a target column, and (for binary targets) a complete label
            mapping to continue.
          </Alert>
        )}
      </div>

      <NavButtons onBack={onBack} onNext={onNext} nextDisabled={!canProceed} />
    </div>
  );
}
