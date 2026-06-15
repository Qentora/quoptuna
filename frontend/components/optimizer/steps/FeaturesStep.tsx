'use client';

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
        title="Feature Selection"
        subtitle="Select input features, the target column, and map labels for binary classification"
      />

      {workflowData.dataset && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-start gap-3">
          <FileText className="w-5 h-5 text-blue-600 mt-0.5" />
          <div>
            <p className="font-medium text-blue-900">{workflowData.dataset.name}</p>
            <p className="text-sm text-blue-700">
              {workflowData.dataset.rows} rows × {workflowData.dataset.columns.length} columns
            </p>
          </div>
        </div>
      )}

      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-900">Columns</h3>
          <div className="flex gap-2 text-sm">
            <button type="button" onClick={selectAll} className="text-blue-600 hover:text-blue-700">
              Select All as Features
            </button>
            <span className="text-gray-300">|</span>
            <button type="button" onClick={clearAll} className="text-blue-600 hover:text-blue-700">
              Clear Selection
            </button>
          </div>
        </div>

        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-50 border-b border-gray-200">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-semibold text-gray-900">Column</th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-900">
                  Feature
                </th>
                <th className="px-4 py-3 text-center text-sm font-semibold text-gray-900">
                  Target
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {(workflowData.dataset?.columns ?? []).map((column) => (
                <tr key={column} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-900">{column}</td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="checkbox"
                      checked={selectedFeatures.includes(column)}
                      onChange={() => toggleFeature(column)}
                      disabled={targetColumn === column}
                      className="w-4 h-4 text-blue-600 rounded disabled:opacity-50"
                    />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <input
                      type="radio"
                      name="target"
                      checked={targetColumn === column}
                      onChange={() => setTarget(column)}
                      className="w-4 h-4 text-blue-600"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {targetColumn && needsMapping && (
          <div className="bg-purple-50 border border-purple-200 rounded-lg p-4 space-y-3">
            <p className="font-medium text-purple-900">Label Mapping (binary classification)</p>
            <p className="text-sm text-purple-700">
              Quantum models require labels encoded as -1 / 1. Map each target value below.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <label className="text-sm">
                <span className="block mb-1 text-gray-700">Maps to -1 (negative)</span>
                <select
                  value={labelMapping.neg === null ? '' : String(labelMapping.neg)}
                  onChange={(e) => setMapping('neg', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="">Select...</option>
                  {targetValues.map((v) => (
                    <option key={String(v)} value={String(v)}>
                      {String(v)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-sm">
                <span className="block mb-1 text-gray-700">Maps to 1 (positive)</span>
                <select
                  value={labelMapping.pos === null ? '' : String(labelMapping.pos)}
                  onChange={(e) => setMapping('pos', e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                >
                  <option value="">Select...</option>
                  {targetValues.map((v) => (
                    <option key={String(v)} value={String(v)}>
                      {String(v)}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        )}

        <div className="bg-gray-50 rounded-lg p-4 space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600">Selected Features:</span>
            <span className="font-medium text-gray-900">{selectedFeatures.length} column(s)</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600">Target Column:</span>
            <span className="font-medium text-gray-900">{targetColumn || 'Not selected'}</span>
          </div>
        </div>

        {!canProceed && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-md text-sm">
            Select at least one feature, a target column, and (for binary targets) a complete label
            mapping to continue.
          </div>
        )}
      </div>

      <NavButtons onBack={onBack} onNext={onNext} nextDisabled={!canProceed} />
    </div>
  );
}
