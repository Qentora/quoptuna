'use client';

import { generateSHAP, getMetrics } from '@/lib/api';
import * as Tabs from '@radix-ui/react-tabs';
import { BarChart3, Download, Loader2 } from 'lucide-react';
import { useState } from 'react';
import { ErrorBanner, NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

const PLOT_TABS: Array<{ id: string; label: string }> = [
  { id: 'bar', label: 'Bar' },
  { id: 'beeswarm', label: 'Beeswarm' },
  { id: 'violin', label: 'Violin' },
  { id: 'heatmap', label: 'Heatmap' },
  { id: 'waterfall', label: 'Waterfall' },
];

function downloadDataUrl(dataUrl: string, filename: string) {
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

export function AnalyzeStep({ onNext, onBack, workflowData, setWorkflowData }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useProba, setUseProba] = useState(true);
  const [subsetSize, setSubsetSize] = useState(50);
  const [sampleIndex, setSampleIndex] = useState(0);

  const { optimization, analysis } = workflowData;
  const hasSHAP = analysis.featureImportance !== null;

  const runAnalysis = async () => {
    if (!optimization.executionId) {
      setError('No optimization results available');
      return;
    }
    setIsGenerating(true);
    setError(null);
    try {
      const [shap, metrics] = await Promise.all([
        generateSHAP({
          optimization_id: optimization.executionId,
          trial_number: optimization.selectedTrial ?? undefined,
          sample_index: sampleIndex,
          use_proba: useProba,
          subset_size: subsetSize,
        }),
        getMetrics(optimization.executionId, optimization.selectedTrial ?? undefined).catch(
          () => null
        ),
      ]);

      setWorkflowData((prev) => ({
        ...prev,
        analysis: {
          featureImportance: shap.feature_importance,
          plots: shap.plots,
          metrics: metrics?.metrics ?? null,
          confusionMatrixPlot: metrics?.confusion_matrix_plot ?? null,
        },
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'SHAP generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const availableTabs = PLOT_TABS.filter((t) => analysis.plots[t.id]);

  return (
    <div className="space-y-6">
      <StepHeader
        title="SHAP Analysis & Visualizations"
        subtitle="Explain the selected model with SHAP plots and classification metrics"
      />

      {optimization.bestValue !== null && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-start gap-3">
          <BarChart3 className="w-5 h-5 text-green-600 mt-0.5" />
          <p className="text-sm text-green-700">
            Analyzing trial <strong>#{optimization.selectedTrial ?? 'best'}</strong> · Best F1{' '}
            <strong>{optimization.bestValue.toFixed(4)}</strong>
          </p>
        </div>
      )}

      <ErrorBanner message={error} />

      <div className="bg-white border border-gray-200 rounded-lg p-4 grid grid-cols-1 md:grid-cols-3 gap-4">
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={useProba}
            onChange={(e) => setUseProba(e.target.checked)}
            className="w-4 h-4 text-blue-600 rounded"
          />
          Use prediction probabilities
        </label>
        <label className="text-sm text-gray-700">
          <span className="block mb-1">Subset size</span>
          <input
            type="number"
            min={10}
            max={500}
            value={subsetSize}
            onChange={(e) => setSubsetSize(Number(e.target.value))}
            className="w-full px-3 py-1.5 border border-gray-300 rounded-md"
          />
        </label>
        <label className="text-sm text-gray-700">
          <span className="block mb-1">Waterfall sample index</span>
          <input
            type="number"
            min={0}
            value={sampleIndex}
            onChange={(e) => setSampleIndex(Number(e.target.value))}
            className="w-full px-3 py-1.5 border border-gray-300 rounded-md"
          />
        </label>
      </div>

      <div className="text-center">
        <button
          type="button"
          onClick={runAnalysis}
          disabled={isGenerating}
          className="px-8 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-semibold inline-flex items-center gap-2 disabled:bg-gray-300"
        >
          {isGenerating ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <BarChart3 className="w-5 h-5" />
          )}
          {hasSHAP ? 'Re-run Analysis' : 'Generate SHAP Analysis'}
        </button>
      </div>

      {hasSHAP && (
        <div className="space-y-6">
          {availableTabs.length > 0 && (
            <Tabs.Root defaultValue={availableTabs[0].id} className="w-full">
              <Tabs.List className="flex gap-2 overflow-x-auto border-b border-gray-200 mb-4">
                {availableTabs.map((t) => (
                  <Tabs.Trigger
                    key={t.id}
                    value={t.id}
                    className="px-4 py-2 text-sm font-medium text-gray-600 data-[state=active]:text-blue-600 data-[state=active]:border-b-2 data-[state=active]:border-blue-600"
                  >
                    {t.label}
                  </Tabs.Trigger>
                ))}
              </Tabs.List>
              {availableTabs.map((t) => (
                <Tabs.Content key={t.id} value={t.id}>
                  <div className="border border-gray-200 rounded-lg p-4">
                    <img
                      src={analysis.plots[t.id]}
                      alt={`${t.label} SHAP plot`}
                      className="mx-auto max-h-[480px]"
                    />
                    <div className="text-right mt-2">
                      <button
                        type="button"
                        onClick={() => downloadDataUrl(analysis.plots[t.id], `shap-${t.id}.png`)}
                        className="text-sm text-blue-600 hover:text-blue-700 inline-flex items-center gap-1"
                      >
                        <Download className="w-4 h-4" /> Download
                      </button>
                    </div>
                  </div>
                </Tabs.Content>
              ))}
            </Tabs.Root>
          )}

          {analysis.featureImportance && analysis.featureImportance.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance</h3>
              <div className="space-y-3">
                {analysis.featureImportance.map((item, index) => {
                  const max = analysis.featureImportance?.[0]?.importance || 1;
                  return (
                    <div key={item.feature} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium text-gray-700">{item.feature}</span>
                        <span className="text-gray-500">{item.importance.toFixed(3)}</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${(item.importance / max) * 100}%`,
                            backgroundColor: `hsl(${220 - index * 18}, 70%, 50%)`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {(analysis.confusionMatrixPlot || analysis.metrics) && (
            <div className="bg-white border border-gray-200 rounded-lg p-6 grid grid-cols-1 md:grid-cols-2 gap-6">
              {analysis.confusionMatrixPlot && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Confusion Matrix</h3>
                  <img
                    src={analysis.confusionMatrixPlot}
                    alt="Confusion matrix"
                    className="max-h-80 mx-auto"
                  />
                </div>
              )}
              {analysis.metrics && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">Metrics</h3>
                  <MetricsList metrics={analysis.metrics} />
                </div>
              )}
            </div>
          )}
        </div>
      )}

      <NavButtons
        onBack={onBack}
        onNext={onNext}
        backDisabled={isGenerating}
        nextDisabled={!hasSHAP || isGenerating}
      />
    </div>
  );
}

function MetricsList({ metrics }: { metrics: Record<string, any> }) {
  const scalarEntries = Object.entries(metrics).filter(
    ([, v]) => typeof v === 'number' || typeof v === 'string'
  );
  const report = metrics.classification_report;

  return (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-2 text-sm">
        {scalarEntries
          .filter(([k]) => k !== 'classification_report')
          .map(([key, value]) => (
            <div key={key} className="flex justify-between border-b border-gray-100 py-1">
              <dt className="text-gray-600">{key}</dt>
              <dd className="font-medium text-gray-900">
                {typeof value === 'number' ? value.toFixed(4) : String(value)}
              </dd>
            </div>
          ))}
      </dl>
      {typeof report === 'string' && (
        <pre className="text-xs bg-gray-50 border border-gray-200 rounded-md p-3 overflow-x-auto">
          {report}
        </pre>
      )}
    </div>
  );
}
