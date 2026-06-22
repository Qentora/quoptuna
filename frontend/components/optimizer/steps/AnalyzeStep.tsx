'use client';

import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { generateSHAP, getMetrics } from '@/lib/api';
import * as Tabs from '@radix-ui/react-tabs';
import { BarChart3, Download, Loader2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { ErrorBanner } from '../NavButtons';
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

export function AnalyzeStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useProba, setUseProba] = useState(true);
  const [subsetSize, setSubsetSize] = useState(50);
  const [sampleIndex, setSampleIndex] = useState(0);

  const { optimization, analysis } = workflowData;
  const hasSHAP = analysis.featureImportance !== null;
  const autoRan = useRef(false);

  useEffect(() => {
    setFooter({ canContinue: hasSHAP, nextBusy: isGenerating, backDisabled: isGenerating });
  }, [hasSHAP, isGenerating, setFooter]);

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

  // Auto-run SHAP once on entering the step when results exist and none yet.
  // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount.
  useEffect(() => {
    if (!autoRan.current && optimization.executionId && !hasSHAP) {
      autoRan.current = true;
      void runAnalysis();
    }
  }, []);

  const availableTabs = PLOT_TABS.filter((t) => analysis.plots[t.id]);

  return (
    <div className="space-y-6">
      <StepHeader
        step={5}
        title="SHAP Analysis & Visualizations"
        subtitle="Explain the selected model with SHAP plots and classification metrics"
      />

      {optimization.bestValue !== null && (
        <Alert variant="success">
          <span className="text-sm">
            Analyzing trial <strong>#{optimization.selectedTrial ?? 'best'}</strong> · Best F1{' '}
            <strong>{optimization.bestValue.toFixed(4)}</strong>
          </span>
        </Alert>
      )}

      <ErrorBanner message={error} />

      <details className="rounded-lg border border-border p-4">
        <summary className="cursor-pointer text-sm font-medium">Advanced options</summary>
        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useProba}
              onChange={(e) => setUseProba(e.target.checked)}
              className="h-4 w-4 rounded accent-brand"
            />
            Use prediction probabilities
          </label>
          <Field label="Subset size">
            <Input
              type="number"
              min={10}
              max={500}
              value={subsetSize}
              onChange={(e) => setSubsetSize(Number(e.target.value))}
            />
          </Field>
          <Field label="Waterfall sample index">
            <Input
              type="number"
              min={0}
              value={sampleIndex}
              onChange={(e) => setSampleIndex(Number(e.target.value))}
            />
          </Field>
        </div>
      </details>

      {isGenerating && !hasSHAP && (
        <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Running SHAP analysis…
        </div>
      )}

      <div className="text-center">
        <Button type="button" size="lg" onClick={runAnalysis} disabled={isGenerating}>
          {isGenerating ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <BarChart3 className="h-5 w-5" />
          )}
          {hasSHAP ? 'Re-run Analysis' : 'Generate SHAP Analysis'}
        </Button>
      </div>

      {hasSHAP && (
        <div className="space-y-6">
          {availableTabs.length > 0 && (
            <Tabs.Root defaultValue={availableTabs[0].id} className="w-full">
              <Tabs.List className="mb-4 flex gap-2 overflow-x-auto border-b border-border">
                {availableTabs.map((t) => (
                  <Tabs.Trigger
                    key={t.id}
                    value={t.id}
                    className="border-b-2 border-transparent px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground data-[state=active]:border-brand data-[state=active]:text-brand"
                  >
                    {t.label}
                  </Tabs.Trigger>
                ))}
              </Tabs.List>
              {availableTabs.map((t) => (
                <Tabs.Content key={t.id} value={t.id}>
                  <div className="rounded-lg border border-border bg-card p-4">
                    <img
                      src={analysis.plots[t.id]}
                      alt={`${t.label} SHAP plot`}
                      className="mx-auto max-h-[480px]"
                    />
                    <div className="mt-2 text-right">
                      <Button
                        type="button"
                        variant="link"
                        size="sm"
                        onClick={() => downloadDataUrl(analysis.plots[t.id], `shap-${t.id}.png`)}
                      >
                        <Download className="h-4 w-4" /> Download
                      </Button>
                    </div>
                  </div>
                </Tabs.Content>
              ))}
            </Tabs.Root>
          )}

          {analysis.featureImportance && analysis.featureImportance.length > 0 && (
            <Card className="p-6">
              <h3 className="mb-4 text-base font-semibold">Feature Importance</h3>
              <div className="max-h-72 space-y-3 overflow-y-auto pr-2">
                {analysis.featureImportance.map((item) => {
                  const max = analysis.featureImportance?.[0]?.importance || 1;
                  return (
                    <div key={item.feature} className="space-y-1">
                      <div className="flex items-center justify-between text-sm">
                        <span className="font-medium">{item.feature}</span>
                        <span className="text-muted-foreground">{item.importance.toFixed(3)}</span>
                      </div>
                      <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-gradient-to-r from-brand to-brand/60"
                          style={{ width: `${(item.importance / max) * 100}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </Card>
          )}

          {(analysis.confusionMatrixPlot || analysis.metrics) && (
            <Card className="grid grid-cols-1 gap-6 p-6 md:grid-cols-2">
              {analysis.confusionMatrixPlot && (
                <div>
                  <h3 className="mb-2 text-base font-semibold">Confusion Matrix</h3>
                  <img
                    src={analysis.confusionMatrixPlot}
                    alt="Confusion matrix"
                    className="mx-auto max-h-80"
                  />
                </div>
              )}
              {analysis.metrics && (
                <div>
                  <h3 className="mb-2 text-base font-semibold">Metrics</h3>
                  <MetricsList metrics={analysis.metrics} />
                </div>
              )}
            </Card>
          )}
        </div>
      )}
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
            <div key={key} className="flex justify-between border-b border-border py-1">
              <dt className="text-muted-foreground">{key}</dt>
              <dd className="font-medium">
                {typeof value === 'number' ? value.toFixed(4) : String(value)}
              </dd>
            </div>
          ))}
      </dl>
      {typeof report === 'string' && (
        <pre className="overflow-x-auto rounded-md border border-border bg-muted p-3 font-mono text-xs">
          {report}
        </pre>
      )}
    </div>
  );
}
