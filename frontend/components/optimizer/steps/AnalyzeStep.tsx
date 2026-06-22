'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Metric } from '@/components/ui/metric';
import { generateSHAP, getCurves, getMetrics, getStudyPlots } from '@/lib/api';
import { cn } from '@/lib/utils';
import * as Tabs from '@radix-ui/react-tabs';
import { BarChart3, Download, Loader2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { ErrorBanner } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

const SHAP_TABS: Array<{ id: string; label: string }> = [
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

const fmt = (v: unknown) =>
  typeof v === 'number' ? v.toFixed(4) : v === null || v === undefined ? '—' : String(v);

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
    const id = optimization.executionId;
    const trial = optimization.selectedTrial ?? undefined;
    setIsGenerating(true);
    setError(null);
    try {
      const [shap, metrics, curves, study] = await Promise.all([
        generateSHAP({
          optimization_id: id,
          trial_number: trial,
          sample_index: sampleIndex,
          use_proba: useProba,
          subset_size: subsetSize,
        }),
        getMetrics(id, trial).catch(() => null),
        getCurves(id, trial, useProba, subsetSize).catch(() => null),
        getStudyPlots(id).catch(() => null),
      ]);

      setWorkflowData((prev) => ({
        ...prev,
        analysis: {
          featureImportance: shap.feature_importance,
          plots: {
            ...shap.plots,
            ...(curves?.roc_curve_plot ? { rocCurve: curves.roc_curve_plot } : {}),
            ...(curves?.pr_curve_plot ? { prCurve: curves.pr_curve_plot } : {}),
            ...(study?.optimization_history_plot
              ? { optimizationHistory: study.optimization_history_plot }
              : {}),
            ...(study?.param_importances_plot
              ? { paramImportances: study.param_importances_plot }
              : {}),
          },
          metrics: metrics?.metrics ?? null,
          confusionMatrixPlot: metrics?.confusion_matrix_plot ?? null,
          rocAuc: curves?.roc_auc ?? null,
          averagePrecision: curves?.average_precision ?? null,
        },
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed');
    } finally {
      setIsGenerating(false);
    }
  };

  // Auto-run once on entering the step when results exist and none yet.
  // biome-ignore lint/correctness/useExhaustiveDependencies: run once on mount.
  useEffect(() => {
    if (!autoRan.current && optimization.executionId && !hasSHAP) {
      autoRan.current = true;
      void runAnalysis();
    }
  }, []);

  const metrics = analysis.metrics ?? {};
  const shapTabs = SHAP_TABS.filter((t) => analysis.plots[t.id]);
  const hasCurves = !!analysis.plots.rocCurve || !!analysis.plots.prCurve;

  const rocAuc = analysis.rocAuc ?? (metrics.roc_auc_score as number | undefined) ?? null;

  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="shrink-0 space-y-4">
        <StepHeader
          step={5}
          title="Analysis & Visualizations"
          subtitle="Explain the selected model with metrics, SHAP, curves and study plots"
        />

        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted-foreground">
            Analyzing trial{' '}
            <span className="font-medium text-brand">#{optimization.selectedTrial ?? 'best'}</span>
            {optimization.bestValue !== null && (
              <>
                {' · '}best F1{' '}
                <span className="font-medium text-accent-emerald-foreground">
                  {optimization.bestValue.toFixed(4)}
                </span>
              </>
            )}
          </p>
          <Button type="button" size="sm" onClick={runAnalysis} disabled={isGenerating}>
            {isGenerating ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <BarChart3 className="h-4 w-4" />
            )}
            {hasSHAP ? 'Re-run analysis' : 'Run analysis'}
          </Button>
        </div>

        <ErrorBanner message={error} />

        <details className="rounded-lg border border-border bg-card">
          <summary className="cursor-pointer px-4 py-3 text-sm font-semibold">
            Advanced options
          </summary>
          <div className="grid grid-cols-1 gap-4 border-t border-border p-4 md:grid-cols-3">
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

        {/* Headline metric cards */}
        {analysis.metrics && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <MetricCard label="F1 score" value={metrics.f1_score} tone="emerald" />
            <MetricCard label="Accuracy" value={metrics.accuracy} tone="brand" />
            <MetricCard label="ROC AUC" value={rocAuc} tone="brand" />
            <MetricCard label="Precision" value={metrics.precision} />
            <MetricCard label="Recall" value={metrics.recall} />
          </div>
        )}
      </div>

      {isGenerating && !hasSHAP ? (
        <div className="flex min-h-0 flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Running analysis…
        </div>
      ) : hasSHAP ? (
        <Tabs.Root defaultValue="overview" className="flex min-h-0 flex-1 flex-col">
          <Tabs.List className="flex shrink-0 gap-1 overflow-x-auto border-b border-border">
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'shap', label: 'SHAP' },
              { id: 'curves', label: 'Curves' },
              { id: 'study', label: 'Study' },
              { id: 'importance', label: 'Feature importance' },
            ].map((t) => (
              <Tabs.Trigger
                key={t.id}
                value={t.id}
                className="whitespace-nowrap border-b-2 border-transparent px-4 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground data-[state=active]:border-brand data-[state=active]:text-brand"
              >
                {t.label}
              </Tabs.Trigger>
            ))}
          </Tabs.List>

          <div className="min-h-0 flex-1 overflow-y-auto pt-4">
            {/* Overview */}
            <Tabs.Content value="overview" className="space-y-4">
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
                {analysis.confusionMatrixPlot && (
                  <PlotCard
                    title="Confusion matrix"
                    image={analysis.confusionMatrixPlot}
                    filename="confusion-matrix.png"
                  />
                )}
                {analysis.plots.rocCurve && (
                  <PlotCard
                    title="ROC curve"
                    image={analysis.plots.rocCurve}
                    filename="roc-curve.png"
                  />
                )}
                {analysis.plots.prCurve && (
                  <PlotCard
                    title="Precision-recall"
                    image={analysis.plots.prCurve}
                    filename="pr-curve.png"
                  />
                )}
              </div>

              {analysis.metrics && <SecondaryMetrics metrics={metrics} />}

              {typeof metrics.classification_report === 'string' && (
                <details className="rounded-lg border border-border bg-card">
                  <summary className="cursor-pointer px-4 py-3 text-sm font-semibold">
                    Classification report
                  </summary>
                  <pre className="overflow-x-auto border-t border-border bg-muted p-4 font-mono text-xs">
                    {metrics.classification_report}
                  </pre>
                </details>
              )}
            </Tabs.Content>

            {/* SHAP */}
            <Tabs.Content value="shap">
              {shapTabs.length > 0 ? (
                <Tabs.Root defaultValue={shapTabs[0].id}>
                  <Tabs.List className="mb-4 flex gap-1 overflow-x-auto border-b border-border">
                    {shapTabs.map((t) => (
                      <Tabs.Trigger
                        key={t.id}
                        value={t.id}
                        className="whitespace-nowrap border-b-2 border-transparent px-3 py-2 text-sm font-medium text-muted-foreground transition-colors hover:text-foreground data-[state=active]:border-brand data-[state=active]:text-brand"
                      >
                        {t.label}
                      </Tabs.Trigger>
                    ))}
                  </Tabs.List>
                  {shapTabs.map((t) => (
                    <Tabs.Content key={t.id} value={t.id}>
                      <PlotCard
                        title={`SHAP ${t.label}`}
                        image={analysis.plots[t.id]}
                        filename={`shap-${t.id}.png`}
                      />
                    </Tabs.Content>
                  ))}
                </Tabs.Root>
              ) : (
                <EmptyState message="No SHAP plots available." />
              )}
            </Tabs.Content>

            {/* Curves */}
            <Tabs.Content value="curves">
              {hasCurves ? (
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  {analysis.plots.rocCurve && (
                    <PlotCard
                      title="ROC curve"
                      caption={rocAuc !== null ? `AUC ${rocAuc.toFixed(3)}` : undefined}
                      image={analysis.plots.rocCurve}
                      filename="roc-curve.png"
                    />
                  )}
                  {analysis.plots.prCurve && (
                    <PlotCard
                      title="Precision-recall curve"
                      caption={
                        analysis.averagePrecision !== null
                          ? `AP ${analysis.averagePrecision.toFixed(3)}`
                          : undefined
                      }
                      image={analysis.plots.prCurve}
                      filename="pr-curve.png"
                    />
                  )}
                </div>
              ) : (
                <EmptyState message="Curves unavailable for this model (needs binary probabilities)." />
              )}
            </Tabs.Content>

            {/* Study */}
            <Tabs.Content value="study">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {analysis.plots.optimizationHistory ? (
                  <PlotCard
                    title="Optimization history"
                    image={analysis.plots.optimizationHistory}
                    filename="optimization-history.png"
                  />
                ) : (
                  <EmptyState message="Optimization history not available." />
                )}
                {analysis.plots.paramImportances ? (
                  <PlotCard
                    title="Parameter importances"
                    image={analysis.plots.paramImportances}
                    filename="param-importances.png"
                  />
                ) : (
                  <EmptyState message="Parameter importances need ≥2 trials and ≥2 params." />
                )}
              </div>
            </Tabs.Content>

            {/* Feature importance */}
            <Tabs.Content value="importance">
              {analysis.featureImportance && analysis.featureImportance.length > 0 ? (
                <div className="rounded-lg border border-border bg-card">
                  <div className="border-b border-border bg-muted px-4 py-3">
                    <h4 className="text-sm font-semibold">Feature importance (mean |SHAP|)</h4>
                  </div>
                  <div className="space-y-3 p-4">
                    {analysis.featureImportance.map((item) => {
                      const max = analysis.featureImportance?.[0]?.importance || 1;
                      return (
                        <div key={item.feature} className="space-y-1">
                          <div className="flex items-center justify-between text-sm">
                            <span className="truncate font-medium">{item.feature}</span>
                            <span className="tabular-nums text-muted-foreground">
                              {item.importance.toFixed(3)}
                            </span>
                          </div>
                          <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
                            <div
                              className="h-full rounded-full bg-brand"
                              style={{ width: `${(item.importance / max) * 100}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <EmptyState message="No feature importance available." />
              )}
            </Tabs.Content>
          </div>
        </Tabs.Root>
      ) : (
        <div className="flex min-h-0 flex-1 items-center justify-center">
          <EmptyState message="Run the analysis to see metrics and plots." />
        </div>
      )}
    </div>
  );
}

function MetricCard({
  label,
  value,
  tone = 'default',
}: {
  label: string;
  value: unknown;
  tone?: 'default' | 'emerald' | 'brand' | 'amber';
}) {
  const display = typeof value === 'number' ? value.toFixed(3) : '—';
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <Metric value={display} tone={tone} className="mt-1 block text-2xl" />
    </div>
  );
}

function SecondaryMetrics({ metrics }: { metrics: Record<string, unknown> }) {
  const rows: Array<[string, unknown]> = [
    ['Average precision', metrics.average_precision_score],
    ['MCC', metrics.mcc],
    ["Cohen's kappa", metrics.cohens_kappa],
    ['Log loss', metrics.log_loss],
  ];
  return (
    <div className="rounded-lg border border-border bg-card">
      <div className="border-b border-border bg-muted px-4 py-3">
        <h4 className="text-sm font-semibold">Secondary metrics</h4>
      </div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 p-4 text-sm md:grid-cols-4">
        {rows.map(([k, v]) => (
          <div key={k} className="flex flex-col">
            <dt className="text-xs text-muted-foreground">{k}</dt>
            <dd className="font-medium tabular-nums">{fmt(v)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function PlotCard({
  title,
  image,
  filename,
  caption,
}: {
  title: string;
  image: string;
  filename: string;
  caption?: string;
}) {
  return (
    <div className="flex flex-col rounded-lg border border-border bg-card">
      <div className="flex items-center justify-between gap-2 border-b border-border bg-muted px-4 py-3">
        <h4 className="text-sm font-semibold">{title}</h4>
        {caption && <Badge variant="secondary">{caption}</Badge>}
      </div>
      <div className="p-4">
        <img src={image} alt={title} className="mx-auto max-h-[440px]" />
        <div className="mt-2 text-right">
          <Button
            type="button"
            variant="link"
            size="sm"
            onClick={() => downloadDataUrl(image, filename)}
          >
            <Download className="h-4 w-4" /> Download
          </Button>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return (
    <div className={cn('rounded-lg border border-dashed border-border p-8 text-center')}>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
