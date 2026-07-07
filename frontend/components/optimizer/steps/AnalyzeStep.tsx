'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Metric } from '@/components/ui/metric';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  type ConfusionMatrixData,
  type CurvesData,
  type FeatureImportanceData,
  type ShapData,
  generateFairness,
  generateSHAP,
  getConfusionMatrixData,
  getCurves,
  getCurvesData,
  getFeatureImportanceData,
  getMetrics,
  getShapData,
  getStudyPlots,
} from '@/lib/api';
import { cn } from '@/lib/utils';
import { BarChart3, Download, Loader2, Scale } from 'lucide-react';
import { Fragment, useEffect, useRef, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from 'recharts';
import { ErrorBanner } from '../NavButtons';
import { PlotSkeleton, PlotlyFigure } from '../PlotlyFigure';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';
import type { FairnessMetrics } from '../types';

const STUDY_PLOTS: Array<{ id: string; label: string; wide?: boolean }> = [
  { id: 'optimization_history', label: 'Optimization history', wide: true },
  { id: 'param_importances', label: 'Parameter importances' },
  { id: 'parallel_coordinate', label: 'Parallel coordinates', wide: true },
  { id: 'slice', label: 'Slice', wide: true },
  { id: 'timeline', label: 'Timeline' },
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

  const { optimization, analysis, features } = workflowData;
  const [isMitigating, setIsMitigating] = useState(false);
  const [fairnessError, setFairnessError] = useState<string | null>(null);
  // Plots may be absent even when metrics exist: the localStorage autosave
  // strips them when the payload exceeds the quota. Re-fetch in that case.
  const hasPlots = Object.keys(analysis.plots).length > 0;
  const hasSHAP = analysis.featureImportance !== null;
  const autoRan = useRef(false);

  // Native chart data (JSON endpoints); each falls back to the PNG plot when null.
  const [curvesData, setCurvesData] = useState<CurvesData | null>(null);
  const [confusionData, setConfusionData] = useState<ConfusionMatrixData | null>(null);
  const [importanceData, setImportanceData] = useState<FeatureImportanceData | null>(null);
  const [shapData, setShapData] = useState<ShapData | null>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: useProba/subsetSize are read at fetch time only.
  useEffect(() => {
    const id = optimization.executionId;
    if (!id) return;
    let cancelled = false;
    const body = { optimization_id: id, trial_number: optimization.selectedTrial ?? undefined };
    void Promise.all([
      getCurvesData({ ...body, use_proba: useProba, subset_size: subsetSize }).catch(() => null),
      getConfusionMatrixData(body).catch(() => null),
      getFeatureImportanceData(body).catch(() => null),
      getShapData({ ...body, subset_size: subsetSize }).catch(() => null),
    ]).then(([curves, cm, fi, shap]) => {
      if (cancelled) return;
      setCurvesData(curves);
      setConfusionData(cm);
      setImportanceData(fi);
      setShapData(shap);
    });
    return () => {
      cancelled = true;
    };
  }, [optimization.executionId, optimization.selectedTrial]);

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
    setFairnessError(null);
    try {
      const [shap, metrics, curves, study, fairness] = await Promise.all([
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
        // Runs only when a protected attribute was stored with the run or is
        // set in this session; the endpoint 400s harmlessly otherwise.
        generateFairness({
          optimization_id: id,
          trial_number: trial,
          sensitive_feature: features.sensitiveFeature ?? undefined,
        }).catch((err) => {
          setFairnessError(err instanceof Error ? err.message : 'Fairness audit failed');
          return null;
        }),
      ]);

      setWorkflowData((prev) => ({
        ...prev,
        analysis: {
          featureImportance: shap.feature_importance,
          plots: {
            ...shap.plots,
            ...(curves?.roc_curve_plot ? { rocCurve: curves.roc_curve_plot } : {}),
            ...(curves?.pr_curve_plot ? { prCurve: curves.pr_curve_plot } : {}),
          },
          studyPlots: study?.plots ?? null,
          metrics: metrics?.metrics ?? null,
          confusionMatrixPlot: metrics?.confusion_matrix_plot ?? null,
          rocAuc: curves?.roc_auc ?? null,
          averagePrecision: curves?.average_precision ?? null,
          fairness,
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
    if (!autoRan.current && optimization.executionId && (!hasSHAP || !hasPlots)) {
      autoRan.current = true;
      void runAnalysis();
    }
  }, []);

  const runFairness = async () => {
    if (!optimization.executionId) return;
    setIsMitigating(true);
    setFairnessError(null);
    try {
      const result = await generateFairness({
        optimization_id: optimization.executionId,
        trial_number: optimization.selectedTrial ?? undefined,
        sensitive_feature: features.sensitiveFeature ?? undefined,
      });
      setWorkflowData((prev) => ({
        ...prev,
        analysis: { ...prev.analysis, fairness: result },
      }));
    } catch (err) {
      setFairnessError(err instanceof Error ? err.message : 'Fairness audit failed');
    } finally {
      setIsMitigating(false);
    }
  };

  const runMitigation = async () => {
    if (!optimization.executionId || !analysis.fairness) return;
    setIsMitigating(true);
    setError(null);
    try {
      const result = await generateFairness({
        optimization_id: optimization.executionId,
        trial_number: optimization.selectedTrial ?? undefined,
        sensitive_feature: analysis.fairness.sensitive_feature,
        mitigate: true,
      });
      setWorkflowData((prev) => ({
        ...prev,
        analysis: { ...prev.analysis, fairness: result },
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Mitigation failed');
    } finally {
      setIsMitigating(false);
    }
  };

  const metrics = analysis.metrics ?? {};
  const hasCurves =
    !!curvesData?.roc || !!curvesData?.pr || !!analysis.plots.rocCurve || !!analysis.plots.prCurve;

  const rocAuc = analysis.rocAuc ?? (metrics.roc_auc_score as number | undefined) ?? null;

  return (
    <div className="flex flex-col gap-4">
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
          <div className="space-y-2">
            <Label htmlFor="analyze-subset-size">Subset size</Label>
            <Input
              id="analyze-subset-size"
              type="number"
              min={10}
              max={500}
              value={subsetSize}
              onChange={(e) => setSubsetSize(Number(e.target.value))}
            />
          </div>
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

      {isGenerating && !hasSHAP ? (
        <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" /> Running analysis…
        </div>
      ) : hasSHAP ? (
        <Tabs defaultValue="overview" className="gap-4">
          <TabsList
            variant="line"
            className="h-auto w-full flex-wrap justify-start border-b border-border"
          >
            {[
              { id: 'overview', label: 'Overview' },
              { id: 'shap', label: 'SHAP' },
              { id: 'curves', label: 'Curves' },
              { id: 'study', label: 'Study' },
              { id: 'fairness', label: 'Fairness' },
              { id: 'importance', label: 'Feature importance' },
            ].map((t) => (
              <TabsTrigger key={t.id} value={t.id} className="flex-none px-3 py-2 text-sm">
                {t.label}
              </TabsTrigger>
            ))}
          </TabsList>

          {/* Overview */}
          <TabsContent value="overview" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
              {confusionData ? (
                <ConfusionMatrixCard data={confusionData} />
              ) : (
                analysis.confusionMatrixPlot && (
                  <PlotCard
                    title="Confusion matrix"
                    image={analysis.confusionMatrixPlot}
                    filename="confusion-matrix.png"
                  />
                )
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
          </TabsContent>

          {/* SHAP */}
          <TabsContent value="shap">
            {shapData || Object.keys(analysis.plots).length > 0 ? (
              <div className="grid gap-4 md:grid-cols-2">
                {shapData ? (
                  <BeeswarmCard shap={shapData} />
                ) : (
                  analysis.plots.beeswarm && (
                    <PlotCard
                      title="SHAP beeswarm"
                      image={analysis.plots.beeswarm}
                      filename="shap-beeswarm.png"
                      className="md:col-span-2"
                    />
                  )
                )}
                {importanceData && importanceData.features.length > 0 ? (
                  <FeatureImportanceCard data={importanceData} />
                ) : shapData ? (
                  <FeatureImportanceCard data={importanceFromShap(shapData)} />
                ) : (
                  analysis.plots.bar && (
                    <PlotCard title="SHAP bar" image={analysis.plots.bar} filename="shap-bar.png" />
                  )
                )}
                {shapData ? (
                  <HeatmapCard shap={shapData} />
                ) : (
                  analysis.plots.heatmap && (
                    <PlotCard
                      title="SHAP heatmap"
                      image={analysis.plots.heatmap}
                      filename="shap-heatmap.png"
                    />
                  )
                )}
                {shapData ? (
                  <WaterfallCard
                    shap={shapData}
                    sampleIndex={sampleIndex}
                    onSampleIndexChange={setSampleIndex}
                  />
                ) : (
                  analysis.plots.waterfall && (
                    <PlotCard
                      title="SHAP waterfall"
                      image={analysis.plots.waterfall}
                      filename="shap-waterfall.png"
                    />
                  )
                )}
                {analysis.plots.violin && (
                  <PlotCard
                    title="SHAP violin"
                    image={analysis.plots.violin}
                    filename="shap-violin.png"
                  />
                )}
              </div>
            ) : (
              <EmptyState message="No SHAP plots available." />
            )}
          </TabsContent>

          {/* Curves */}
          <TabsContent value="curves">
            {hasCurves ? (
              <div className="grid gap-4 md:grid-cols-2">
                {curvesData?.roc ? (
                  <RocCurveCard roc={curvesData.roc} />
                ) : (
                  analysis.plots.rocCurve && (
                    <PlotCard
                      title="ROC curve"
                      caption={rocAuc !== null ? `AUC ${rocAuc.toFixed(3)}` : undefined}
                      image={analysis.plots.rocCurve}
                      filename="roc-curve.png"
                    />
                  )
                )}
                {curvesData?.pr ? (
                  <PrCurveCard pr={curvesData.pr} />
                ) : (
                  analysis.plots.prCurve && (
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
                  )
                )}
              </div>
            ) : (
              <EmptyState message="Curves unavailable for this model (needs binary probabilities)." />
            )}
          </TabsContent>

          {/* Study */}
          <TabsContent value="study">
            {analysis.studyPlots ? (
              <div className="grid gap-4 md:grid-cols-2">
                {STUDY_PLOTS.map(({ id, label, wide }) => {
                  const figure = analysis.studyPlots?.[id];
                  const span = wide ? 'md:col-span-2' : undefined;
                  return figure ? (
                    <PlotlyFigure key={id} title={label} figure={figure} className={span} />
                  ) : (
                    <EmptyState
                      key={id}
                      message={`${label} not available for this study.`}
                      className={span}
                    />
                  );
                })}
              </div>
            ) : isGenerating ? (
              <div className="grid gap-4 md:grid-cols-2">
                {STUDY_PLOTS.map(({ id, label, wide }) => (
                  <PlotSkeleton
                    key={id}
                    title={label}
                    className={wide ? 'md:col-span-2' : undefined}
                  />
                ))}
              </div>
            ) : (
              <EmptyState message="Study plots not available." />
            )}
          </TabsContent>

          {/* Fairness */}
          <TabsContent value="fairness" className="space-y-4">
            {analysis.fairness ? (
              <>
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-sm text-muted-foreground">
                    Audit grouped by{' '}
                    <span className="font-medium text-brand">
                      {analysis.fairness.sensitive_feature}
                    </span>{' '}
                    (fairlearn)
                  </p>
                  {!analysis.fairness.mitigation && (
                    <Button type="button" size="sm" onClick={runMitigation} disabled={isMitigating}>
                      {isMitigating ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Scale className="h-4 w-4" />
                      )}
                      Run mitigation (ThresholdOptimizer)
                    </Button>
                  )}
                </div>

                <DisparitySummary disparities={analysis.fairness.metrics.disparities} />
                <GroupMetricsTable metrics={analysis.fairness.metrics} />

                {Object.keys(analysis.fairness.metrics.by_group).length > 0 ? (
                  <GroupMetricsChart metrics={analysis.fairness.metrics} />
                ) : (
                  <div className="grid gap-4 md:grid-cols-2">
                    {Object.entries(analysis.fairness.plots).map(([name, image]) => (
                      <PlotCard
                        key={name}
                        title={`${name.replace(/_/g, ' ')} by group`}
                        image={image}
                        filename={`fairness-${name}.png`}
                      />
                    ))}
                  </div>
                )}

                {analysis.fairness.mitigation && (
                  <Card>
                    <CardHeader>
                      <CardTitle>
                        Mitigation — ThresholdOptimizer (
                        {analysis.fairness.mitigation.constraint.replace(/_/g, ' ')})
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid gap-4 md:grid-cols-2">
                        <MitigationDisparities
                          title="Before"
                          disparities={analysis.fairness.mitigation.before.disparities}
                        />
                        <MitigationDisparities
                          title="After"
                          disparities={analysis.fairness.mitigation.after.disparities}
                        />
                      </div>
                      {Object.keys(analysis.fairness.mitigation.before.disparities).length > 0 ? (
                        <MitigationComparisonChart
                          before={analysis.fairness.mitigation.before.disparities}
                          after={analysis.fairness.mitigation.after.disparities}
                        />
                      ) : (
                        <PlotCard
                          title="Disparity before vs after"
                          image={analysis.fairness.mitigation.comparison_plot}
                          filename="fairness-mitigation.png"
                        />
                      )}
                    </CardContent>
                  </Card>
                )}
              </>
            ) : (
              <div className="space-y-3">
                <EmptyState
                  message={
                    fairnessError
                      ? `Fairness audit failed: ${fairnessError}`
                      : features.sensitiveFeature
                        ? `Protected attribute "${features.sensitiveFeature}" selected — run the audit below.`
                        : 'No fairness audit available. Select a protected attribute in the Features step, then run the audit.'
                  }
                />
                {features.sensitiveFeature && (
                  <div className="text-center">
                    <Button type="button" size="sm" onClick={runFairness} disabled={isMitigating}>
                      {isMitigating ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Scale className="h-4 w-4" />
                      )}
                      Run fairness audit
                    </Button>
                  </div>
                )}
              </div>
            )}
          </TabsContent>

          {/* Feature importance */}
          <TabsContent value="importance">
            {importanceData && importanceData.features.length > 0 ? (
              <FeatureImportanceCard data={importanceData} />
            ) : analysis.featureImportance && analysis.featureImportance.length > 0 ? (
              <Card>
                <CardHeader>
                  <CardTitle>Feature importance (mean |SHAP|)</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
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
                </CardContent>
              </Card>
            ) : (
              <EmptyState message="No feature importance available." />
            )}
          </TabsContent>
        </Tabs>
      ) : (
        <div className="py-16">
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
    <Card size="sm">
      <CardContent>
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
        <Metric value={display} tone={tone} className="mt-1 block text-2xl" />
      </CardContent>
    </Card>
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
    <Card>
      <CardHeader>
        <CardTitle>Secondary metrics</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm md:grid-cols-4">
          {rows.map(([k, v]) => (
            <div key={k} className="flex flex-col">
              <dt className="text-xs text-muted-foreground">{k}</dt>
              <dd className="font-medium tabular-nums">{fmt(v)}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}

function PlotCard({
  title,
  image,
  filename,
  caption,
  className,
}: {
  title: string;
  image: string;
  filename: string;
  caption?: string;
  className?: string;
}) {
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardAction className="flex items-center gap-2">
          {caption && <Badge variant="secondary">{caption}</Badge>}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => downloadDataUrl(image, filename)}
            aria-label={`Download ${title}`}
          >
            <Download className="h-4 w-4" />
          </Button>
        </CardAction>
      </CardHeader>
      <CardContent>
        <img src={image} alt={title} className="h-auto w-full rounded-md" />
      </CardContent>
    </Card>
  );
}

const axisTick = { fontSize: 12 } as const;
const pct = (v: number) => v.toFixed(1);

function RocCurveCard({ roc }: { roc: NonNullable<CurvesData['roc']> }) {
  const data = roc.fpr.map((fpr, i) => ({ fpr, tpr: roc.tpr[i] }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>ROC curve</CardTitle>
        <CardDescription>True vs false positive rate · AUC {roc.auc.toFixed(3)}</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={{ tpr: { label: 'TPR', color: 'var(--chart-1)' } }}>
          <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              tickFormatter={pct}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              width={34}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              tickFormatter={pct}
            />
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              strokeDasharray="4 4"
              ifOverflow="hidden"
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  indicator="line"
                  labelFormatter={(_, payload) =>
                    `FPR ${Number(payload?.[0]?.payload?.fpr ?? 0).toFixed(3)}`
                  }
                />
              }
            />
            <Line
              dataKey="tpr"
              type="monotone"
              stroke="var(--color-tpr)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

function PrCurveCard({ pr }: { pr: NonNullable<CurvesData['pr']> }) {
  const data = pr.recall
    .map((recall, i) => ({ recall, precision: pr.precision[i] }))
    .sort((a, b) => a.recall - b.recall);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Precision-recall curve</CardTitle>
        <CardDescription>
          Precision vs recall · AP {pr.average_precision.toFixed(3)}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={{ precision: { label: 'Precision', color: 'var(--chart-2)' } }}>
          <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              dataKey="recall"
              type="number"
              domain={[0, 1]}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              tickFormatter={pct}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              width={34}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              tickFormatter={pct}
            />
            <ChartTooltip
              content={
                <ChartTooltipContent
                  indicator="line"
                  labelFormatter={(_, payload) =>
                    `Recall ${Number(payload?.[0]?.payload?.recall ?? 0).toFixed(3)}`
                  }
                />
              }
            />
            <Line
              dataKey="precision"
              type="monotone"
              stroke="var(--color-precision)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

function ConfusionMatrixCard({ data }: { data: ConfusionMatrixData }) {
  const n = data.labels.length;
  const colSums = data.labels.map((_, j) => data.matrix.reduce((s, row) => s + (row[j] ?? 0), 0));
  const rowSums = data.matrix.map((row) => row.reduce((s, c) => s + c, 0));
  const perClass = data.labels.map((label, i) => ({
    label,
    precision: colSums[i] > 0 ? (data.matrix[i]?.[i] ?? 0) / colSums[i] : null,
    recall: rowSums[i] > 0 ? (data.matrix[i]?.[i] ?? 0) / rowSums[i] : null,
  }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Confusion matrix</CardTitle>
        <CardDescription>Rows: actual · columns: predicted</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex items-stretch gap-1">
          <div className="flex w-5 items-center justify-center">
            <span className="-rotate-90 whitespace-nowrap text-xs font-medium text-muted-foreground">
              Actual
            </span>
          </div>
          <div
            className="grid flex-1 gap-1 text-xs"
            style={{ gridTemplateColumns: `auto repeat(${n}, minmax(0, 1fr))` }}
          >
            <div />
            <div
              className="pb-0.5 text-center font-medium text-muted-foreground"
              style={{ gridColumn: `span ${n}` }}
            >
              Predicted
            </div>
            <div />
            {data.labels.map((label) => (
              <div
                key={label}
                className="truncate px-1 pb-1 text-center font-medium text-muted-foreground"
                title={label}
              >
                {label}
              </div>
            ))}
            {data.matrix.map((row, i) => (
              <Fragment key={data.labels[i]}>
                <div className="flex items-center justify-end pr-2 font-medium text-muted-foreground">
                  {data.labels[i]}
                </div>
                {row.map((count, j) => {
                  const norm = data.normalized[i]?.[j] ?? 0;
                  return (
                    <div
                      key={data.labels[j]}
                      className={cn(
                        'flex aspect-square max-h-28 flex-col items-center justify-center rounded-md border border-border/50 tabular-nums',
                        norm > 0.5 && 'text-white'
                      )}
                      style={{
                        backgroundColor: `color-mix(in oklab, var(--chart-1) ${Math.round(norm * 70)}%, var(--card))`,
                      }}
                      title={`Actual ${data.labels[i]} · predicted ${data.labels[j]}: ${count} (${(norm * 100).toFixed(1)}%)`}
                    >
                      <span className="text-sm font-semibold">{count}</span>
                      <span
                        className={cn(
                          'text-[10px]',
                          norm > 0.5 ? 'text-white/80' : 'text-muted-foreground'
                        )}
                      >
                        {(norm * 100).toFixed(1)}%
                      </span>
                    </div>
                  );
                })}
              </Fragment>
            ))}
          </div>
        </div>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
          {perClass.map(({ label, precision, recall }) => (
            <span key={label} className="tabular-nums">
              <span className="font-medium text-foreground">{label}</span>
              {' · P '}
              {precision === null ? '—' : precision.toFixed(3)}
              {' · R '}
              {recall === null ? '—' : recall.toFixed(3)}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function FeatureImportanceCard({ data }: { data: FeatureImportanceData }) {
  const rows = data.features
    .map((feature, i) => ({ feature, importance: data.importances[i] ?? 0 }))
    .sort((a, b) => b.importance - a.importance);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Feature importance</CardTitle>
        <CardDescription>Mean |SHAP| value per feature</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{ importance: { label: 'Importance', color: 'var(--chart-1)' } }}
          className="aspect-auto w-full"
          style={{ height: Math.min(350, Math.max(160, rows.length * 32)) }}
        >
          <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 12, left: 8 }}>
            <CartesianGrid horizontal={false} />
            <XAxis type="number" tickLine={false} axisLine={false} tick={axisTick} />
            <YAxis
              type="category"
              dataKey="feature"
              width={120}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
            />
            <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
            <Bar
              dataKey="importance"
              fill="var(--color-importance)"
              radius={4}
              isAnimationActive={false}
            />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

/** Mean |SHAP| per feature, shaped like the feature-importance endpoint payload. */
function importanceFromShap(shap: ShapData): FeatureImportanceData {
  const importances = shap.feature_names.map(
    (_, j) =>
      shap.values.reduce((sum, row) => sum + Math.abs(row[j] ?? 0), 0) /
      Math.max(1, shap.values.length)
  );
  return { features: shap.feature_names, importances };
}

/** Indices of the top-n features by mean |SHAP|, descending. */
function topFeatureIndices(shap: ShapData, n: number): number[] {
  const { importances } = importanceFromShap(shap);
  return importances
    .map((importance, index) => ({ importance, index }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, n)
    .map((r) => r.index);
}

const shapFmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3));

interface BeePoint {
  x: number;
  y: number;
  feature: string;
  featureValue: number;
  fill: string;
}

function BeeswarmCard({ shap }: { shap: ShapData }) {
  const top = topFeatureIndices(shap, 10);
  const names = top.map((fi) => shap.feature_names[fi]);
  const points: BeePoint[] = [];
  top.forEach((fi, row) => {
    const shapVals = shap.values.map((r) => r[fi] ?? 0);
    const rawVals = shap.data.map((r) => r[fi] ?? 0);
    const min = Math.min(...shapVals);
    const max = Math.max(...shapVals);
    const binSize = Math.max((max - min) * 0.05, 1e-9);
    const rawMin = Math.min(...rawVals);
    const rawMax = Math.max(...rawVals);
    const rawRange = rawMax - rawMin;
    // Deterministic beeswarm jitter: bin SHAP values, then stack points within
    // a bin at alternating ± offsets from the feature's band center.
    const binCounts = new Map<number, number>();
    shapVals.forEach((x, s) => {
      const bin = Math.round(x / binSize);
      const k = binCounts.get(bin) ?? 0;
      binCounts.set(bin, k + 1);
      const offset = Math.min(Math.ceil(k / 2) * 0.09, 0.38) * (k % 2 === 1 ? 1 : -1);
      const raw = rawVals[s];
      const norm = rawRange > 0 ? (raw - rawMin) / rawRange : 0.5;
      points.push({
        x,
        y: row + offset,
        feature: shap.feature_names[fi],
        featureValue: raw,
        fill: `color-mix(in oklab, var(--chart-1) ${Math.round(norm * 100)}%, var(--chart-2))`,
      });
    });
  });
  return (
    <Card className="md:col-span-2">
      <CardHeader>
        <CardTitle>SHAP beeswarm</CardTitle>
        <CardDescription>
          Per-sample SHAP values for the top {top.length} features · color: feature value (low →
          high)
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{}}
          className="aspect-auto w-full"
          style={{ height: Math.max(160, top.length * 30 + 40) }}
        >
          <ScatterChart margin={{ top: 8, right: 12, left: 8, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis
              type="number"
              dataKey="x"
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[-0.5, top.length - 0.5]}
              ticks={names.map((_, i) => i)}
              tickFormatter={(v: number) => names[Math.round(v)] ?? ''}
              width={130}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              reversed
            />
            <ReferenceLine x={0} strokeDasharray="4 4" />
            <ChartTooltip
              cursor={false}
              content={({ active, payload }) => {
                const p = payload?.[0]?.payload as BeePoint | undefined;
                if (!active || !p) return null;
                return (
                  <div className="grid min-w-32 gap-1 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
                    <span className="font-medium">{p.feature}</span>
                    <span className="flex justify-between gap-3">
                      <span className="text-muted-foreground">SHAP</span>
                      <span className="font-mono tabular-nums">{shapFmt(p.x)}</span>
                    </span>
                    <span className="flex justify-between gap-3">
                      <span className="text-muted-foreground">Value</span>
                      <span className="font-mono tabular-nums">{shapFmt(p.featureValue)}</span>
                    </span>
                  </div>
                );
              }}
            />
            <Scatter data={points} isAnimationActive={false}>
              {points.map((p, i) => (
                <Cell key={`${p.feature}-${i}`} fill={p.fill} />
              ))}
            </Scatter>
          </ScatterChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

const HEATMAP_MAX_ROWS = 60;

function HeatmapCard({ shap }: { shap: ShapData }) {
  const top = topFeatureIndices(shap, 10);
  const rows = shap.values.slice(0, HEATMAP_MAX_ROWS);
  const maxAbs = Math.max(1e-9, ...rows.flatMap((row) => top.map((fi) => Math.abs(row[fi] ?? 0))));
  return (
    <Card>
      <CardHeader>
        <CardTitle>SHAP heatmap</CardTitle>
        <CardDescription>
          Samples × top {top.length} features · negative vs positive SHAP
          {shap.values.length > rows.length ? ` · first ${rows.length} samples` : ''}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className="grid gap-px"
          style={{ gridTemplateColumns: `repeat(${top.length}, minmax(0, 1fr))` }}
        >
          {top.map((fi) => (
            <div
              key={fi}
              className="truncate pb-1 text-center text-[10px] font-medium text-muted-foreground"
              title={shap.feature_names[fi]}
            >
              {shap.feature_names[fi]}
            </div>
          ))}
          {rows.map((row, s) =>
            top.map((fi) => {
              const v = row[fi] ?? 0;
              const intensity = Math.round((Math.abs(v) / maxAbs) * 85);
              const hue = v >= 0 ? 'var(--chart-1)' : 'var(--chart-2)';
              return (
                <div
                  key={`${s}-${fi}`}
                  className="h-2.5 rounded-[2px]"
                  style={{
                    backgroundColor: `color-mix(in oklab, ${hue} ${intensity}%, var(--card))`,
                  }}
                  title={`Sample ${s} · ${shap.feature_names[fi]}: ${shapFmt(v)}`}
                />
              );
            })
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function WaterfallCard({
  shap,
  sampleIndex,
  onSampleIndexChange,
}: {
  shap: ShapData;
  sampleIndex: number;
  onSampleIndexChange: (index: number) => void;
}) {
  const idx = Math.min(Math.max(0, sampleIndex), Math.max(0, shap.values.length - 1));
  const row = shap.values[idx] ?? [];
  const raw = shap.data[idx] ?? [];
  const contributions = shap.feature_names
    .map((feature, j) => ({ feature, value: row[j] ?? 0, raw: raw[j] }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const top = contributions.slice(0, 10);
  const rest = contributions.slice(10);
  const steps = [
    ...top.map((c) => ({
      name: `${c.feature} = ${typeof c.raw === 'number' ? shapFmt(c.raw) : '—'}`,
      value: c.value,
    })),
    ...(rest.length > 0
      ? [
          {
            name: `other (${rest.length} features)`,
            value: rest.reduce((s, c) => s + c.value, 0),
          },
        ]
      : []),
  ];
  let cursor = shap.base_value;
  const bars = steps.map((s) => {
    const start = cursor;
    cursor += s.value;
    return {
      name: s.name,
      range: [start, cursor] as [number, number],
      delta: s.value,
      fill: s.value >= 0 ? 'var(--chart-1)' : 'var(--chart-2)',
    };
  });
  const prediction = cursor;
  bars.push({
    name: `f(x) = ${shapFmt(prediction)}`,
    range: [shap.base_value, prediction],
    delta: prediction - shap.base_value,
    fill: prediction - shap.base_value >= 0 ? 'var(--chart-1)' : 'var(--chart-2)',
  });
  return (
    <Card>
      <CardHeader>
        <CardTitle>SHAP waterfall</CardTitle>
        <CardDescription>
          Sample {idx} · base value E[f(X)] = {shapFmt(shap.base_value)}
        </CardDescription>
        <CardAction className="flex items-center gap-2">
          <Label htmlFor="analyze-sample-index" className="text-xs text-muted-foreground">
            Sample
          </Label>
          <Input
            id="analyze-sample-index"
            type="number"
            min={0}
            max={Math.max(0, shap.values.length - 1)}
            value={sampleIndex}
            onChange={(e) => onSampleIndexChange(Number(e.target.value))}
            className="h-8 w-20"
          />
        </CardAction>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{ range: { label: 'Contribution', color: 'var(--chart-1)' } }}
          className="aspect-auto w-full"
          style={{ height: Math.max(200, bars.length * 30 + 40) }}
        >
          <BarChart data={bars} layout="vertical" margin={{ top: 4, right: 12, left: 8 }}>
            <CartesianGrid horizontal={false} />
            <XAxis
              type="number"
              domain={['auto', 'auto']}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
              tickFormatter={(v: number) => v.toFixed(2)}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tickLine={false}
              axisLine={false}
              tick={axisTick}
            />
            <ReferenceLine x={shap.base_value} strokeDasharray="4 4" />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  hideIndicator
                  formatter={(_, __, item) => {
                    const p = item.payload as (typeof bars)[number];
                    return (
                      <div className="flex w-full justify-between gap-3">
                        <span className="text-muted-foreground">contribution</span>
                        <span className="font-mono font-medium tabular-nums">
                          {p.delta >= 0 ? '+' : ''}
                          {shapFmt(p.delta)}
                        </span>
                      </div>
                    );
                  }}
                />
              }
            />
            <Bar dataKey="range" radius={4} isAnimationActive={false}>
              {bars.map((b) => (
                <Cell key={b.name} fill={b.fill} />
              ))}
            </Bar>
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

const GROUP_COLORS = [
  'var(--chart-1)',
  'var(--chart-2)',
  'var(--chart-3)',
  'var(--chart-4)',
  'var(--chart-5)',
];

function GroupMetricsChart({ metrics }: { metrics: FairnessMetrics }) {
  const metricNames = Object.keys(metrics.by_group).filter((name) => name !== 'count');
  const groups = metricNames.length > 0 ? Object.keys(metrics.by_group[metricNames[0]]) : [];
  if (metricNames.length === 0 || groups.length === 0) return null;
  const data = metricNames.map((name) => ({
    metric: name.replace(/_/g, ' '),
    ...Object.fromEntries(groups.map((g) => [g, metrics.by_group[name][g] ?? 0])),
  }));
  const config = Object.fromEntries(
    groups.map((g, i) => [g, { label: g, color: GROUP_COLORS[i % GROUP_COLORS.length] }])
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle>Metric comparison by group</CardTitle>
        <CardDescription>One bar per group for each fairness metric</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={config} className="aspect-auto w-full" style={{ height: 300 }}>
          <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="metric" tickLine={false} axisLine={false} tick={axisTick} />
            <YAxis width={40} tickLine={false} axisLine={false} tick={axisTick} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            {groups.map((g) => (
              <Bar
                key={g}
                dataKey={g}
                fill={`var(--color-${g})`}
                radius={4}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

function MitigationComparisonChart({
  before,
  after,
}: {
  before: Record<string, number>;
  after: Record<string, number>;
}) {
  const data = Object.keys(before).map((key) => ({
    metric: DISPARITY_LABELS[key] ?? key.replace(/_/g, ' '),
    before: before[key] ?? 0,
    after: after[key] ?? 0,
  }));
  return (
    <Card>
      <CardHeader>
        <CardTitle>Disparity before vs after</CardTitle>
        <CardDescription>ThresholdOptimizer effect on disparity metrics</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer
          config={{
            before: { label: 'Before', color: 'var(--chart-2)' },
            after: { label: 'After', color: 'var(--chart-1)' },
          }}
          className="aspect-auto w-full"
          style={{ height: 280 }}
        >
          <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="metric" tickLine={false} axisLine={false} tick={axisTick} />
            <YAxis width={40} tickLine={false} axisLine={false} tick={axisTick} />
            <ChartTooltip content={<ChartTooltipContent />} />
            <ChartLegend content={<ChartLegendContent />} />
            <Bar dataKey="before" fill="var(--color-before)" radius={4} isAnimationActive={false} />
            <Bar dataKey="after" fill="var(--color-after)" radius={4} isAnimationActive={false} />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  );
}

const DISPARITY_LABELS: Record<string, string> = {
  demographic_parity_difference: 'Demographic parity diff',
  demographic_parity_ratio: 'Demographic parity ratio',
  equalized_odds_difference: 'Equalized odds diff',
};

function DisparitySummary({ disparities }: { disparities: Record<string, number> }) {
  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
      {Object.entries(disparities).map(([key, value]) => {
        // Ratio: 1 is ideal (≥0.8 commonly acceptable); differences: 0 is ideal.
        const isRatio = key.endsWith('_ratio');
        const concerning = isRatio ? value < 0.8 : value > 0.1;
        return (
          <Card key={key} size="sm">
            <CardContent>
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {DISPARITY_LABELS[key] ?? key}
              </p>
              <Metric
                value={value.toFixed(3)}
                tone={concerning ? 'amber' : 'emerald'}
                className="mt-1 block text-2xl"
              />
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}

function GroupMetricsTable({ metrics }: { metrics: FairnessMetrics }) {
  const metricNames = Object.keys(metrics.by_group);
  const groups = metricNames.length > 0 ? Object.keys(metrics.by_group[metricNames[0]]) : [];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Metrics by group</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted-foreground">
              <th className="px-4 py-2 font-medium">Metric</th>
              {groups.map((g) => (
                <th key={g} className="px-4 py-2 font-medium">
                  {g}
                </th>
              ))}
              <th className="px-4 py-2 font-medium">Overall</th>
            </tr>
          </thead>
          <tbody>
            {metricNames.map((name) => (
              <tr key={name} className="border-b border-border last:border-0">
                <td className="px-4 py-2 font-medium">{name.replace(/_/g, ' ')}</td>
                {groups.map((g) => (
                  <td key={g} className="px-4 py-2 tabular-nums">
                    {name === 'count'
                      ? metrics.by_group[name][g]
                      : metrics.by_group[name][g]?.toFixed(3)}
                  </td>
                ))}
                <td className="px-4 py-2 tabular-nums text-muted-foreground">
                  {name === 'count' ? metrics.overall[name] : metrics.overall[name]?.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function MitigationDisparities({
  title,
  disparities,
}: {
  title: string;
  disparities: Record<string, number>;
}) {
  return (
    <div className="rounded-lg border border-border bg-muted/40 p-3">
      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{title}</p>
      <dl className="mt-2 space-y-1 text-sm">
        {Object.entries(disparities).map(([key, value]) => (
          <div key={key} className="flex items-center justify-between gap-2">
            <dt className="text-muted-foreground">{DISPARITY_LABELS[key] ?? key}</dt>
            <dd className="font-medium tabular-nums">{value.toFixed(3)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function EmptyState({ message, className }: { message: string; className?: string }) {
  return (
    <div className={cn('rounded-lg border border-dashed border-border p-8 text-center', className)}>
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  );
}
