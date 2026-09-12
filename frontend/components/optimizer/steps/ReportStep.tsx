'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Field, FieldDescription, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { StatusDot } from '@/components/ui/status-dot';
import { Textarea } from '@/components/ui/textarea';
import {
  type PersistedReport,
  type ReportResponse,
  downloadResearchBundle,
  generateReport,
  getReportContext,
  listSnapshotReports,
} from '@/lib/api';
import { buildFigureMap, resolveFigureSrc } from '@/lib/figures';
import { loadReportSettings, reportSettingsPayload } from '@/lib/reportSettings';
import { type ApiKeys, loadApiKeys } from '@/lib/settings';
import {
  Check,
  ChevronRight,
  Copy,
  Download,
  FileArchive,
  FileJson,
  FileText,
  Loader2,
  Settings2,
} from 'lucide-react';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ErrorBanner } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

type Provider = 'google' | 'openai' | 'anthropic';

// Current GA model IDs (verified June 2026). The backend passes model_name straight to LiteLLM,
// so any valid provider ID works; "Custom…" lets users enter newer ones.
const PROVIDER_MODELS: Record<Provider, { label: string; models: string[] }> = {
  google: {
    label: 'Google (Gemini)',
    models: ['gemini-3.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite'],
  },
  openai: { label: 'OpenAI', models: ['gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini'] },
  anthropic: {
    label: 'Anthropic (Claude)',
    models: ['claude-fable-5', 'claude-opus-4-8', 'claude-sonnet-5', 'claude-haiku-4-5-20251001'],
  },
};

export function ReportStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<Provider>('google');
  const [modelName, setModelName] = useState(PROVIDER_MODELS.google.models[0]);
  const [customModel, setCustomModel] = useState('');
  const [datasetDescription, setDatasetDescription] = useState('');
  const [keys, setKeys] = useState<ApiKeys>({ openai: '', anthropic: '', google: '' });
  const [copied, setCopied] = useState(false);
  // Diagnostics from the last generation: figures used/dropped, residual
  // markdown issues, and what the agent was actually given.
  const [diagnostics, setDiagnostics] = useState<ReportResponse | null>(null);
  // Every report persisted for this snapshot, newest first.
  const [history, setHistory] = useState<PersistedReport[]>([]);
  // Read once per generation so a mid-run Settings change can't half-apply.
  const [reportSettings, setReportSettings] = useState(loadReportSettings);

  useEffect(() => {
    loadApiKeys()
      .then(setKeys)
      .catch(() => undefined);
    setReportSettings(loadReportSettings());
  }, []);

  const { optimization, analysis, report } = workflowData;
  const apiKey = keys[provider];
  // `figures/<id>.png` references in the report resolve to these data URLs, so
  // the plots the agent wrote about render inline instead of breaking.
  const figureMap = useMemo(
    () =>
      buildFigureMap({
        plots: analysis.plots,
        confusionMatrixPlot: analysis.confusionMatrixPlot,
        fairness: analysis.fairness,
      }),
    [analysis.plots, analysis.confusionMatrixPlot, analysis.fairness]
  );
  // Guard against a non-string payload (e.g. structured LLM content blocks) —
  // ReactMarkdown throws on non-string children and would crash the whole app.
  const hasReport = typeof report.markdown === 'string' && report.markdown.length > 0;
  const effectiveModel = modelName === '__custom__' ? customModel.trim() : modelName;

  const refreshHistory = useCallback(async (): Promise<PersistedReport[]> => {
    if (!analysis.snapshotId) return [];
    const reports = await listSnapshotReports(analysis.snapshotId).catch(() => []);
    setHistory(reports);
    return reports;
  }, [analysis.snapshotId]);

  useEffect(() => {
    void refreshHistory();
  }, [refreshHistory]);

  useEffect(() => {
    if (!analysis.snapshotId || report.markdown) return;
    void listSnapshotReports(analysis.snapshotId)
      .then((reports) => {
        const latest = reports.find(
          (item) =>
            item.status === 'completed' && item.snapshot_revision === analysis.snapshotRevision
        );
        if (latest?.markdown) {
          setWorkflowData((prev) => ({ ...prev, report: { markdown: latest.markdown } }));
        }
      })
      .catch(() => undefined);
  }, [analysis.snapshotId, analysis.snapshotRevision, report.markdown, setWorkflowData]);

  useEffect(() => {
    setFooter({ canContinue: false, hideNext: true, backDisabled: isGenerating });
  }, [isGenerating, setFooter]);

  const handleProvider = (p: Provider) => {
    setProvider(p);
    setModelName(PROVIDER_MODELS[p].models[0]);
  };

  const run = async () => {
    if (!optimization.executionId) {
      setError('No optimization results available');
      return;
    }
    if (
      !analysis.snapshotId ||
      analysis.snapshotRevision === null ||
      analysis.status !== 'completed'
    ) {
      setError('Run and complete analysis before generating a report.');
      return;
    }
    if (!apiKey) {
      setError(`No ${provider} API key configured. Add one on the Settings page.`);
      return;
    }
    setIsGenerating(true);
    setError(null);
    const settings = loadReportSettings();
    setReportSettings(settings);
    try {
      const result = await generateReport({
        optimization_id: optimization.executionId,
        analysis_snapshot_id: analysis.snapshotId,
        analysis_revision: analysis.snapshotRevision,
        trial_number: optimization.selectedTrial ?? undefined,
        llm_provider: provider,
        api_key: apiKey,
        model_name: effectiveModel || PROVIDER_MODELS[provider].models[0],
        dataset_description: datasetDescription || undefined,
        // Evidence toggles and agent prompts from Settings → AI Report Generation.
        ...reportSettingsPayload(settings),
      });
      const markdown =
        typeof result.report_markdown === 'string'
          ? result.report_markdown
          : JSON.stringify(result.report_markdown);
      setWorkflowData((prev) => ({ ...prev, report: { markdown } }));
      setDiagnostics(result);
      // Pull the new row in so the superseded report stays reachable.
      void refreshHistory();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
      // A failed attempt is persisted with its error; surface it in the history.
      void refreshHistory();
    } finally {
      setIsGenerating(false);
    }
  };

  const saveBlob = (blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const download = () => {
    saveBlob(
      new Blob([report.markdown || ''], { type: 'text/markdown' }),
      `optimization-report-${Date.now()}.md`
    );
  };

  /**
   * One click, one archive: the structured context, the exact evidence the
   * agents read, every figure as a file, the CSV tables, the prompts and the
   * generated reports. `figures/<id>.png` links in the report resolve inside it.
   */
  const downloadBundle = async () => {
    if (!analysis.snapshotId) return;
    setIsDownloading(true);
    setError(null);
    try {
      const { blob, filename } = await downloadResearchBundle(analysis.snapshotId);
      saveBlob(blob, filename);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not download the research dump');
    } finally {
      setIsDownloading(false);
    }
  };

  /** The same evidence bundle as JSON, for scripted analysis or an appendix. */
  const downloadContextJson = async () => {
    if (!analysis.snapshotId) return;
    setIsDownloading(true);
    setError(null);
    try {
      const { context, evidence_markdown } = await getReportContext(analysis.snapshotId);
      saveBlob(
        new Blob([JSON.stringify({ context, evidence_markdown }, null, 2)], {
          type: 'application/json',
        }),
        `quoptuna-report-context-${analysis.snapshotId}.json`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not download the report context');
    } finally {
      setIsDownloading(false);
    }
  };

  const copy = async () => {
    if (!report.markdown) return;
    try {
      await navigator.clipboard.writeText(report.markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <div className="space-y-4">
      <StepHeader
        step={6}
        title="AI Report"
        subtitle="Generate a written summary of the run using your configured LLM provider"
      />

      <ErrorBanner message={error} />

      {/* Two-column layout: report flows with page scroll; controls live in a sticky rail. */}
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[1fr_320px]">
        {/* Report column */}
        <Card>
          <CardHeader>
            <CardTitle>Generated report</CardTitle>
            {hasReport && !isGenerating && (
              <CardAction className="flex items-center gap-2">
                <Button type="button" variant="ghost" size="sm" onClick={copy}>
                  {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
                <Button type="button" variant="secondary" size="sm" onClick={download}>
                  <Download className="h-4 w-4" /> Download .md
                </Button>
              </CardAction>
            )}
          </CardHeader>
          <CardContent>
            {/* Generation takes precedence over the previous report: showing stale
                prose while a new one is being written reads as if nothing is
                happening. The old report is persisted and stays reachable under
                "Report history" below. */}
            {isGenerating ? (
              <GeneratingReport
                model={effectiveModel || PROVIDER_MODELS[provider].models[0]}
                provider={PROVIDER_MODELS[provider].label}
                reviewed={reportSettings.enableReview}
                hadPrevious={hasReport}
              />
            ) : hasReport && report.markdown ? (
              <ReportMarkdown markdown={report.markdown} figures={figureMap} />
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
                <FileText className="h-8 w-8 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  {analysis.status === 'completed'
                    ? 'Generate a report from the persisted metrics and analysis artifacts.'
                    : 'Run and complete analysis before generating a report.'}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Report history sits under the report, in the same column. */}
        {history.length > 0 && (
          <div className="lg:col-start-1">
            <ReportHistory
              reports={history}
              currentRevision={analysis.snapshotRevision}
              currentMarkdown={report.markdown}
              figures={figureMap}
              onSave={saveBlob}
            />
          </div>
        )}

        {/* Controls rail */}
        <Card className="self-start lg:sticky lg:top-4">
          <CardHeader>
            <CardTitle>Report settings</CardTitle>
            <CardAction>
              <StatusDot
                status={apiKey ? 'online' : 'offline'}
                label={apiKey ? `${provider} key ready` : 'No API key'}
              />
            </CardAction>
          </CardHeader>
          <CardContent className="space-y-4">
            <Field>
              <FieldLabel htmlFor="report-provider">Provider</FieldLabel>
              <Select value={provider} onValueChange={(v) => handleProvider(v as Provider)}>
                <SelectTrigger id="report-provider" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(Object.keys(PROVIDER_MODELS) as Provider[]).map((p) => (
                    <SelectItem key={p} value={p}>
                      {PROVIDER_MODELS[p].label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>
            <Field>
              <FieldLabel htmlFor="report-model">Model</FieldLabel>
              <Select value={modelName} onValueChange={setModelName}>
                <SelectTrigger id="report-model" className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {PROVIDER_MODELS[provider].models.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                  <SelectItem value="__custom__">Custom…</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            {modelName === '__custom__' && (
              <Field>
                <FieldLabel htmlFor="report-custom-model">Custom model name</FieldLabel>
                <Input
                  id="report-custom-model"
                  type="text"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="e.g. gemini-2.0-flash"
                />
              </Field>
            )}
            <Field>
              <FieldLabel htmlFor="report-dataset-description">
                Dataset description (optional)
              </FieldLabel>
              <Textarea
                id="report-dataset-description"
                value={datasetDescription}
                onChange={(e) => setDatasetDescription(e.target.value)}
                rows={2}
                placeholder="Briefly describe the dataset and prediction goal…"
              />
            </Field>

            <div className="space-y-3 border-t border-border pt-4">
              <Button
                type="button"
                variant="brand"
                className="w-full"
                onClick={run}
                disabled={isGenerating || !apiKey || analysis.status !== 'completed'}
              >
                {isGenerating ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <FileText className="h-4 w-4" />
                )}
                {hasReport ? 'Regenerate report' : 'Generate report'}
              </Button>
              {!apiKey && (
                <p className="text-sm text-muted-foreground">
                  No {provider} key found.{' '}
                  <Link href="/settings" className="font-medium text-brand hover:underline">
                    Add one in Settings
                  </Link>
                  .
                </p>
              )}
              {isGenerating && (
                <p className="text-sm text-muted-foreground">
                  Sending the full run evidence &amp; plots to the model — this can take a minute.
                </p>
              )}

              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => void downloadBundle()}
                disabled={isDownloading || !analysis.snapshotId || analysis.status !== 'completed'}
              >
                {isDownloading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <FileArchive className="h-4 w-4" />
                )}
                Download research dump (.zip)
              </Button>
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => void downloadContextJson()}
                disabled={isDownloading || !analysis.snapshotId || analysis.status !== 'completed'}
              >
                <FileJson className="h-4 w-4" /> Download context (.json)
              </Button>
              <p className="text-sm text-muted-foreground">
                Everything behind the report in one archive: the structured context (
                <code className="font-mono text-xs">context.json</code>), the exact evidence the
                agents read, every figure as a file, CSV tables, the prompts used, and all generated
                reports. The JSON is the same bundle on its own, with the figure manifest instead of
                the image files.
              </p>

              <p className="text-sm text-muted-foreground">
                <Settings2 className="mr-1 inline h-3.5 w-3.5" />
                Prompts and what goes into the evidence bundle are configured in{' '}
                <Link href="/settings" className="font-medium text-brand hover:underline">
                  Settings → AI Report Generation
                </Link>
                .
              </p>
            </div>

            {diagnostics && <ReportDiagnostics result={diagnostics} />}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

/**
 * What the agents were given and what came back.
 *
 * Surfaced because the failure modes that matter are invisible in the prose: a
 * fairness-aware run whose search details were excluded, a Pareto front the
 * report never mentions, or figure references the agent invented.
 */
/**
 * Renders report markdown, resolving figure references to the in-memory plots.
 *
 * A report references its plots as `figures/<id>.png`, which is a real file
 * inside the downloaded research dump but not in the browser.
 */
function ReportMarkdown({
  markdown,
  figures,
}: {
  markdown: string;
  figures: Record<string, string>;
}) {
  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          table: ({ children }) => (
            <div className="my-4 overflow-x-auto rounded-md border border-border">
              <table className="m-0 w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border-b border-border bg-muted px-3 py-2 text-left font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-b border-border px-3 py-2 align-top last:border-r-0">
              {children}
            </td>
          ),
          img: ({ src, alt }) => {
            const resolved = resolveFigureSrc(typeof src === 'string' ? src : undefined, figures);
            if (!resolved) {
              return (
                <span className="text-sm text-muted-foreground italic">
                  [{alt || 'figure'} — not available in this view]
                </span>
              );
            }
            return (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={resolved}
                alt={alt || 'Report figure'}
                className="mx-auto my-3 max-w-full rounded-md border border-border bg-white"
              />
            );
          },
        }}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}

/**
 * Loading state shown in place of the previous report while a new one is written.
 *
 * States what the request actually is (provider, model, whether the reviewer
 * pass runs) instead of a bare spinner, since a two-agent generation can take a
 * minute and the wait is otherwise indistinguishable from a hang.
 */
function GeneratingReport({
  model,
  provider,
  reviewed,
  hadPrevious,
}: {
  model: string;
  provider: string;
  reviewed: boolean;
  hadPrevious: boolean;
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <Loader2 className="h-6 w-6 animate-spin text-brand" />
      <p className="font-medium text-sm">Writing report…</p>
      <p className="max-w-md text-sm text-muted-foreground">
        Sending the full run evidence and plots to <span className="font-mono">{model}</span> (
        {provider}).{' '}
        {reviewed
          ? 'A second reviewer pass then re-checks every number against the evidence, so this takes about twice as long.'
          : 'The reviewer pass is off, so this is a single call.'}
      </p>
      {hadPrevious && (
        <p className="text-sm text-muted-foreground">
          The previous report is kept under <span className="font-medium">Report history</span>{' '}
          below.
        </p>
      )}
      <div className="mt-2 w-full max-w-md space-y-2" aria-hidden>
        {['w-3/4', 'w-full', 'w-5/6', 'w-2/3'].map((width) => (
          <div key={width} className={`h-3 animate-pulse rounded bg-muted ${width}`} />
        ))}
      </div>
    </div>
  );
}

/**
 * Previous reports for this trial's analysis snapshot, newest first.
 *
 * The revision badge is the point of this panel: two reports about the same
 * trial can legitimately disagree because they were grounded in different
 * analysis revisions (for example, before and after the fairness audit was
 * computed). Native `<details>` keeps it collapsed and needs no extra library.
 */
function ReportHistory({
  reports,
  currentRevision,
  currentMarkdown,
  figures,
  onSave,
}: {
  reports: PersistedReport[];
  currentRevision: number | null;
  currentMarkdown: string | null;
  figures: Record<string, string>;
  onSave: (blob: Blob, filename: string) => void;
}) {
  const shown = reports.findIndex(
    (item) => item.status === 'completed' && item.markdown === currentMarkdown
  );
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Report history ({reports.length})</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-sm text-muted-foreground">
          Every report generated for this trial's analysis snapshot. Reports from an earlier
          revision were grounded in different evidence — for instance, before a fairness audit was
          added.
        </p>
        {reports.map((item, index) => (
          <details
            key={item.id}
            className="group rounded-md border border-border bg-muted/30 px-3 py-2"
          >
            <summary className="flex cursor-pointer flex-wrap items-center gap-2 text-sm">
              <ChevronRight className="h-4 w-4 shrink-0 transition-transform group-open:rotate-90" />
              <span className="font-mono text-xs">{formatTimestamp(item.created_at)}</span>
              <Badge variant="secondary">{item.model_name}</Badge>
              <Badge
                variant={item.snapshot_revision === currentRevision ? 'secondary' : 'outline'}
                title={
                  item.snapshot_revision === currentRevision
                    ? 'Grounded in the analysis revision currently loaded'
                    : 'Grounded in an earlier analysis revision'
                }
              >
                rev {item.snapshot_revision}
              </Badge>
              {item.status === 'failed' && <Badge variant="destructive">failed</Badge>}
              {index === shown && <Badge variant="emerald">shown above</Badge>}
            </summary>
            <div className="mt-3 border-t border-border pt-3">
              {item.status === 'completed' && item.markdown ? (
                <>
                  <div className="mb-2 flex justify-end">
                    <Button
                      type="button"
                      variant="secondary"
                      size="sm"
                      onClick={() =>
                        onSave(
                          new Blob([item.markdown ?? ''], { type: 'text/markdown' }),
                          `optimization-report-rev${item.snapshot_revision}-${item.id.slice(0, 8)}.md`
                        )
                      }
                    >
                      <Download className="h-4 w-4" /> Download .md
                    </Button>
                  </div>
                  <ReportMarkdown markdown={item.markdown} figures={figures} />
                </>
              ) : (
                <p className="text-sm text-muted-foreground">
                  {item.status === 'failed'
                    ? `This generation failed${item.error ? `: ${item.error}` : '.'}`
                    : 'This report is still being generated.'}
                </p>
              )}
            </div>
          </details>
        ))}
      </CardContent>
    </Card>
  );
}

/** ISO timestamp -> locale string, falling back to the raw value. */
function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function ReportDiagnostics({ result }: { result: ReportResponse }) {
  const summary = result.context_summary;
  const issues = result.markdown_issues ?? [];
  const dropped = result.dropped_figures ?? [];
  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant={issues.length ? 'secondary' : 'emerald'}>
          {issues.length ? `${issues.length} markdown issue(s)` : 'Valid markdown'}
        </Badge>
        <Badge variant={result.reviewed ? 'emerald' : 'secondary'}>
          {result.reviewed ? 'Reviewed' : 'Draft only'}
        </Badge>
        {summary && (
          <>
            <Badge variant="secondary">{summary.figures} figures</Badge>
            <Badge variant="secondary">
              {summary.trials_included}/{summary.trials_recorded ?? 0} trials
            </Badge>
            {summary.fairness_mode && summary.fairness_mode !== 'off' && (
              <Badge variant="emerald">fairness: {summary.fairness_mode}</Badge>
            )}
            {summary.fairness_audit_included && <Badge variant="emerald">audit included</Badge>}
            {summary.pareto_points > 0 && (
              <Badge variant="emerald">{summary.pareto_points} Pareto points</Badge>
            )}
          </>
        )}
      </div>
      {summary && summary.omissions > 0 && (
        <p className="text-sm text-muted-foreground">
          {summary.omissions} evidence gap(s) were declared to the agent so it would not speculate;
          see <code className="font-mono text-xs">context.json</code> in the research dump.
        </p>
      )}
      {dropped.length > 0 && (
        <p className="text-sm text-muted-foreground">
          Removed {dropped.length} reference(s) to figures that do not exist:{' '}
          {dropped.map((id) => (
            <code key={id} className="mr-1 font-mono text-xs">
              {id}
            </code>
          ))}
        </p>
      )}
      {issues.length > 0 && (
        <ul className="list-disc space-y-1 pl-5 text-sm text-muted-foreground">
          {issues.slice(0, 5).map((issue) => (
            <li key={issue}>{issue}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
