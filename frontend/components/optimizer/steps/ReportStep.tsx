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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
  Copy,
  Download,
  Eye,
  FileArchive,
  FileJson,
  FileText,
  History,
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

type ReportTab = 'report' | 'history';

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
  const [tab, setTab] = useState<ReportTab>('report');
  // Id of a report opened from History; null shows the current one.
  const [viewingId, setViewingId] = useState<string | null>(null);
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
  // The report being read: an entry opened from History, else the current one.
  const viewing = viewingId ? history.find((item) => item.id === viewingId) : undefined;
  const shownMarkdown = viewing?.markdown ?? (hasReport ? report.markdown : null);

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
    // Return to the live report and the Report tab, so the result is what is read.
    setViewingId(null);
    setTab('report');
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
      new Blob([shownMarkdown || ''], { type: 'text/markdown' }),
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
    if (!shownMarkdown) return;
    try {
      await navigator.clipboard.writeText(shownMarkdown);
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

      {/* Two-column layout: report flows with page scroll; controls live in a sticky
          rail. The left column is a nested stack rather than extra grid children —
          auto-placement would otherwise push the rail into row 2 once the history
          card claims column 1. */}
      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <Card>
            {/* The Tabs wrapper becomes the Card's only child, so it has to carry
                the card's own spacing or the header/content gap halves. */}
            <Tabs
              value={tab}
              onValueChange={(value) => setTab(value as ReportTab)}
              className="gap-(--card-spacing)"
            >
              <CardHeader>
                <TabsList>
                  <TabsTrigger value="report">Report</TabsTrigger>
                  <TabsTrigger value="history" disabled={history.length === 0}>
                    History
                    {history.length > 0 && (
                      <span className="ml-1.5 tabular-nums opacity-70">{history.length}</span>
                    )}
                  </TabsTrigger>
                </TabsList>
                {tab === 'report' && shownMarkdown && !isGenerating && (
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
                <TabsContent value="report">
                  {/* A report opened from History is labelled, so an older revision is
                      never mistaken for the current one. */}
                  {viewing && (
                    <div className="mb-4 flex flex-wrap items-center gap-2 rounded-md border border-border bg-muted/50 px-3 py-2 text-sm">
                      <History className="h-4 w-4 shrink-0 text-muted-foreground" />
                      <span>
                        Viewing an earlier report from{' '}
                        <span className="font-medium">{formatTimestamp(viewing.created_at)}</span>
                      </span>
                      <Badge variant="secondary">{viewing.model_name}</Badge>
                      <RevisionBadge
                        revision={viewing.snapshot_revision}
                        current={analysis.snapshotRevision}
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="ml-auto"
                        onClick={() => setViewingId(null)}
                      >
                        Back to latest
                      </Button>
                    </div>
                  )}
                  {/* Generation takes precedence over the previous report: stale prose
                      while a new one is written reads as if nothing is happening. The
                      superseded report stays reachable on the History tab. */}
                  {isGenerating ? (
                    <GeneratingReport
                      model={effectiveModel || PROVIDER_MODELS[provider].models[0]}
                      provider={PROVIDER_MODELS[provider].label}
                      reviewed={reportSettings.enableReview}
                      historyCount={history.length}
                    />
                  ) : shownMarkdown ? (
                    <ReportMarkdown markdown={shownMarkdown} figures={figureMap} />
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
                </TabsContent>
                <TabsContent value="history">
                  <ReportHistoryTable
                    reports={history}
                    currentRevision={analysis.snapshotRevision}
                    currentMarkdown={report.markdown}
                    viewingId={viewingId}
                    onView={(id) => {
                      setViewingId(id);
                      setTab('report');
                    }}
                    onSave={saveBlob}
                  />
                </TabsContent>
              </CardContent>
            </Tabs>
          </Card>
        </div>

        {/* Controls rail */}
        <Card className="self-start lg:sticky lg:top-4">
          <CardHeader>
            <CardTitle>Generate a report</CardTitle>
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

            <div className="space-y-2 border-t border-border pt-4">
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
              {!apiKey ? (
                <p className="text-sm text-muted-foreground">
                  No {provider} key found.{' '}
                  <Link href="/settings" className="font-medium text-brand hover:underline">
                    Add one in Settings
                  </Link>
                  .
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">
                  <Settings2 className="mr-1 inline h-3.5 w-3.5" />
                  <Link href="/settings" className="font-medium text-brand hover:underline">
                    Prompts &amp; evidence settings
                  </Link>
                </p>
              )}
            </div>

            <RailSection title="Exports">
              <div className="grid grid-cols-2 gap-2">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void downloadBundle()}
                  disabled={
                    isDownloading || !analysis.snapshotId || analysis.status !== 'completed'
                  }
                  title="Research dump: context, evidence, every figure, CSV tables, prompts and all reports"
                >
                  {isDownloading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <FileArchive className="h-4 w-4" />
                  )}
                  .zip
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void downloadContextJson()}
                  disabled={
                    isDownloading || !analysis.snapshotId || analysis.status !== 'completed'
                  }
                  title="The same evidence bundle as JSON, with the figure manifest instead of image files"
                >
                  <FileJson className="h-4 w-4" /> .json
                </Button>
              </div>
              <p className="text-sm text-muted-foreground">
                Everything behind the report — evidence, figures, tables, prompts.
              </p>
            </RailSection>

            {diagnostics && (
              <RailSection title="Last run">
                <ReportDiagnostics result={diagnostics} />
              </RailSection>
            )}
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
/**
 * A labelled group in the 320px rail.
 *
 * The rail holds three unrelated things — the model to call, the exports, and
 * the last run's diagnostics. Without headed groups they read as one long list
 * and the primary action gets lost among them.
 */
function RailSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2 border-t border-border pt-4">
      <p className="font-medium text-muted-foreground text-xs uppercase tracking-wide">{title}</p>
      {children}
    </div>
  );
}

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
  historyCount,
}: {
  model: string;
  provider: string;
  reviewed: boolean;
  historyCount: number;
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
      {historyCount > 0 && (
        <p className="text-sm text-muted-foreground">
          Earlier reports stay on the <span className="font-medium">History</span> tab.
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
 * Marks which analysis revision a report was grounded in.
 *
 * This is the point of the history: two reports about the same trial can
 * legitimately disagree because they read different evidence — for example one
 * written before the fairness audit was computed and one after.
 */
function RevisionBadge({ revision, current }: { revision: number; current: number | null }) {
  const isCurrent = revision === current;
  return (
    <Badge
      variant={isCurrent ? 'secondary' : 'outline'}
      title={
        isCurrent
          ? 'Grounded in the analysis revision currently loaded'
          : 'Grounded in an earlier analysis revision, so its evidence differed'
      }
    >
      rev {revision}
    </Badge>
  );
}

/**
 * Every report persisted for this trial's analysis snapshot, newest first.
 *
 * An aligned table rather than a stack of expanders: the columns are what make
 * provenance scannable, and opening a row reuses the single reading area instead
 * of nesting a full report inside a card.
 */
function ReportHistoryTable({
  reports,
  currentRevision,
  currentMarkdown,
  viewingId,
  onView,
  onSave,
}: {
  reports: PersistedReport[];
  currentRevision: number | null;
  currentMarkdown: string | null;
  viewingId: string | null;
  onView: (id: string) => void;
  onSave: (blob: Blob, filename: string) => void;
}) {
  const latestId = reports.find(
    (item) => item.status === 'completed' && item.markdown === currentMarkdown
  )?.id;
  const revisions = new Set(reports.map((item) => item.snapshot_revision));
  return (
    <div className="space-y-3">
      {revisions.size > 1 && (
        <p className="text-sm text-muted-foreground">
          These reports span {revisions.size} analysis revisions, so they were not all grounded in
          the same evidence — check the revision column before comparing them.
        </p>
      )}
      <div className="overflow-x-auto rounded-md border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Generated</TableHead>
              <TableHead>Model</TableHead>
              <TableHead>Evidence</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {reports.map((item) => {
              const readable = item.status === 'completed' && Boolean(item.markdown);
              return (
                <TableRow key={item.id} className={item.id === viewingId ? 'bg-muted/50' : ''}>
                  <TableCell className="whitespace-nowrap font-mono text-xs">
                    {formatTimestamp(item.created_at)}
                  </TableCell>
                  <TableCell className="whitespace-nowrap">
                    <span className="font-mono text-xs">{item.model_name}</span>
                    <span className="ml-2 text-muted-foreground text-xs">{item.provider}</span>
                  </TableCell>
                  <TableCell>
                    <RevisionBadge revision={item.snapshot_revision} current={currentRevision} />
                  </TableCell>
                  <TableCell>
                    {item.status === 'failed' ? (
                      <span
                        className="text-destructive text-xs"
                        title={item.error ?? 'Generation failed'}
                      >
                        failed
                      </span>
                    ) : item.id === latestId ? (
                      <Badge variant="emerald">latest</Badge>
                    ) : item.status === 'running' ? (
                      <span className="text-muted-foreground text-xs">generating…</span>
                    ) : (
                      <span className="text-muted-foreground text-xs">superseded</span>
                    )}
                  </TableCell>
                  <TableCell className="text-right whitespace-nowrap">
                    {readable ? (
                      <div className="flex items-center justify-end gap-1">
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={() => onView(item.id)}
                        >
                          <Eye className="h-4 w-4" /> View
                        </Button>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          aria-label="Download this report as markdown"
                          onClick={() =>
                            onSave(
                              new Blob([item.markdown ?? ''], { type: 'text/markdown' }),
                              `report-rev${item.snapshot_revision}-${item.id.slice(0, 8)}.md`
                            )
                          }
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      </div>
                    ) : (
                      <span className="text-muted-foreground text-xs">
                        {item.error ? 'see status' : '—'}
                      </span>
                    )}
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
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
    <div className="space-y-2">
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
