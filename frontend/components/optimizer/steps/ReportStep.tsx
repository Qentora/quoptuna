'use client';

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
import { generateReport, listSnapshotReports } from '@/lib/api';
import { type ApiKeys, loadApiKeys } from '@/lib/settings';
import { Check, Copy, Download, FileText, Loader2 } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
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
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<Provider>('google');
  const [modelName, setModelName] = useState(PROVIDER_MODELS.google.models[0]);
  const [customModel, setCustomModel] = useState('');
  const [datasetDescription, setDatasetDescription] = useState('');
  const [keys, setKeys] = useState<ApiKeys>({ openai: '', anthropic: '', google: '' });
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadApiKeys()
      .then(setKeys)
      .catch(() => undefined);
  }, []);

  const { optimization, analysis, report } = workflowData;
  const apiKey = keys[provider];
  // Guard against a non-string payload (e.g. structured LLM content blocks) —
  // ReactMarkdown throws on non-string children and would crash the whole app.
  const hasReport = typeof report.markdown === 'string' && report.markdown.length > 0;
  const effectiveModel = modelName === '__custom__' ? customModel.trim() : modelName;

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
      });
      const markdown =
        typeof result.report_markdown === 'string'
          ? result.report_markdown
          : JSON.stringify(result.report_markdown);
      setWorkflowData((prev) => ({ ...prev, report: { markdown } }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Report generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const download = () => {
    const blob = new Blob([report.markdown || ''], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `optimization-report-${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
            {hasReport && (
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
            {hasReport && report.markdown ? (
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
                  }}
                >
                  {report.markdown}
                </ReactMarkdown>
              </div>
            ) : isGenerating ? (
              <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground">
                <Loader2 className="h-5 w-5 animate-spin" /> Writing report…
              </div>
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
                  Sending metrics &amp; plots to the model — this can take a minute.
                </p>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
