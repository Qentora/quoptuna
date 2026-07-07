'use client';

import { Button } from '@/components/ui/button';
import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { StatusDot } from '@/components/ui/status-dot';
import { Textarea } from '@/components/ui/textarea';
import { generateReport } from '@/lib/api';
import { type ApiKeys, loadApiKeys } from '@/lib/settings';
import { Check, Copy, Download, FileText, Loader2 } from 'lucide-react';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
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

  const { optimization, report } = workflowData;
  const apiKey = keys[provider];
  // Guard against a non-string payload (e.g. structured LLM content blocks) —
  // ReactMarkdown throws on non-string children and would crash the whole app.
  const hasReport = typeof report.markdown === 'string' && report.markdown.length > 0;
  const effectiveModel = modelName === '__custom__' ? customModel.trim() : modelName;

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
    if (!apiKey) {
      setError(`No ${provider} API key configured. Add one on the Settings page.`);
      return;
    }
    setIsGenerating(true);
    setError(null);
    try {
      const result = await generateReport({
        optimization_id: optimization.executionId,
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
    <div className="flex h-full min-h-0 flex-col gap-4">
      <div className="shrink-0 space-y-4">
        <StepHeader
          step={6}
          title="AI Report"
          subtitle="Generate a written summary of the run using your configured LLM provider"
        />

        <ErrorBanner message={error} />

        {/* Generation config */}
        <section className="rounded-lg border border-border bg-card">
          <div className="flex items-center justify-between gap-3 border-b border-border bg-muted px-4 py-3">
            <h4 className="text-sm font-semibold">Report settings</h4>
            <StatusDot
              status={apiKey ? 'online' : 'offline'}
              label={apiKey ? `${provider} key ready` : 'No API key'}
            />
          </div>
          <div className="grid grid-cols-1 gap-4 p-4 md:grid-cols-2">
            <Field label="Provider">
              <Select value={provider} onChange={(e) => handleProvider(e.target.value as Provider)}>
                {(Object.keys(PROVIDER_MODELS) as Provider[]).map((p) => (
                  <option key={p} value={p}>
                    {PROVIDER_MODELS[p].label}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label="Model">
              <Select value={modelName} onChange={(e) => setModelName(e.target.value)}>
                {PROVIDER_MODELS[provider].models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
                <option value="__custom__">Custom…</option>
              </Select>
            </Field>
            {modelName === '__custom__' && (
              <Field label="Custom model name" className="md:col-span-2">
                <Input
                  type="text"
                  value={customModel}
                  onChange={(e) => setCustomModel(e.target.value)}
                  placeholder="e.g. gemini-2.0-flash"
                />
              </Field>
            )}
            <Field label="Dataset description (optional)" className="md:col-span-2">
              <Textarea
                value={datasetDescription}
                onChange={(e) => setDatasetDescription(e.target.value)}
                rows={2}
                placeholder="Briefly describe the dataset and prediction goal…"
              />
            </Field>
          </div>

          <div className="flex flex-wrap items-center gap-3 border-t border-border p-4">
            <Button type="button" variant="brand" onClick={run} disabled={isGenerating || !apiKey}>
              {isGenerating ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <FileText className="h-4 w-4" />
              )}
              {hasReport ? 'Regenerate report' : 'Generate report'}
            </Button>
            {!apiKey && (
              <span className="text-sm text-muted-foreground">
                No {provider} key found.{' '}
                <Link href="/settings" className="font-medium text-brand hover:underline">
                  Add one in Settings
                </Link>
                .
              </span>
            )}
            {isGenerating && (
              <span className="text-sm text-muted-foreground">
                Sending metrics &amp; plots to the model — this can take a minute.
              </span>
            )}
          </div>
        </section>
      </div>

      {/* Report output fills remaining height */}
      <div className="flex min-h-0 flex-1 flex-col rounded-lg border border-border bg-card">
        <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border bg-muted px-4 py-3">
          <h4 className="text-sm font-semibold">Generated report</h4>
          {hasReport && (
            <div className="flex items-center gap-2">
              <Button type="button" variant="ghost" size="sm" onClick={copy}>
                {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                {copied ? 'Copied' : 'Copy'}
              </Button>
              <Button type="button" variant="secondary" size="sm" onClick={download}>
                <Download className="h-4 w-4" /> Download .md
              </Button>
            </div>
          )}
        </div>
        {hasReport && report.markdown ? (
          <div className="min-h-0 flex-1 overflow-y-auto p-6">
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>{report.markdown}</ReactMarkdown>
            </div>
          </div>
        ) : isGenerating ? (
          <div className="flex min-h-0 flex-1 items-center justify-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" /> Writing report…
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 p-8 text-center">
            <FileText className="h-8 w-8 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Generate a report to see an AI-written summary of the best trial, metrics and SHAP
              findings.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
