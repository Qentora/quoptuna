'use client';

import { Alert } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Field } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { generateReport } from '@/lib/api';
import { type ApiKeys, loadApiKeys } from '@/lib/settings';
import { Download, FileText, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { ErrorBanner } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

type Provider = 'google' | 'openai';

const DEFAULT_MODELS: Record<Provider, string> = {
  google: 'gemini-1.5-flash',
  openai: 'gpt-4o',
};

export function ReportStep({ workflowData, setWorkflowData, setFooter }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<Provider>('google');
  const [modelName, setModelName] = useState(DEFAULT_MODELS.google);
  const [datasetDescription, setDatasetDescription] = useState('');
  const [keys, setKeys] = useState<ApiKeys>({ openai: '', anthropic: '', google: '' });

  useEffect(() => {
    loadApiKeys()
      .then(setKeys)
      .catch(() => undefined);
  }, []);

  const { optimization, report } = workflowData;
  const apiKey = provider === 'google' ? keys.google : keys.openai;
  const hasReport = report.markdown !== null;

  useEffect(() => {
    setFooter({ canContinue: false, hideNext: true, backDisabled: isGenerating });
  }, [isGenerating, setFooter]);

  const handleProvider = (p: Provider) => {
    setProvider(p);
    setModelName(DEFAULT_MODELS[p]);
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
        model_name: modelName,
        dataset_description: datasetDescription || undefined,
      });
      setWorkflowData((prev) => ({ ...prev, report: { markdown: result.report_markdown } }));
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

  return (
    <div className="space-y-6">
      <StepHeader
        step={6}
        title="Generate Summary Report"
        subtitle="AI-powered analysis report using your configured LLM provider"
      />

      <ErrorBanner message={error} />

      <Card className="grid grid-cols-1 gap-4 p-4 md:grid-cols-2">
        <Field label="Provider">
          <Select value={provider} onChange={(e) => handleProvider(e.target.value as Provider)}>
            <option value="google">Google (Gemini)</option>
            <option value="openai">OpenAI</option>
          </Select>
        </Field>
        <Field label="Model">
          <Input type="text" value={modelName} onChange={(e) => setModelName(e.target.value)} />
        </Field>
        <Field label="Dataset description (optional)" className="md:col-span-2">
          <Textarea
            value={datasetDescription}
            onChange={(e) => setDatasetDescription(e.target.value)}
            rows={2}
            placeholder="Briefly describe the dataset and prediction goal..."
          />
        </Field>
      </Card>

      {!apiKey && (
        <Alert variant="warning">
          No {provider} API key found. Add one on the Settings page to enable report generation.
        </Alert>
      )}

      <div className="text-center">
        <Button
          type="button"
          variant="brand"
          size="lg"
          onClick={run}
          disabled={isGenerating || !apiKey}
        >
          {isGenerating ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <FileText className="h-5 w-5" />
          )}
          {hasReport ? 'Regenerate Report' : 'Generate AI Report'}
        </Button>
        {isGenerating && (
          <p className="mt-3 text-sm text-muted-foreground">
            Sending plots and metrics to the model — this can take a minute.
          </p>
        )}
      </div>

      {hasReport && report.markdown && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-semibold">Generated Report</h3>
            <Button type="button" size="md" onClick={download}>
              <Download className="h-4 w-4" /> Download .md
            </Button>
          </div>
          <Card className="max-h-[500px] overflow-y-auto p-6">
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>{report.markdown}</ReactMarkdown>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
