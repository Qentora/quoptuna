'use client';

import { generateReport } from '@/lib/api';
import { type ApiKeys, loadApiKeys } from '@/lib/settings';
import { Download, FileText, Loader2 } from 'lucide-react';
import { useEffect, useState } from 'react';
import { ErrorBanner, NavButtons } from '../NavButtons';
import { StepHeader } from '../Wizard';
import type { StepProps } from '../Wizard';

type Provider = 'google' | 'openai';

const DEFAULT_MODELS: Record<Provider, string> = {
  google: 'gemini-1.5-flash',
  openai: 'gpt-4o',
};

export function ReportStep({ onBack, workflowData, setWorkflowData }: StepProps) {
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [provider, setProvider] = useState<Provider>('google');
  const [modelName, setModelName] = useState(DEFAULT_MODELS.google);
  const [datasetDescription, setDatasetDescription] = useState('');
  const [keys, setKeys] = useState<ApiKeys>({ openai: '', anthropic: '', google: '' });

  useEffect(() => {
    setKeys(loadApiKeys());
  }, []);

  const { optimization, report } = workflowData;
  const apiKey = provider === 'google' ? keys.google : keys.openai;
  const hasReport = report.markdown !== null;

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
        title="Generate Summary Report"
        subtitle="AI-powered analysis report using your configured LLM provider"
      />

      <ErrorBanner message={error} />

      <div className="bg-white border border-gray-200 rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
        <label className="text-sm text-gray-700">
          <span className="block mb-1">Provider</span>
          <select
            value={provider}
            onChange={(e) => handleProvider(e.target.value as Provider)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          >
            <option value="google">Google (Gemini)</option>
            <option value="openai">OpenAI</option>
          </select>
        </label>
        <label className="text-sm text-gray-700">
          <span className="block mb-1">Model</span>
          <input
            type="text"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
          />
        </label>
        <label className="text-sm text-gray-700 md:col-span-2">
          <span className="block mb-1">Dataset description (optional)</span>
          <textarea
            value={datasetDescription}
            onChange={(e) => setDatasetDescription(e.target.value)}
            rows={2}
            className="w-full px-3 py-2 border border-gray-300 rounded-md"
            placeholder="Briefly describe the dataset and prediction goal..."
          />
        </label>
      </div>

      {!apiKey && (
        <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded-md text-sm">
          No {provider} API key found. Add one on the Settings page to enable report generation.
        </div>
      )}

      <div className="text-center">
        <button
          type="button"
          onClick={run}
          disabled={isGenerating || !apiKey}
          className="px-8 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 font-semibold inline-flex items-center gap-2 disabled:bg-gray-300"
        >
          {isGenerating ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <FileText className="w-5 h-5" />
          )}
          {hasReport ? 'Regenerate Report' : 'Generate AI Report'}
        </button>
        {isGenerating && (
          <p className="text-sm text-gray-500 mt-3">
            Sending plots and metrics to the model — this can take a minute.
          </p>
        )}
      </div>

      {hasReport && report.markdown && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-gray-900">Generated Report</h3>
            <button
              type="button"
              onClick={download}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm inline-flex items-center gap-2"
            >
              <Download className="w-4 h-4" /> Download .md
            </button>
          </div>
          <div className="bg-white border border-gray-200 rounded-lg p-6 max-h-[500px] overflow-y-auto">
            <pre className="whitespace-pre-wrap text-sm text-gray-700 font-sans">
              {report.markdown}
            </pre>
          </div>
        </div>
      )}

      <NavButtons onBack={onBack} backDisabled={isGenerating} hideNext />
    </div>
  );
}
