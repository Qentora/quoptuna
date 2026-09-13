'use client';

/**
 * TEMPORARY layout preview for the Report step. Delete before committing.
 *
 * Mounts the real ReportStep with stubbed wizard state and intercepts the
 * snapshot-reports fetch, so the tabs, the history table and the rail can be
 * checked visually without a completed optimization run behind them.
 */

import { type WorkflowData, initialWorkflowData } from '@/components/optimizer/types';
import { ReportStep } from '@/components/optimizer/steps/ReportStep';
import { saveApiKeys } from '@/lib/settings';
import { useEffect, useState } from 'react';

const PNG =
  'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABLCAYAAAC2Jl5AAAAAaUlEQVR42u3QMQEAAADCoPZP4Q1EBgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgH8GJHAAAeW0TxYAAAAASUVORK5CYII=';

const REPORTS = [
  {
    id: 'aaaaaaaa-1111-4111-8111-111111111111',
    snapshot_id: 'snap-1',
    snapshot_revision: 2,
    status: 'completed',
    provider: 'google',
    model_name: 'gemini-2.5-pro',
    created_at: '2026-09-12T15:04:11',
    markdown: '# Adult — DataReuploading Evaluation Report\n\nLatest revision.\n',
  },
  {
    id: 'bbbbbbbb-2222-4222-8222-222222222222',
    snapshot_id: 'snap-1',
    snapshot_revision: 2,
    status: 'completed',
    provider: 'anthropic',
    model_name: 'claude-sonnet-5',
    created_at: '2026-09-12T14:31:02',
    markdown:
      '# Adult — DataReuploading Evaluation Report\n\n## Executive Summary\n\n- An earlier attempt on the same evidence.\n',
  },
  {
    id: 'cccccccc-3333-4333-8333-333333333333',
    snapshot_id: 'snap-1',
    snapshot_revision: 1,
    status: 'completed',
    provider: 'openai',
    model_name: 'gpt-5.5',
    created_at: '2026-09-12T11:02:47',
    markdown: '# Adult — Evaluation Report\n\nWritten before the fairness audit existed.\n',
  },
  {
    id: 'dddddddd-4444-4444-8444-444444444444',
    snapshot_id: 'snap-1',
    snapshot_revision: 1,
    status: 'failed',
    provider: 'google',
    model_name: 'gemini-3.5-flash',
    created_at: '2026-09-12T09:20:15',
    markdown: null,
    error: 'litellm.RateLimitError: quota exceeded for project',
  },
];

const MARKDOWN = `# Adult — DataReuploadingClassifier Evaluation Report

## Executive Summary

- Binary income classification on the UCI Adult dataset (48,842 rows, 3 features selected).
- Winning configuration: \`DataReuploadingClassifier\` with \`n_layers = 3\`, test F1 0.8734.
- A multi-objective fairness search returned a 3-point Pareto front; the reported model is the highest-F1 point, not the fairest.
- Largest risk: recall for group \`Female\` is 0.612 against 0.884 overall, a 27.2 point shortfall.
- Recommendation: adopt with conditions, pending the mitigation trade-off review below.

## 1. Experimental Setup

| Setting | Value |
| --- | --- |
| Dataset | Adult (UCI, id 2) |
| Rows | 48,842 |
| Target | \`salary\` |
| Resampling | \`oversample\` (train split only) |
| Categorical encoding | \`onehot\` |

The split is seeded, so the reported metrics are reproducible from the study database alone.

## 5. Predictive Performance

| Metric | Value |
| --- | ---: |
| F1 | 0.8734 |
| Precision | 0.9012 |
| Recall | 0.8471 |
| ROC AUC | 0.9345 |

![Figure 1 — Confusion matrix](figures/confusion_matrix.png)
*Figure 1 — Confusion matrix on the held-out test split.*

## 6. Explainability

![Figure 2 — SHAP bar plot](figures/shap_bar.png)
*Figure 2 — Mean absolute SHAP per feature; \`age\` dominates at 0.412.*
`;

function patchFetch() {
  if (typeof window === 'undefined') return;
  const flag = '__previewFetchPatched';
  if ((window as unknown as Record<string, boolean>)[flag]) return;
  (window as unknown as Record<string, boolean>)[flag] = true;
  const real = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === 'string' ? input : input.toString();
    if (url.includes('/reports')) {
      return new Response(JSON.stringify({ reports: REPORTS }), {
        headers: { 'Content-Type': 'application/json' },
      });
    }
    if (url.includes('/report-prompts')) {
      return new Response(
        JSON.stringify({ prompts: { analyst: 'A', reviewer: 'R' }, settings: [] }),
        { headers: { 'Content-Type': 'application/json' } }
      );
    }
    return real(input, init);
  };
}

export default function PreviewPage() {
  const [ready, setReady] = useState(false);
  const [data, setData] = useState<WorkflowData>(() => ({
    ...initialWorkflowData,
    dataset: { id: '2', name: 'Adult', source: 'uci', rows: 48842, columns: ['age', 'salary'] },
    optimization: { ...initialWorkflowData.optimization, executionId: 'opt_abc123' },
    analysis: {
      ...initialWorkflowData.analysis,
      snapshotId: 'snap-1',
      snapshotRevision: 2,
      status: 'completed',
      plots: { bar: PNG },
      confusionMatrixPlot: PNG,
    },
    report: { markdown: MARKDOWN },
  }));

  useEffect(() => {
    patchFetch();
    saveApiKeys({ openai: '', anthropic: '', google: 'preview-key' })
      .catch(() => undefined)
      .finally(() => setReady(true));
  }, []);

  if (!ready) return null;
  return (
    <div className="min-h-screen bg-background p-6">
      <ReportStep
        onNext={() => undefined}
        onBack={() => undefined}
        workflowData={data}
        setWorkflowData={setData}
        setFooter={() => undefined}
      />
    </div>
  );
}
