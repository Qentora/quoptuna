---
title: Generate an AI analysis report
description: Produce a governance-ready report summarizing performance, SHAP interpretation, and recommendations.
---

After a run, QuOptuna can generate a governance-ready report summarizing performance, SHAP interpretation, and recommendations. The report is produced by an analyst + reviewer agent pair, built on the OpenAI Agents SDK + LiteLLM.

## Requirements

- Internet access.
- A provider API key, entered in the web UI's Settings and sent with each report request (the `api_key` field). Requests without a key are rejected with 400.

## Providers

Set `llm_provider` to one of:

| `llm_provider` | Provider |
| --- | --- |
| `openai` | OpenAI |
| `google` (default) | Google Gemini |
| `anthropic` | Anthropic Claude |

Pick the model with `model_name` (default `gpt-4o`; choose one that matches your provider). Requests are routed through LiteLLM as `<provider>/<model_name>`.

## Generate from the web UI

The report is the final wizard step, **Report**. Run through the wizard, then produce the report from that step.

## Generate via API

```bash
POST /api/v1/analysis/report
```

Call this after a study and its analysis snapshot have completed. Required body fields: `optimization_id`, `analysis_snapshot_id`, `analysis_revision`, `api_key`. Optional: `trial_number`, `llm_provider`, `model_name`, `dataset_description`, `sensitive_feature`, `analyst_instructions`/`reviewer_instructions` (prompt overrides), and `enable_review` (default `true`; runs the reviewer pass).

:::caution
Report generation calls an external LLM provider, so it needs internet access and a valid provider API key. Without both, the step fails.
:::

## Next steps

- [Configuration reference](/reference/configuration/)
- [Tune for speed and search quality](/how-to/tune-for-speed-and-quality/)
- [CLI reference](/reference/cli/)
