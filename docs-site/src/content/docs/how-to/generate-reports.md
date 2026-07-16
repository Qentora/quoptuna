---
title: Generate an AI analysis report
description: Produce a governance-ready report summarizing performance, SHAP interpretation, and recommendations.
---

After a run, QuOptuna can generate a governance-ready report summarizing performance, SHAP interpretation, and recommendations. The report is produced by an analyst + reviewer agent pair, built on the OpenAI Agents SDK + LiteLLM.

## Requirements

- Internet access.
- A provider API key (see the [configuration reference](/reference/configuration/) for the exact environment variable names).

## Providers

You can select and key one of:

| Provider | |
| --- | --- |
| OpenAI | Selected/keyed via environment variables. |
| Google Gemini | Selected/keyed via environment variables. |
| Anthropic Claude | Selected/keyed via environment variables. |

The provider and its API key are set through environment variables. See the [configuration reference](/reference/configuration/) for the exact variable names.

## Generate from the web UI

The report is the final wizard step, **Report**. Run through the wizard, then produce the report from that step.

## Generate via API

```bash
POST /api/v1/analysis/report
```

Call this after a study has completed to generate the report programmatically.

:::caution
Report generation calls an external LLM provider, so it needs internet access and a valid provider API key. Without both, the step fails.
:::

## Next steps

- [Configuration reference](/reference/configuration/)
- [Tune for speed and search quality](/how-to/tune-for-speed-and-quality/)
- [CLI reference](/reference/cli/)
