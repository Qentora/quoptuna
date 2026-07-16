---
title: Feature overview
description: A catalog of QuOptuna's major capabilities and where to read more about each.
---

QuOptuna combines quantum and classical machine learning under a single hyperparameter-optimization workflow built on Optuna and PennyLane. This page is a hub: a short description of each major capability with a link to the page that covers it in depth.

## Running optimizations

- **Guided wizard** — a 6-step Next.js UI walks you from dataset to report. See [Architecture](/explanation/architecture/) for how the UI, server, and engine connect.
- **Headless CLI** — drive the same pipeline for scripted, reproducible runs. See the [CLI reference](/reference/cli/).
- **Python API** — use the `Optimizer` engine directly in your own code. See [How the optimization engine works](/explanation/optimization-engine/).

## Search and tuning

- **Conditional search space** — the objective suggests `model_type` first, then only that model's relevant parameters, keeping the TPE model clean. See [How the optimization engine works](/explanation/optimization-engine/).
- **Samplers and pruners** — `tpe`/`random`/`grid` samplers and `asha`/`hyperband`/`none` pruners, with early stopping for iterative quantum models. See [How the optimization engine works](/explanation/optimization-engine/).
- **Fairness-aware search** — off, constrained (feasibility constraint on disparity), or multi-objective (F1 vs disparity Pareto front), using equal-opportunity, disparate-impact, and demographic-parity metrics on a sensitive feature. See [How the optimization engine works](/explanation/optimization-engine/).
- **Multiclass and One-vs-Rest** — macro-F1 scoring with OvR-wrapped variational models for K-class problems. See [How the optimization engine works](/explanation/optimization-engine/).

## Analysis and reporting

- **SHAP and XAI** — SHAP plots and feature-importance analysis for the best model. See [The workflow engine](/explanation/workflow-engine/).
- **LLM reports** — analyst + reviewer agents (OpenAI/Gemini/Anthropic providers) generate a written report of results. See [The workflow engine](/explanation/workflow-engine/).

## Data and persistence

- **UCI + CSV ingestion** — load data from an uploaded CSV or a UCI dataset. See [The workflow engine](/explanation/workflow-engine/).
- **Persistence and crash rehydration** — durable SQLite stores (`run_store`, `analysis_store`, `dataset_registry`) plus Optuna studies as source of truth; stale runs are recovered on restart. See [Architecture](/explanation/architecture/).

## Deployment and access

- **Optional Auth0** — cookie-session authentication that is a no-op when unconfigured, so local runs stay unauthenticated. See [Architecture](/explanation/architecture/).
- **Single-port packaged deploy** — `uvx quoptuna` serves the built UI and API from one uvicorn process. See [Architecture](/explanation/architecture/).
- **Legacy Streamlit UI** — a fallback dashboard kept for compatibility. See [Legacy Streamlit UI](/legacy/streamlit-ui/).

## Next steps

- [Architecture](/explanation/architecture/)
- [How the optimization engine works](/explanation/optimization-engine/)
- [The workflow engine](/explanation/workflow-engine/)
