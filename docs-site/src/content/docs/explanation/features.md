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
- **Class-imbalance resampling** — optional `oversample`/`undersample` of the inner training split (default `none`), with the sensitive column resampled in lockstep. See [How the optimization engine works](/explanation/optimization-engine/).

## Analysis and reporting

- **SHAP and XAI** — SHAP plots, metrics, curves, confusion matrices and feature importance for the best (or any chosen) trial, run as a resumable, cancellable background job with revision history. See [Analysis pipeline](/explanation/analysis-pipeline/).
- **LLM reports** — analyst + reviewer agents (OpenAI/Gemini/Anthropic providers) generate a written report grounded in a specific analysis revision. See [Analysis pipeline](/explanation/analysis-pipeline/).
- **Bulk research dumps** — select runs on the Runs page and download one zip (`POST /api/v1/analysis/bundles/bulk`, 1–100 runs).

## Data and persistence

- **Bundled, UCI and CSV ingestion** — load an uploaded CSV, a dataset bundled with the package (gzipped CSVs, available offline), or a UCI dataset; targets are labelled with a class-balance profile. See [The workflow engine](/explanation/workflow-engine/).
- **Persistence and crash rehydration** — a SQLModel application database (SQLite by default, PostgreSQL supported) behind `run_store`, `analysis_store` and `dataset_registry`, plus Optuna studies as source of truth; stale runs are recovered on restart. See [Architecture](/explanation/architecture/).

## Deployment and access

- **Optional Auth0** — cookie-session authentication that is a no-op when unconfigured, so local runs stay unauthenticated. See [Architecture](/explanation/architecture/).
- **Single-port packaged deploy** — `uvx quoptuna` serves the built UI and API from one uvicorn process. See [Architecture](/explanation/architecture/).
- **Legacy Streamlit UI** — a fallback dashboard kept for compatibility. See [Legacy Streamlit UI](/legacy/streamlit-ui/).

## Next steps

- [Architecture](/explanation/architecture/)
- [How the optimization engine works](/explanation/optimization-engine/)
- [The workflow engine](/explanation/workflow-engine/)
