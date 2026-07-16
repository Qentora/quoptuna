---
title: REST API reference
description: Endpoint groups exposed by the QuOptuna Next FastAPI server.
---

The FastAPI server (title **QuOptuna Next API**) serves interactive OpenAPI docs at [`http://localhost:8000/api/docs`](http://localhost:8000/api/docs). Treat those docs as the source of truth for request and response schemas.

:::note
All `/api/v1/*` routes require an authenticated user **only when Auth0 is configured**. When Auth0 is unset, the API is open (suitable for local/dev). See [Configuration reference](/reference/configuration/).
:::

## System — `/api/v1`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Health check |
| GET | `/info` | Server info |
| GET | `/models` | List available model types |

## Data — `/api/v1/data`

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/upload` | CSV upload |
| GET | `/uci` | List UCI datasets |
| POST | `/uci/{dataset_id}/load` | Load a UCI dataset |
| GET | `/uci/{dataset_id}` | UCI dataset metadata |
| GET | `/{dataset_id}/preview` | Preview dataset rows |
| GET | `/{dataset_id}` | Dataset details |

## Optimize — `/api/v1/optimize`

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/` | Start a run (background task) |
| GET | `/` | List runs |
| GET | `/{optimization_id}` | Run status |
| GET | `/{optimization_id}/detail` | Run detail |
| GET | `/{optimization_id}/trials` | Live trial polling |
| DELETE | `/{optimization_id}` | Cancel / delete run |

The main request contract is the **`OptimizationRequest`** (dataset, features, target, model set, sampler/pruner, fairness, encoding, device). Refer to [`/api/docs`](http://localhost:8000/api/docs) for its full schema.

## Analysis — `/api/v1/analysis`

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/jobs` | Start an async analysis job |
| GET | `/jobs/{job_id}` | Analysis job status |
| GET | `/snapshots` | List snapshots |
| GET | `/snapshots/{id}` | Snapshot detail |
| GET | `/snapshots/{id}/reports` | Snapshot reports |
| POST | `/snapshots/{id}/fairness` | Snapshot fairness audit |
| POST | `/shap` | SHAP analysis |
| POST | `/shap/data` | SHAP plot data |
| POST | `/metrics` | Metrics |
| POST | `/curves` | Curves |
| POST | `/curves/data` | Curve plot data |
| POST | `/confusion-matrix/data` | Confusion-matrix data |
| POST | `/feature-importance/data` | Feature-importance data |
| POST | `/study-plots` | Optuna study plots |
| POST | `/fairness` | Fairness metrics |
| POST | `/report` | AI report |

## Auth — `/auth`

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/login` | Begin Auth0 login |
| GET | `/callback` | Auth0 callback |
| GET | `/logout` | Log out |
| GET | `/profile` | Authenticated user profile |

## See also

- [CLI reference](/reference/cli/)
- [Python API reference](/reference/python-api/)
- [Configuration reference](/reference/configuration/)
