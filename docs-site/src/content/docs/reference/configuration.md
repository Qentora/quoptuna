---
title: Configuration reference
description: Environment variables and settings for the QuOptuna server and optimizer.
---

Settings can be supplied via a `.env` file or the process environment.

## Server / app

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./db/quoptuna_app.db` | SQLModel application database URL; use `postgresql+psycopg://...` for Supabase |
| `OPTUNA_DATABASE_URL` | empty | PostgreSQL/Supabase URL for Optuna studies; falls back to local SQLite when empty |
| `OPTUNA_DB_SCHEMA` | `optuna` | PostgreSQL schema used for Optuna tables |
| `UPLOAD_DIR` | `./uploads` | Directory for uploaded datasets |
| `MAX_UPLOAD_SIZE` | 100 MB | Maximum upload size |
| `DEFAULT_N_TRIALS` | `100` | Default number of Optuna trials |
| `DEFAULT_TIMEOUT` | `3600` s | Default run timeout |
| `CORS_ORIGINS` | — | Comma-separated list of allowed origins |
| `APP_BASE_URL` | `http://localhost:8000` | Base URL for generated links |

## Artifact and dataset storage

Analysis images and uploaded datasets use local files by default. Set
`ARTIFACT_STORAGE=s3` to use AWS S3, Supabase Storage's S3-compatible endpoint,
MinIO, or another compatible provider.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARTIFACT_STORAGE` | `local` | `local` or `s3` |
| `ARTIFACT_ROOT` | `db/analysis` | Local artifact directory |
| `S3_ENDPOINT_URL` | empty | S3-compatible endpoint; empty for AWS S3 |
| `S3_BUCKET` | empty | Bucket name |
| `S3_REGION` | empty | Provider region |
| `S3_ACCESS_KEY_ID` | empty | S3 access key; keep server-side only |
| `S3_SECRET_ACCESS_KEY` | empty | S3 secret; keep server-side only |
| `S3_PREFIX` | `quoptuna` | Object-key prefix |
| `S3_SIGNED_URL_TTL` | `900` | Signed URL lifetime in seconds |

## Supabase example

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
OPTUNA_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
OPTUNA_DB_SCHEMA=optuna
ARTIFACT_STORAGE=s3
S3_ENDPOINT_URL=https://YOUR-S3-ENDPOINT
S3_BUCKET=YOUR-BUCKET
S3_REGION=YOUR-REGION
S3_ACCESS_KEY_ID=YOUR-ACCESS-KEY
S3_SECRET_ACCESS_KEY=YOUR-SECRET-KEY
S3_PREFIX=quoptuna
```

Never commit this file with real credentials. The backend uses these credentials;
the browser should never receive them.

## LLM report providers

Set the API key for your chosen report provider. All default to empty.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | OpenAI provider key |
| `ANTHROPIC_API_KEY` | empty | Anthropic provider key |
| `GOOGLE_API_KEY` | empty | Google provider key |

## Auth0 (optional)

Authentication is enforced **only when all** of these are set. When unset, `/api/v1/*` is open (good for local/dev). Sessions are stored in encrypted, httponly cookies.

| Variable | Purpose |
| --- | --- |
| `AUTH0_DOMAIN` | Auth0 tenant domain |
| `AUTH0_CLIENT_ID` | Auth0 application client id |
| `AUTH0_CLIENT_SECRET` | Auth0 application client secret |
| `AUTH0_SECRET` | 64-char hex session secret (`openssl rand -hex 32`) |

## Optimizer performance toggles

| Variable | Default | Purpose |
| --- | --- | --- |
| `QUOPTUNA_JAX_X64` | on | Set `0` for float32 — roughly halves simulator memory/compute |
| `QUOPTUNA_OVR_N_JOBS` | off | Thread count for multiclass OvR sub-fits |
| `QUOPTUNA_LOG_FILE` | `db/logs/quoptuna.log` | Override the rotating log path |

## See also

- [CLI reference](/reference/cli/)
- [REST API reference](/reference/rest-api/)
- [Python API reference](/reference/python-api/)
