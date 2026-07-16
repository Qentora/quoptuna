---
title: Configuration reference
description: Environment variables and settings for the QuOptuna server and optimizer.
---

Settings can be supplied via a `.env` file or the process environment.

## Server / app

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./quoptuna.db` | Application database URL |
| `UPLOAD_DIR` | `./uploads` | Directory for uploaded datasets |
| `MAX_UPLOAD_SIZE` | 100 MB | Maximum upload size |
| `DEFAULT_N_TRIALS` | `100` | Default number of Optuna trials |
| `DEFAULT_TIMEOUT` | `3600` s | Default run timeout |
| `CORS_ORIGINS` | — | Comma-separated list of allowed origins |
| `APP_BASE_URL` | `http://localhost:8000` | Base URL for generated links |

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
