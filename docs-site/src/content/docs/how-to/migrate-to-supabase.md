---
title: Move persistence to Supabase and S3
description: Configure PostgreSQL application metadata, Optuna studies, and S3-compatible artifacts.
---

This guide moves QuOptuna's durable data out of local SQLite files.

## Install the deployment dependencies

```bash
uv sync
```

## Configure the environment

Copy `.env.example` to `.env`, then set your Supabase PostgreSQL connection:

```dotenv
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
OPTUNA_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/postgres
OPTUNA_DB_SCHEMA=optuna
```

`DATABASE_URL` stores QuOptuna application data such as runs, sessions,
datasets, analysis snapshots, and reports. `OPTUNA_DATABASE_URL` stores Optuna
studies and trials. They may point to the same Supabase database; QuOptuna
places Optuna tables in the separate `optuna` schema.

## Migrate application metadata

Preview the migration:

```bash
uv run quoptuna migrate-supabase \
  --source-db db/quoptuna_app.db \
  --database-url "$DATABASE_URL" \
  --dry-run
```

Run it after reviewing the counts:

```bash
uv run quoptuna migrate-supabase \
  --source-db db/quoptuna_app.db \
  --database-url "$DATABASE_URL"
```

The local database remains available as a backup.

## Migrate the active Optuna database

Identify the database used by the runs, then migrate that file. For example:

```bash
uv run quoptuna migrate-optuna db/results-trial-june15.db
```

The command migrates every study in that file. Use `--study-name` when only one
study is needed. It copies trials, parameters, values, states, and study/trial
attributes.

## Configure S3-compatible artifacts

To store analysis images and uploaded datasets remotely:

```dotenv
ARTIFACT_STORAGE=s3
S3_ENDPOINT_URL=https://YOUR-S3-ENDPOINT
S3_BUCKET=YOUR-BUCKET
S3_REGION=YOUR-REGION
S3_ACCESS_KEY_ID=YOUR-ACCESS-KEY
S3_SECRET_ACCESS_KEY=YOUR-SECRET-KEY
S3_PREFIX=quoptuna
S3_SIGNED_URL_TTL=900
```

For AWS S3, leave `S3_ENDPOINT_URL` empty. For Supabase Storage or MinIO, use
the provider's S3-compatible endpoint. Create the bucket before starting the
server.

## Verify the migration

Start the server:

```bash
uv run quoptuna run --no-browser
```

Create a new run and analysis, then confirm application records in Supabase:

```sql
select job_id, session_id, study_name, status
from public.quoptuna_runs
order by created_at desc;
```

Confirm Optuna tables are isolated in the configured schema:

```sql
select table_schema, table_name
from information_schema.tables
where table_schema = 'optuna'
order by table_name;
```

If S3 is enabled, check that objects appear under:

```text
quoptuna/runs/<run-id>/analysis/<snapshot-id>/revisions/<revision>/
```

Restart the server and reopen the run. Successful rehydration confirms that
the run metadata is coming from Supabase and analysis artifacts are coming from
the configured object storage.

## Troubleshooting

### `No module named psycopg2`

Use the `psycopg` URL form and reinstall dependencies:

```bash
uv sync
```

The URL should be accepted as either `postgresql://...` or
`postgresql+psycopg://...`; QuOptuna normalizes the former.

### `copy_study() got an unexpected keyword argument study_name`

Use the current CLI command from this version. It uses Optuna's
`to_study_name` API internally:

```bash
uv run quoptuna migrate-optuna db/results-trial-june15.db
```

### A study already exists

The source may have been partially migrated. Inspect the destination before
rerunning; do not delete the source SQLite database until the trial counts have
been verified.
