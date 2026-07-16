---
title: Installation
description: Install QuOptuna and learn the three ways to run it — packaged, dev, and Docker.
---

This page is for anyone setting up QuOptuna for the first time. By the end you'll have the package installed and the app running at `http://localhost:8000`.

## Requirements

| Requirement | Details |
| --- | --- |
| Python | 3.11 or 3.12 |
| RAM | 4 GB minimum, 8 GB recommended for quantum models |
| Internet | Needed for UCI datasets and LLM-generated reports |
| Node.js | 18+ — only if you run the frontend in dev mode |

## Install the package

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install quoptuna
```

Or with pip:

```bash
pip install quoptuna
```

For an editable development install with extra tooling:

```bash
uv pip install -e ".[dev]"
```

## Choose a run mode

QuOptuna can run three ways. Most users want **packaged mode**.

| Mode | Command | Ports | When to use |
| --- | --- | --- | --- |
| Packaged | `uvx quoptuna` or `uv run quoptuna run` | `:8000` (UI + API) | Everyday use; simplest, one process |
| Dev | `make dev` | `:8000` API + `:3000` UI | Contributing with hot reload |
| Docker | `docker compose up --build` | `:3000` frontend + `:8000` backend | Isolated, reproducible environment |

### Packaged mode (one process)

Serves the web UI and JSON API on a single port. It opens your browser
automatically and prints a gradient ASCII banner.

```bash
uvx quoptuna
```

Running `uvx quoptuna` (or `uv run quoptuna`) with no subcommand is the same as
`quoptuna run`. Useful options:

```bash
uv run quoptuna run --port 8001 --no-browser
uv run quoptuna run --host 0.0.0.0
uv run quoptuna run --streamlit   # launch the legacy Streamlit dashboard
```

:::note
The default port is `8000`. If it's busy, QuOptuna auto-increments to the next
free port.
:::

### Dev mode (two processes, hot reload)

Runs FastAPI and the Next.js frontend separately so both hot-reload. This needs
Node 18+ for the frontend.

```bash
make dev        # FastAPI on :8000 + Next.js on :3000
```

Individual targets are also available: `make run_backend`, `make run_frontend`,
`make run_streamlit`, and `make run_cli`.

### Docker

```bash
docker compose up --build
```

The `docker-compose.yml` defines the frontend on `:3000` and the backend on
`:8000`.

## Verify it works

1. Start the app: `uvx quoptuna`.
2. Open the web UI at [http://localhost:8000](http://localhost:8000) — you
   should see the 6-step wizard.
3. Open the interactive API docs at
   [http://localhost:8000/api/docs](http://localhost:8000/api/docs) to confirm
   the API is live.

The JSON API lives under `http://localhost:8000/api/v1/...`.

## Next steps

- [Quickstart](/getting-started/quickstart/)
- [First optimization in the UI](/tutorials/first-optimization-ui/)
