---
title: Contributing
description: Set up a dev environment, run the checks, and open a pull request for QuOptuna.
---

Contributions are welcome. QuOptuna is Apache 2.0 licensed. This page covers the dev environment, the Make targets, the quality checks, and the PR flow.

## Requirements

- **Python 3.11–3.12**
- **Node 18+** — only needed for frontend development

## Setup

Install the package with dev extras:

```bash
uv pip install -e ".[dev]"
```

Backend and frontend dependencies can also be installed via Make:

```bash
make install_backend
make install_frontend
```

## Make targets

| Target | What it does |
| --- | --- |
| `make dev` / `make run_cli` | Run backend + frontend together |
| `make run_backend` | Run the FastAPI server |
| `make run_frontend` | Run the Next.js dev server |
| `make run_streamlit` | Launch the legacy Streamlit UI |
| `make build` | Build wheel / sdist |
| `make build_package` | Build frontend, bundle into package, then build the wheel (needed for `uvx`) |
| `make format` | Format the code |
| `make lint` | Lint with ruff + mypy |
| `make lint-fix` | Auto-fix lint issues |
| `make tests` | Run the pytest suite |
| `make coverage` | Run tests with coverage |
| `make pre-commit` | Run pre-commit hooks |

## Windows: npm equivalents

The Makefile needs bash, `pkill`, and `lsof`, so it does not run on a stock
Windows install. `scripts/run.mjs` reimplements every target in Node (no extra
dependencies — Node 18+ is all you need), exposed as npm scripts in the root
`package.json`. These also work on macOS and Linux.

| Make target | npm equivalent |
| --- | --- |
| `make dev` / `make run_cli` | `npm run dev` |
| `make run_backend` | `npm run dev:backend` |
| `make run_frontend` | `npm run dev:frontend` |
| `make run_streamlit` | `npm run dev:streamlit` |
| `make install_backend` | `npm run install:backend` |
| `make install_frontend` | `npm run install:frontend` |
| — | `npm run install:all` (both of the above) |
| `make init` | `npm run init` (clears caches, then `uv sync`) |
| `make build` | `npm run build` |
| `make build_package` | `npm run build:package` |
| `make format` | `npm run format` |
| `make lint` | `npm run lint` |
| `make lint-fix` | `npm run lint:fix` |
| `make tests` | `npm run test` |
| `make coverage` | `npm run coverage` |
| `make pre-commit` | `npm run precommit` |
| `make clean` | `npm run clean` (frees ports 8000/3000 via `netstat` + `taskkill`) |
| `make clean_python_cache` | `npm run clean:cache` |
| `make clean_all` | `npm run clean:all` |
| `make help` | `npm run help` |

`npm run dev` starts both servers and stops both on Ctrl+C, killing the whole
process tree so ports 8000 and 3000 are released. It also creates
`frontend/.env.local` from `.env.example` when missing, so the UI on `:3000`
calls the API on `:8000` rather than 404-ing against itself.

## Quality checks

Run these before opening a PR:

```bash
uv run pytest          # coverage fail-under 35
uv run ruff check .
uv run mypy .
```

:::note
`make build_package` is the target to use when you need the packaged app (frontend built and bundled) so that `uvx quoptuna` ships the UI. A plain `make build` does not rebuild the frontend.
:::

## Project layout

- **`src/quoptuna/`** — the canonical shipped package: backend engine, server, CLI, `web/` (built UI), and the legacy `frontend/`.
- **`frontend/`** — Next.js dev source.
- **`backend/`** — legacy dev sub-project (its server code now lives in `src/quoptuna/server`).
- **`db/`** — SQLite stores.
- **`docs-site/`** — this Astro docs site.
- **`tests/`** — the pytest suite.

## Contributing docs

Documentation is Markdown under `docs-site/src/content/docs/`. To preview locally:

```bash
cd docs-site
npm install
npm run dev
```

## Pull request flow

1. **Fork** the repository.
2. Create a **feature branch**.
3. Make your change and ensure **tests pass** and **ruff + mypy** are clean.
4. Open a **PR** against `main`.

## License

QuOptuna is released under the **Apache 2.0** license.

## See also

- [Architecture](/explanation/architecture/)
- [Feature overview](/explanation/features/)
