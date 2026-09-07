# Contributing to QuOptuna

Thanks for your interest in improving QuOptuna! This is the short, practical guide; the long-form contributor documentation lives at [Qentora.github.io/quoptuna/contributing](https://Qentora.github.io/quoptuna/contributing/).

## Development setup

Requirements: Python 3.11 or 3.12, [uv](https://docs.astral.sh/uv/), and (only for frontend work) Node.js 18+.

```bash
git clone https://github.com/Qentora/quoptuna.git && cd quoptuna
uv sync                      # install Python dependencies
uv run pre-commit install    # lint/format hooks
cp frontend/.env.example frontend/.env.local   # only for frontend work
```

`frontend/.env.local` sets `NEXT_PUBLIC_API_URL=http://localhost:8000`. It is
gitignored and defaults to empty (same-origin) for the packaged build, so
without it the dev UI on `:3000` calls itself and every API request 404s from
Next.js. `npm run dev` creates it automatically.

Useful commands (see the `Makefile` for the full list):

```bash
uv run pytest                # test suite (coverage gate: 35%)
uv run pytest -m "not slow"  # skip the long quantum-training tests
uv run ruff check .          # lint
uv run mypy .                # type-check
make run_backend             # FastAPI on :8000 (--reload)
make run_frontend            # Next.js dev server on :3000
cd docs-site && npm run dev  # documentation site (Astro/Starlight)
```

On Windows (no `make`, bash, or `lsof` needed — just Node 18+), every target has
an npm equivalent backed by `scripts/run.mjs`:

```powershell
npm run dev            # backend + frontend, Ctrl+C stops both
npm run dev:backend    # = make run_backend
npm run dev:frontend   # = make run_frontend
npm run install:all    # = make install_backend + make install_frontend
npm run lint           # = make lint
npm run test           # = make tests
npm run help           # list every task
```

## Making changes

1. **Fork** the repo and create a feature branch from `main`.
2. Make your change, **with tests** — new behavior needs a test that fails without it.
3. Run `uv run pytest` and `uv run ruff check .` locally before pushing.
4. Open a pull request using the PR template. Keep PRs focused; smaller is faster to review.

Tips:

- Backend code lives in `src/quoptuna/backend/` (models, optimizer, XAI/fairness) and `src/quoptuna/server/` (FastAPI + services); the web UI is in `frontend/`.
- Quantum model implementations under `src/quoptuna/backend/base/pennylane_models/` follow the [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) conventions.
- Mark tests that train real quantum models with `@pytest.mark.slow`.

## Reporting bugs & requesting features

Use the [issue templates](https://github.com/Qentora/quoptuna/issues/new/choose). For security issues, see [SECURITY.md](SECURITY.md) — please don't open public issues for vulnerabilities.

## Code of Conduct

By participating you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).
