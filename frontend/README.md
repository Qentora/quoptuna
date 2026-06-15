# QuOptuna Frontend (Next.js)

The QuOptuna web UI, built on the Next.js App Router. It provides a guided
optimizer wizard with Streamlit feature parity: dataset loading (UCI / CSV),
data preview, feature & target selection with label remapping, hyperparameter
optimization, SHAP analysis, classification metrics, and AI-generated reports.

## Stack

| Concern          | Tool                          |
| ---------------- | ----------------------------- |
| Framework        | Next.js (App Router, SSR)     |
| UI library       | React + TypeScript            |
| Data fetching    | TanStack React Query          |
| Styling          | Tailwind CSS                  |
| UI primitives    | Radix UI                      |
| State            | Zustand                       |
| Animation        | Motion                        |
| Linting / format | Biome                         |
| E2E testing      | Playwright                    |

## Getting started

```bash
npm install
cp .env.example .env.local   # set NEXT_PUBLIC_API_URL if your backend is elsewhere
npm run dev                  # http://localhost:3000
```

The backend (FastAPI) must be running on the URL set by `NEXT_PUBLIC_API_URL`
(default `http://localhost:8000`). From the repo root, `make run_cli` starts
both services.

## Scripts

| Command            | Description                       |
| ------------------ | --------------------------------- |
| `npm run dev`      | Start the dev server (port 3000)  |
| `npm run build`    | Production build                  |
| `npm run start`    | Serve the production build        |
| `npm run lint`     | Lint with Biome                   |
| `npm run format`   | Format with Biome                 |
| `npm run test:e2e` | Run Playwright end-to-end tests   |

## Structure

```
app/                 App Router routes
  layout.tsx         Root layout (sidebar shell + providers)
  providers.tsx      QueryClient + Toaster
  page.tsx           Home (SSR dashboard)
  optimizer/         Optimizer wizard (client)
  settings/          API key management
components/
  Sidebar.tsx        Navigation
  optimizer/         Wizard step components
lib/
  api.ts             FastAPI client
  hooks.ts           React Query hooks
  settings.ts        localStorage-backed API keys
```
