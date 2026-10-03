---
title: Deploy the documentation site
description: Build, preview, and deploy the Astro + Starlight docs site to GitHub Pages.
---

This docs site is an **Astro + Starlight** project in `docs-site/`. It deploys to GitHub Pages at <https://quoptuna.org> (custom domain, set by `docs-site/public/CNAME`) via a GitHub Actions workflow.

## Build locally

```bash
cd docs-site
npm install
npm run build
```

Static output lands in `docs-site/dist/`.

## Preview locally

```bash
cd docs-site
npm run dev
```

The dev server defaults to <http://localhost:4321>.

## Deploy flow

The `Deploy Documentation` workflow (`.github/workflows/docs.yml`) triggers on pushes to `main` or `master` that touch `docs-site/**` (or the workflow itself), and can be run manually. It runs `npm ci && npm run build` on Node 20, uploads `dist/` with `actions/upload-pages-artifact`, and publishes it with `actions/deploy-pages`.

The site is served from the root of the custom domain, so `astro.config.mjs` sets:

| Setting | Value | Overridable via |
| --- | --- | --- |
| `site` | `https://quoptuna.org` | `DOCS_SITE` env |
| `base` | `/` | `DOCS_BASE` env |

The env overrides exist for PR preview builds.

## Pull request previews

Pull requests that touch `docs-site/**` run the `Docs Preview` workflow (`.github/workflows/docs-preview.yml`), which builds the site and posts a sticky comment on the PR. No preview host is configured yet, so the comment confirms a successful build rather than linking to a live preview; wiring a host into the workflow's `TODO(host)` block makes it post the preview URL.

## Next steps

- [Generate an AI analysis report](/how-to/generate-reports/)
- [Configuration reference](/reference/configuration/)
- [CLI reference](/reference/cli/)
