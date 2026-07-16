---
title: Deploy the documentation site
description: Build, preview, and deploy the Astro + Starlight docs site to GitHub Pages.
---

This docs site is an **Astro + Starlight** project in `docs-site/`. It deploys to GitHub Pages at <https://Qentora.github.io/quoptuna> via a GitHub Actions workflow.

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

A GitHub Actions workflow triggers on pushes to `main` that touch `docs-site/**`. The workflow runs `npm ci && npm run build` and publishes `dist/` with `actions/deploy-pages`.

Because this is a project page (not a user/org root page), `astro.config.mjs` sets:

| Setting | Value | Overridable via |
| --- | --- | --- |
| `site` | `https://Qentora.github.io` | `DOCS_SITE` env |
| `base` | `/quoptuna` | `DOCS_BASE` env |

The env overrides exist for PR preview builds.

## Pull request previews

Pull requests get an automatic preview deployment. The preview URL is posted as a comment on the PR, so reviewers can see the rendered docs before merge.

## Next steps

- [Generate an AI analysis report](/how-to/generate-reports/)
- [Configuration reference](/reference/configuration/)
- [CLI reference](/reference/cli/)
