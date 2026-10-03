# QuOptuna documentation site

The QuOptuna docs, built with [Astro](https://astro.build) +
[Starlight](https://starlight.astro.build) and organized on the
[Diátaxis](https://diataxis.fr) framework (tutorials · how-to · reference ·
explanation).

## Develop

```bash
npm install
npm run dev        # live preview at http://localhost:4321/
npm run build      # static output in dist/ (mirrors CI)
npm run preview    # serve the built dist/
```

## Structure

- `src/content/docs/**` — page content (Markdown / MDX with Starlight frontmatter)
- `src/content/docs/index.mdx` — marketing landing page (renders `src/components/Landing.astro`)
- `src/components/` — landing-page components, `SiteTitle.astro`, and `landing.css`
- `src/styles/theme.css` — brand theme (maps QuOptuna's quantum-purple / classical-orange onto Starlight tokens)
- `astro.config.mjs` — site config + Diátaxis sidebar. `site`/`base` default to
  `https://quoptuna.org` and `/` and can be overridden with the `DOCS_SITE` /
  `DOCS_BASE` environment variables (used for PR previews).

## Deploy

Pushed to GitHub Pages at `https://quoptuna.org` by
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) on changes to
`docs-site/`. Pull requests get a build check via
[`docs-preview.yml`](../.github/workflows/docs-preview.yml); no preview host is
configured yet, so it posts a build-status comment rather than a preview URL.
