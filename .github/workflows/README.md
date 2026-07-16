# GitHub Actions Workflows

## Documentation

The documentation site lives in [`docs-site/`](../../docs-site) and is built
with **[Astro](https://astro.build) + [Starlight](https://starlight.astro.build)**.
It deploys to GitHub Pages at `https://Qentora.github.io/quoptuna/`.

### docs.yml — Deploy Documentation

**Triggers:**
- Push to `main` or `master`
- Changes to `docs-site/**` or the workflow file
- Manual trigger via `workflow_dispatch`

**How it works:**
1. `npm ci` in `docs-site/`
2. `npm run build` (Astro produces static output in `docs-site/dist/`)
3. Publishes `dist/` to GitHub Pages via `actions/deploy-pages`

### docs-preview.yml — Docs Preview

**Triggers:**
- Pull requests touching `docs-site/**` (or the docs workflows)

**How it works:**
1. Builds the site (catching broken links / build breaks on every PR)
2. Deploys a per-PR preview and posts the URL as a sticky PR comment

:::note
The preview **deploy host is intentionally pluggable** — fill in the
`TODO(host)` block in `docs-preview.yml` (Cloudflare Pages / Netlify / Vercel /
GitHub Pages subpath) and add the matching repo secrets. Until then the workflow
still builds the site and comments that no preview host is configured.
:::

The `docs` job in `test.yml` also runs `npm run build` on every PR as a fast
build check.

## Other Workflows

### test.yml
Runs unit tests on push and pull requests.

### draft_release.yml
Creates draft releases automatically.

### release.yml
Publishes releases to PyPI.

### dependencies.yml
Manages dependency updates.

### autofix.yml (`autofix.ci`)
Runs ruff safe lint fixes (excluding `F401` to preserve side-effect imports)
and `ruff format` on the Python files changed in a pull request, then hands the
working-tree diff to the [autofix.ci](https://github.com/apps/autofix-ci) App,
which pushes the fix commit back to the PR branch. Covers both the root package
(`src/quoptuna`, `tests`) and the `backend/` directory; ruff resolves the
closest `pyproject.toml` per file.

### lint.yml
Strict checks on the Python files changed in a pull request: `ruff check
--no-fix` (lint violations fail the build) plus `mypy`, split by project so each
file is checked with its own config (`src/quoptuna` from `src/`, `tests` from
the repo root, `backend/app` from `backend/`).

### update-uv-lock.yml
Regenerates `uv.lock` and `backend/uv.lock` whenever `pyproject.toml` or
`backend/pyproject.toml` changes (push to `main`, pull requests, or manual
dispatch) and commits the refreshed lockfiles back via the `github-actions[bot]`
identity.

### codeflash-optimize.yaml
Code optimization checks.

### cookiecutter.yml
Template updates for the project.

## Setup Instructions

1. **Enable GitHub Pages:**
   - Go to repository **Settings** → **Pages**
   - Set Source to **GitHub Actions**

2. **Push to trigger deployment:**
   ```bash
   git add docs-site/
   git commit -m "Update documentation"
   git push origin main
   ```

3. **Check deployment:**
   - Go to the **Actions** tab and monitor the *Deploy Documentation* run
   - Visit `https://Qentora.github.io/quoptuna/` once complete

## Local Testing

Build and preview the docs locally before pushing:

```bash
cd docs-site
npm install
npm run build      # static output in dist/ (mirrors CI)
npm run dev        # live preview at http://localhost:4321/quoptuna/
```

## Troubleshooting

### Build Fails
1. **Actions** tab → click the failed workflow and review the logs
2. Reproduce locally with `cd docs-site && npm run build`

### Pages Not Deploying
1. Ensure GitHub Pages is enabled with source **GitHub Actions**
2. Verify the *Deploy Documentation* workflow completed successfully

## Resources

- [Astro](https://astro.build/)
- [Starlight](https://starlight.astro.build/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [GitHub Pages](https://docs.github.com/en/pages)
