# GitHub Actions Workflows

## Documentation Deployment

### docs-versioned.yml (Recommended)

**Triggers:**
- Push to `main`, `master`, or `development` branches
- Changes to `docs/**`, `mkdocs.yml`, or the workflow file itself
- Manual trigger via `workflow_dispatch`

**Features:**
- Versioned documentation using [mike](https://github.com/jimporter/mike)
- Separate versions for stable and development
- Version selector in documentation
- Automatic version management

**URLs:**
- Stable: `https://qentora.github.io/quoptuna/latest/`
- Development: `https://qentora.github.io/quoptuna/dev/`

**How it works:**
1. Detects branch and assigns version:
   - `main/master` → version: "latest"
   - `development` → version: "dev"
2. Builds documentation with mkdocs
3. Deploys to `gh-pages` branch using mike
4. Updates version selector

### docs.yml

**Triggers:**
- Push to `main` or `master` branch only
- Changes to `docs/**`, `mkdocs.yml`, or the workflow file
- Manual trigger via `workflow_dispatch`

**Features:**
- Simple single-version deployment
- No versioning support
- Faster deployment (no version management overhead)

**URL:**
- `https://qentora.github.io/quoptuna/`

**Use case:**
- Simple documentation without version history
- Quick setup for single-branch projects

### docs-dev.yml

**Triggers:**
- Push to `development` branch
- Pull requests targeting `development` branch
- Changes to `docs/**`, `mkdocs.yml`, or the workflow file

**Features:**
- Builds on PR to catch errors early
- Only deploys on push (not on PR)
- Separate environment for development docs

**URL:**
- `https://qentora.github.io/quoptuna/` (when used alone)

**Use case:**
- Testing documentation changes before merging
- Separate deployment pipeline for development

## Other Workflows

### test.yml
Runs unit tests on push and pull requests.

### draft_release.yml
Creates draft releases automatically.

### release.yml
Publishes releases to PyPI.

### dependencies.yml
Manages dependency updates.

### autofix.yml
Automatically fixes code formatting issues.

### codeflash-optimize.yaml
Code optimization checks.

### cookiecutter.yml
Template updates for the project.

## Choosing a Workflow

**For most projects, use `docs-versioned.yml` because:**
- ✅ Supports multiple versions
- ✅ Easy to maintain documentation history
- ✅ Users can access both stable and development docs
- ✅ Version selector for easy navigation

**Use `docs.yml` if:**
- You only need single-version documentation
- You don't need to support multiple branches
- You want simpler setup

**Use `docs-dev.yml` if:**
- You want separate deployment for development
- You need PR previews
- You want isolated development documentation

## Setup Instructions

See [GITHUB_PAGES_SETUP.md](../../GITHUB_PAGES_SETUP.md) for detailed setup instructions.

## Quick Start

1. **Enable GitHub Pages:**
   - Go to repository **Settings** → **Pages**
   - Set Source to **GitHub Actions**

2. **Push to trigger deployment:**
   ```bash
   git add docs/
   git commit -m "Update documentation"
   git push origin development  # or main/master
   ```

3. **Check deployment:**
   - Go to **Actions** tab
   - Monitor the workflow run
   - Visit the deployed URL once complete

## Troubleshooting

### Build Fails

Check the workflow logs:
1. **Actions** tab → Click failed workflow
2. Review error messages
3. Fix issues locally:
   ```bash
   mkdocs build --strict
   ```

### Permissions Error

1. **Settings** → **Actions** → **General**
2. Set Workflow permissions to "Read and write permissions"

### Pages Not Deploying

1. Ensure GitHub Pages is enabled
2. Check that source is set to "GitHub Actions"
3. Verify workflow completed successfully

## Local Testing

Test documentation locally before pushing:

```bash
# Install dependencies
pip install -r requirements.txt

# Or install with project
pip install -e ".[dev]"

# Build documentation
mkdocs build --strict

# Serve locally
mkdocs serve
# Visit http://127.0.0.1:8000

# Test versioned deployment
mike deploy test
mike serve
```

## Resources

- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Mike](https://github.com/jimporter/mike)
- [GitHub Actions](https://docs.github.com/en/actions)
- [GitHub Pages](https://docs.github.com/en/pages)
