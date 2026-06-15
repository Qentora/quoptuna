# GitHub Pages Setup Guide

## Overview

QuOptuna uses GitHub Actions to automatically build and deploy documentation to GitHub Pages. The setup supports multiple versions:

- **Latest (Stable)**: Documentation from `main` or `master` branch
- **Development**: Documentation from `development` branch

## Architecture

We use [mike](https://github.com/jimporter/mike) for documentation versioning, which allows us to maintain multiple versions of the documentation simultaneously.

### Workflow Files

Three GitHub Actions workflows handle documentation deployment:

1. **`docs-versioned.yml`** (Recommended):
   - Deploys versioned docs using mike
   - Supports both `main/master` and `development` branches
   - URLs:
     - Stable: `https://<org>.github.io/<repo>/latest/`
     - Dev: `https://<org>.github.io/<repo>/dev/`

2. **`docs.yml`**:
   - Simple deployment for main/master branch only
   - Single version deployment

3. **`docs-dev.yml`**:
   - Development branch specific deployment
   - Builds on PRs but only deploys on push

## Initial Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select:
   - Source: **GitHub Actions**
   - (No need to select a branch when using GitHub Actions)

### 2. Configure Branch Protection (Optional but Recommended)

For the `development` branch:

1. Go to **Settings** → **Branches**
2. Add a branch protection rule for `development`
3. Enable:
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - Select the "build" check from the docs workflow

## How It Works

### Versioned Deployment (docs-versioned.yml)

When you push to `main/master` or `development`:

1. **Build Phase**:
   - Checkout code
   - Install Python and dependencies (mkdocs-material, mike, etc.)
   - Determine version based on branch:
     - `main/master` → version: "latest"
     - `development` → version: "dev"

2. **Deploy Phase**:
   - Use `mike deploy` to publish versioned docs
   - Update the `gh-pages` branch with the new version
   - Set default version to "latest" (for main/master only)

### Version Selector

Users can switch between versions using the version selector in the docs (top of the page).

### Directory Structure on gh-pages

```
gh-pages/
├── latest/           # Stable docs from main/master
│   ├── index.html
│   └── ...
├── dev/              # Development docs
│   ├── index.html
│   └── ...
└── versions.json     # Version metadata for mike
```

## Manual Deployment

### Using Mike Locally

You can also deploy documentation manually:

```bash
# Install mike
pip install mike

# Deploy a new version
mike deploy <version> <alias> --push

# Examples:
mike deploy 1.0.0 latest --push          # Deploy version 1.0.0 as latest
mike deploy dev --push                    # Deploy dev version
mike deploy 2.0.0 stable --push --update-aliases  # Deploy and update alias

# Set default version
mike set-default latest --push

# List all versions
mike list

# Delete a version
mike delete <version> --push

# Serve locally to preview
mike serve
```

## Customization

### Change Version Names

Edit `.github/workflows/docs-versioned.yml`:

```yaml
- name: Determine version name
  id: version
  run: |
    if [[ "${{ github.ref }}" == "refs/heads/main" ]]; then
      echo "version=v1.0" >> $GITHUB_OUTPUT    # Change version name
      echo "alias=stable" >> $GITHUB_OUTPUT    # Change alias
    fi
```

### Add More Branches

Add more branches to the workflow trigger:

```yaml
on:
  push:
    branches:
      - main
      - master
      - development
      - staging        # Add new branch
      - beta          # Add another branch
```

Then add logic in the "Determine version name" step:

```yaml
elif [[ "${{ github.ref }}" == "refs/heads/staging" ]]; then
  echo "version=staging" >> $GITHUB_OUTPUT
  echo "alias=staging" >> $GITHUB_OUTPUT
```

### Custom Domain

To use a custom domain:

1. Add a `CNAME` file to your `docs/` directory:
   ```
   docs.yourdomain.com
   ```

2. Configure DNS:
   - Add a CNAME record pointing to `<org>.github.io`

3. Update `mkdocs.yml`:
   ```yaml
   site_url: https://docs.yourdomain.com
   ```

## Workflow Triggers

### On Push to Development

```yaml
on:
  push:
    branches:
      - development
    paths:
      - 'docs/**'
      - 'mkdocs.yml'
```

This triggers when:
- Changes are pushed to the `development` branch
- Files in `docs/` or `mkdocs.yml` are modified

### On Pull Request

```yaml
on:
  pull_request:
    branches:
      - development
```

This builds (but doesn't deploy) docs when a PR is created to `development`, allowing you to catch build errors before merging.

## Permissions

The workflows require these permissions:

```yaml
permissions:
  contents: write    # For pushing to gh-pages branch
  pages: write       # For deploying to GitHub Pages
  id-token: write    # For authentication
```

## Troubleshooting

### Build Fails

**Error: `mkdocs build` fails**

Check the build logs:
1. Go to **Actions** tab
2. Click on the failed workflow
3. Review the "Build documentation" step

Common issues:
- Missing dependencies in workflow
- Broken links in markdown
- Invalid mkdocs.yml syntax

**Fix:**
```bash
# Test locally
mkdocs build --strict --verbose
```

### Deployment Fails

**Error: Permission denied**

Check repository settings:
1. **Settings** → **Actions** → **General**
2. Under **Workflow permissions**, select:
   - ✅ Read and write permissions

**Error: Pages deployment failed**

Ensure GitHub Pages is enabled:
1. **Settings** → **Pages**
2. Source should be **GitHub Actions**

### Version Not Showing

**Version selector is empty**

1. Ensure `extra.version.provider: mike` is in `mkdocs.yml`
2. Check that `mike` is installed in the workflow
3. Verify `versions.json` exists in gh-pages branch

## Testing

### Test Locally

```bash
# Install dependencies
pip install mkdocs-material mike mkdocs-autorefs mkdocstrings[python] mkdocs-glightbox

# Build locally
mkdocs build

# Serve locally
mkdocs serve

# Test versioned deployment locally
mike deploy test --push
mike serve
```

### Test Workflow

Create a test branch:

```bash
git checkout -b test-docs
# Make changes to docs
git add docs/
git commit -m "Test docs changes"
git push -u origin test-docs
```

Then create a PR to `development` to trigger the build workflow.

## Best Practices

### 1. Always Review Locally First

```bash
mkdocs serve
# Visit http://127.0.0.1:8000
```

### 2. Use Strict Mode

This catches issues early:

```bash
mkdocs build --strict
```

### 3. Keep Versions Clean

Delete old versions when no longer needed:

```bash
mike delete old-version --push
```

### 4. Use Aliases Wisely

Aliases make it easy to update versions:

```bash
# Users can always go to /latest/ to get current stable version
mike deploy 1.2.0 latest --update-aliases --push
```

### 5. Document Breaking Changes

When deploying a new major version, consider:
- Keeping old version accessible
- Adding migration guide
- Using version warnings

## URLs

After setup, your documentation will be available at:

- **Stable/Latest**: `https://<org>.github.io/<repo>/latest/`
- **Development**: `https://<org>.github.io/<repo>/dev/`
- **Root**: `https://<org>.github.io/<repo>/` (redirects to default version)

For QuOptuna:
- **Stable**: `https://qentora.github.io/quoptuna/latest/`
- **Development**: `https://qentora.github.io/quoptuna/dev/`

## Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [Mike Documentation](https://github.com/jimporter/mike)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

## Support

If you encounter issues:

1. Check workflow logs in the **Actions** tab
2. Review the [GitHub Pages Status](https://www.githubstatus.com/)
3. Test locally with `mkdocs build --strict`
4. Open an issue in the repository

## License

This setup is part of QuOptuna and follows the same MIT license.
