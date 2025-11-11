# GitHub Settings Configuration Guide

This guide walks you through the required GitHub repository settings to enable automatic documentation deployment.

## Prerequisites

- You must have **admin access** to the repository
- The workflows must be committed to the repository
- You need to be on GitHub.com (not local)

---

## Step 1: Enable GitHub Pages

### 1.1 Navigate to Repository Settings

1. Go to your repository on GitHub: `https://github.com/Qentora/quoptuna`
2. Click the **⚙️ Settings** tab (top right of the repository page)
   - If you don't see this tab, you don't have admin access

### 1.2 Access Pages Settings

1. In the left sidebar, scroll down to the **Code and automation** section
2. Click on **Pages**

### 1.3 Configure Pages Source

1. Under **Build and deployment** section:
   - **Source**: Select **GitHub Actions** from the dropdown
     ```
     Before: Deploy from a branch
     After:  GitHub Actions ✓
     ```
   - You should see: "Your site is ready to be published at https://qentora.github.io/quoptuna/"

2. **Do NOT select a branch** - GitHub Actions will handle deployment

3. The page should now show:
   ```
   Source
   GitHub Actions

   Your GitHub Pages site is currently being built from the GitHub Actions workflow.
   ```

4. Click anywhere else to auto-save (no save button needed)

### 1.4 Optional: Custom Domain (Skip if not needed)

If you want to use a custom domain like `docs.quoptuna.com`:

1. Under **Custom domain**, enter your domain
2. Click **Save**
3. Wait for DNS check to complete
4. Enable **Enforce HTTPS** after DNS verification

---

## Step 2: Configure Workflow Permissions

This is **critical** - without this, the workflows cannot deploy to GitHub Pages.

### 2.1 Navigate to Actions Settings

1. Still in **⚙️ Settings**, find **Code and automation** section in left sidebar
2. Click on **Actions**
3. Click on **General** (under Actions)

### 2.2 Set Workflow Permissions

Scroll down to the **Workflow permissions** section:

1. Select: **Read and write permissions** (this is critical!)
   ```
   ○ Read repository contents and packages permissions (default)
   ● Read and write permissions ← SELECT THIS
   ```

2. Keep checked: ✅ **Allow GitHub Actions to create and approve pull requests**

3. Click **Save** button at the bottom

### 2.3 Why This Is Important

The workflows need write permissions to:
- Push to the `gh-pages` branch
- Deploy documentation to GitHub Pages
- Update version information with mike

---

## Step 3: Verify Branch Protection (Optional but Recommended)

### 3.1 Navigate to Branches

1. In **⚙️ Settings**, click **Branches** (under Code and automation)

### 3.2 Add Branch Protection Rule for Development

1. Click **Add branch protection rule**

2. Configure:
   - **Branch name pattern**: `development`

3. Enable these settings:
   - ✅ **Require a pull request before merging**
     - ✅ Require approvals: 1 (or more)

   - ✅ **Require status checks to pass before merging**
     - ✅ Require branches to be up to date before merging
     - Search and select: `build` (from docs workflows)

   - ✅ **Require conversation resolution before merging**

   - ✅ **Do not allow bypassing the above settings**

4. Click **Create** or **Save changes**

### 3.3 Branch Protection for Main/Master (Recommended)

Repeat the same process for your main branch:
- **Branch name pattern**: `main` (or `master`)
- Same settings as development
- This prevents accidental direct pushes to production

---

## Step 4: Enable GitHub Actions (if disabled)

### 4.1 Check if Actions Are Enabled

1. In **⚙️ Settings** → **Actions** → **General**
2. Under **Actions permissions**, ensure one of these is selected:
   - ✅ **Allow all actions and reusable workflows** (recommended for public repos)
   - ✅ **Allow enterprise, and select non-enterprise, actions and reusable workflows**

3. If actions were disabled, click **Save**

---

## Step 5: Configure Environments (Automatic)

GitHub will automatically create these environments when workflows run:
- `github-pages` (for stable docs from main)
- `github-pages-dev` (for dev docs from development)

You can view/configure them later at:
- **⚙️ Settings** → **Environments**

### Optional Environment Configuration

After the first deployment, you can add protection rules:

1. Go to **Settings** → **Environments**
2. Click on `github-pages` or `github-pages-dev`
3. Add **Deployment protection rules**:
   - Required reviewers
   - Wait timer
   - Deployment branches (restrict to specific branches)

---

## Step 6: Trigger Your First Deployment

### Option A: Push to Development Branch

```bash
# If you're on the claude branch, merge to development
git checkout development
git merge claude/jupyter-dataset-experiments-011CUsbYuf2cF6t4P6NWtFfg
git push origin development
```

### Option B: Merge Pull Request

1. Create a PR from your claude branch to `development`
2. The workflow will build (but not deploy) the docs
3. After PR is merged, the workflow will deploy to `/dev/`

### Option C: Manual Trigger

1. Go to **Actions** tab in your repository
2. Select **Deploy Versioned Docs** workflow
3. Click **Run workflow** button
4. Select branch: `development`
5. Click **Run workflow**

---

## Step 7: Monitor Deployment

### 7.1 Check Workflow Status

1. Go to **Actions** tab
2. You should see a workflow running: **Deploy Versioned Docs**
3. Click on it to see progress
4. Wait for all jobs to complete (green checkmarks)

### 7.2 Check for Errors

If the workflow fails:

1. Click on the failed job
2. Expand the step that failed (red X)
3. Read the error message
4. Common issues:
   - **Permission denied**: Go back to Step 2, ensure "Read and write permissions" is selected
   - **Pages not enabled**: Go back to Step 1
   - **Build errors**: Check the documentation locally with `mkdocs build --strict`

### 7.3 Verify Deployment

After workflow succeeds:

1. Go back to **Settings** → **Pages**
2. You should see:
   ```
   Your site is live at https://qentora.github.io/quoptuna/
   ```
3. Click **Visit site** to view your documentation

---

## Step 8: Access Your Documentation

### Development Documentation

After pushing to `development` branch:
- **URL**: `https://qentora.github.io/quoptuna/dev/`
- Updates automatically on every push to development

### Stable Documentation

After merging to `main` or `master`:
- **URL**: `https://qentora.github.io/quoptuna/latest/`
- Also available at: `https://qentora.github.io/quoptuna/`

### Version Selector

Users can switch between versions using the selector in the documentation header.

---

## Troubleshooting

### Issue 1: "Workflow not permitted to deploy to environment"

**Solution:**
1. **Settings** → **Actions** → **General**
2. Ensure "Read and write permissions" is selected
3. Click **Save**

### Issue 2: "Pages build and deployment failed"

**Solution:**
1. **Settings** → **Pages**
2. Ensure source is set to **GitHub Actions** (not "Deploy from a branch")

### Issue 3: Workflow runs but docs don't update

**Solution:**
1. Check if the `gh-pages` branch exists
2. Go to repository → **Branches**
3. If `gh-pages` doesn't exist, the first deployment creates it
4. Wait a few minutes and check again

### Issue 4: 404 error when visiting docs URL

**Possible causes:**
1. Workflow hasn't completed yet - wait and refresh
2. GitHub Pages not enabled - go to Step 1
3. Path is wrong - use `/dev/` or `/latest/` not just root

**Solution:**
1. Verify workflow completed successfully (green checkmark)
2. Wait 2-3 minutes for GitHub Pages to propagate
3. Clear browser cache
4. Try incognito/private browsing mode

### Issue 5: Old version still showing

**Solution:**
```bash
# Clear version cache with mike
pip install mike
mike delete old-version --push
mike deploy new-version latest --push --update-aliases
```

---

## Quick Reference Checklist

Before workflows can deploy docs, ensure:

- [ ] GitHub Pages enabled (Settings → Pages → Source: GitHub Actions)
- [ ] Workflow permissions set to "Read and write" (Settings → Actions → General)
- [ ] Workflows committed and pushed to repository
- [ ] Branch exists (`development` or `main`)
- [ ] Changes to `docs/` or `mkdocs.yml` pushed

---

## Summary of Settings

| Setting | Location | Required Value |
|---------|----------|----------------|
| Pages Source | Settings → Pages | GitHub Actions |
| Workflow Permissions | Settings → Actions → General | Read and write permissions |
| Actions | Settings → Actions → General | Allow all actions |
| Branch | N/A | `development` or `main` exists |

---

## Next Steps After Setup

1. **Test the deployment:**
   - Make a small change to any file in `docs/`
   - Commit and push to `development`
   - Check Actions tab for workflow run
   - Visit the dev docs URL

2. **Set up branch protection** (recommended)
   - Protect `development` and `main` branches
   - Require PR reviews
   - Require status checks

3. **Customize documentation:**
   - Update `docs/index.md`
   - Add more documentation pages
   - Customize `mkdocs.yml` theme

4. **Monitor deployments:**
   - Subscribe to workflow notifications
   - Check Actions tab regularly
   - Set up deployment notifications

---

## Getting Help

If you encounter issues not covered here:

1. Check workflow logs in **Actions** tab
2. Review [GitHub Pages Setup](github-pages-setup.md) for detailed troubleshooting
3. Consult [GitHub Pages documentation](https://docs.github.com/en/pages)
4. Check [mike documentation](https://github.com/jimporter/mike)
5. Open an issue in the repository

---

## Visual Guide

### Settings → Pages
```
┌─────────────────────────────────────┐
│ Build and deployment                │
│                                     │
│ Source                              │
│ [GitHub Actions ▼]  ← SELECT THIS  │
│                                     │
│ Your site is live at:               │
│ https://qentora.github.io/quoptuna/ │
└─────────────────────────────────────┘
```

### Settings → Actions → General → Workflow permissions
```
┌─────────────────────────────────────┐
│ Workflow permissions                │
│                                     │
│ ○ Read repository contents and     │
│   packages permissions              │
│                                     │
│ ● Read and write permissions        │
│   ← SELECT THIS                     │
│                                     │
│ ✓ Allow GitHub Actions to create   │
│   and approve pull requests         │
│                                     │
│ [Save]                              │
└─────────────────────────────────────┘
```

---

**Last Updated**: November 2024
**Repository**: https://github.com/Qentora/quoptuna
