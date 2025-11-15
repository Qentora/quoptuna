# QuOptuna Documentation

This directory contains the source files for the QuOptuna documentation website, built with [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## 📁 Documentation Structure

```
docs/
├── README.md                    # This file - documentation overview
├── index.md                     # Homepage - Quick start and overview
├── user_guide.md                # Complete UI workflow guide
├── api.md                       # Auto-generated API reference
├── examples.md                  # Code examples and tutorials
├── changelog.md                 # Version history
├── assets/                      # Images and static files
│   └── logo.png
└── guides/                      # Detailed guides
    ├── python-api-guide.md      # Python API usage guide
    ├── streamlit-guide.md       # Streamlit interface guide
    ├── github-pages-setup.md    # GitHub Pages deployment
    └── github-settings-guide.md # GitHub repository setup
```

## 📖 Documentation Pages

### Core Documentation

- **[index.md](index.md)** - Landing page with quick start, features, and installation
- **[user_guide.md](user_guide.md)** - Step-by-step guide for the Streamlit UI
- **[api.md](api.md)** - Auto-generated API reference using mkdocstrings
- **[examples.md](examples.md)** - Practical code examples and use cases
- **[changelog.md](changelog.md)** - Version history and release notes

### Guides (guides/)

- **[python-api-guide.md](guides/python-api-guide.md)** - Comprehensive Python API guide with examples
- **[streamlit-guide.md](guides/streamlit-guide.md)** - Detailed Streamlit interface documentation
- **[github-pages-setup.md](guides/github-pages-setup.md)** - How to deploy docs to GitHub Pages
- **[github-settings-guide.md](guides/github-settings-guide.md)** - Repository settings configuration

## 🏗️ Building Documentation

### Local Development

Build and serve documentation locally:

```bash
# Install dependencies
pip install mkdocs-material mkdocs-autorefs mkdocstrings[python] mkdocs-glightbox

# Serve locally with live reload
mkdocs serve

# Build static site
mkdocs build

# Build with strict mode (catches errors)
mkdocs build --strict
```

The documentation will be available at `http://127.0.0.1:8000`

### Deployment

Documentation is automatically deployed to GitHub Pages using GitHub Actions:

- **Main/Master branch** → `https://qentora.github.io/quoptuna/latest/`
- **Development branch** → `https://qentora.github.io/quoptuna/dev/`

See [github-pages-setup.md](guides/github-pages-setup.md) for detailed deployment information.

## ✏️ Contributing to Documentation

### Adding a New Page

1. Create a new markdown file in the appropriate location:
   - Core docs: `docs/`
   - Guides: `docs/guides/`

2. Update `mkdocs.yml` navigation:
   ```yaml
   nav:
     - Home: 'index.md'
     - Your Page: 'your-page.md'
   ```

3. Test locally:
   ```bash
   mkdocs serve
   ```

### Writing Style Guidelines

- Use clear, concise language
- Include code examples where appropriate
- Add links to related documentation
- Use proper markdown formatting
- Include screenshots for UI features
- Test all code examples before committing

### Markdown Features

The documentation supports these MkDocs/Material extensions:

- **Code blocks** with syntax highlighting
- **Admonitions** (notes, warnings, tips)
- **Tabs** for organizing content
- **Mermaid diagrams** for flowcharts
- **Math** expressions with MathJax
- **Icons** and emojis
- **Code annotations**

Example:
```python
def example():
    """Example with annotation."""
    return "Hello"  # (1)!
```

1. This is a code annotation

## 🔗 Links and References

### Internal Links

Use relative paths for internal links:

```markdown
See the [User Guide](user_guide.md) for more information.
See the [Python API Guide](guides/python-api-guide.md) for details.
```

### API Documentation

The `api.md` file uses mkdocstrings to auto-generate API docs from docstrings:

```markdown
::: quoptuna.Optimizer
    options:
      members: true
      show_root_heading: true
```

### Cross-References

Use autorefs for automatic cross-referencing:

```markdown
See [DataPreparation][quoptuna.DataPreparation] for data handling.
```

## 📋 Documentation Checklist

Before submitting documentation changes:

- [ ] Tested locally with `mkdocs serve`
- [ ] All links work correctly
- [ ] Code examples are tested and working
- [ ] Spelling and grammar checked
- [ ] Images are optimized and properly referenced
- [ ] Navigation updated in `mkdocs.yml` if needed
- [ ] Build passes with `mkdocs build --strict`

## 🐛 Common Issues

### Build Fails

**Problem:** `mkdocs build` fails with errors

**Solutions:**
- Check for broken internal links
- Verify all referenced files exist
- Ensure valid YAML in front matter
- Check for unsupported markdown syntax

### Auto-generated API Docs Not Working

**Problem:** mkdocstrings not generating API docs

**Solutions:**
- Verify `mkdocstrings[python]` is installed
- Check that Python modules are importable
- Ensure docstrings follow Google or NumPy style
- Check `mkdocs.yml` mkdocstrings configuration

### Images Not Displaying

**Problem:** Images don't show in docs

**Solutions:**
- Use relative paths: `![Logo](assets/logo.png)`
- Ensure images are in `docs/assets/` or subdirectories
- Check image file extensions are lowercase
- Verify image files are committed to git

## 📚 Additional Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings Documentation](https://mkdocstrings.github.io/)
- [Markdown Guide](https://www.markdownguide.org/)

## 📝 License

Documentation is part of QuOptuna and follows the same MIT license.

---

**Need Help?** Open an issue on [GitHub](https://github.com/Qentora/quoptuna/issues)
