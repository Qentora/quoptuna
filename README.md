# QuOptuna

<div align="center">
  <img src="https://raw.githubusercontent.com/Qentora/quoptuna/refs/heads/main/assets/logo.png" alt="QuOptuna Logo" width="200" height="auto" />
  <h1>QuOptuna</h1>
  <p>
    Bridging quantum computing and hyperparameter optimization for next-generation machine learning
  </p>

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Qentora/quoptuna?utm_source=oss&utm_medium=github&utm_campaign=Qentora%2Fquoptuna&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![License](https://img.shields.io/github/license/Qentora/quoptuna.svg)](https://github.com/Qentora/quoptuna/blob/master/LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%E2%80%933.12-blue.svg)](https://www.python.org/downloads/)

[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Qentora/quoptuna)

</div>

---

## 📚 Table of Contents

- [About](#-about)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌟 About

QuOptuna seamlessly integrates quantum computing capabilities with the powerful Optuna hyperparameter optimization framework. By leveraging quantum algorithms, QuOptuna enables researchers and practitioners to explore optimization landscapes more efficiently, pushing the boundaries of what's possible in machine learning and computational research.

Whether you're working with quantum machine learning models or classical algorithms, QuOptuna provides the tools you need to find optimal hyperparameters faster and more effectively.

## ✨ Key Features

- **🔬 Quantum-Enhanced Optimization**: Specialized hyperparameter tuning algorithms designed specifically for quantum machine learning workflows
- **🎯 Hybrid Model Support**: Seamlessly optimize both quantum and classical models
  - **Quantum Models**: Circuit-Centric Classifier, Data Reuploading Classifier, Quantum Kitchen Sinks, and more
  - **Classical Models**: SVC, MLP Classifier, Perceptron, and other scikit-learn compatible models
- **📊 Interactive Dashboard**: Real-time visualization of optimization progress through an intuitive Streamlit interface
- **🔍 Explainable AI**: Built-in interpretability tools to understand model decisions and optimization trajectories
- **🔌 Extensible Architecture**: Plugin-friendly design for easy integration with custom models and optimization strategies

## 📦 Installation

QuOptuna requires Python 3.11 or 3.12. Install using your preferred package manager:

### Using UV (Recommended)

```bash
uv pip install quoptuna
```

### Using pip

```bash
pip install quoptuna
```

### Development Installation

For contributors and developers:

```bash
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna
uv pip install -e ".[dev]"
```

## 🚀 Quick Start

Get up and running in minutes with this simple example:

```python
import quoptuna as qo

# Define your objective function
def objective(trial):
    """
    Example: Minimize a simple quadratic function
    """
    x = trial.suggest_float('x', -10, 10)
    return x ** 2

# Create and run optimization study
study = qo.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Display results
print(f"Best value: {study.best_value}")
print(f"Best parameters: {study.best_params}")
```

### 🖥️ Launch the Application

QuOptuna bundles a pre-built web UI inside the Python package, so a single command boots the whole app — no Node.js, no repository checkout, just Python:

```bash
# Run straight from PyPI without installing anything permanently
uvx quoptuna

# ...or, in a project/venv that already has quoptuna installed
uv run quoptuna run
```

This starts one FastAPI/uvicorn process that serves both the JSON API and the bundled UI on a single port (defaulting to `:8000`, auto-incrementing if busy), greets you with a gradient ASCII banner, and opens your browser automatically.

| URL | What |
| --- | --- |
| `http://localhost:8000` | Web UI |
| `http://localhost:8000/api/v1/...` | JSON API |
| `http://localhost:8000/api/docs` | Interactive API docs |

Common options:

```bash
# Pick an explicit port and skip auto-opening the browser
uv run quoptuna run --port 8001 --no-browser

# Launch the legacy Streamlit dashboard instead of the full stack
uv run quoptuna run --streamlit
```

Running `uv run quoptuna` (or `uvx quoptuna`) with no subcommand is equivalent to `uv run quoptuna run`.

> **Dev vs packaged.** The command above is the *packaged* mode (one process, one port, served from the wheel). For frontend development with hot reload, use the *dev* mode — two processes — via `make run_cli` (Next.js dev server on `:3000` + FastAPI on `:8000`). See the docs for details.

## 📖 Documentation

Comprehensive documentation, tutorials, and API references are available at:

**[https://Qentora.github.io/quoptuna](https://Qentora.github.io/quoptuna)**

Topics covered include:
- Detailed installation guides
- Quantum algorithm integration
- Advanced optimization techniques
- Custom sampler implementation
- API reference

## 🛠️ Development

We welcome contributions from the community! Here's how to set up your development environment:

### Prerequisites

- Python 3.11 or 3.12
- UV package manager (recommended) or pip
- Node.js 18+ — only for frontend development; the published package ships a pre-built UI, so running QuOptuna does not require Node
- Git

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna

# Install development dependencies
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=quoptuna

# Generate HTML coverage report
uv run pytest --cov=quoptuna --cov-report=html
```

### Code Quality

Maintain code quality with our linting and type-checking tools:

```bash
# Run linter
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Type checking
uv run mypy .
```

## 🤝 Contributing

We're excited to have you contribute to QuOptuna! Here's how you can help:

1. **Fork the repository** on GitHub
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and write tests
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to the branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

Please ensure your code:
- Passes all tests (`pytest`)
- Follows our style guide (`ruff check`)
- Includes appropriate documentation
- Has type hints where applicable

For detailed guidelines, see our [Contributing Guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the Apache 2.0 License. See the [LICENSE](https://github.com/Qentora/quoptuna/blob/master/LICENSE) file for full details.

## 🙏 Acknowledgments

This project builds on the excellent work of:

- **[Wolt Python Package Cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter)** - Project template and structure
- **[XanaduAI's qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks)** - Quantum machine learning benchmarking tools (Apache License 2.0)
- **[Optuna](https://optuna.org/)** - The hyperparameter optimization framework that powers QuOptuna

Special thanks to all our [contributors](https://github.com/Qentora/quoptuna/graphs/contributors) who help make QuOptuna better!

---

## 📊 Project Activity

[![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Recent activity

[![Timeline graph](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_timeline.svg)](https://github.com/Qentora/quoptuna/commits)
[![Issue status graph](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_issues.svg)](https://github.com/Qentora/quoptuna/issues)
[![Pull request status graph](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_prs.svg)](https://github.com/Qentora/quoptuna/pulls)
[![Top contributors](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_users.svg)](https://github.com/Qentora/quoptuna/graphs/contributors)

---

<div align="center">

**[Documentation](https://Qentora.github.io/quoptuna)** • **[Report Bug](https://github.com/Qentora/quoptuna/issues)** • **[Request Feature](https://github.com/Qentora/quoptuna/issues)**

Made with ❤️ by the Qentora team

</div>
