# QuOptuna

<div align="center">
  <img src="assets/logo.png" alt="QuOptuna Logo" width="200" height="auto" />
  <h1>QuOptuna</h1>
  <p>
    Bridging quantum computing and hyperparameter optimization for next-generation machine learning
  </p>

![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Qentora/quoptuna?utm_source=oss&utm_medium=github&utm_campaign=Qentora%2Fquoptuna&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![License](https://img.shields.io/github/license/Qentora/quoptuna.svg)](https://github.com/Qentora/quoptuna/blob/master/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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

QuOptuna requires Python 3.8 or higher. Install using your preferred package manager:

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

### 📈 Launch Interactive Dashboard

Monitor your optimization progress in real-time:

```bash
quoptuna --start
```

This launches a Streamlit dashboard where you can visualize optimization history, parameter importance, and convergence patterns.

## 📖 Documentation

Comprehensive documentation is available at **[https://Qentora.github.io/quoptuna](https://Qentora.github.io/quoptuna)**

### Documentation Structure

- **[User Guide](https://qentora.github.io/quoptuna/user_guide/)** - Step-by-step Streamlit interface walkthrough
- **[API Reference](https://qentora.github.io/quoptuna/api/)** - Auto-generated API documentation
- **[Examples](https://qentora.github.io/quoptuna/examples/)** - Practical code examples and tutorials
- **[Python API Guide](https://qentora.github.io/quoptuna/guides/python-api-guide/)** - Comprehensive Python usage guide
- **[Streamlit Guide](https://qentora.github.io/quoptuna/guides/streamlit-guide/)** - Detailed interface documentation
- **[Changelog](https://qentora.github.io/quoptuna/changelog/)** - Version history and updates

### Local Documentation

To build and view documentation locally:

```bash
# Install documentation dependencies
pip install mkdocs-material mkdocs-autorefs mkdocstrings[python] mkdocs-glightbox

# Serve documentation locally
mkdocs serve

# Build static documentation
mkdocs build
```

For contributors, see [docs/about-docs.md](docs/about-docs.md) for documentation guidelines.

## 🛠️ Development

We welcome contributions from the community! Here's how to set up your development environment:

### Prerequisites

- Python 3.8 or higher
- UV package manager (recommended) or pip
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
