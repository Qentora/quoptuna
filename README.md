# QuOptuna

<div align="center">
  <img src="assets/logo.png" alt="logo" width="200" height="auto" />
  <h1>QuOptuna</h1>
  <p>
    Integrating quantum computing with Optuna for enhanced hyperparameter optimization.
  </p>
  <p>
    <a href="https://img.shields.io/pypi/v/quoptuna?style=flat-round">
      <img src="https://img.shields.io/pypi/v/quoptuna?style=flat-round" alt="PyPI Version" />
    </a>
    <a href="https://img.shields.io/pypi/pyversions/quoptuna?style=flat-round">
      <img src="https://img.shields.io/pypi/pyversions/quoptuna?style=flat-round" alt="Python Versions" />
    </a>
    <a href="https://img.shields.io/pypi/l/quoptuna?style=flat-round">
      <img src="https://img.shields.io/pypi/l/quoptuna?style=flat-round" alt="License" />
    </a>
    <a href="https://readthedocs.org/projects/quoptuna/badge/?version=latest">
      <img src="https://readthedocs.org/projects/quoptuna/badge/?version=latest" alt="Documentation Status" />
    </a>
    <a href="https://codecov.io/gh/Qentora/quoptuna">
      <img src="https://codecov.io/gh/Qentora/quoptuna/graph/badge.svg?token=6QE861D1CB" alt="codecov" />
    </a>
    <a href="https://github.com/Qentora/quoptuna/graphs/contributors">
      <img src="https://img.shields.io/github/contributors/Qentora/quoptuna" alt="contributors" />
    </a>
    <a href="https://github.com/Qentora/quoptuna/network/members">
      <img src="https://img.shields.io/github/forks/Qentora/quoptuna" alt="forks" />
    </a>
    <a href="https://github.com/Qentora/quoptuna/stargazers">
      <img src="https://img.shields.io/github/stars/Qentora/quoptuna" alt="stars" />
    </a>
    <a href="https://github.com/Qentora/quoptuna/issues/">
      <img src="https://img.shields.io/github/issues/Qentora/quoptuna" alt="open issues" />
    </a>
    <a href="https://github.com/Qentora/quoptuna/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/Qentora/quoptuna.svg" alt="license" />
    </a>
  </p>
</div>

---

## :notebook_with_decorative_cover: Table of Contents

- [About the Project](#star2-about-the-project)
- [Features](#dart-features)
- [Installation](#gear-installation)
- [Quick Start](#eyes-quick-start)
- [Documentation](#book-documentation)
- [Development](#toolbox-development)
- [Contributing](#wave-contributing)
- [License](#warning-license)
- [Acknowledgments](#gem-acknowledgments)

---

## :star2: About the Project

QuOptuna combines quantum computing with advanced hyperparameter optimization to push the boundaries of machine learning and computational tasks.

### :dart: Features

- **Quantum-ML Optimization**: Specialized hyperparameter tuning for Quantum algorithms
- **Multiple Model Support**: Compatible with both classical and quantum models, including:
  - Quantum Models: Circuit-Centric Classifier, Data Reuploading Classifier, Quantum Kitchen Sinks, and more
  - Classical Models: SVC, MLP Classifier, Perceptron
- **Interactive Dashboard**: Visualize optimization progress in real-time
- **Explainable AI**: Built-in tools for model interpretability
- **Extensible Framework**: Easy integration with custom models and algorithms

## :gear: Installation

Install QuOptuna using UV (recommended) or pip:

```sh
# Using UV (recommended)
uv pip install quoptuna

# Using pip
pip install quoptuna
```

## :eyes: Quick Start

Here's how you can get started:

```python
import quoptuna as qo

def objective(trial):
    # Define your optimization logic here
    return trial.suggest_float('x', -10, 10) ** 2

study = qo.create_study()
study.optimize(objective, n_trials=100)

print("Best trial:", study.best_trial)
```

### Interactive Dashboard

QuOptuna includes a Streamlit-based dashboard for real-time optimization monitoring:

```sh
quoptuna --start
```

## :book: Documentation

Comprehensive documentation is available at [https://Qentora.github.io/quoptuna](https://Qentora.github.io/quoptuna).

## :toolbox: Development

To contribute to QuOptuna, follow these steps:

### Clone the Repository

```sh
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna
```

### Install Requirements

1. Ensure UV and Python 3.8+ are installed.
2. Install development dependencies:

```sh
uv pip install -e ".[dev]"
```

### Testing

Run tests with:

```sh
uv run pytest
```

To check code coverage:

```sh
uv run pytest --cov=quoptuna
```

### Code Quality

Run linting and type checks:

```sh
uv run ruff check .
uv run mypy .
```

## :wave: Contributing

We welcome contributions! To get started:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Refer to the Contributing Guidelines for more details.

## :warning: License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details [here](https://github.com/Qentora/quoptuna?tab=Apache-2.0-1-ov-file).

## :gem: Acknowledgments

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.

This project includes code from the [qml-benchmarks project by XanaduAI](https://github.com/XanaduAI/qml-benchmarks). Licensed under the Apache License, Version 2.0.

---

## [![Repography logo](https://images.repography.com/logo.svg)](https://repography.com) / Recent activity [![Time period](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_badge.svg)](https://repography.com)
[![Timeline graph](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_timeline.svg)](https://github.com/Qentora/quoptuna/commits)
[![Issue status graph](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_issues.svg)](https://github.com/Qentora/quoptuna/issues)
[![Pull request status graph](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_prs.svg)](https://github.com/Qentora/quoptuna/pulls)
[![Trending topics](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_words.svg)](https://github.com/Qentora/quoptuna/commits)
[![Top contributors](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_users.svg)](https://github.com/Qentora/quoptuna/graphs/contributors)
[![Activity map](https://images.repography.com/61358072/Qentora/quoptuna/recent-activity/NFJ5verQB-dv9RS0RuDOgSE84SRTHP39DKgPw9dAbEI/BGB39P4WjaaHESiizHx0odYYf_6ZBBoXmK4Gh8qIAD4_map.svg)](https://github.com/Qentora/quoptuna/commits)


