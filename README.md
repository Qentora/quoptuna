# QuOptuna

[![PyPI Version](https://img.shields.io/pypi/v/quoptuna?style=flat-round)](https://pypi.org/project/quoptuna/)
[![Python Versions](https://img.shields.io/pypi/pyversions/quoptuna?style=flat-round)](https://pypi.org/project/quoptuna/)
[![License](https://img.shields.io/pypi/l/quoptuna?style=flat-round)](https://pypi.org/project/quoptuna/)
[![Documentation Status](https://readthedocs.org/projects/quoptuna/badge/?version=latest)](https://Qentora.github.io/quoptuna)
[![codecov](https://codecov.io/gh/Qentora/quoptuna/graph/badge.svg?token=6QE861D1CB)](https://codecov.io/gh/Qentora/quoptuna)


<!-- [![Build Status](https://github.com/Qentora/quoptuna/actions/workflows/build.yml/badge.svg)](https://github.com/Qentora/quoptuna/actions) -->
QuOptuna integrates quantum computing with [Optuna](https://optuna.org/), enhancing hyperparameter optimization through quantum algorithms.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Development](#development)
  - [Testing](#testing)
  - [Building Documentation](#building-documentation)
  - [Releasing](#releasing)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Quantum Optimization**: Leverage quantum algorithms for enhanced hyperparameter tuning.
- **Seamless Integration**: Compatible with Optuna's workflow.
- **Extensible**: Supports multiple quantum backends and custom algorithms.

---

## Installation

Install QuOptuna using pip:

```sh
pip install quoptuna
```

## Quick Start

Here’s how you can get started:

```python
import quoptuna as qo

def objective(trial):
    # Define your optimization logic here
    return trial.suggest_float('x', -10, 10) ** 2

study = qo.create_study()
study.optimize(objective, n_trials=100)

print("Best trial:", study.best_trial)
```

For more examples, visit the examples directory.

## Documentation

Comprehensive documentation is available at [https://Qentora.github.io/quoptuna](https://Qentora.github.io/quoptuna). It includes:
- Tutorials
- API references
- Usage guides

## Development

To contribute to QuOptuna, follow these steps:

### Clone the Repository

```sh
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna
```

### Install Requirements

1. Ensure Poetry and Python 3.8+ are installed.
2. Install dependencies:

```sh
poetry install
```

3. Activate the virtual environment:

```sh
poetry shell
```

### Testing

Run tests with:

```sh
pytest
```

To check code coverage:

```sh
pytest --cov=quoptuna
```

### Building Documentation

Generate and preview documentation locally:

```sh
mkdocs serve
```

### Releasing

1. Trigger the Draft Release Workflow by clicking Run workflow.
2. Publish the draft release from GitHub releases.
3. The release workflow will automatically:
   - Publish to PyPI.
   - Deploy updated documentation.

### Pre-commit Hooks

Install pre-commit hooks to ensure consistent code quality:

```sh
pre-commit install
```

To manually run hooks:

```sh
pre-commit run --all-files
```

## Contributing

We welcome contributions! To get started:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Refer to the Contributing Guidelines for more details.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.

QuOptuna combines quantum computing with advanced hyperparameter optimization to push the boundaries of machine learning and computational tasks.

---
