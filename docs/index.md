# QuOptuna Documentation

Welcome to QuOptuna, a quantum-enhanced hyperparameter optimization framework that combines quantum computing with Optuna for advanced model tuning.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Features](#features)
4. [Usage](#usage)
5. [API Documentation](#api-documentation)
6. [Contributing](#contributing)

## Introduction

QuOptuna integrates quantum computing capabilities with Optuna's hyperparameter optimization framework to enhance the model tuning process. It supports both classical and quantum models, providing a unified interface for optimization tasks.

## Installation

QuOptuna can be installed using UV (recommended) or pip:

```bash
# Using UV (recommended)
uv pip install quoptuna

# Using pip
pip install quoptuna
```

For development setup:

```bash
# Clone the repository
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna

# Install dependencies using UV
uv pip install -e .
```

## Features

- **Quantum-ML Optimization**:  hyperparameter tuning for Quantum algorithms
- **Multiple Model Support**: Compatible with both classical and quantum models
- **Interactive Dashboard**: Visualize optimization progress in real-time
- **Explainable AI**: Built-in tools for model interpretability
- **Extensible Framework**: Easy integration with custom models and algorithms

## Usage

### Basic Example

```python
import quoptuna as qo

Add Example Here
```

### Interactive Dashboard

QuOptuna provides a Streamlit-based dashboard for real-time optimization monitoring:

```bash
quoptuna --start
```

### Model Types

QuOptuna supports various quantum and classical models:

- Quantum Models:
  - Circuit-Centric Classifier
  - Data Reuploading Classifier
  - Quantum Kitchen Sinks
  - Quantum Metric Learner
  - Dressed Quantum Circuit Classifier

- Classical Models:
  - SVC (Support Vector Classification)
  - MLP Classifier
  - Perceptron

## API Documentation

For detailed API documentation, please refer to our [API Documentation](api_docs.md).

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run mypy .
```

For more information about the development process, refer to our [Development Guide](CONTRIBUTING.md).



## Backend Optimizer Implementation

```python
class Optimizer:
    def __init__(self, db_name: str, dataset_name: str = "", data: Optional[dict] = None, study_name: str = ""):
        self.db_name = db_name
        self.dataset_name = dataset_name
        self.data = data or {}
        self.data_path = f"db/{self.db_name}.db"
        self.study_name = study_name
```

## Frontend Application

```python
def main():
    if "run" in sys.argv:
        sys.argv = ["streamlit", "run", "src/quoptuna/frontend/app.py"]
        stcli.main()
    if "--start" in sys.argv:
        sys.argv = ["streamlit", "run", "src/quoptuna/frontend/app.py"]
        stcli.main()
    else:
        initialize_session_state()
        main_page()
        with st.sidebar:
            handle_sidebar()
        update_plot()
...
if __name__ == "__main__":
    main()
```

