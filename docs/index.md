# QuOptuna Documentation

Welcome to QuOptuna, a quantum-enhanced hyperparameter optimization framework that combines quantum computing with Optuna for advanced model tuning and explainable AI.

## Overview

QuOptuna provides a comprehensive platform for:
- 🎯 **Automated hyperparameter optimization** for quantum and classical ML models
- 🔍 **SHAP-based explainable AI** with rich visualizations
- 📊 **UCI ML Repository integration** for easy dataset access
- 📝 **AI-powered report generation** for model analysis
- 🖥️ **Interactive Streamlit interface** for the complete workflow

## Quick Start

### Installation

Install QuOptuna using UV (recommended) or pip:

```bash
# Using UV (recommended)
uv pip install quoptuna

# Using pip
pip install quoptuna
```

### Launch the Application

Start the full stack (FastAPI backend + Next.js frontend) in production mode with a single command. A gradient ASCII banner appears while the services boot:

```bash
uv run quoptuna run
```

This builds the frontend, starts both services on the first free ports (defaults: `:8000` API, `:3000` UI), waits for readiness, prints the access links, and opens your browser. Running `uv run quoptuna` with no subcommand does the same thing.

Useful options:

```bash
# Custom ports and no auto-opened browser
uv run quoptuna run --backend-port 8001 --frontend-port 3001 --no-browser

# Legacy Streamlit dashboard instead of the full stack
uv run quoptuna run --streamlit
```

Build and server logs are written to files under `${TMPDIR}/quoptuna/`; their paths are shown beneath the banner.

### Basic Python Usage

```python
from quoptuna import DataPreparation, Optimizer
from quoptuna.backend.utils.data_utils.data import mock_csv_data
import pandas as pd

# Load and prepare data
df = pd.read_csv("your_data.csv")
df["target"] = df["target"].replace({0: -1, 1: 1})

# Save data
file_path = mock_csv_data(df, tmp_path="data", file_name="my_data")

# Prepare for training
data_prep = DataPreparation(
    file_path=file_path,
    x_cols=[col for col in df.columns if col != "target"],
    y_col="target"
)
data_dict = data_prep.get_data(output_type="2")

# Run optimization
optimizer = Optimizer(db_name="experiment", study_name="trial_1", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=100)

print(f"Best F1 Score: {best_trials[0].value:.4f}")
print(f"Best Model: {best_trials[0].params['model_type']}")
```

## Key Features

### 🎯 Hyperparameter Optimization

Automated optimization using Optuna with support for:
- Multiple quantum models (Data Reuploading, Circuit-Centric, Quantum Kitchen Sinks, etc.)
- Classical baselines (SVC, MLP, Perceptron)
- Multi-objective optimization
- Parallel trial execution
- Persistent storage with SQLite

### 🔍 Explainable AI

Comprehensive SHAP analysis with multiple visualization types:
- **Bar Plot**: Feature importance ranking
- **Beeswarm Plot**: Feature value impact distribution
- **Violin Plot**: SHAP value distributions
- **Heatmap**: Instance-level feature contributions
- **Waterfall Plot**: Individual prediction explanations
- **Confusion Matrix**: Model performance visualization

### 📊 Dataset Management

- **UCI ML Repository**: Direct access to 100+ datasets
- **Custom Upload**: Support for CSV files
- **Automatic Preprocessing**: Handle missing values, feature scaling
- **Target Transformation**: Binary classification support (-1/+1 encoding)

### 📝 AI-Powered Reports

Generate comprehensive analysis reports using:
- Google Gemini
- OpenAI GPT
- Anthropic Claude

Reports include performance metrics, SHAP interpretations, and governance recommendations.

## Supported Models

### Quantum Models

- **Data Reuploading Classifier**: Quantum circuit with data re-uploading
- **Circuit-Centric Classifier**: Parameterized quantum circuits
- **Quantum Kitchen Sinks**: Quantum feature maps
- **Quantum Metric Learner**: Metric learning with quantum circuits
- **Dressed Quantum Circuit Classifier**: Hybrid quantum-classical

### Classical Models

- **Support Vector Classifier (SVC)**: With multiple kernels
- **Multi-Layer Perceptron (MLP)**: Neural network classifier
- **Perceptron**: Simple linear classifier

## Documentation

### 🚀 Getting Started
- **[Quick Start Guide](getting-started/quickstart.md)** - Get up and running in 5 minutes
- **[QuOptuna Next](getting-started/quoptuna-next.md)** - Modern drag-and-drop workflow interface
- **[Running Without Docker](getting-started/run-without-docker.md)** - Local development setup

### 📖 User Guides
- **[User Guide](guides/user-guide.md)** - Complete walkthrough of the Streamlit interface
- **[Streamlit Guide](guides/streamlit-guide.md)** - Streamlit-specific features and tips
- **[Workflow Builder Guide](guides/workflow-builder-guide.md)** - Create and manage workflows
- **[Frontend Quick Reference](guides/frontend-quick-reference.md)** - Quick reference for frontend features

### 🏗️ Architecture & Design
- **[Frontend Architecture](architecture/frontend-architecture.md)** - Frontend design and components
- **[Frontend Architecture Diagram](architecture/frontend-architecture-diagram.md)** - Visual architecture overview
- **[Optimizer Architecture](architecture/optimizer-architecture.md)** - Backend optimization system design
- **[New Frontend Design](architecture/new-frontend-design.md)** - Latest frontend improvements

### 🛠️ Development
- **[Testing Checklist](development/testing-checklist.md)** - Testing best practices
- **[Workflow Testing](development/workflow-testing.md)** - Test workflow components
- **[Implementation Summary](development/implementation-summary.md)** - Recent implementation updates

### ⚙️ Configuration
- **[GitHub Settings Guide](configuration/github-settings-guide.md)** - Repository configuration
- **[GitHub Pages Setup](configuration/github-pages-setup.md)** - Deploy documentation site

### 📚 API Documentation
- **[API Reference](api/api-reference.md)** - Detailed API documentation for Python usage
- **[API Docs](api/api-docs.md)** - Additional API documentation

### 💡 Examples
- **[Examples](examples/examples.md)** - Code examples for common use cases

### 🗺️ Roadmap
- **[Implementation Roadmap](roadmap/implementation-roadmap.md)** - Future plans and features

### 📋 Changelog
- **[Changelog](../CHANGELOG.md)** - Version history and updates

## Workflow

QuOptuna provides a structured workflow:

1. **Dataset Selection**
   - Load from UCI ML Repository or upload CSV
   - Configure features and target
   - Apply preprocessing

2. **Optimization**
   - Prepare train/test splits
   - Run hyperparameter optimization
   - Review best performing models

3. **Model Training**
   - Select best trial
   - Train model with optimized parameters
   - Evaluate performance

4. **SHAP Analysis**
   - Calculate SHAP values
   - Generate multiple visualization types
   - Understand feature importance

5. **Report Generation**
   - Create AI-powered analysis
   - Export results
   - Share insights

## System Requirements

- Python 3.11 or 3.12
- Node.js 18+ (for the Next.js frontend)
- 4GB+ RAM (8GB recommended for quantum models)
- Internet connection (for UCI datasets and LLM reports)

### Dependencies

Core dependencies:
- `optuna` - Hyperparameter optimization
- `streamlit` - Web interface
- `shap` - Explainable AI
- `pennylane` - Quantum computing
- `scikit-learn` - Classical ML models
- `pandas`, `numpy` - Data processing

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/Qentora/quoptuna.git
cd quoptuna

# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run mypy .
```

### Project Structure

```
quoptuna/
├── src/quoptuna/
│   ├── backend/         # Core optimization and model code
│   │   ├── models/      # Model implementations
│   │   ├── tuners/      # Optuna integration
│   │   ├── xai/         # SHAP analysis
│   │   └── utils/       # Utilities
│   └── frontend/        # Streamlit interface
│       ├── pages/       # Multi-page app
│       ├── app.py       # Main application
│       └── support.py   # Helper functions
├── docs/                # Documentation
├── experiments/         # Example notebooks
└── tests/              # Unit tests
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](https://github.com/Qentora/quoptuna/blob/main/CONTRIBUTING.md).

### Ways to Contribute

- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests
- ⭐ Star the repository

## Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/Qentora/quoptuna/issues)
- **Discussions**: [Join the community](https://github.com/Qentora/quoptuna/discussions)
- **Documentation**: [Full docs](https://qentora.github.io/quoptuna)

## License

QuOptuna is released under the MIT License. See [LICENSE](https://github.com/Qentora/quoptuna/blob/main/LICENSE) for details.

## Citation

If you use QuOptuna in your research, please cite:

```bibtex
@software{quoptuna,
  title = {QuOptuna: Quantum-Enhanced Machine Learning Optimization},
  author = {QuOptuna Team},
  year = {2024},
  url = {https://github.com/Qentora/quoptuna}
}
```

## Acknowledgments

Built with:
- [Optuna](https://optuna.org/) - Hyperparameter optimization framework
- [PennyLane](https://pennylane.ai/) - Quantum machine learning
- [SHAP](https://shap.readthedocs.io/) - Explainable AI
- [Streamlit](https://streamlit.io/) - Web framework

---

**Ready to get started?** Check out the [Quick Start Guide](getting-started/quickstart.md) or [User Guide](guides/user-guide.md), then launch the app with `uv run quoptuna run`!

