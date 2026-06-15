# API Reference

## Overview

QuOptuna provides a comprehensive Python API for quantum-enhanced machine learning optimization. This reference covers the main classes and functions available for programmatic use.

## Core Classes

### DataPreparation

Handles data loading, preprocessing, and splitting.

```python
from quoptuna import DataPreparation

data_prep = DataPreparation(
    file_path="path/to/data.csv",
    x_cols=["feature1", "feature2", "feature3"],
    y_col="target"
)

# Get preprocessed data
data_dict = data_prep.get_data(output_type="2")
```

**Parameters:**
- `file_path` (str): Path to the CSV data file
- `x_cols` (list): List of feature column names
- `y_col` (str): Target column name
- `test_size` (float, optional): Proportion of data for testing (default: 0.25)
- `random_state` (int, optional): Random seed for reproducibility

**Methods:**

#### `get_data(output_type="2")`

Returns preprocessed data dictionary.

**Parameters:**
- `output_type` (str): Format of output
  - `"1"`: Returns pandas DataFrames
  - `"2"`: Returns numpy arrays (recommended for optimization)

**Returns:**
- dict: Dictionary with keys `train_x`, `test_x`, `train_y`, `test_y`

### Optimizer

Manages hyperparameter optimization using Optuna.

```python
from quoptuna import Optimizer

optimizer = Optimizer(
    db_name="my_experiment",
    study_name="trial_001",
    data=data_dict
)

# Run optimization
study, best_trials = optimizer.optimize(n_trials=100)
```

**Parameters:**
- `db_name` (str): Database name for storing results
- `study_name` (str): Unique study identifier
- `data` (dict): Data dictionary from DataPreparation
- `dataset_name` (str, optional): Human-readable dataset name

**Attributes:**
- `storage_location` (str): SQLite database URI
- `study` (optuna.Study): Optuna study object
- `best_trials` (list): List of best performing trials

**Methods:**

#### `optimize(n_trials=100, timeout=None)`

Run hyperparameter optimization.

**Parameters:**
- `n_trials` (int): Number of optimization trials
- `timeout` (int, optional): Maximum optimization time in seconds

**Returns:**
- `study` (optuna.Study): Completed study
- `best_trials` (list): List of Pareto-optimal trials

#### `load_study()`

Load previously saved study from database.

**Returns:**
- `study` (optuna.Study): Loaded study object

### Model Creation

Create models with optimized hyperparameters.

```python
from quoptuna.backend.models import create_model

# From trial parameters
model = create_model(**trial.params)

# Or specify directly
model = create_model(
    model_type="DataReuploadingClassifier",
    n_layers=10,
    learning_rate=0.1,
    batch_size=32
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

**Supported Models:**

#### Quantum Models

**DataReuploadingClassifier**
- `n_layers` (int): Number of quantum layers
- `learning_rate` (float): Learning rate for optimization
- `batch_size` (int): Batch size for training
- `n_input_copies` (int): Number of data reuploads
- `observable_type` (str): Type of observable ("all" or "half")

**CircuitCentricClassifier**
- `n_layers` (int): Circuit depth
- `learning_rate` (float): Learning rate
- `n_qubits` (int): Number of qubits

**QuantumKitchenSinks**
- `n_episodes` (int): Number of training episodes
- `learning_rate` (float): Learning rate
- `gamma` (float): Kernel parameter

#### Classical Models

**SVC (Support Vector Classifier)**
- `C` (float): Regularization parameter
- `gamma` (str or float): Kernel coefficient
- `kernel` (str): Kernel type

**MLPClassifier**
- `hidden_layer_sizes` (tuple): Hidden layer configuration
- `learning_rate` (str): Learning rate schedule
- `alpha` (float): L2 regularization parameter

### XAI (Explainable AI)

Generate SHAP explanations and visualizations.

```python
from quoptuna import XAI
from quoptuna.backend.xai.xai import XAIConfig

# Configure XAI
config = XAIConfig(
    use_proba=True,
    onsubset=True,
    subset_size=50
)

# Create XAI instance
xai = XAI(
    model=trained_model,
    data=data_dict,
    config=config
)

# Get evaluation report
report = xai.get_report()
```

**XAIConfig Parameters:**
- `use_proba` (bool): Use probability predictions
- `onsubset` (bool): Use subset of data
- `subset_size` (int): Size of subset

**XAI Methods:**

#### `get_report()`

Generate classification report.

**Returns:**
- dict: Contains confusion matrix, classification report, and ROC curve data

#### `get_plot(plot_type, **kwargs)`

Generate SHAP visualization.

**Parameters:**
- `plot_type` (str): Type of plot
  - `"bar"`: Feature importance bar plot
  - `"beeswarm"`: SHAP value distribution
  - `"violin"`: Violin plot of SHAP values
  - `"heatmap"`: Instance-level SHAP heatmap
  - `"waterfall"`: Individual prediction explanation
- `max_display` (int): Maximum features to display
- `class_index` (int): Class to explain (for binary: 0 or 1)
- `index` (int): Sample index (for waterfall plot)
- `save_config` (dict, optional): Configuration for saving plot

**Returns:**
- str: Base64 encoded image

#### `plot_confusion_matrix()`

Generate confusion matrix plot.

**Returns:**
- matplotlib.figure.Figure: Confusion matrix figure

#### `generate_report_with_langchain(provider, api_key, model_name, dataset_info=None)`

Generate AI-powered analysis report.

**Parameters:**
- `provider` (str): LLM provider ("google", "openai", "anthropic")
- `api_key` (str): API key for the provider
- `model_name` (str): Model identifier
- `dataset_info` (dict, optional): Dataset metadata

**Returns:**
- str: Markdown formatted report

## Utility Functions

### mock_csv_data

Save DataFrame to CSV file.

```python
from quoptuna.backend.utils.data_utils.data import mock_csv_data

file_path = mock_csv_data(
    dataframe,
    tmp_path="data",
    file_name="my_dataset"
)
```

**Parameters:**
- `dataframe` (pd.DataFrame): Data to save
- `tmp_path` (str): Directory path
- `file_name` (str): File name (without .csv extension)

**Returns:**
- str: Full path to saved file

## Complete Example

Here's a complete example workflow:

```python
import pandas as pd
from ucimlrepo import fetch_ucirepo
from quoptuna import DataPreparation, Optimizer, XAI
from quoptuna.backend.models import create_model
from quoptuna.backend.xai.xai import XAIConfig
from quoptuna.backend.utils.data_utils.data import mock_csv_data

# 1. Load dataset
dataset = fetch_ucirepo(id=143)  # Statlog dataset
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)

# 2. Prepare data
df["target"] = df["A15"].replace({0: -1, 1: 1})
df = df.drop(columns=["A15"])
df = df.dropna()

# 3. Save to file
file_path = mock_csv_data(df, tmp_path="data", file_name="Statlog")

# 4. Prepare for training
data_prep = DataPreparation(
    file_path=file_path,
    x_cols=list(df.columns.difference(["target"])),
    y_col="target"
)
data_dict = data_prep.get_data(output_type="2")

# Convert to numpy arrays
data_dict["train_x"] = data_dict["train_x"].values
data_dict["test_x"] = data_dict["test_x"].values
data_dict["train_y"] = data_dict["train_y"].values
data_dict["test_y"] = data_dict["test_y"].values

# 5. Run optimization
optimizer = Optimizer(
    db_name="Statlog",
    study_name="Statlog",
    data=data_dict
)
study, best_trials = optimizer.optimize(n_trials=100)

# 6. Train best model
best_trial = best_trials[0]
model = create_model(**best_trial.params)
model.fit(data_dict["train_x"], data_dict["train_y"])

# 7. SHAP analysis
config = XAIConfig(use_proba=True, onsubset=True, subset_size=50)
xai = XAI(model=model, data=data_dict, config=config)

# 8. Generate visualizations
bar_plot = xai.get_plot("bar", max_display=10, class_index=1)
beeswarm_plot = xai.get_plot("beeswarm", max_display=10, class_index=1)

# 9. Generate report
report = xai.generate_report_with_langchain(
    provider="google",
    api_key="your-api-key",
    model_name="models/gemini-2.0-flash-exp",
    dataset_info={
        "Name": "Statlog Credit Approval",
        "URL": "https://archive.ics.uci.edu/dataset/143",
        "Description": "Credit card application dataset"
    }
)

print(report)
```

## Data Format Requirements

### Input Data

**CSV Format:**
- Must have header row with column names
- Target column should contain binary values
- Features can be numeric or categorical
- Missing values will be removed

**Target Encoding:**
- Binary classification: Must use `-1` and `1`
- QuOptuna does not currently support multi-class classification

### Data Dictionary Format

After preprocessing, data should be in this format:

```python
data_dict = {
    "train_x": np.ndarray,  # Shape: (n_train_samples, n_features)
    "test_x": np.ndarray,   # Shape: (n_test_samples, n_features)
    "train_y": np.ndarray,  # Shape: (n_train_samples,)
    "test_y": np.ndarray    # Shape: (n_test_samples,)
}
```

## Advanced Usage

### Custom Optimization Objectives

You can customize the optimization objective:

```python
import optuna

def custom_objective(trial):
    # Define your custom objective
    params = {
        "model_type": trial.suggest_categorical("model_type", ["SVC", "MLPClassifier"]),
        "C": trial.suggest_float("C", 0.1, 10.0)
    }

    model = create_model(**params)
    model.fit(train_x, train_y)

    # Return custom metric
    return custom_metric(model, test_x, test_y)

study = optuna.create_study(direction="maximize")
study.optimize(custom_objective, n_trials=100)
```

### Parallel Optimization

Run multiple trials in parallel:

```python
optimizer = Optimizer(db_name="my_db", study_name="my_study", data=data_dict)

# Use n_jobs for parallel execution
study, best_trials = optimizer.optimize(
    n_trials=100,
    n_jobs=4  # Run 4 trials in parallel
)
```

### Saving and Loading Models

```python
import joblib

# Save model
joblib.dump(model, "model.pkl")

# Load model
loaded_model = joblib.load("model.pkl")
```

## Error Handling

Common errors and solutions:

```python
try:
    optimizer.optimize(n_trials=100)
except ValueError as e:
    # Handle data validation errors
    print(f"Data error: {e}")
except RuntimeError as e:
    # Handle optimization errors
    print(f"Optimization error: {e}")
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Use Subsets for SHAP**: Analyze 50-100 samples for faster computation
2. **Increase Trials Gradually**: Start with 50 trials, increase as needed
3. **Use Caching**: Reuse loaded studies when possible
4. **Monitor Memory**: Large datasets may require subset analysis
5. **Parallel Processing**: Use `n_jobs` parameter for faster optimization

## See Also

- [User Guide](user_guide.md) - Step-by-step usage instructions
- [Examples](examples.md) - Common use cases and tutorials
- [GitHub Repository](https://github.com/Qentora/quoptuna) - Source code and issues
