# Examples

## Introduction

This page provides practical examples for common QuOptuna use cases. Each example includes complete, runnable code.

## Table of Contents

1. [Basic Workflow](#basic-workflow)
2. [UCI Dataset Analysis](#uci-dataset-analysis)
3. [Custom Dataset](#custom-dataset)
4. [SHAP Analysis](#shap-analysis)
5. [Comparing Models](#comparing-models)
6. [Report Generation](#report-generation)
7. [Batch Processing](#batch-processing)

## Basic Workflow

Complete workflow from data loading to SHAP analysis:

```python
from quoptuna import DataPreparation, Optimizer, XAI
from quoptuna.backend.models import create_model
from quoptuna.backend.xai.xai import XAIConfig
from quoptuna.backend.utils.data_utils.data import mock_csv_data
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Ensure target is -1 and 1
df["target"] = df["target"].replace({0: -1, 1: 1})

# Save to file
file_path = mock_csv_data(df, tmp_path="data", file_name="my_data")

# Prepare data
data_prep = DataPreparation(
    file_path=file_path,
    x_cols=[col for col in df.columns if col != "target"],
    y_col="target"
)
data_dict = data_prep.get_data(output_type="2")

# Convert to numpy
for key in ["train_x", "test_x", "train_y", "test_y"]:
    data_dict[key] = data_dict[key].values

# Optimize
optimizer = Optimizer(db_name="my_experiment", study_name="run_1", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=50)

# Train best model
best_model = create_model(**best_trials[0].params)
best_model.fit(data_dict["train_x"], data_dict["train_y"])

# SHAP analysis
xai = XAI(
    model=best_model,
    data=data_dict,
    config=XAIConfig(use_proba=True, onsubset=True, subset_size=50)
)

# Generate plots
bar_plot = xai.get_plot("bar", max_display=10, class_index=1)
print("Analysis complete!")
```

## UCI Dataset Analysis

Working with UCI ML Repository datasets:

```python
from ucimlrepo import fetch_ucirepo
from quoptuna import DataPreparation, Optimizer
from quoptuna.backend.utils.data_utils.data import mock_csv_data
import pandas as pd

# Fetch dataset from UCI
dataset = fetch_ucirepo(id=143)  # Statlog Credit Approval

# Get metadata
print("Dataset:", dataset.metadata["name"])
print("Instances:", dataset.metadata["num_instances"])
print("Features:", dataset.metadata["num_features"])

# Prepare data
X = dataset.data.features
y = dataset.data.targets
df = pd.concat([X, y], axis=1)

# Transform target
target_col = dataset.metadata["target_col"][0]
df["target"] = df[target_col].replace({0: -1, 1: 1})
df = df.drop(columns=[target_col])
df = df.dropna()

# Save and prepare
file_path = mock_csv_data(df, tmp_path="data", file_name="Statlog")

data_prep = DataPreparation(
    file_path=file_path,
    x_cols=list(df.columns.difference(["target"])),
    y_col="target"
)
data_dict = data_prep.get_data(output_type="2")

# Convert to numpy
for key in data_dict.keys():
    data_dict[key] = data_dict[key].values

# Run optimization
optimizer = Optimizer(db_name="Statlog", study_name="Statlog", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=100)

# Show results
for i, trial in enumerate(best_trials[:3]):
    print(f"\n=== Best Trial {i+1} ===")
    print(f"Model: {trial.params['model_type']}")
    print(f"Quantum F1: {trial.user_attrs.get('Quantum_f1_score', 0):.4f}")
    print(f"Classical F1: {trial.user_attrs.get('Classical_f1_score', 0):.4f}")
```

## Custom Dataset

Loading and processing a custom CSV file:

```python
import pandas as pd
from quoptuna import DataPreparation, Optimizer
from quoptuna.backend.utils.data_utils.data import mock_csv_data

# Load custom dataset
df = pd.read_csv("my_dataset.csv")

# Explore data
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:", df.isnull().sum().sum())

# Handle missing values
df = df.dropna()

# Transform target to -1 and 1
# Example: If target is 'Yes'/'No'
df["target"] = df["outcome"].map({"Yes": 1, "No": -1})

# Drop original target column
df = df.drop(columns=["outcome"])

# Select features
feature_cols = ["age", "income", "credit_score", "debt_ratio"]

# Keep only selected columns
df = df[feature_cols + ["target"]]

# Save processed data
file_path = mock_csv_data(df, tmp_path="data", file_name="custom_dataset")

# Prepare for training
data_prep = DataPreparation(
    file_path=file_path,
    x_cols=feature_cols,
    y_col="target"
)
data_dict = data_prep.get_data(output_type="2")

# Convert to numpy
for key in data_dict.keys():
    data_dict[key] = data_dict[key].values

# Optimize
optimizer = Optimizer(
    db_name="custom_experiment",
    study_name="trial_001",
    data=data_dict
)
study, best_trials = optimizer.optimize(n_trials=100)

print(f"\nFound {len(best_trials)} best trials")
```

## SHAP Analysis

Comprehensive SHAP analysis with all plot types:

```python
from quoptuna import XAI
from quoptuna.backend.models import create_model
from quoptuna.backend.xai.xai import XAIConfig
import os

# Assuming you have optimized model and data_dict from previous steps
# Load best trial parameters
best_params = best_trials[0].params

# Train model
model = create_model(**best_params)
model.fit(data_dict["train_x"], data_dict["train_y"])

# Configure XAI
config = XAIConfig(
    use_proba=True,
    onsubset=True,
    subset_size=100
)

# Create XAI instance
xai = XAI(model=model, data=data_dict, config=config)

# Create output directory
os.makedirs("outputs/shap_plots", exist_ok=True)

# Generate and save all plot types
plot_types = ["bar", "beeswarm", "violin", "heatmap"]

for plot_type in plot_types:
    print(f"Generating {plot_type} plot...")

    plot = xai.get_plot(
        plot_type,
        max_display=10,
        class_index=1,
        save_config={
            "save_path": "outputs/shap_plots",
            "save_name": f"{plot_type}_plot",
            "save_format": "png",
            "save_dpi": 300
        }
    )

    print(f"Saved {plot_type} plot")

# Generate waterfall plots for first 5 samples
for i in range(5):
    waterfall = xai.get_plot(
        "waterfall",
        index=i,
        class_index=1,
        save_config={
            "save_path": "outputs/shap_plots",
            "save_name": f"waterfall_sample_{i}",
            "save_format": "png",
            "save_dpi": 300
        }
    )

    print(f"Saved waterfall plot for sample {i}")

# Get classification report
report = xai.get_report()

print("\n=== Classification Report ===")
print(report["classification_report"])

# Plot confusion matrix
import matplotlib.pyplot as plt

fig = xai.plot_confusion_matrix()
plt.savefig("outputs/shap_plots/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nAll SHAP plots saved to outputs/shap_plots/")
```

## Comparing Models

Compare quantum vs classical models:

```python
from quoptuna import Optimizer
from quoptuna.backend.models import create_model
import pandas as pd

# Run optimization (assumes data_dict is prepared)
optimizer = Optimizer(db_name="comparison", study_name="quantum_vs_classical", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=100)

# Separate quantum and classical trials
quantum_trials = []
classical_trials = []

for trial in study.get_trials():
    model_type = trial.params.get("model_type", "")

    # Determine if quantum or classical
    if "Classifier" in model_type and any(
        q in model_type
        for q in ["Reuploading", "Circuit", "Quantum", "Kitchen", "Dressed"]
    ):
        quantum_trials.append(trial)
    else:
        classical_trials.append(trial)

# Compare performance
def get_f1_score(trial):
    q_f1 = trial.user_attrs.get("Quantum_f1_score", 0)
    c_f1 = trial.user_attrs.get("Classical_f1_score", 0)
    return max(q_f1, c_f1)

# Get best from each category
best_quantum = max(quantum_trials, key=get_f1_score) if quantum_trials else None
best_classical = max(classical_trials, key=get_f1_score) if classical_trials else None

print("=== Model Comparison ===\n")

if best_quantum:
    print("Best Quantum Model:")
    print(f"  Type: {best_quantum.params['model_type']}")
    print(f"  F1 Score: {get_f1_score(best_quantum):.4f}")
    print(f"  Trial: {best_quantum.number}")

if best_classical:
    print("\nBest Classical Model:")
    print(f"  Type: {best_classical.params['model_type']}")
    print(f"  F1 Score: {get_f1_score(best_classical):.4f}")
    print(f"  Trial: {best_classical.number}")

# Create comparison DataFrame
comparison_data = []

for trial in quantum_trials + classical_trials:
    comparison_data.append({
        "Trial": trial.number,
        "Model Type": trial.params["model_type"],
        "Category": "Quantum" if trial in quantum_trials else "Classical",
        "F1 Score": get_f1_score(trial),
        "State": trial.state.name
    })

df_comparison = pd.DataFrame(comparison_data)
df_comparison = df_comparison.sort_values("F1 Score", ascending=False)

print("\n=== Top 10 Models ===")
print(df_comparison.head(10))

# Save results
df_comparison.to_csv("outputs/model_comparison.csv", index=False)
```

## Report Generation

Generate comprehensive AI reports:

```python
from quoptuna import XAI
from quoptuna.backend.xai.xai import XAIConfig
import os

# Train model (from previous steps)
model = create_model(**best_trials[0].params)
model.fit(data_dict["train_x"], data_dict["train_y"])

# Create XAI instance
xai = XAI(
    model=model,
    data=data_dict,
    config=XAIConfig(use_proba=True, onsubset=True, subset_size=50)
)

# Dataset information for better reports
dataset_info = {
    "Name": "Credit Card Approval",
    "URL": "https://archive.ics.uci.edu/dataset/143",
    "Description": """
        This dataset concerns credit card applications.
        It contains a mix of continuous and categorical features
        for predicting credit approval decisions.
    """,
    "Features": ["Age", "Income", "Credit Score", "Employment Status"],
    "Target": "Approval Decision",
    "Instances": 690,
    "Task": "Binary Classification"
}

# Generate report with Google Gemini
report = xai.generate_report_with_langchain(
    provider="google",
    api_key=os.getenv("GOOGLE_API_KEY"),
    model_name="models/gemini-2.0-flash-exp",
    dataset_info=dataset_info
)

# Save report
with open("outputs/analysis_report.md", "w") as f:
    f.write(report)

print("Report saved to outputs/analysis_report.md")

# Generate with OpenAI GPT
report_gpt = xai.generate_report_with_langchain(
    provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4",
    dataset_info=dataset_info
)

with open("outputs/analysis_report_gpt4.md", "w") as f:
    f.write(report_gpt)

print("GPT-4 report saved to outputs/analysis_report_gpt4.md")
```

## Batch Processing

Process multiple datasets:

```python
from quoptuna import DataPreparation, Optimizer
from quoptuna.backend.utils.data_utils.data import mock_csv_data
import pandas as pd
import os

# List of datasets to process
datasets = [
    {"id": 143, "name": "Statlog"},
    {"id": 176, "name": "Blood"},
    {"id": 267, "name": "Banknote"},
]

results = []

for dataset_info in datasets:
    print(f"\n{'='*50}")
    print(f"Processing: {dataset_info['name']}")
    print('='*50)

    try:
        # Fetch dataset
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=dataset_info["id"])

        # Prepare data
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)

        # Get target column name
        target_col = dataset.metadata["target_col"][0]
        df["target"] = df[target_col].replace({0: -1, 1: 1})
        df = df.drop(columns=[target_col])
        df = df.dropna()

        # Save
        file_path = mock_csv_data(
            df,
            tmp_path="data/batch",
            file_name=dataset_info["name"]
        )

        # Prepare
        data_prep = DataPreparation(
            file_path=file_path,
            x_cols=list(df.columns.difference(["target"])),
            y_col="target"
        )
        data_dict = data_prep.get_data(output_type="2")

        # Convert to numpy
        for key in data_dict.keys():
            data_dict[key] = data_dict[key].values

        # Optimize
        optimizer = Optimizer(
            db_name=f"batch_{dataset_info['name']}",
            study_name=dataset_info["name"],
            data=data_dict
        )
        study, best_trials = optimizer.optimize(n_trials=50)

        # Record results
        best_f1 = max(
            best_trials[0].user_attrs.get("Quantum_f1_score", 0),
            best_trials[0].user_attrs.get("Classical_f1_score", 0)
        )

        results.append({
            "Dataset": dataset_info["name"],
            "Best Model": best_trials[0].params["model_type"],
            "Best F1": best_f1,
            "Trials": len(study.trials),
            "Status": "Success"
        })

        print(f"✓ Completed: {dataset_info['name']}")
        print(f"  Best F1: {best_f1:.4f}")
        print(f"  Model: {best_trials[0].params['model_type']}")

    except Exception as e:
        print(f"✗ Failed: {dataset_info['name']}")
        print(f"  Error: {e}")

        results.append({
            "Dataset": dataset_info["name"],
            "Best Model": None,
            "Best F1": None,
            "Trials": 0,
            "Status": f"Failed: {str(e)}"
        })

# Save summary
df_results = pd.DataFrame(results)
df_results.to_csv("outputs/batch_processing_results.csv", index=False)

print("\n" + "="*50)
print("BATCH PROCESSING COMPLETE")
print("="*50)
print(df_results)
```

## Advanced: Custom Objective Function

Define custom optimization objectives:

```python
import optuna
from quoptuna.backend.models import create_model
from sklearn.metrics import f1_score, precision_score, recall_score

def custom_objective(trial, data_dict):
    """Custom objective balancing F1 score and model complexity."""

    # Suggest model type
    model_type = trial.suggest_categorical(
        "model_type",
        ["SVC", "MLPClassifier", "DataReuploadingClassifier"]
    )

    # Suggest hyperparameters based on model type
    if model_type == "SVC":
        params = {
            "model_type": model_type,
            "C": trial.suggest_float("C", 0.1, 10.0),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
        }
    elif model_type == "MLPClassifier":
        params = {
            "model_type": model_type,
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                ["(10,)", "(50,)", "(10, 10)"]
            ),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.1, log=True)
        }
    else:  # DataReuploadingClassifier
        params = {
            "model_type": model_type,
            "n_layers": trial.suggest_int("n_layers", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5)
        }

    # Create and train model
    model = create_model(**params)
    model.fit(data_dict["train_x"], data_dict["train_y"])

    # Evaluate
    y_pred = model.predict(data_dict["test_x"])
    y_true = data_dict["test_y"]

    # Calculate metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Store metrics as user attributes
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)

    # Return weighted score
    # Prefer higher F1 but penalize complex models
    complexity_penalty = 0.01 if model_type == "DataReuploadingClassifier" else 0
    return f1 - complexity_penalty

# Create study
study = optuna.create_study(direction="maximize")

# Optimize
study.optimize(
    lambda trial: custom_objective(trial, data_dict),
    n_trials=100
)

# Show results
print(f"Best F1 Score: {study.best_value:.4f}")
print(f"Best Parameters: {study.best_params}")
print(f"Precision: {study.best_trial.user_attrs['precision']:.4f}")
print(f"Recall: {study.best_trial.user_attrs['recall']:.4f}")
```

## Next Steps

- Review the [API Reference](api_reference.md) for detailed class documentation
- Check the [User Guide](user_guide.md) for the Streamlit interface
- Visit [GitHub](https://github.com/Qentora/quoptuna) for more examples
