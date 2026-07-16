---
title: Python API reference
description: The public quoptuna package API — Optimizer, DataPreparation, XAI, XAIConfig, create_model.
---

The public API is exported from the `quoptuna` package (`__all__`): `Optimizer`, `DataPreparation`, `XAI`, `XAIConfig`, `create_model`.

## `DataPreparation(file_path, x_cols, y_col)`

Loads and prepares a CSV for training.

| Parameter | Description |
| --- | --- |
| `file_path` | Path to the CSV dataset |
| `x_cols` | List of feature column names |
| `y_col` | Target column name |

Method `.get_data(output_type="2")` returns a data dict consumed by `Optimizer`.

:::note
Binary targets should follow the `{-1, +1}` convention; multiclass uses raw codes `0..K-1`.
:::

## `Optimizer(db_name, study_name, data, ...)`

Optuna-based hyperparameter search over quantum and classical models.

| Parameter | Description |
| --- | --- |
| `db_name` | SQLite storage database name |
| `study_name` | Optuna study name |
| `data` | Data dict from `DataPreparation.get_data()` |
| `model_types` | List of model names (default set) |
| `search_space` | Optional custom categorical search space |
| `sampler` | `tpe` \| `random` \| `grid` |
| `sampler_seed` | Sampler seed |
| `pruner` | `none` \| `asha` \| `hyperband` |

The full constructor parameter set lives in source.

Method `.optimize(n_trials=...) -> (study, best_trials)`.

- `best_trials[0].value` — best (validation) F1
- `best_trials[0].params["model_type"]` — winning model

:::note
`study.best_value` is validation F1 (possibly threshold-tuned). Test metrics live in each trial's `user_attrs`.
:::

## `XAI` / `XAIConfig`

SHAP-based explainability. `XAIConfig` configures the analysis; `XAI` produces SHAP plots (bar, beeswarm, violin, heatmap, waterfall) and other diagnostics for a trained model. Detailed plotting options are configured via `XAIConfig` fields — refer to source or [`/api/docs`](http://localhost:8000/api/docs) for the exact field names.

## `create_model(model_type, **kwargs)`

Construct a single model instance by name (quantum or classical) without running a search. See the [Model catalog](/reference/models/) for valid `model_type` values.

## End-to-end example

```python
from quoptuna import DataPreparation, Optimizer

data_prep = DataPreparation(file_path="your_data.csv", x_cols=["f1","f2","f3"], y_col="target")
data_dict = data_prep.get_data(output_type="2")

optimizer = Optimizer(db_name="experiment", study_name="trial_1", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=100)
print(best_trials[0].value, best_trials[0].params["model_type"])
```

:::caution
Do NOT use `create_study` — it is not part of the QuOptuna API. Use `Optimizer`.
:::

## See also

- [Model catalog](/reference/models/)
- [CLI reference](/reference/cli/)
- [Configuration reference](/reference/configuration/)
