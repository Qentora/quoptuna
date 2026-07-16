---
title: Python API tutorial
description: Run a QuOptuna optimization from Python with DataPreparation and Optimizer, then explain it with SHAP.
---

This tutorial walks through the QuOptuna Python API end to end: prepare data with
`DataPreparation`, search with `Optimizer`, inspect the best trials, and briefly
explain the result with SHAP. Everything here uses real public exports from
`quoptuna`. Plan on about 10 minutes.

## Prepare your data

`DataPreparation` loads a CSV, selects the feature and target columns, and
returns a data dictionary ready for the optimizer.

```python
from quoptuna import DataPreparation

data_prep = DataPreparation(
    file_path="your_data.csv",
    x_cols=["f1", "f2", "f3"],   # feature columns
    y_col="target",
)
data_dict = data_prep.get_data(output_type="2")
```

:::note
Targets are encoded for you. Binary targets use the {-1, +1} convention;
multiclass targets use codes 0..K-1 and are scored with macro-F1 (OvR-wrapped).
:::

## Run the optimization

Create an `Optimizer` with a database name, a study name, and the prepared data,
then call `optimize`.

```python
from quoptuna import Optimizer

optimizer = Optimizer(db_name="experiment", study_name="trial_1", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=100)
```

`Optimizer` also accepts (among others) `sampler`, `sampler_seed`, `pruner`,
`model_types`, and `search_space`, so you can control which models are searched
and how trials are proposed.

## Inspect the best trials

`optimize` returns the Optuna `study` and a list of `best_trials`. The first
entry is the best result:

```python
print(f"Best F1: {best_trials[0].value:.4f}")
print(f"Best model: {best_trials[0].params['model_type']}")
```

Expected output looks like:

```text
Best F1: 0.9667
Best model: SVC
```

The `study` object is a standard Optuna study, so you can reuse Optuna's tools
for further analysis.

## Explain with SHAP

QuOptuna exposes `XAI` and `XAIConfig` for SHAP-based explanations, producing
plots such as bar, beeswarm, violin, heatmap, and waterfall.

```python
from quoptuna import XAI, XAIConfig
```

For configuration details and the full set of plots, see the Python API
reference.

## Next steps

- [Python API reference](/reference/python-api/)
- [Optimize from the CLI](/tutorials/optimize-from-cli/)
