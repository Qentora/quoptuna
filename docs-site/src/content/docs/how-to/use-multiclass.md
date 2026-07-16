---
title: Optimize a multiclass problem
description: Run QuOptuna on targets with more than two classes, including OvR-wrapped quantum models.
---

QuOptuna handles binary and multiclass targets with different conventions:

| | Binary | Multiclass |
| --- | --- | --- |
| Target encoding | `{-1, +1}` | codes `0..K-1` |
| Objective metric | binary F1 | macro-F1 |
| Variational quantum models | native | automatically OvR-wrapped (one-vs-rest, K binary sub-models) |
| Kernel / classical models | native | native |

Variational quantum models are automatically wrapped in one-vs-rest (OvR): K binary sub-models, one per class. Kernel and classical sklearn models handle multiclass natively without wrapping.

## Run a 3-class problem (CLI)

Iris (`--uci-id 53`) has 3 classes and is a good smoke test:

```bash
quoptuna optimize --uci-id 53
```

QuOptuna detects the number of classes, switches the objective to macro-F1, and OvR-wraps any variational quantum models in the search space.

## Python path

From Python, pass the raw target column — QuOptuna reads the class codes directly:

```python
from quoptuna.backend.tuners.optimizer import Optimizer

Optimizer(
    study_name="iris-3class",
    db_name="iris",
    data=data,            # raw target column with codes 0..K-1
    model_types=models,
    search_space=space,
).optimize(n_trials=100)
```

## Speed up OvR sub-fits

OvR sub-fits are serial by default. To fit the K binary sub-models concurrently on threads, set the environment variable:

```bash
export QUOPTUNA_OVR_N_JOBS=3   # = K, the number of classes
```

Threads only — QuOptuna never uses processes here because pickling JAX models is unsafe.

## Multiclass fairness

For fairness-aware multiclass search you must set `favorable_class` (which class code counts as the favorable outcome). See [Run fairness-aware search](/how-to/run-fairness-aware-search/).

:::note[Pruning and OvR]
The pruner has no effect on OvR-wrapped multiclass models — they don't expose per-step intermediate reports the same way single binary models do. See [Choose samplers and pruners](/how-to/choose-samplers-and-pruners/).
:::

## Next steps

- [Run fairness-aware search](/how-to/run-fairness-aware-search/)
- [Choose samplers and pruners](/how-to/choose-samplers-and-pruners/)
- [CLI reference](/reference/cli/)
