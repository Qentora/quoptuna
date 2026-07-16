---
title: Choose samplers and pruners
description: Configure how QuOptuna proposes hyperparameter configs and early-stops unpromising quantum-model trainings.
---

QuOptuna's optimizer supports a **sampler** (proposes hyperparameter configs) and an optional **pruner** (early-stops unpromising quantum-model trainings).

| Setting | Values | Default |
| --- | --- | --- |
| `sampler` | `tpe` (Bayesian), `random`, `grid` | `tpe` |
| `pruner` | `asha` (asynchronous successive halving), `hyperband`, `none` | `asha` |
| `pruner_min_resource` | int — rung-0 resource, in units of intermediate reports | `1` |
| `pruner_reduction_factor` | int — fraction of trials promoted per rung (1/η) | `3` |
| `intermediate_metric` | `f1` (validation F1, same averaging as the objective), `accuracy`, `neg_loss` (negated recent training loss) | `f1` |
| `max_steps` | optional cap on training steps for iterative models | model default |

All fields are on the `POST /api/v1/optimize` request and the `Optimizer` constructor; the web UI exposes sampler + pruner in the Configure step.

## How pruning works

Iterative (JAX-trained) quantum models report an intermediate value every `convergence_interval` steps. The pruner compares against other trials at the same rung and stops the bottom fraction — they end PRUNED (not FAILED) and never win best_trial. The pruner operates on the report index (1st/2nd/3rd report), so `pruner_min_resource=1` = "may stop after first report". f1/accuracy metrics are evaluated on a capped subset (first 128 rows) of the VALIDATION split to bound circuit cost. Default metric is f1 because it matches the objective: on imbalanced data, accuracy-based pruning keeps majority-class predictors alive and kills trials slowly learning the minority class.

Kernel models (IQPKernel, ProjectedQuantumKernel, QuantumKitchenSinks, SeparableKernelClassifier) and classical sklearn models have no training steps — they always train to completion regardless of pruner.

:::note[Non-converging trials]
A trial reaching `max_steps` without meeting the flat-loss convergence criterion is NOT discarded: the partially-trained model is scored and the trial completes with user attribute `converged: false`. To stop slow trials from consuming the full default budget (10,000 steps), set `max_steps` smaller — pruning alone cannot stop a trial whose intermediate metric stays competitive while its loss keeps drifting.
:::

:::caution[neg_loss caveat]
`intermediate_metric="neg_loss"` costs zero extra circuit evals, but raw training losses are only comparable across trials of the SAME model type. Use only for single-model studies; default f1 is comparable across the whole search space.
:::

:::note[Grid search]
`sampler="grid"` builds the grid from the categorical search space and may exhaust every combination before num_trials — the study simply stops early.
:::

## Tune performance

The web UI's Settings → Optimizer Performance card (and `POST /api/v1/optimize` fields) exposes three knobs dominating trial wall-clock:

- `max_vmap` (UI default 32): circuit evaluations vectorized per JAX call. Historical value 1 evaluates one sample at a time; on small tabular datasets 32 is severalfold faster. Must divide batch size (32): valid 1/2/4/8/16/32.
- `max_steps` (UI default 2000): training-step cap per trial (model default 10,000). Trials hitting the cap unconverged are still scored.
- `convergence_interval` (UI default 100): steps between flat-loss checks and pruning reports. Lower = converged trials exit sooner + ASHA earlier decisions, at cost of noisier convergence estimates.

## Account for resource use

Every completed/pruned trial records user attrs: `training_time` (wall-clock s), `n_steps`, `batch_size`, `pruned`/`pruned_at_step`. Total quantum calls to solution:

```python
from optuna import load_study
study = load_study(storage="sqlite:///db/results.db", study_name="my-study")
total_steps = sum(t.user_attrs.get("n_steps", 0) for t in study.trials)
total_circuit_evals = sum(t.user_attrs.get("n_steps", 0) * t.user_attrs.get("batch_size", 0) for t in study.trials)
```

## Recipe: TPE+ASHA vs random search

```python
from quoptuna.backend.tuners.optimizer import Optimizer
common = dict(db_name="rq1", data=data, model_types=models, search_space=space)
Optimizer(study_name="rq1-random", sampler="random", sampler_seed=0, **common).optimize(n_trials=100)
Optimizer(study_name="rq1-tpe-asha", sampler="tpe", sampler_seed=0, pruner="asha", **common).optimize(n_trials=100)
```

Plot running max of trial values vs cumulative n_steps per study — TPE+ASHA should reach the same best score with materially fewer quantum evaluations.

## Next steps

- [Tune for speed and search quality](/how-to/tune-for-speed-and-quality/)
- [Optimization engine](/explanation/optimization-engine/)
- [CLI reference](/reference/cli/)
