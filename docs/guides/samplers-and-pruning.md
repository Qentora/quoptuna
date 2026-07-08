# Samplers and Pruning (ASHA / Hyperband)

QuOptuna's optimizer supports configurable search strategies: a **sampler** that
proposes hyperparameter configurations and an optional **pruner** that
early-stops unpromising quantum-model trainings to save compute.

## Options

| Setting | Values | Default |
| --- | --- | --- |
| `sampler` | `tpe` (Bayesian), `random`, `grid` | `tpe` |
| `pruner` | `none`, `asha` (asynchronous successive halving), `hyperband` | `none` |
| `pruner_min_resource` | int — rung-0 resource, in units of intermediate reports | `1` |
| `pruner_reduction_factor` | int — fraction of trials promoted per rung (1/η) | `3` |
| `intermediate_metric` | `accuracy` (validation accuracy) or `neg_loss` (negated recent training loss) | `accuracy` |
| `max_steps` | optional cap on training steps for iterative models | model default |

All fields are available on the `POST /api/v1/optimize` request and on the
`Optimizer` constructor; the web UI exposes sampler and pruner in the
Configure step.

## How pruning works

Iterative (JAX-trained) quantum models report an intermediate value every
`convergence_interval` training steps. The pruner compares that value against
other trials at the same rung and stops trials in the bottom fraction — they
end in the `PRUNED` state (not `FAILED`) and never win `best_trial`.

The pruner operates on the **report index** (1st, 2nd, 3rd report...), so
`pruner_min_resource=1` means "a trial may be stopped after its first
intermediate report". The `accuracy` metric is evaluated on a fixed validation
subset (first 128 test rows) to bound the circuit cost of each report.

Kernel models (IQPKernel, ProjectedQuantumKernel, QuantumKitchenSinks,
SeparableKernelClassifier) and classical sklearn models have no training steps;
they always train to completion regardless of the pruner.

!!! note "Non-converging trials"
    A trial that reaches `max_steps` without meeting the flat-loss convergence
    criterion is **not** discarded: the partially-trained model is scored and
    the trial completes with user attribute `converged: false`. To stop slow
    trials from consuming the full default budget (10,000 steps), set
    `max_steps` to a smaller cap — pruning alone cannot stop a trial whose
    intermediate accuracy stays competitive while its loss keeps drifting.

!!! warning "`neg_loss` caveat"
    `intermediate_metric="neg_loss"` costs zero extra circuit evaluations, but
    raw training losses are only comparable across trials of the **same model
    type**. Use it only for single-model studies; the default `accuracy` is
    comparable across the whole search space.

!!! note "Grid search"
    `sampler="grid"` builds the grid from the categorical search space and may
    exhaust every combination before `num_trials` — the study simply stops
    early.

## Resource accounting

Every completed or pruned trial records user attributes for
resource-efficiency analysis:

- `training_time` — wall-clock seconds spent in the training loop
- `n_steps` — training steps actually executed
- `batch_size` — samples per step
- `pruned` / `pruned_at_step` — whether and when the pruner stopped the trial

Total "quantum calls to solution" for a study:

```python
from optuna import load_study

study = load_study(storage="sqlite:///db/results.db", study_name="my-study")
total_steps = sum(t.user_attrs.get("n_steps", 0) for t in study.trials)
total_circuit_evals = sum(
    t.user_attrs.get("n_steps", 0) * t.user_attrs.get("batch_size", 0)
    for t in study.trials
)
```

## Recipe: TPE+ASHA vs. random search (RQ1-style comparison)

Run the same dataset and search space under two studies and compare
best-score-vs-cumulative-steps:

```python
from quoptuna.backend.tuners.optimizer import Optimizer

common = dict(db_name="rq1", data=data, model_types=models, search_space=space)

Optimizer(study_name="rq1-random", sampler="random", sampler_seed=0,
          **common).optimize(n_trials=100)
Optimizer(study_name="rq1-tpe-asha", sampler="tpe", sampler_seed=0,
          pruner="asha", **common).optimize(n_trials=100)
```

Plot the running maximum of trial values against cumulative `n_steps` (or
`n_steps × batch_size`) per study — the TPE+ASHA curve should reach the same
best score with materially fewer quantum evaluations.
