---
title: Tune for speed and search quality
description: What the July 2026 execution-speed and search-quality changes did, and why each was made.
---

Two batches of changes (July 2026): execution-speed and search-quality improvements, and WHY each was made. See the [samplers-and-pruners how-to](/how-to/choose-samplers-and-pruners/) for the day-to-day knob reference.

## Why the changes were made

1. Runs were slow — the default left speed on the table: circuits evaluated one sample at a time, pruner off, every trial on the slow default simulator at 64-bit.
2. Search plateaued on imbalanced data — e.g. blood transfusion (~24% positives), best F1 stalled ~0.4 regardless of trials. Structural causes: pruning judged by accuracy (rewards majority predictors); nothing handled imbalance; every trial sampled ~25 hyperparameters of which ~5 were used by the chosen model (polluting TPE); the objective was scored directly on the unstratified test split.

## Execution-speed changes

| Change | Before | After | Why |
| --- | --- | --- | --- |
| `max_vmap` default | 1 | 32 | With batch_size=32, max_vmap=1 ran 32 size-1 circuit evals per step; 32 vectorizes the batch in one JAX vmap — severalfold speedup, no accuracy impact. Non-dividing pairs now fail fast at config time. |
| Pruner default | none | asha | ASHA existed but was off; stops bottom fraction per rung. (Coerced to none for `fairness_mode=multi_objective`, where Optuna doesn't support pruning.) |
| Simulator device `dev_type` | hard-coded `default.qubit` | selectable `default.qubit` / `lightning.qubit` (API field + Settings) | Falls back to `default.qubit` with a warning if unavailable. On small circuits `default.qubit`'s JAX path is usually faster; lightning wins as qubit counts grow. |
| JAX precision | 64-bit forced at import | `QUOPTUNA_JAX_X64` env toggle (default on) | Set `QUOPTUNA_JAX_X64=0` for float32 — ~halves simulator memory/compute; classification accuracy impact typically negligible. |
| OvR multiclass sub-fits | serial | opt-in threading via `QUOPTUNA_OVR_N_JOBS` | OvR wraps K binary sub-models; they can fit concurrently on threads (never processes — pickling JAX models is unsafe). Off by default; set `QUOPTUNA_OVR_N_JOBS=K`. |

## Search-quality changes

| Change | Before | After | Why |
| --- | --- | --- | --- |
| Pruning metric | accuracy | f1 (default) | Objective is F1; accuracy-based pruning kept majority-class trials alive. Now same F1 (same averaging) as the objective. |
| Search space | flat (~25 params every trial) | conditional (model_type first, then only that model's params) | Irrelevant params are noise in TPE. Each trial records only the hyperparameters its model uses (see `MODEL_PARAM_KEYS` in `quoptuna.backend.models`). Grid mode keeps the flat product (grid must be static). |
| Data splits | unstratified + objective on test | stratified + validation split carved from train | Objective + pruning scored on validation (20% of train) so selection doesn't tune to test. Test metrics still computed per trial + stored in user attrs; Analyze tab recomputes from the stored test split. Tiny datasets fall back to validating on test, with a warning. |
| Class imbalance | none | `class_weight` search + decision-threshold tuning | `class_weight: [None, "balanced"]` searched for classical models supporting it (SVC, LinearSVC, Perceptron). For binary models exposing predict_proba, the optimizer sweeps a 19-point threshold grid on validation — recovers minority-class F1 without retraining. |

## Read results after these changes

`study.best_value` is now validation F1 (possibly threshold-tuned). Test metrics live in each trial's user attrs: `Quantum_f1_score`/`Classical_f1_score`, `..._accuracy`, `val_f1_score`. Threshold-tuned trials also record `decision_threshold`, `val_f1_unthresholded`, `f1_score_thresholded` (test-side). The plain test F1 attr stays unthresholded to match the Analyze tab.

## Operational notes

1. **Resuming studies** — an Optuna study's parameter choices are immutable; because the defaults changed the search space, resuming a pre-change study fails fast with a clear error ("use a new study name...") instead of a cryptic Optuna error.
2. **Log file** — all backend/server/Optuna logs mirror in real time to `db/logs/quoptuna.log` (rotating 5MB×3) plus terminal; override with `QUOPTUNA_LOG_FILE`.

## Next steps

- [Choose samplers and pruners](/how-to/choose-samplers-and-pruners/)
- [Optimization engine](/explanation/optimization-engine/)
- [Configuration reference](/reference/configuration/)
