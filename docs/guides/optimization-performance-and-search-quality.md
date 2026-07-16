# Optimization Performance and Search Quality

This page documents two batches of changes to the optimization pipeline
(July 2026): **execution-speed** improvements and **search-quality**
improvements, and — importantly — *why* each was made. See
[Samplers and Pruning](samplers-and-pruning.md) for the day-to-day knob
reference.

## Why these changes

Two independent problems were observed on real runs:

1. **Runs were slow.** A performance audit found the default configuration
   left most of the available speed on the table: circuits were evaluated one
   sample at a time, the pruner was off, and every trial trained on the slow
   default simulator with 64-bit precision.
2. **The search plateaued on imbalanced data.** On datasets like blood
   transfusion (~24% positives), the best F1 stalled around 0.4 regardless of
   trial count. This was structural, not bad luck: pruning judged trials by
   *accuracy* (which rewards majority-class predictors), nothing handled class
   imbalance, every trial sampled ~25 hyperparameters of which only ~5 were
   used by the chosen model (polluting TPE's model of the search space), and
   the objective was scored directly on the unstratified test split.

## Execution-speed changes

| Change | Before | After | Why |
| --- | --- | --- | --- |
| `max_vmap` default | `1` | `32` | With `batch_size=32`, `max_vmap=1` ran 32 separate size-1 circuit evaluations per training step. `32` vectorizes the whole batch in one JAX `vmap` call — a severalfold speedup with no accuracy impact. Non-dividing `max_vmap`/`batch_size` pairs now fail fast at configuration time instead of failing every trial. |
| Pruner default | `none` | `asha` | The ASHA pruner existed but was off, so a hopeless configuration could burn its full step budget. ASHA stops the bottom fraction of trials at each rung. (Coerced back to `none` automatically for `fairness_mode="multi_objective"`, where Optuna does not support pruning.) |
| Simulator device (`dev_type`) | hard-coded `default.qubit` | selectable `default.qubit` / `lightning.qubit` | Exposed end-to-end (API field, Settings page). Falls back to `default.qubit` with a warning if the requested device is unavailable. **Note:** on small circuits (few qubits) `default.qubit`'s JAX path is usually *faster* than lightning; lightning wins as qubit counts grow. |
| JAX precision | 64-bit forced at import | `QUOPTUNA_JAX_X64` env toggle (default on) | Set `QUOPTUNA_JAX_X64=0` to train in float32 — roughly halves simulator memory and compute; accuracy impact for classification is typically negligible. |
| OvR multiclass sub-fits | strictly serial | opt-in threading via `QUOPTUNA_OVR_N_JOBS` | One-vs-rest wraps K binary sub-models. They can now fit concurrently on threads (never processes — pickling the JAX models is unsafe). Off by default until benchmarked; set `QUOPTUNA_OVR_N_JOBS=K` to enable. |

## Search-quality changes

| Change | Before | After | Why |
| --- | --- | --- | --- |
| Pruning metric | `accuracy` | `f1` (default) | The objective is F1. On imbalanced data, accuracy-based pruning kept "predict the majority class" trials alive and killed trials that were slowly learning the minority class — the exact trials the search needed. Pruning now uses the same F1 (same averaging) as the objective. |
| Search space | flat: all ~25 params sampled every trial | conditional: `model_type` first, then only that model's parameters | Irrelevant parameters are noise in TPE's model of good/bad regions. Each trial now records only the hyperparameters its model actually uses (see `MODEL_PARAM_KEYS` in `quoptuna.backend.models`). Grid mode keeps the flat product (a grid must be static). |
| Data splits | unstratified, objective scored on test | stratified splits + validation split carved from train | Stratification keeps minority proportions consistent across splits. The objective and pruning reports are now scored on a validation split (20% of train), so model selection no longer tunes to the test set. Test metrics are still computed per trial and stored in user attrs; the Analyze tab is unaffected (it recomputes from the stored test split). On tiny datasets the optimizer falls back to validating on test, with a warning. |
| Class imbalance | none | `class_weight` search + decision-threshold tuning | `class_weight: [None, "balanced"]` is searched for the classical models that support it (SVC, LinearSVC, Perceptron). For binary models exposing `predict_proba`, the optimizer sweeps a 19-point threshold grid on validation: unweighted losses often produce well-*ranked* probabilities with a bad default 0.5 cutoff, and re-cutting recovers minority-class F1 without retraining. |

### Reading the results after these changes

- `study.best_value` is now the **validation** F1 (possibly threshold-tuned).
- Test-set metrics live in each trial's user attributes: `Quantum_f1_score` /
  `Classical_f1_score`, `..._accuracy`, plus `val_f1_score`.
- Threshold-tuned trials additionally record `decision_threshold`,
  `val_f1_unthresholded`, and `f1_score_thresholded` (test-side). The plain
  test F1 attribute stays **unthresholded** so it matches what the Analyze tab
  computes; the tuned threshold is informational until analysis applies it
  (flagged follow-up).

## Operational notes

- **Resuming studies:** an Optuna study's parameter choices are immutable.
  Because these changes altered the default search space, resuming a study
  created before them now fails fast with a clear error ("use a new study
  name...") instead of failing every trial with Optuna's cryptic
  `CategoricalDistribution does not support dynamic value space`.
- **Log file:** all backend/server/Optuna log lines are mirrored in real time
  to `db/logs/quoptuna.log` (rotating, 5 MB × 3 backups) in addition to the
  terminal, so runs can be inspected after the fact. Override the path with
  `QUOPTUNA_LOG_FILE`.
