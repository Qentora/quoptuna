---
title: Analysis pipeline
description: How the analysis job retrains a trial, what runs before SHAP, and how results and history are persisted.
---

The Analyze step explains one trial from a finished optimization. It runs as a
durable background job so a browser reload never loses a result, and it
persists every run so earlier analyses stay inspectable. This page describes
what the job does, in what order, and why.

## Which model is analysed

The analysis explains **one trial**, not the study as a whole.

By default that is the study's best trial. The UI seeds `selectedTrial` from
the highest-scoring successful trial when the optimization finishes, and any
row in the trials table can be clicked to choose a different one. If the client
sends no `trial_number`, the backend independently resolves the best trial via
`study_best_trial`, so the default holds on both sides.

For multi-objective (fairness-constrained) studies `study.best_trial` is
undefined, so `study_best_trial` takes the Pareto-front point with the highest
objective 0 — F1 in every mode. That is the most accurate point on the front,
which is not necessarily the best fairness trade-off; pick a trial explicitly
when the trade-off matters.

### Retraining fidelity

The model is **rebuilt and refitted**, not loaded from a serialized artifact.
`build_xai` reads the trial's sampled hyperparameters from Optuna and refits on
`x_train`. The split is deterministic under a fixed `random_state`, so the
training data matches what the trial saw.

Optuna records only what it *sampled*, and the search passes three further
arguments to `create_model` that are not hyperparameters:

- `max_steps`
- `convergence_interval`
- `dev_type`

These are absent from `trial.params`. Rebuilding without them silently refits
at the model class defaults — for `CircuitCentricClassifier` that is
`max_steps=10000` against a searched value of `200`. `_training_budget` in
`workflow_service.py` recovers them from the persisted request and replays them
in both `build_xai` and the deployment path. Values that are `None` are omitted
so runs predating these knobs keep the class defaults.

Only the iterative JAX-trained models consume this budget; `create_model`
applies it via `hasattr`, so kernel and classical models are unaffected.

### Known gap: the decision threshold

`_tune_decision_threshold` sweeps the binary decision cutoff on the validation
split and records it as a `decision_threshold` user attr — not in
`trial.params`. The rebuilt model therefore predicts at the default 0.5, and
Analyze reports the unthresholded metrics. This is deliberate and internally
consistent (the reported `f1_score` attr is also unthresholded), but it means
the analysed classifier is not the thresholded one whose score won the search.

The attr only exists for binary runs on models with `predict_proba` where the
sweep beat the baseline, so many studies have nothing to restore.

## What runs before SHAP

Three steps precede SHAP, and the first two can dominate the total time:

1. **`preparing`** — `_get_completed_result`. For a run whose in-memory job was
   lost to a restart, `_rehydrate_result` re-executes the data-prep workflow
   (load, feature selection, split, encoding) to reproduce the exact
   DataFrames.
2. **`training`** — `build_xai` calls `model.fit(...)`. On a variational
   quantum model at the searched budget this is the slowest step of the entire
   analysis.
3. **`shap`** — the SHAP computation itself.

Each publishes its own `current_section` so the UI can report honest progress.
All three previously reported as `shap`, which made a working job look frozen.

These steps form a hard dependency chain — the model cannot be fitted before
the data exists, and SHAP cannot run before the model is fitted — so they
cannot be reordered or overlapped. Reducing this cost means caching the fitted
model rather than parallelising it (see *Planned work* below).

## Section execution

After SHAP and metrics complete, the job publishes them immediately via
`publish_partial` so the UI can render core results while the rest continue.

The derived sections — curves, curve data, confusion data, feature importance,
SHAP data — then run as one `asyncio.gather(..., return_exceptions=True)`
group, with study plots and fairness as a second group. Grouping gives two
guarantees: one section failing records a warning instead of cancelling its
siblings, and there are no serialised per-section stalls.

:::caution[This is ordering, not true parallelism]
These endpoints are `async def` but perform blocking CPU work (SHAP, sklearn,
matplotlib) with no internal `await`. `asyncio.gather` overlaps awaits, not
compute, so on a single event loop the sections still execute one at a time.

Real parallelism requires `run_in_threadpool`, which is blocked on matplotlib:
the plotting endpoints use the stateful `pyplot` API, which is not
thread-safe. Migrating them to the object-oriented API is a prerequisite.
:::

### Shared state

All sections share one `XAI` instance through a `ContextVar` to avoid refitting
per section. `XAI.shap_values`, `.predictions` and `.predictions_proba` memoise
into unguarded attributes, so `_warm_xai_caches` computes them before the
concurrent group runs, leaving the sections as pure readers. Warming is
best-effort per property: a model without `predict_proba` must not fail the
analysis.

## Persistence and history

A **snapshot** is keyed by `(optimization_id, config)`, where the config
includes the trial number, SHAP cost knobs and class/sample indices. Re-running
the same configuration produces a new *revision* of the same snapshot.

Every completed revision is recorded in `quoptuna_analysis_revisions` with its
payload, the job that produced it, and the trial and model type it explains.
Revision rows are kept indefinitely. Only the on-disk *artifacts* are pruned,
beyond `ANALYSIS_HISTORY_LIMIT` (default 5), and the row is flagged
`artifacts_pruned` so the UI can present it as metadata-only. Set the limit to
`1` for the old keep-latest-only behaviour, or `0` to disable pruning.

History is exposed at:

- `GET /api/v1/analysis/snapshots/{id}/revisions`
- `GET /api/v1/analysis/snapshots/{id}/revisions/{n}`

## Report provenance

Each snapshot stores an `analysed_model` block recording which model the
results describe: the resolved trial number, whether it was chosen explicitly
or as the best trial, the model type, hyperparameters, and the training budget
used to refit it. "Best" is resolved to a concrete trial number **at analysis
time**, so the record cannot drift if the study is later extended.

The report renders this explicitly rather than leaving it implied:

> All SHAP values, metrics, curves and fairness figures in this report describe
> a **DataReuploadingClassifier** model from trial **#7**, which is also the
> best trial of the search.

An analysis of a non-winning trial is flagged as such. Snapshots written before
provenance existed fall back to the request config and omit the line.

Reports always build against the **latest** revision of a snapshot. A client
holding a stale revision number is served the current one, and the report
records the revision it used. Only a revision the server never produced is
rejected.

## Planned work

Known gaps and their designs are specified in `SPEC.md` at the repository root:

- **SPEC-001** — cache the fitted model across analyses (the dominant cost in
  the pipeline).
- **SPEC-002** — reattach the tuned decision threshold.
- **SPEC-003** — distinguish validation and test metrics in the UI.
- **SPEC-004** — true parallelism for the analysis sections, blocked on
  matplotlib thread-safety.
