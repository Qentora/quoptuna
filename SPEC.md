# QuOptuna — Planned work specifications

Specs for work that is designed but not built. Each entry states the problem,
the evidence, the proposed design, open questions, and how we will know it is
done.

Conventions:

- **Status** — `Proposed` (agreed problem, design open) · `Ready` (design
  settled, safe to implement) · `In progress` · `Done` (move to the changelog).
- A spec is the source of truth until the work lands; update it when the design
  changes, not after the code does.
- Background on the current pipeline lives in
  `docs-site/src/content/docs/explanation/analysis-pipeline.md`.

---

## SPEC-001 — Cache the fitted model across analyses

**Status:** Proposed
**Area:** `server/services/workflow_service.py`, `server/api/v1/analysis.py`
**Motivation:** Analysis latency

### Problem

Every analysis run refits the selected trial's model, including an identical
re-analysis of the same trial. `build_xai` calls `model.fit(...)` on each
invocation. For a variational quantum model at the searched budget
(`max_steps=200` on `lightning.qubit`) this is the single largest cost in the
pipeline, and the UI cannot show anything until it completes.

The step is not parallelisable: data prep → fit → SHAP is a hard dependency
chain, and the fit is one call inside the model class. Caching is the only
lever.

### Evidence

- `build_xai` fits unconditionally (`workflow_service.py`, the `model.fit` call
  after `create_model`).
- The deployment path (`_execute_shap_analysis`) performs the same refit.
- Nine of eleven stored runs carry a non-default training budget, so the fit is
  genuinely expensive for the common case.

### Proposed design

Persist the fitted estimator keyed by everything that determines it:

```
(optimization_id, trial_number, training_budget, data_prep_fingerprint)
```

- `training_budget` — the `_training_budget(opt_result)` dict. A budget change
  **must** miss the cache; this is the failure mode that would silently serve a
  model trained at the wrong step count.
- `data_prep_fingerprint` — covers feature selection, split, encoding and
  resampling. The split is deterministic under a fixed `random_state`, but the
  *request* that produced it is not immutable.
- Store alongside existing analysis artifacts, under the run's artifact
  directory, with the same local/S3 backends.

### Open questions

1. **Serialization.** Which model classes are safely picklable? JAX-trained
   models may hold device state that does not round-trip. Probably an
   allowlist, falling back to refit.
2. **Size bound.** What is the eviction policy, and does it follow
   `ANALYSIS_HISTORY_LIMIT` or get its own budget?
3. **Correctness check.** Should a cache hit verify the model reproduces the
   trial's recorded metrics before being trusted?
4. **Invalidation on code change.** A change to `create_model` or a model class
   invalidates every cached fit. Include a version stamp?

### Acceptance criteria

- Re-analysing the same trial with the same config performs no refit.
- Changing `max_steps`, the trial, or any data-prep input forces a refit.
- A corrupt or unreadable cache entry falls back to refitting rather than
  failing the analysis.
- Cache hits and misses are visible in the job's section reporting.

### Non-goals

Parallelising the fit itself, or caching across different optimization runs.

---

## SPEC-002 — Reattach the tuned decision threshold

**Status:** Proposed
**Area:** `backend/tuners/optimizer.py`, `server/services/workflow_service.py`
**Motivation:** Analysis fidelity

### Problem

`_tune_decision_threshold` sweeps the binary decision cutoff on the validation
split and records it as a `decision_threshold` trial user attr — not in
`trial.params`. The rebuilt model therefore predicts at the default 0.5, so
SHAP values and metrics describe a different classifier from the one whose
score won the search.

### Evidence

From `ILPD_Mac_faireness_100trials`, best trial #57:

```
val_f1_unthresholded : 0.8296
decision_threshold   : 0.7
val_f1_score         : 0.8871   ← objective, after tuning, on validation
f1_score_thresholded : 0.2933   ← same threshold, on test
Quantum_f1_score     : 0.3261   ← unthresholded test (what Analyze reports)
```

Two conclusions:

1. The threshold is overfit to the validation split — it does not transfer.
2. Reattaching it would make Analyze report **0.293 instead of 0.326**, i.e.
   worse. This is a fidelity fix, not an improvement, and must be framed as
   such.

Coverage is uneven: the attr exists only for binary runs on models with
`predict_proba` where the sweep beat the baseline. Multiclass runs (e.g. the
`wifi_signal_*` studies) will never have it.

### Proposed design

Options, in increasing order of ambition:

- **(a) Report both.** Keep Analyze unthresholded, and surface the thresholded
  metrics alongside with the threshold stated. No behaviour change, closes the
  information gap.
- **(b) Reattach the threshold** so the analysed model matches the searched one,
  and state the threshold prominently wherever metrics appear.
- **(c) Select the threshold more robustly** (cross-validated, or nested) so it
  generalises, then reattach.

(a) is safe and immediately useful. (c) addresses the real defect — the ILPD
numbers show the current sweep is not trustworthy on a small validation split.

### Open questions

1. Is a threshold tuned on one small validation split ever worth reattaching,
   or should it be recomputed?
2. Should the search objective itself stop using the tuned value, given it
   inflates the reported score?

### Acceptance criteria

- Whichever option ships, every reported F1 states which split and which
  threshold produced it.
- Runs with no recorded threshold behave exactly as today.
- Existing snapshots remain readable.

---

## SPEC-003 — Distinguish validation and test metrics in the UI

**Status:** Ready
**Area:** `frontend/components/optimizer/steps/OptimizeStep.tsx`,
`AnalyzeStep.tsx`
**Motivation:** Correctness of presentation

### Problem

The search objective is **validation** F1 (`_ensure_validation_split` carves a
validation set out of train so selection does not tune to test). Analyze
computes metrics on **test** (`xai.get_f1_score` over `y_test`). The UI shows
"best F1" beside a test-split F1 card with nothing marking them as different
splits, so a normal generalization gap reads as a bug.

On ILPD trial #57 the two differ by 0.887 vs 0.326, which looks alarming and
prompted exactly this confusion.

### Proposed design

1. Label the Optimize header value as **validation F1** and the Analyze metric
   card as **test F1**.
2. Where both are visible, show the gap explicitly and note that a large gap
   indicates overfitting to the validation split.
3. Where a `decision_threshold` exists, state it next to the validation figure
   — it is a large part of why the numbers diverge.

### Acceptance criteria

- No user-facing F1 appears without its split named.
- Trials with no validation split (the small-dataset fallback path) are
  labelled accurately rather than mislabelled as validation.

### Non-goals

Changing how either metric is computed.

---

## SPEC-004 — True parallelism for analysis sections

**Status:** Proposed
**Area:** `server/api/v1/analysis.py`
**Motivation:** Analysis latency
**Blocked by:** matplotlib pyplot thread-safety

### Problem

The derived sections run under `asyncio.gather(..., return_exceptions=True)`,
which gives failure isolation and removes per-section stalls but **not**
parallelism: these endpoints are `async def` yet perform blocking CPU work with
no internal `await`, so they still execute one at a time on the event loop.

### Proposed design

Move the sections to `run_in_threadpool`. This requires two prerequisites:

1. **matplotlib.** The plotting endpoints use the stateful `pyplot` API
   (`plt.subplots`, `plt.close`), which is not thread-safe. They must migrate to
   the object-oriented API (`Figure(...)` directly) first.
2. **Shared `XAI`.** Sections share one instance via `ContextVar`.
   `_warm_xai_caches` makes the readers pure, which is sufficient for the
   current serialised execution; under real threads the memoised attributes
   need verifying, or locking.

### Open questions

1. Is the speedup worth it once SPEC-001 removes the dominant cost? The fit,
   not the sections, is the bottleneck — this may not be worth doing.
2. Does the GIL limit the gain, given the work is numpy/sklearn heavy (which
   releases it) mixed with Python-level plotting (which does not)?

### Acceptance criteria

- Sections execute concurrently, demonstrated by wall-clock measurement on a
  real run, not a synthetic one.
- Failure isolation is preserved.
- No figure corruption under concurrent rendering.

### Non-goals

Process-level parallelism or a task queue.

---

## SPEC-005 — Keep tests out of the application database

**Status:** Ready
**Area:** `tests/server/`, `server/services/database.py`
**Motivation:** Developer safety

### Problem

`_database_url()` prefers `run_store.APP_DB_PATH` over `settings.DATABASE_URL`,
and `get_engine` is `lru_cache`d. A test that patches only `DATABASE_URL` still
resolves to the real `db/quoptuna_app.db`, and a cached engine keeps it even
after patching. Test fixtures wrote rows (`run-1`, `run-2`, `opt_old`) into the
developer's live database.

### Evidence

Observed directly: `quoptuna_analysis_snapshots` contained `run-1`, `run-2` and
`opt_old` alongside real `opt_*` runs. Fixed for the provenance suite by also
patching `run_store.APP_DB_PATH` and clearing `get_engine`'s cache, but the
trap remains for any new test.

### Proposed design

A shared autouse fixture in `tests/server/conftest.py` that points every
database-touching test at a temporary path — patching `DATABASE_URL` **and**
`run_store.APP_DB_PATH`, and clearing `get_engine.cache_clear()` before and
after. Optionally, a guard that fails loudly if a test resolves to a path under
`db/`.

### Acceptance criteria

- Running the full suite leaves `db/quoptuna_app.db` byte-identical.
- A test that forgets isolation fails rather than writing to the real database.
