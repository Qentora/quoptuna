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

## SPEC-002 — Decide whether the threshold sweep survives

**Status:** Proposed (the reattachment half shipped; the sweep's value is open)
**Area:** `backend/tuners/optimizer.py`
**Motivation:** Analysis fidelity

### What already shipped

Option (b) below landed with the split-leakage fix (see the changelog's
Unreleased → Fixed): `build_xai` passes the trial's `decision_threshold` into
`XAIConfig`, `XAI.predictions` applies it, and the snapshot's `analysed_model`
records it. Probability-based metrics (ROC-AUC, average precision, log loss)
are unaffected by construction — only the label decision rule changed.

### Problem that remains

`_tune_decision_threshold` picks the best of 19 cutoffs on the validation
split. That is a maximum over 19 noisy estimates on ~88 rows, so the sweep may
be fitting validation noise rather than recovering minority-class signal. If it
does not transfer, the right fix is to delete the sweep, not to keep
propagating its output.

### Evidence

From `ILPD_Mac_faireness_100trials`, best trial #57:

```
val_f1_unthresholded : 0.8296
decision_threshold   : 0.7
val_f1_score         : 0.8871   ← objective, after tuning, on validation
f1_score_thresholded : 0.2933   ← same threshold, on test
Quantum_f1_score     : 0.3261   ← unthresholded test
```

The tuned cutoff made test F1 *worse* (0.326 → 0.293). But this run predates
the leakage fix: its validation split contained duplicated training rows, so
the sweep was optimising memorised rows and these numbers cannot settle the
question. **Every stored `decision_threshold` is contaminated the same way.**
A fresh run is required.

Coverage is uneven regardless: the attr exists only for binary runs on models
with `predict_proba` where the sweep beat the baseline. Multiclass runs (e.g.
the `wifi_signal_*` studies) will never have it.

### Options

- **(a) Report both.** Surface thresholded and unthresholded metrics side by
  side. Superseded by what shipped, which reports one classifier consistently.
- **(b) Reattach the threshold.** *Shipped.*
- **(c) Select the threshold robustly** (cross-validated or nested), then
  reattach — or drop the sweep entirely if the lift does not generalise.

### Open questions

1. On a clean validation split, does the sweep's validation lift transfer to
   test? Measure `val_f1_unthresholded` vs `val_f1_score` against test F1
   across trials of one post-fix run.
2. If it does not: delete `_tune_decision_threshold`, or nest it inside the
   folds of SPEC-006's cross-validation?

### Acceptance criteria

- The question in (1) is answered from a post-fix run, not from the
  contaminated studies in `db/results-trial-june15.db`.
- Whichever way it resolves, every reported F1 states which split and which
  threshold produced it.
- Runs with no recorded threshold behave exactly as today.

---

## SPEC-003 — Distinguish validation and test metrics in the UI

**Status:** Ready
**Area:** `frontend/components/optimizer/steps/OptimizeStep.tsx`,
`AnalyzeStep.tsx`
**Motivation:** Correctness of presentation

### Problem

The search objective is **validation** F1; Analyze computes metrics on **test**
(`xai.get_f1_score` over `y_test`). The UI shows "best F1" beside a test-split
F1 card with nothing marking them as different splits, so a genuine
generalization gap reads as a bug.

The gap that originally motivated this spec — ILPD trial #57, 0.887 vs 0.326 —
turned out to be a leakage bug, not a labelling problem, and is fixed (see the
changelog). Post-fix the two agree closely (ILPD: val 0.280 vs test 0.268), so
this spec is now about presentation only, and the design below is unchanged.

Two residual reasons the numbers still differ, both worth naming in the UI:

- `best_value` is a maximum over trials, so it carries selection optimism even
  when the split is clean (SPEC-006).
- A `decision_threshold`, where one exists, is chosen on validation.

### Proposed design

1. Label the Optimize header value as **validation F1** (selection score) and
   the Analyze metric card as **test F1** (the result).
2. Where both are visible, show the gap explicitly. A large gap now means
   selection optimism or a genuinely hard split — no longer leakage.
3. Where a `decision_threshold` exists, state it next to both figures; after
   the reattachment fix it applies to the test metrics too.

### Acceptance criteria

- No user-facing F1 appears without its split named.
- `best_value` is never presented as the run's result.
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

---

## SPEC-006 — k-fold cross-validated trial scoring

**Status:** Proposed
**Area:** `backend/tuners/optimizer.py` (`objective`, `_ensure_validation_split`,
`_make_pruning_callback`, `_tune_decision_threshold`)
**Motivation:** Selection quality, and honesty of `best_value`
**Depends on:** the split-ordering fix (folds must be cut before resampling,
for the same reason the holdout is)

### Problem

The objective scores each trial on a single stratified holdout — 20% of train,
one seed (`random_state=42`). `best_value` is then the **maximum** over ~100
such estimates. The maximum of many noisy estimates is biased upward, so
`best_value` overstates the winning trial's true skill, and trials whose real
skill differs by less than the measurement noise are ranked essentially at
random.

This is *not* the leakage bug fixed earlier. That one made validation
anti-correlated with test (Spearman −0.75 on `ilpd_oversample-nofair_100`) and
inflated the headline by 0.56; it produced false claims. This is an estimator
efficiency problem: the reported test metrics stay unbiased because no trial is
selected on test, but the *choice* of trial is noisier than it should be, and
`best_value` should never be quoted as a result.

### Evidence

Simulated on the post-fix ILPD validation geometry (88 rows, 25 positive, 100
trials, all trials given identical true skill so every difference is noise):

```
single-trial val F1 : mean 0.462, sd 0.084
max over 100 trials : mean 0.657   -> selection optimism +0.196
```

A real search has genuine skill spread, so the true optimism is smaller — but
the per-trial sd of 0.084 is the measured quantity that matters: configurations
within ~0.1 F1 of each other are being ordered by noise.

Related unmeasured quantity: on `ilpd_oversample-nofair_100` the shipped trial
scored test F1 0.340 while the search had already trained one at 0.631. How
much of that 0.291 a clean single holdout recovers, versus how much needs
variance reduction, is the open question below.

### Proposed design

Replace the single holdout with stratified k-fold CV inside `objective`. The
trial's reported value becomes the mean fold score (report the sd too — it is
what tells you whether a ranking is meaningful).

Mechanics that are not optional:

- **Folds are cut before resampling.** Each fold's training portion is
  resampled independently, inside the loop. Resampling first and folding after
  reproduces exactly the leakage bug this spec builds on — duplicated rows on
  both sides of the boundary.
- **The threshold sweep (SPEC-002) runs per fold**, on that fold's held-out
  part. A single threshold swept across pooled out-of-fold predictions is the
  alternative; decide which after SPEC-002 answers whether the sweep survives
  at all.
- **The fairness disparity is averaged over folds** the same way, since it is
  an objective in `multi_objective` mode.
- **Pruning needs rethinking.** `_make_pruning_callback` reports intermediate
  values per training step of one fit. With k fits per trial the report index
  no longer means what ASHA assumes. Either prune on fold 1 and only then run
  the rest (cheap, biased), or report the running fold mean (correct, but k×
  the cost before a trial can be killed). This is the main design decision.
- **Cost.** `k ×` training per trial. On the JAX-trained quantum models at
  `max_steps=200` this is the dominant cost in the whole pipeline; on kernel
  and sklearn models it is near-free. Consider `k` per model family, or CV only
  for the cheap families, before accepting a flat `k` everywhere.

Cheaper alternatives worth measuring first, since they may capture most of the
benefit:

- **Never present `best_value` as a result** (SPEC-003). Free; removes the harm
  of the bias without touching the search at all.
- **Re-rank the shortlist.** Re-score only the top N trials on 3 extra
  validation seeds and pick by the mean: `3 × N` refits instead of `k × 100`,
  aimed exactly at the trials whose ordering is contested.

Independently of which ships: report test metrics with a confidence interval
and the trial count wherever a result is published.

### Open questions

1. **Pruning.** Prune on the first fold before spending the rest, or report a
   running fold mean? The first is `k×` cheaper per killed trial but prunes on
   exactly the noisy single-fold estimate this spec exists to replace.
2. Does shortlist re-ranking recover most of CV's benefit at a fraction of the
   cost? Measurable by re-ranking a completed study offline — no new training
   for the kernel and classical models.
3. Should `k` adapt to dataset size, or to model family? The variance problem
   is worst where the holdout is smallest, which is also where CV is cheapest;
   the cost problem is worst on the variational quantum models.
4. Does the fairness disparity need per-fold treatment, or is the mean enough?
   It drives an objective in `multi_objective` mode, so it inherits the
   identical variance problem.
5. Does Optuna's `study.best_value` semantics still read sensibly when the
   value is a fold mean, and does the stored `val_f1_score` attr become the
   mean, the per-fold list, or both?

### Acceptance criteria

- The per-trial validation sd is measured on a real post-fix run, not simulated.
- Folds are verifiably cut before resampling: the leakage assertion in
  `tests/test_split_leakage.py` extends to every fold's train/validation pair.
- Selection quality is demonstrated by the selected trial's **test** F1
  improving against the current single-holdout baseline on at least one
  imbalanced dataset — not by `best_value` moving.
- The added wall-clock cost per trial is measured per model family, not
  assumed.
- `best_value` is documented everywhere as a selection score.

### Non-goals

Nested cross-validation for unbiased performance estimation. The held-out test
split already serves that purpose.

---

## SPEC-007 — Reliable rare-class and small-sample optimization

**Status:** Proposed  
**Area:** `backend/tuners/optimizer.py`, `server/services/workflow_service.py`,
`server/api/v1/optimize.py`, `server/api/v1/analysis.py`, frontend optimization
and analysis metrics  
**Motivation:** Prevent selection optimism and invalid fairness claims on
datasets such as UCI Fertility (100 rows, 12 positive outcomes).

### Problem

The current three-way split can leave a tiny validation set with only one or
two positive examples. A search over many model families, hyperparameters, and
19 decision thresholds can then select a validation-noise winner. The held-out
test set correctly exposes the gap, but it is currently predicted during every
trial and its metrics are stored on each trial, weakening the intended
train/validation/test separation.

Fertility demonstrates the failure mode:

```text
rows                       100
class counts               N=88, O=12
trial 24 validation F1     1.0000
trial 24 threshold          0.45
trial 24 thresholded test F1 0.0000
test confusion matrix      TN=21, FP=1, FN=3, TP=0
```

The reported 84% test accuracy is majority-class performance, not useful
positive-class detection. Its fairness audit is also unsupported: the
`child_diseases` groups contain 4 and 21 test rows. In addition, the search
currently records fairness from default `predict()` labels while its F1
objective may use thresholded labels; those are different classifiers.

### Relationship to existing specs

- **SPEC-002** decides whether threshold selection survives and how it is
  estimated.
- **SPEC-003** labels validation and test results distinctly in the UI.
- **SPEC-006** provides k-fold trial scoring.

This spec adds the missing operational safety rules: strict test isolation,
minimum-support gates, threshold/fairness consistency, small-data capacity
controls, and a reproducible resampling comparison.

### Proposed design

#### 1. Strict test isolation

`Optimizer.objective` must fit and score only training and validation data.
It must not call `predict`, `predict_proba`, threshold selection, fairness
metrics, or metric recording on `test_x`/`test_y`.

After Optuna selects a trial:

1. reconstruct the selected configuration;
2. fit it only on the development data allowed by the selected protocol;
3. evaluate the untouched test split exactly once;
4. persist final test metrics separately from per-trial selection metrics.

The only exception is an explicitly marked research/debug mode that never
publishes test results as final evidence.

#### 2. One label rule per evaluated classifier

Threshold selection must return both the chosen threshold and its thresholded
validation predictions. F1, fairness disparity, per-group metrics, analysis,
and the final test evaluation must use that same label rule.

The stored record must include:

```text
metric split             train | validation | test
decision rule            model default | threshold=<value>
threshold source         none | validation OOF predictions
```

Do not compare default-rule fairness with thresholded-rule F1.

#### 3. Support-aware protocol selection

Before optimization, calculate class and protected-group support after the
planned split. Surface the chosen protocol in the request and report.

| Condition | Required behavior |
| --- | --- |
| Fewer than 10 validation or out-of-fold positive examples | Disable automatic threshold sweep; use the model default or a user-specified domain threshold. |
| Fewer than 20 examples in any protected group, or fewer than 5 positive and 5 negative outcomes in a group | Disable fairness optimization and label the audit insufficiently supported. |
| Rare-class data with adequate total support | Prefer PR-AUC, recall, F1, and class-wise confusion counts over accuracy. |
| Small dataset with no viable validation support | Require repeated stratified CV or refuse a model-selection claim; never silently choose a lucky holdout. |

Exact thresholds are configuration defaults, recorded with the run, and may be
raised by a domain policy. They must not be silently relaxed.

#### 4. Stable selection

Implement SPEC-006's repeated stratified CV or shortlist re-ranking. Select by
mean validation score and record standard deviation, fold scores, class counts,
and selected threshold per fold. Thresholds are either selected per fold or
from pooled out-of-fold probabilities; the choice must be measured against
SPEC-002 before it ships.

The UI must show a selection score as a distribution, for example:

```text
validation F1: 0.54 ± 0.11 across five folds
```

not as a single result claim.

#### 5. Small-data capacity policy

Add a data-size-aware preset for iterative quantum models:

- cap encoding/re-uploading layers and maximum steps on low-support data;
- favour simpler classical baselines and shallow quantum models;
- use convergence/early-stop diagnostics;
- show a warning when parameter capacity is high relative to minority-class
  support.

The preset constrains search only; it does not fabricate a generalization
claim.

#### 6. Resampling as an experiment

Keep resampling strictly inside each training fold. For rare-class datasets,
offer a paired comparison:

```text
none versus RandomOverSampler
same fold assignments, model families, threshold policy, and seed
```

Report unique minority rows before resampling and duplicated rows after it.
Never default to undersampling when it would discard most of a small majority
class. Oversampling is a recall-oriented option, not a substitute for
independent positive examples.

#### 7. Confidence intervals and publication rules

Final test reports must include class counts and a confidence interval or
explicit small-sample warning for F1, recall, and PR-AUC. Runs that fail
support gates can remain inspectable, but cannot be marked as a successful
fairness or generalization result.

### Acceptance criteria

- No per-trial code path reads `test_x` or `test_y`; a regression test fails
  when `objective` touches either.
- A thresholded objective and fairness metric consume identical label arrays.
- The Fertility support gate disables fairness optimization and automatic
  threshold tuning under the documented default thresholds.
- An oversampling comparison duplicates rows only inside a training fold; no
  source row appears in both that fold's training and validation/test data.
- Selection output includes fold scores, support counts, mean, and standard
  deviation.
- Test metrics are produced once for the selected trial and are labelled as
  final held-out results.
- A report cannot describe a support-gated run as a fairness success.

### Non-goals

Improving the intrinsic predictive signal of datasets with too few independent
positive outcomes. Collecting more positive examples or choosing a larger
dataset remains necessary for a credible fertility/fairness study.
