# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [1.0.0] - 2026-09-25
First stable release. The major bump marks two things: results produced by earlier
versions are not directly comparable (the validation-leakage, target-shape and
determinism fixes below change the scores of affected runs), and several Python APIs
changed shape. Re-run any study whose conclusions depend on resampled runs,
fairness-aware search, SVC winners, or `QuantumKitchenSinks`.

### Breaking changes
- `Optimizer(sensitive_test=...)` is renamed to `sensitive_val`. The fairness
  disparity is now measured on the **validation** split, so the sensitive column must
  align positionally with `val_y`; a length mismatch raises `ValueError`.
- `Optimizer` now reads the validation split from `data["val_x"]` / `data["val_y"]`.
  Callers that resample the train split MUST supply it; without those keys a
  stratified 20% of train is still carved as before.
- `report_agent.generate_report` takes an evidence bundle (`report_context`) and
  returns a result dict (final markdown, draft, rendered evidence,
  referenced/dropped figures, lint findings) instead of a bare string.
- Bundled datasets ship as gzipped CSVs (`*.csv.gz`). Code that opened the old
  `*.csv` files by path must switch to the new names (pandas reads both).
- Scores of existing runs are not reproduced bit-for-bit by this version wherever the
  fixes below apply; stored trials are kept as-is, but a re-analysis uses the
  corrected pipeline.

### Added
- Class-imbalance resampling for optimization runs: `resampling` on the optimize
  request (`"none"`, `"oversample"`, `"undersample"`; default `"none"`), exposed in the
  UI. Only the inner training portion is resampled, and the sensitive column is
  resampled in lockstep so fairness checks stay row-aligned. Adds the
  `imbalanced-learn` dependency.
- Resumable, cancellable analysis jobs. Jobs run off the event loop (one at a time),
  report progress through preparation, training and SHAP (with per-row SHAP
  counters), and publish partial SHAP/metrics payloads while later sections finish.
  `GET /api/v1/analysis/jobs?optimization_id=...` lets a refreshed browser reattach to
  a running job, `POST /api/v1/analysis/jobs/{job_id}/cancel` stops one, and jobs
  left pending/running by a restart are marked failed on startup.
- Analysis revision history: `GET /api/v1/analysis/snapshots/{id}/revisions` and
  `.../revisions/{revision}`. The Analyze step browses earlier revisions; the Report
  step shows which revision a report was grounded in and warns when it differs from
  the one loaded. Report generation returns `analysis_revision`.
- Bulk research dumps: select runs on the Runs page and download one zip via
  `POST /api/v1/analysis/bundles/bulk` (1–100 optimization ids). Runs with a
  completed analysis get the full research bundle; runs without one get a
  metadata/trial/Pareto archive.
- Target class-balance profiles (`quoptuna.datasets.balance.target_balance_profile`):
  bundled catalog entries and dataset previews (`target_balance_by_column`) label a
  target as balanced, moderately imbalanced, imbalanced or multiclass, shown in the
  Dataset and Features steps. The label describes prevalence only, not fairness.
- Offline bundled datasets (loaded before any network fetch):
  - UCI: Adult / Census Income, Statlog (German Credit), Breast Cancer Wisconsin
    (Original), Contraceptive Method Choice, Iris, Mammographic Mass, Seeds, User
    Knowledge Modeling, Banknote Authentication, Occupancy Detection, Exasens, Rice
    (Cammeo and Osmancik), Raisin, Wireless Indoor Localization.
  - Fairness benchmarks and public surveys: COMPAS Two-Year Recidivism, ACS 2023
    California Employment, ACS 2023 California Travel Time, ACS PUMS 2023 Internet
    Access, NTIA Internet Use Survey 2023 Wearable Use.
  - RECS 2020 Housing Tenure (owned vs rented), as a full set (id 9021) and a seeded,
    class-stratified 2,000-row subset sized for kernel models (id 9020). Ids from
    9000 up are reserved for bundled datasets with no UCI entry.
  - Regeneration scripts under `scripts/` (`prepare_uci_catalog.py`,
    `prepare_public_surveys.py`, `prepare_fairness_benchmarks.py`,
    `prepare_recs2020.py`).
- Cross-platform (Windows-friendly) npm task runner: a root `package.json` and
  `scripts/run.mjs` mirroring the Makefile targets (`npm run dev`, `test`, `lint`,
  `build`, ...), seeding `frontend/.env.local` for local API routing.
- GPU development workflow with WSL2 support: `npm run gpu:check`, `gpu:setup` and
  `dev:gpu`.
- `QUOPTUNA_SAMPLING_FLOAT32` override and adaptive sampling precision for shot-based
  PennyLane circuits.
- `psycopg2-binary` dependency for PostgreSQL-backed deployments.
- Report agents now receive a complete, structured evidence bundle for the run
  (`report_context`): every configuration option chosen, the trial history and
  per-model-family aggregates, hyperparameter importances, the fairness-aware search
  settings, the Pareto front with its knee point, per-feature SHAP statistics, and an
  explicit list of what the run does *not* contain.
- One-click research dump per analysis snapshot:
  `GET /api/v1/analysis/snapshots/{id}/bundle` returns a zip with `context.json`, the
  exact evidence markdown the agents read, every figure as a file, the Optuna study
  figures as Plotly JSON, CSV tables, the prompts, and all generated reports;
  `.../context` serves the same bundle as JSON. Both are downloadable from the Report
  step.
- Settings → AI Report Generation exposes the analyst and reviewer prompts plus the
  evidence toggles (fairness audit, fairness-aware search, Pareto front, SHAP detail,
  trial history, study diagnostics, figure attachment, trial-row cap), each with an
  inline explanation of what it does and when to use it. Defaults are served from
  `GET /api/v1/analysis/report-prompts` so the UI and the agents never drift.
- Generated reports are normalised to a strict GitHub-Flavored Markdown contract in
  code (`markdown_report`): wrapping code fences and chatty preambles removed, one H1,
  no skipped heading levels, repaired pipe tables, and figure references validated
  against the manifest. Residual issues are reported instead of shipped.
- Reports reference figures as `figures/<id>.png`, which resolves to a real file inside
  the downloaded bundle and to the in-browser plot in the Report step.
- Documented the `quoptuna infra`, `active-work`, and `deployment-check` CLI commands.
- New "Deploy the AWS infrastructure" how-to covering prerequisites, the deployment
  file, the Textual console options, every operation and script flag, and troubleshooting.
- Documented `AUTH_ALLOWED_EMAILS`, `AUTH_REQUIRE_VERIFIED_EMAIL`, and `APP_ENV`,
  including the production start-up check and the public `/api/v1/health` path.
- Added `QuantumBoltzmannMachine` to the model catalog and a new image-shaped models
  section for `QuanvolutionalNeuralNetwork`, `WeiNet`, and `ConvolutionalNeuralNetwork`.
- Refit consistency check. The search already scores every trial on the test split;
  the analysis recomputes the same number after retraining it. Those must agree, and
  every silent-divergence bug above moved that delta while leaving other metrics
  plausible. Analyses now record `refit_consistency` (both values, the drift, and
  whether it is within tolerance) in `analysed_model`, and raise a snapshot warning —
  visible to the UI and the report agent — when the analysed model does not reproduce
  the trial that was selected.
- End-to-end pipeline canary (`tests/test_pipeline_canary.py`) on Banknote
  Authentication, which is linearly separable: a correct pipeline scores ~1.0, so any
  of the failure modes above shows up as a number below the floor. Asserts perfect
  F1/accuracy, a non-null ROC-AUC, search/analysis agreement, and a confusion matrix
  that uses both classes. Runs in ~5s.
- Determinism regression tests: `tests/test_model_isolation.py` (instances never share
  a fitted estimator; a second fit cannot change an earlier model's predictions;
  identical configurations refit identically) and an end-to-end check that two
  identical runs of the shot-based model produce identical metrics.

### Changed
- The report pipeline no longer sends the agents a `str()` of six metric keys. The
  analyst and reviewer prompts were rewritten around a fixed section skeleton, a
  grounding rule, and the markdown contract; conditional sections (fairness audit,
  fairness-aware search, Pareto front) are dropped when their evidence is absent
  instead of being invented.
- The analysis reviewer pass can be turned off, and a reviewer that returns almost
  nothing no longer replaces a good draft.
- Dataset selection (API and Streamlit) loads bundled files first and only falls back
  to a network fetch for datasets that are not bundled. All bundled datasets are
  gzipped, shrinking the original three files from ~1.9 MB to ~223 KB.
- `SeparableKernelClassifier` precomputes its kernel in vectorized blocks bounded by
  `max_vmap`, reducing peak memory and removing the per-pair Python loop.
- Clarified that the model catalog lists registry keys, while `/api/v1/models`
  returns display names.

### Fixed
- **Train rows leaked into the validation split on resampled runs.** The train split
  was oversampled (`RandomOverSampler` duplicates minority rows verbatim) *before*
  `Optimizer` carved its validation split out of it, so a row and its own copy landed
  on opposite sides of that boundary and the objective scored memorisation. On ILPD
  (71/29 imbalance) 43% of validation rows — 84% of the positive class — were exact
  copies of training rows, reporting F1 0.90 where the true value was 0.27, and
  ranking trials *against* their real test performance (Spearman -0.75). The
  validation split is now carved in the split node before any resampling, and only
  the inner training portion is resampled; `Optimizer` consumes the split instead of
  deriving one. Undersampled and unresampled runs were unaffected.
- The fairness-aware search measured its disparity on the **test** split, so
  constrained and multi-objective runs selected against the same data the post-hoc
  audit reports. Disparity is now computed on validation, with the sensitive column
  carried through the carve and the resampling in lockstep.
- Analyze refit the selected trial on a different frame than the trial trained on and
  scored it at a 0.5 cutoff while the objective had been maximised over a tuned
  `decision_threshold`. It now refits on the trial's own training frame and applies
  that threshold to every label-based metric (`decision_threshold` is recorded in the
  snapshot's `analysed_model`); probability-based metrics are unchanged.
- **Analyze trained every model on a mis-shaped target.** `build_xai` (and the SHAP
  node) passed `y_train.values` straight to `model.fit`, but the label-encoding node
  stores `y` as a DataFrame, so the target arrived as `(n, 1)` instead of `(n,)`. The
  fit does not raise: it collapses the model's probabilities into a narrow band
  around 0.5 (observed range 0.474-0.512 on ILPD) and degrades its ranking (test
  ROC-AUC 0.54 versus 0.72 for the same configuration fitted correctly). Argmax
  predictions stayed plausible, which is why it went unnoticed — until a probability
  threshold was applied to that band and put every row on one side, reporting F1
  0.000 with zero predicted positives. Both refit paths now ravel the target.
- A stored `decision_threshold` that falls outside the refit's probability range is
  now discarded rather than applied: the analysis falls back to the model's own
  `predict` and records `decision_threshold_discarded` in `analysed_model`. A
  threshold chosen against one fit is not guaranteed to be meaningful against
  another, and an analysis must never report an all-one-class score for a model that
  predicts both.
- `SVC` was constructed with scikit-learn's default `probability=False`, so it has no
  `predict_proba`. Whenever an SVC won the search — the common case on easy datasets
  — ROC-AUC, average precision and log loss came back `null`, the decision-threshold
  sweep was skipped, and the UI's analysis (which requests probability mode by
  default) failed outright with "Model does not have a predict_proba method". SVC now
  fits Platt scaling with a pinned `random_state`, and `build_xai` degrades to label
  mode for models that genuinely cannot produce probabilities (`LinearSVC`,
  `Perceptron`) instead of failing the analysis.
- **Scores changed on every re-analysis.** Two independent causes, both now fixed.
  1. The kernel-head models (`ProjectedQuantumKernel`, `IQPKernelClassifier`,
     `SeparableKernelClassifier`, `QuantumKitchenSinks`) took their inner sklearn
     estimator as a *mutable default argument* (`svm=SVC(...)`). Python evaluates a
     default once at import, so every instance of the class shared one estimator
     object, and each `fit` refitted it in place. In the analysis job — which fits its
     model and then lets the fairness section build another — the second fit rewrote
     the first model's classifier head underneath the metrics being computed.
     Measured: 27 of 40 predictions flipped in an already-fitted model purely because
     a second instance was fitted. Each class now builds its own estimator.
  2. `QuantumKitchenSinks`, the only shot-based model, sampled from an unseeded
     simulator device, so the same fitted model returned different predictions on
     every call. The device is now seeded with a `jax.random.PRNGKey` derived from
     `random_state`, which is reused verbatim per execution, making repeated predicts
     and independent refits bit-identical.
- Report figure selection called `random.sample(range(n), n)` — a whole-population
  sample, so the result was always `range(n)` but the call still consumed global RNG
  state. Replaced with `range(n)`.
- The refit consistency check compared an analysis F1 computed at the trial's tuned
  decision threshold against the trial's unthresholded F1, reporting the threshold's
  own effect as drift (0.131 on RECS 2020). It now compares against
  `f1_score_thresholded` when a threshold was applied and records which rule it used.
- UCI Adult mixed `>50K.` / `<=50K.` (test file) with `>50K` / `<=50K` (training
  file), producing four target classes instead of two. Targets are now canonicalized
  on load.
- `QuantumKitchenSinks` failed on Windows with a dtype callback error during
  shot-based sampling; sampling now runs under the adaptive precision context.
- File-based SQLite URLs whose parent directory did not exist failed at start-up; the
  directory is now created before the engine.
- Light-theme text contrast now meets WCAG 2.2 AA; axe-core reports no violations on
  any audited page in either theme (report: `audits/accessibility/2026-09-25-axe-wcag.md`).
  Darkened `--muted-foreground` and the emerald, amber, red, purple and orange accent
  foregrounds (charts using them shift slightly darker), switched the destructive
  badge to the red accent foreground in light mode, removed the 80% opacity from the
  "When to use" help text in Settings and the 70% opacity from the Report history
  count, and made the dashboard's "Open the Optimizer" gradient end at 90% opacity.

## [0.1.5] - 2026-07-27
### Added
- AWS infrastructure: Terraform templates, infra scripts, and AWS CLI tooling.

### Changed
- Refreshed landing page, docs theming, and spacing.

## [0.1.4] - 2026-07-18
### Added
- SQLModel-backed persistence (SQLite or PostgreSQL) replacing the legacy SQLite app
  store, with migration tooling and a `migrate-supabase` CLI command.
- S3 artifact storage with presigned URLs; uploaded datasets record their object key.
- Optuna storage helpers (`ensure_optuna_schema`, `optuna_storage_url`).

### Changed
- README overhaul, branding assets, and community profile files.

## [0.1.3]
### Changed
- Migrated the documentation from MkDocs to an Astro + Starlight site in `docs-site/`,
  reorganized on the Diátaxis framework (tutorials / how-to / reference / explanation)
  with a custom marketing landing page.
- Rewrote docs against the current Next.js + FastAPI app; added architecture, feature,
  CLI, REST API, configuration, and model-catalog pages; consolidated the legacy
  Streamlit docs into a single page.
- Replaced the MkDocs GitHub Actions workflows with an Astro Pages deploy plus a
  per-PR docs preview build; removed MkDocs dependencies from `pyproject.toml`.

### Fixed
- Corrected the fabricated `create_study` example (README and docs) to the real
  `Optimizer` API, and the MIT/Apache-2.0 license inconsistency (Apache 2.0).
