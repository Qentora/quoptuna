# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
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

### Changed
- The report pipeline no longer sends the agents a `str()` of six metric keys. The
  analyst and reviewer prompts were rewritten around a fixed section skeleton, a
  grounding rule, and the markdown contract; conditional sections (fairness audit,
  fairness-aware search, Pareto front) are dropped when their evidence is absent
  instead of being invented.
- `report_agent.generate_report` takes an evidence bundle and returns a result dict
  (final markdown, draft, rendered evidence, referenced/dropped figures, lint findings)
  rather than a bare string; the reviewer pass can be turned off, and a reviewer that
  returns almost nothing no longer replaces a good draft.
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
