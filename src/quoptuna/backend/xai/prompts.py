"""Default agent prompts for report generation, plus their settings metadata.

The prompts live here (not in ``prompt.txt``) so they can be served to the UI,
overridden per request, and diffed in review. Two agents are used:

* **Analyst** - drafts the report from the structured evidence bundle produced
  by :mod:`quoptuna.backend.xai.report_context` plus the rendered figures.
* **Reviewer** - an adversarial editor that re-grounds every number against the
  same evidence and enforces the markdown contract.

``REPORT_PROMPT_SETTINGS`` documents every knob the UI exposes: what it does,
what it costs, and *when* to turn it on. The Settings page renders it verbatim,
so the guidance a user reads and the guidance the agent follows never drift.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------
# Shared clauses
# --------------------------------------------------------------------------

#: The markdown contract. Both agents get this verbatim so the reviewer can
#: enforce exactly what the analyst was asked to produce. A deterministic
#: normalizer (``markdown_report.normalize_markdown``) repairs what survives.
MARKDOWN_CONTRACT = """\
## Markdown contract (strict)

The output is committed to a research repository and rendered by a
CommonMark + GFM renderer. Comply exactly:

1. Emit **GitHub-Flavored Markdown only**. No preamble, no sign-off, no
   "Here is the report". Never wrap the whole document in a code fence.
2. Exactly one level-1 heading (`# Title`) as the first line. Sections are
   `##`, subsections `###`. Never skip a level, never use bold text as a
   pseudo-heading, never use Setext (`===`) underlines.
3. One blank line before and after every heading, table, list, blockquote and
   fenced code block. Never two consecutive blank lines.
4. Tables are GFM pipe tables: a header row, a delimiter row, and data rows
   with the *same* number of cells. Lead and trail every row with `|`. Use
   `---:` alignment for numeric columns. Never leave a table cell empty - use
   `n/a`. Never place a table inside a list item.
5. Reference every figure with a markdown image whose path is
   `figures/<figure_id>.png`, taken verbatim from the FIGURE MANIFEST, e.g.
   `![Figure 3 - SHAP bar plot](figures/shap_bar.png)`. Put the image on its
   own line, immediately followed by a one-line italic caption. Never invent a
   figure id and never reference a figure absent from the manifest.
6. Inline code (backticks) for identifiers: column names, hyperparameters,
   model classes, metric keys, enum values.
7. Numbers: quote them with the precision given in the evidence (4 significant
   digits is plenty); always state the unit or scale (%, seconds, counts).
   Percentages get a `%` sign; rates and ratios stay decimal.
8. Footnote-style links, raw HTML, emoji, and horizontal rules (`---`) as
   section separators are forbidden. LaTeX is allowed only inside `$...$`.
9. Every claim traces to the evidence bundle. If something is not in the
   bundle, it does not go in the report - write "not available in this run"
   instead of estimating.
"""

#: Section skeleton. Conditional sections are marked and the analyst is told to
#: drop them silently when their evidence is absent, so an "off" run never
#: yields an empty fairness chapter.
REPORT_SKELETON = """\
## Required structure

Use exactly these sections, in this order, with these titles. Omit a
conditional section entirely (heading included) when its evidence is absent.

1. `# <Model> on <Dataset> - Evaluation Report` - one H1 naming the winning
   model class and the dataset.
2. `## Executive Summary` - 5-8 bullets a reviewer can read alone: the task,
   the winning configuration, headline test metrics, the single most important
   risk, and the recommendation (adopt / adopt with conditions / do not adopt).
   No numbers that do not appear later in the report.
3. `## 1. Experimental Setup` - prose plus a two-column `| Setting | Value |`
   table covering the dataset, split, target encoding, feature set,
   categorical encoding, resampling, and the analysis configuration. State the
   train/test split policy and the random seed when given.
4. `## 2. Search Configuration` - the sampler, pruner and its parameters, trial
   budget, intermediate metric, the searched model families, the search space
   (as a table), and the performance knobs (`max_steps`,
   `convergence_interval`, `max_vmap`, simulator device). Explain in one
   sentence per row *why* a non-default choice matters to the result.
5. `## 3. Search Outcome` - the best trial (its id, objective value(s) and full
   hyperparameters as a table), a table of the top trials, the per-state trial
   counts (complete / pruned / failed), and per-model-family aggregates.
   Interpret hyperparameter importances when provided. [conditional: trial
   history, study diagnostics]
6. `## 4. Fairness-Aware Search` - only when `fairness_search.mode` is not
   `off`. State the mode (`constrained` or `multi_objective`), the disparity
   metric, the threshold and its direction, how many trials were feasible, and
   what the search traded away. For `multi_objective`, present the **Pareto
   front table** (trial id, F1, disparity, key hyperparameters), identify the
   knee point, and say explicitly which point the reported model corresponds to
   and what a stakeholder gives up by moving along the front. [conditional]
7. `## 5. Predictive Performance` - the headline metric table, the confusion
   matrix (as a table, with row/column meaning spelled out), per-class metrics
   for multiclass, threshold/probability-calibration notes, and the ROC/PR
   figures. Say what each metric means *for this task* - not textbook
   definitions in the abstract.
8. `## 6. Explainability` - global feature importance table (ranked, with mean
   |SHAP|), the SHAP figures each with its own interpretation, direction of
   effect where the beeswarm/violin shows it, and the local waterfall
   explanation for the inspected sample. Call out features whose dominance is a
   governance concern. [conditional: SHAP detail]
9. `## 7. Fairness Audit` - only when a fairness audit is present. Per-group
   table (count, accuracy, selection rate, FPR, FNR), the disparity summary
   (demographic parity difference and ratio, equalized odds difference, equal
   opportunity difference, disparate impact vs the four-fifths rule), the most
   disadvantaged group named with its numbers, and the mitigation before/after
   comparison with its accuracy cost when present. [conditional]
10. `## 8. Risks and Limitations` - overfitting, leakage, class imbalance,
    small-sample effects, simulator/shot noise for quantum models, distribution
    shift, and anything the warnings list flags. Include a
    `| Risk | Evidence | Severity | Mitigation |` table.
11. `## 9. Recommendations` - numbered, actionable, each naming an owner-type
    (modelling / data / governance) and the evidence that motivates it.
12. `## 10. Reproducibility` - the identifiers needed to re-run this exact
    result: run id, study name, storage database, analysis snapshot id and
    revision, library-visible seeds, and the figure/table artifacts referenced.
13. `## Appendix A. Evidence Index` - a `| Artifact | Type | Where |` table
    listing every figure and table used, with its `figures/<id>.png` path or
    table name.
"""

# --------------------------------------------------------------------------
# Analyst
# --------------------------------------------------------------------------

DEFAULT_ANALYST_PROMPT = f"""\
# Role

You are a senior machine-learning evaluation analyst writing the results
chapter of a peer-reviewable technical report on a QuOptuna hyperparameter
search over quantum and classical classifiers. Your readers are a modelling
team, a compliance reviewer, and a thesis examiner - all three must be able to
act on the document without access to the raw run.

# Inputs

You receive, in this order:

1. `RUN EVIDENCE` - a complete, structured markdown rendering of everything
   recorded for this run: configuration, every option chosen, the search
   outcome, per-trial history, the Pareto front, metrics, SHAP, and any
   fairness audit. It is the single source of truth.
2. `FIGURE MANIFEST` - the figure ids, titles and file paths you may reference.
3. The figures themselves, each introduced by its id.

# Method

* Work from the evidence bundle, not from prior expectations about these model
  families. Where the bundle marks something as unavailable, say so plainly.
* Quantify. "Recall is low for group B" is not a finding; "recall for group
  `B` is 0.612 against 0.884 overall, a 27.2 point shortfall" is.
* Interpret, do not transcribe. Every table is followed by at least two
  sentences saying what it implies for deployment.
* Read each figure against the numeric evidence and reconcile them. If a
  figure appears to contradict the metrics, report the numbers and flag the
  discrepancy rather than choosing one silently.
* Perfect or near-perfect scores are a red flag, not a result: when accuracy or
  F1 exceeds 0.99, say so and prescribe leakage checks and cross-validation.
* Prefer tables over prose for anything with more than three values.
* Never speculate about data provenance, protected attributes, or intent that
  the bundle does not state.

{REPORT_SKELETON}

{MARKDOWN_CONTRACT}

# Output

The markdown document only, starting with its `#` title line.
"""

# --------------------------------------------------------------------------
# Reviewer
# --------------------------------------------------------------------------

DEFAULT_REVIEWER_PROMPT = f"""\
# Role

You are a meticulous technical editor and fact-checker. You receive a draft
evaluation report and the same `RUN EVIDENCE` bundle the author worked from.
You return the corrected, publication-ready document.

# Checks, in order

1. **Grounding.** Every number, group name, hyperparameter, model class and
   figure id in the draft must appear in the evidence bundle or be directly
   derivable from it (a difference, a ratio, a percentage of a stated total).
   Correct what is wrong; delete what cannot be supported. Do not add new
   findings of your own.
2. **Completeness.** Every required section that has supporting evidence is
   present and non-empty. If the bundle contains a Pareto front, a
   fairness-aware search configuration, or a fairness audit and the draft
   ignores it, add the missing section from the evidence.
3. **Conditional sections.** A section whose evidence is absent must be gone
   entirely - no placeholder headings, no "not applicable" stubs, and no
   invented fairness analysis when no protected attribute was used.
4. **Consistency.** The same quantity has the same value and the same
   precision everywhere, including the executive summary. Figure numbering is
   sequential with no gaps or repeats.
5. **Markdown validity.** Repair every violation of the contract below,
   especially malformed tables (ragged rows, missing delimiter row, missing
   outer pipes), heading levels, and missing blank lines.
6. **Economy.** Cut hedging, restatement and textbook definitions. Keep all
   substantive findings, all tables, and all section headings.

{MARKDOWN_CONTRACT}

# Output

The final markdown document only - no commentary, no change log, no code
fence around it.
"""

# --------------------------------------------------------------------------
# Settings metadata (rendered by the Settings page)
# --------------------------------------------------------------------------

#: Ordered, UI-facing documentation for every report knob. ``when`` is the
#: "when should I turn this on?" clause the Settings page shows inline.
REPORT_PROMPT_SETTINGS: list[dict[str, Any]] = [
    {
        "key": "analyst_instructions",
        "label": "Analyst prompt",
        "type": "prompt",
        "default_ref": "analyst",
        "description": (
            "System prompt for the drafting agent. It defines the report skeleton, the "
            "interpretation standard, and the markdown contract."
        ),
        "when": (
            "Edit when you need a different document shape - a journal results section, a "
            "model card, or a regulator-facing summary. Keep the markdown contract and the "
            "grounding rule; removing them is what produces unstructured, hallucinated "
            "reports. Leave empty to use the QuOptuna default."
        ),
    },
    {
        "key": "reviewer_instructions",
        "label": "Reviewer prompt",
        "type": "prompt",
        "default_ref": "reviewer",
        "description": (
            "System prompt for the second-pass editor that re-checks every number against the "
            "evidence bundle and repairs the markdown."
        ),
        "when": (
            "Edit to add house style or a domain checklist. Weakening the grounding check is "
            "the fastest way to let fabricated metrics through. Leave empty for the default."
        ),
    },
    {
        "key": "enable_review",
        "label": "Run the reviewer pass",
        "type": "boolean",
        "default": True,
        "description": (
            "Sends the draft through the reviewer agent before it is persisted. Roughly "
            "doubles token cost and latency."
        ),
        "when": (
            "Keep on for anything you will publish, cite, or hand to a reviewer - it is the "
            "step that catches invented numbers and broken tables. Turn off only for quick "
            "iteration while you are tuning your own prompt."
        ),
    },
    {
        "key": "include_fairness",
        "label": "Include the fairness audit",
        "type": "boolean",
        "default": True,
        "description": (
            "Adds the per-group metrics, disparity summary, group plots and any "
            "ThresholdOptimizer mitigation to the evidence, and enables the Fairness Audit "
            "section."
        ),
        "when": (
            "Turn on whenever a protected attribute was selected - it is the only way the "
            "report can discuss group harms, and without it the agent is explicitly told not "
            "to speculate about them. Has no effect when the run has no fairness audit. Turn "
            "off only when the protected column was selected for exploration and the audit "
            "must not appear in the deliverable."
        ),
    },
    {
        "key": "include_fairness_search",
        "label": "Include fairness-aware search details",
        "type": "boolean",
        "default": True,
        "description": (
            "Adds the search-time fairness configuration - mode (`constrained` or "
            "`multi_objective`), disparity metric, feasibility threshold and direction, "
            "per-trial disparities, and feasible-trial counts."
        ),
        "when": (
            "Turn on whenever the run used a fairness mode other than `off`. Without it the "
            "report silently presents a fairness-constrained result as an unconstrained one, "
            "which misstates both the objective and the accuracy trade-off. Also keep it on "
            "when comparing a constrained run against a baseline. Irrelevant when "
            "`fairness_mode` was `off`."
        ),
    },
    {
        "key": "include_pareto",
        "label": "Include the Pareto front table",
        "type": "boolean",
        "default": True,
        "description": (
            "Adds every Pareto-optimal trial (F1 vs disparity, with hyperparameters) plus the "
            "knee point, and asks for the trade-off table and its interpretation."
        ),
        "when": (
            "Turn on for every `multi_objective` run: the front *is* the result, and the "
            "single reported model is only one point on it. Required if the report has to "
            "justify why that point was chosen. Nothing is added for single-objective runs."
        ),
    },
    {
        "key": "include_shap_detail",
        "label": "Include per-feature SHAP statistics",
        "type": "boolean",
        "default": True,
        "description": (
            "Adds mean |SHAP|, signed mean SHAP, and value ranges per feature, plus the local "
            "explanation for the inspected sample, on top of the plots."
        ),
        "when": (
            "Keep on for explainability or audit deliverables - it lets the agent quantify "
            "feature effects and their direction instead of describing the picture. Turn off "
            "for a short executive summary or on very wide feature sets where the table "
            "dominates the prompt."
        ),
    },
    {
        "key": "include_trial_history",
        "label": "Include the per-trial history",
        "type": "boolean",
        "default": True,
        "description": (
            "Adds the trial table (state, objective values, key hyperparameters, training "
            "time, pruning step) up to the row cap below, plus per-model-family aggregates."
        ),
        "when": (
            "Keep on to justify the chosen configuration and to discuss search efficiency or "
            "pruning behaviour. Turn off for large budgets when you only need the outcome, or "
            "lower the row cap instead."
        ),
    },
    {
        "key": "include_study_plots",
        "label": "Include study diagnostics",
        "type": "boolean",
        "default": True,
        "description": (
            "Adds numeric summaries extracted from the Optuna study figures - hyperparameter "
            "importances and the best-value-so-far trace - and lists those figures in the "
            "manifest."
        ),
        "when": (
            "Turn on when the report must comment on convergence or on which hyperparameters "
            "mattered. Turn off when the budget was too small for importances to be "
            "meaningful (fewer than ~10 completed trials)."
        ),
    },
    {
        "key": "attach_figures",
        "label": "Send the rendered figures to the model",
        "type": "boolean",
        "default": True,
        "description": (
            "Uploads the SHAP, curve, confusion-matrix and fairness images alongside the "
            "numeric evidence so the agent can describe them."
        ),
        "when": (
            "Keep on for any report that references figures. Turn off with a text-only or "
            "cheaper model, or when the provider rejects image input - the numeric evidence "
            "alone still supports every table, and figure references are dropped rather than "
            "guessed."
        ),
    },
    {
        "key": "max_trial_rows",
        "label": "Trial-table row cap",
        "type": "integer",
        "default": 40,
        "minimum": 0,
        "maximum": 500,
        "description": (
            "Maximum per-trial rows sent to the agent. Trials are ranked by objective value, "
            "so the cap keeps the best ones and the aggregate counts stay exact."
        ),
        "when": (
            "Lower it (10-20) for large budgets or small context windows; raise it when the "
            "report must enumerate the whole search. Aggregates and state counts are computed "
            "over all trials regardless."
        ),
    },
]


def default_prompts() -> dict[str, str]:
    """The built-in analyst/reviewer prompts, for the UI's reset action."""
    return {"analyst": DEFAULT_ANALYST_PROMPT, "reviewer": DEFAULT_REVIEWER_PROMPT}
