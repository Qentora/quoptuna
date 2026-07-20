# JOSS — Journal of Open Source Software submission notes

**Why this is the single highest-leverage academic move.** JOSS publishes a short, **peer-reviewed, citable paper about the software itself**. For a PhD project this gives you: (1) a DOI others cite when they use QuOptuna, (2) a durable credibility badge on the README, and (3) reviewers who are exactly your target users — the review *is* promotion. It's free and open.

**Site:** https://joss.theoj.org/ · **Docs:** https://joss.readthedocs.io/

---

## Eligibility check (QuOptuna vs. JOSS criteria)

| JOSS requirement | QuOptuna status |
|---|---|
| Open-source, OSI license | ✅ Apache-2.0 |
| Research application / scholarly effort | ✅ AutoML for quantum ML; 3 IEEE papers |
| "Substantial scholarly effort" (not a thin wrapper) | ✅ 21 models, conditional search spaces, fairness-in-loop, JAX-vectorized circuits — argue this explicitly |
| Public version-controlled repo | ✅ github.com/Qentora/quoptuna |
| Documentation (install, example usage, API) | ✅ Astro/Starlight docs — ensure "Statement of need" + example are complete |
| Automated tests | ⚠️ **Verify coverage** — JOSS reviewers check for a test suite + CI. Shore this up before submitting. |
| Community guidelines (contributing, issues, support) | ⚠️ Ensure `CONTRIBUTING.md` + support/issue guidance exist |

> **Pre-submission TODO:** confirm a runnable test suite + CI badge, and a `CONTRIBUTING.md`. These are the two things reviewers most commonly block on.

---

## `paper.md` — draft (put in repo root or `paper/`)

```markdown
---
title: 'QuOptuna: Fairness-aware, explainable AutoML for quantum and classical machine learning'
tags:
  - Python
  - machine learning
  - quantum machine learning
  - AutoML
  - hyperparameter optimization
  - fairness
  - explainability
  - PennyLane
  - Optuna
authors:
  - name: Edwin Jose
    orcid: 0000-0000-0000-0000   # TODO: add your ORCID
    affiliation: 1
  - name: Alvis C. Fong
    affiliation: 1
affiliations:
  - name: Western Michigan University, USA
    index: 1
date: 19 July 2026
bibliography: paper.bib
---

# Summary

QuOptuna is an open-source AutoML framework that performs a unified
hyperparameter search across 21 quantum and classical machine-learning
classifiers. Built on PennyLane and Optuna, it evaluates 17 quantum
classifiers alongside classical baselines within a single study, prunes
unpromising configurations with ASHA and Hyperband, audits candidate
models for fairness during optimization, explains the selected model with
SHAP, and can generate a research report via a two-agent language-model
pipeline. It is usable through a web wizard, a REST API, and a CLI.

# Statement of need

Developing quantum machine-learning models typically requires manually
designing circuits, hand-tuning hyperparameters, and lacks integrated
tooling for fairness and explainability. Classical AutoML systems do not
support quantum models, and general-purpose optimizers such as Optuna
require substantial manual wiring to search heterogeneous quantum and
classical model families together. QuOptuna addresses this gap by
providing conditional per-model search spaces, JAX-vectorized circuit
evaluation, fairness-constrained and multi-objective optimization, and
built-in explainability — enabling researchers to obtain an honest,
governable comparison of quantum and classical models with a single
command.

# Acknowledgements

The quantum classifier implementations build on Xanadu's qml-benchmarks.

# References
```

## `paper.bib` — seed
Add your three IEEE references (SMC Magazine 2026 doi:10.1109/MSMC.2025.3613072, INTCEC 2025, RASSE 2025) plus Optuna, PennyLane, fairlearn, SHAP citations.

## After acceptance
- Add the JOSS DOI badge to the README.
- Add a "How to cite" section + `CITATION.cff` pointing at the JOSS paper.
- Announce the publication as its own mini-launch (LinkedIn + X + r/MachineLearning).
