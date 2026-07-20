---
title: "I built AutoML for quantum machine learning — here's the architecture"
published: false
description: "QuOptuna searches 21 quantum + classical classifiers in one Optuna run, with fairness constraints and SHAP explainability built in. Here's how it works — and how to run it with zero install."
tags: opensource, machinelearning, python, quantum
cover_image: https://raw.githubusercontent.com/Qentora/quoptuna/main/assets/branding/social-preview.png
canonical_url: https://github.com/Qentora/quoptuna
---

> **TL;DR** — `uvx quoptuna` boots a full AutoML web app with no install. It runs one hyperparameter search across **21 quantum and classical classifiers**, prunes weak configs early, audits every model for **fairness**, explains the winner with **SHAP**, and can draft the report for you. Apache-2.0. Repo: [github.com/Qentora/quoptuna](https://github.com/Qentora/quoptuna). Stars and feedback very welcome. ⭐

## The problem

Training a good **quantum** machine-learning model today is painful in three specific ways:

1. **You hand-write circuits.** Every quantum classifier is a different ansatz you code by hand.
2. **You guess hyperparameters.** There's no `GridSearchCV` habit for quantum models the way there is for classical ones.
3. **You can't tell if the result is trustworthy.** No fairness audit, no explanation, no easy comparison to a classical baseline.

Classical AutoML tools (auto-sklearn, FLAML, …) don't touch quantum models. Plain [Optuna](https://optuna.org/) is a great optimizer but makes you wire up every model yourself. I wanted one tool that searches **quantum and classical models together**, tells me honestly which won, and hands me a governable result. That became **QuOptuna** — which is also my PhD project at Western Michigan University.

## What it does, in 30 seconds

```bash
uvx quoptuna
```

That single command boots the whole app on `http://localhost:8000` — a 6-step wizard (**Dataset → Features → Configure → Optimize → Analyze → Report**), a REST API, and live trial monitoring. No Node.js, no install. (More on *why* there's no Node.js below — that's a fun packaging trick.)

Prefer the terminal?

```bash
pip install quoptuna
quoptuna optimize --uci-id 267 --trials 25 --sampler tpe
```

Or the Python API:

```python
from quoptuna import DataPreparation, Optimizer
```

## The architecture

### 1. One search space over 21 heterogeneous models

QuOptuna registers **17 quantum classifiers** (data-reuploading, circuit-centric, IQP & projected quantum kernels, quantum kitchen sinks, quantum metric learner, tree tensor networks, quanvolutional NNs, WeiNet, plus separable/dressed variants) and **4 classical baselines** (SVC, LinearSVC, MLP, Perceptron), with one-vs-rest multiclass. The quantum implementations extend Xanadu's [`qml-benchmarks`](https://github.com/XanaduAI/qml-benchmarks).

The hard part is that each model has a *different* hyperparameter shape. QuOptuna gives every model its own **conditional Optuna search space**, so a single `study` can sample a circuit depth for one trial and an SVC kernel for the next.

### 2. Making a quantum search actually tractable

Evaluating quantum circuits in a loop is slow. QuOptuna leans on:

- **[PennyLane](https://pennylane.ai/)** for the circuits,
- **JAX `vmap`** to vectorize circuit evaluation across a batch, and
- **Optuna pruning** — ASHA and Hyperband — to kill hopeless configs after a few epochs instead of training them to completion.

TPE / random / grid samplers are all available.

### 3. Fairness *inside* the search loop

This is the part I care about most. Fairness usually gets measured *after* you've picked a model — too late. QuOptuna makes it part of optimization, two ways:

- **Constrained mode:** trials whose disparity exceeds a feasibility threshold are penalized, so the search only keeps fair-enough models.
- **Multi-objective mode:** it jointly optimizes accuracy and a fairness metric and returns the **Pareto front**, so you choose your own trade-off.

Metrics come from [fairlearn](https://fairlearn.org/): demographic parity, equalized odds, equal opportunity.

### 4. Explainability + auto-generated reports

For the winning model you get **SHAP** plots (bar, beeswarm, violin, heatmap, waterfall), ROC/PR curves, and confusion matrices. Optionally, a **two-agent LLM pipeline** (an analyst that drafts and a reviewer that critiques) writes a research report — bring your own OpenAI/Anthropic/Gemini key; skip it and you still get every plot and metric.

### 5. The no-Node.js trick

The frontend is **Next.js, statically exported and bundled into the Python wheel**. So `pip install quoptuna` ships the compiled UI, and `uvx quoptuna` serves it from FastAPI on a single port. Users never install Node. This is the detail most people don't expect from a "quantum ML" package.

**Stack:** Python 3.11–3.12 · PennyLane · Optuna · JAX/Flax/Optax · scikit-learn · SHAP · fairlearn · FastAPI + Pydantic v2 + SQLModel · Next.js · Typer + Rich.

## Honest limitations

- It's **Beta (0.1.4)**.
- Quantum models run on **simulators (JAX/CPU)** — no hardware backend yet. This is a research/prototyping tool.
- On most tabular datasets, classical models still win — and QuOptuna will tell you so. I think that honesty is a feature.

## Try it and tell me what breaks

```bash
uvx quoptuna
```

- **Repo:** https://github.com/Qentora/quoptuna
- **Docs:** https://Qentora.github.io/quoptuna
- **PyPI:** https://pypi.org/project/quoptuna/

The approach is written up in three IEEE papers (including *IEEE Systems, Man & Cybernetics Magazine*, 2026) if you want the theory.

If you find this interesting, a ⭐ on the repo genuinely helps a solo PhD project — and I'd love feedback on the conditional search-space design and the fairness-constrained optimization. Drop a comment with what you'd want added.

`#opensource` `#machinelearning` `#python` `#quantum`
