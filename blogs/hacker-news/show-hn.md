# Hacker News — Show HN

> ⚠️ **Read [`README.md`](README.md) in this folder BEFORE posting.** The first launch (20 Jul 2026) was `[flagged]` — not because of the copy, but because it was posted from a brand-new account (karma 1) with zero history. Warm up the account first, fix the URL typo below, and email the mods to recover a flagged post. Details in the README.

## Title (pick one — ≤80 chars, no hype words)

- **`Show HN: QuOptuna – AutoML that searches quantum and classical ML models together`**
- Alt: `Show HN: QuOptuna – Fairness-aware AutoML for quantum + classical ML`
- Alt: `Show HN: QuOptuna – One Optuna search over 21 quantum and classical classifiers`

## URL field
Leave the URL field on the **repo**: `https://github.com/Qentora/quoptuna`
(If a hosted demo exists by launch, use that instead — HN loves a clickable demo.)

## Text body (first comment posts automatically as the top comment — put the story here)

Hi HN — I'm Edwin. QuOptuna is an open-source AutoML tool I built (it's also my PhD work at Western Michigan University) that does something I couldn't find anywhere else: it runs **one hyperparameter search across both quantum and classical ML classifiers** and picks the best one for your dataset.

The problem it started from: training a quantum ML model today means hand-writing circuits, guessing hyperparameters, and having no real way to tell if the result is trustworthy. Classical AutoML tools (auto-sklearn, etc.) don't touch quantum models; plain Optuna makes you wire every model up by hand. QuOptuna sits on top of Optuna + PennyLane and automates the whole thing.

What it does:

- **21 models, one search** — 17 quantum classifiers (data-reuploading, circuit-centric, IQP/projected quantum kernels, quantum kitchen sinks, tree tensor networks, quanvolutional nets, and more) plus classical baselines (SVC, MLP, Perceptron). Optuna TPE/random/grid sampling, ASHA/Hyperband pruning, circuits vectorized with JAX `vmap`.
- **Fairness in the search loop** — not measured after the fact. You can run it as a hard constraint (disparity threshold) or as a multi-objective Pareto front (accuracy vs. fairness) using demographic parity / equalized odds via fairlearn.
- **Explainability built in** — SHAP plots, ROC/PR curves, confusion matrices for the winning model.
- **It writes the report** — a two-agent (analyst + reviewer) LLM pipeline drafts a research write-up (bring your own OpenAI/Anthropic/Gemini key; optional).
- **Zero install** — `uvx quoptuna` boots a 6-step web wizard on localhost:8000. The Next.js frontend is statically bundled into the Python wheel, so there's no Node.js at runtime. There's also a REST API and a headless CLI.

Try it:

    uvx quoptuna                 # full web app, no install
    # or
    pip install quoptuna
    quoptuna optimize --uci-id 267 --trials 25 --sampler tpe

Stack: Python 3.11–3.12, PennyLane, Optuna, JAX/Flax, scikit-learn, SHAP, fairlearn, FastAPI, Next.js. Apache-2.0.

Honest caveats: it's Beta (0.1.4). Quantum models run on simulators (CPU/JAX), so this is for research and prototyping, not production quantum hardware. The quantum classifier implementations build on Xanadu's `qml-benchmarks`.

The approach is written up in three IEEE papers if you want the theory. I'd genuinely love feedback on the search-space design and the fairness-constrained optimization — those were the hardest parts. Repo: https://github.com/Qentora/quoptuna

## First-comment reply drafts (paste as needed — reply FAST)

**"Why quantum ML? Does it actually beat classical?"**
> Fair question. For most tabular datasets today, classical models win or tie — and QuOptuna will tell you that honestly, because it searches both and reports the winner. The value now is (a) making quantum models trivial to *evaluate* against classical baselines instead of hand-coding them, and (b) building the tooling/benchmarks so that when quantum advantage does show up on specific data, you already have the pipeline. I'd rather the tool tell you "classical won" than hide it.

**"How is this different from just using Optuna?"**
> Optuna is the optimizer underneath. QuOptuna adds the parts you'd otherwise write yourself: conditional per-model search spaces for 21 models (quantum circuits included), vectorized circuit evaluation, fairness constraints as part of the objective, SHAP/report generation, and the web wizard + CLI. Think of it as "Optuna, pre-wired for quantum+classical classification with governance built in."

**"Fairness in the loop — how does that work mechanically?"**
> Two modes. Constrained: Optuna trials that exceed your disparity threshold (e.g. demographic-parity difference > 0.1) are penalized/pruned, so the search only keeps feasible models. Multi-objective: it optimizes accuracy and the fairness metric jointly and returns the Pareto front, so you pick your own trade-off. Metrics come from fairlearn (demographic parity, equalized odds, equal opportunity).

**"Does it need API keys / send my data anywhere?"**
> No, by default nothing leaves your machine — training and search run locally. The only optional external call is the LLM report writer, and only if *you* add an OpenAI/Anthropic/Gemini key. Skip it and you still get all the plots and metrics.

**"Show HN etiquette / who are you?"**
> I'm the maintainer — this is my PhD project, first public launch. Reading every comment today, so fire away.
