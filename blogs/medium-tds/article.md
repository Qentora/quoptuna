---
title: "Fairness and Explainability Shouldn't Be Afterthoughts — Even in Quantum Machine Learning"
subtitle: "How QuOptuna builds fairness auditing and SHAP explanations directly into an AutoML search over 21 quantum and classical models"
target_publication: "Towards Data Science (submit via their portal) — or your own Medium profile"
tags: [Machine Learning, AutoML, Responsible AI, Quantum Computing, Open Source]
canonical_url: https://github.com/Qentora/quoptuna
---

*This is my flagship data-science-audience piece. Angle: responsible ML (fairness + explainability), with quantum AutoML as the vehicle. TDS accepts stronger, less promotional writing than dev.to — so this version leads with the idea, not the tool.*

---

Most machine-learning pipelines treat fairness and explainability as things you check **after** the model is chosen. You optimize for accuracy, pick a winner, and then — maybe, if there's time — you run a disparity metric and a SHAP plot to see whether you should be worried. By then the model is already selected. The fairness check can only tell you that you have a problem; it can't help you avoid one.

I think that ordering is backwards, and I built an open-source tool to explore the alternative: **make fairness part of the search itself.** The tool is [QuOptuna](https://github.com/Qentora/quoptuna), an AutoML system that happens to search both **quantum and classical** models — but the responsible-ML idea in it applies to any AutoML pipeline.

## Fairness after the fact vs. fairness in the loop

Consider a standard AutoML run. It samples hundreds of hyperparameter configurations, scores each on validation accuracy, and returns the best. Fairness never enters the objective, so the search has no reason to prefer an equitable model over a marginally more accurate but more biased one.

QuOptuna offers two ways to change that:

- **Constrained optimization.** You set a disparity feasibility threshold (say, a demographic-parity difference below 0.1). Trials that violate it are penalized, so the search *only advances* models that clear the fairness bar. Accuracy is optimized within the feasible region.
- **Multi-objective optimization.** Instead of collapsing to one number, the search jointly optimizes accuracy and a fairness metric and returns the **Pareto front** — the set of models where you can't improve fairness without sacrificing accuracy and vice versa. You, the human, pick the trade-off, with the options laid out explicitly.

Fairness metrics come from [fairlearn](https://fairlearn.org/): demographic parity, equalized odds, and equal opportunity. The point is that "fair" is no longer a report you read at the end — it's a shape the search is forced to respect.

## Explainability as a first-class output, not a debugging afterthought

For whichever model wins, QuOptuna automatically produces **SHAP** explanations — bar, beeswarm, violin, heatmap, and waterfall plots — alongside ROC/PR curves and confusion matrices. The winner arrives already explained. If you've ever selected a model and *then* scrambled to justify it to a stakeholder, you know why bundling this matters.

*(You can embed real SHAP plots from the repo here — e.g. the beeswarm and waterfall in `experiments/basic_dataset_test/blood/`.)*

## Where the quantum part comes in

The reason QuOptuna searches 21 models — **17 quantum classifiers plus 4 classical baselines** — is that quantum ML badly needs this discipline. Quantum models are usually hand-coded circuits with hand-guessed hyperparameters and no governance story at all. By putting quantum and classical models in the *same* fairness-aware, explainable search, you get an honest comparison: often the classical model wins on tabular data, and the tool says so plainly rather than overselling the quantum result.

Under the hood it's [Optuna](https://optuna.org/) (TPE/random/grid, with ASHA/Hyperband pruning) driving a search over [PennyLane](https://pennylane.ai/) circuits vectorized with JAX — but the responsible-ML framing is the part I'd argue generalizes far beyond quantum.

## Try the idea yourself

```bash
uvx quoptuna    # full web app, zero install
# or
pip install quoptuna
```

QuOptuna is Apache-2.0 and still Beta; quantum models run on simulators. It's my PhD project (Western Michigan University), written up in three IEEE papers. If the "fairness in the loop" framing resonates, the repo is [github.com/Qentora/quoptuna](https://github.com/Qentora/quoptuna) — I'd love your thoughts, and a ⭐ helps others find it.

---

**Closing CTA for Medium:** end with a clear ask for claps + comments on how *your* team handles fairness in model selection, and the repo link. Keep the promotional tone lighter here than on dev.to.
