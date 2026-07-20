# arXiv — software/tool note

**Why:** arXiv is where your exact audience (quantum ML + AutoML researchers) discovers new work. A short tool/software note complements the IEEE papers and gives you a stable, citable URL to seed into Papers with Code, awesome-lists, and Google Scholar.

**Category:** primary `quant-ph` (Quantum Physics) or `cs.LG` (Machine Learning); cross-list the other, plus `cs.CY` (fairness/ethics angle). **Requires endorsement** if you haven't posted to that category before — ask your advisor (Alvis C. Fong) or a co-author to endorse.

**License on arXiv:** pick CC BY 4.0 so it's reusable.

---

## Title
**QuOptuna: Fairness-Aware, Explainable AutoML for Quantum and Classical Machine Learning**

## Authors
Edwin Jose, Alvis C. Fong, et al. — Western Michigan University

## Abstract (draft, ~180 words)

Developing quantum machine-learning (QML) models typically requires manual circuit design, ad-hoc hyperparameter tuning, and offers little support for the fairness auditing and explainability increasingly expected of trustworthy machine learning. We present **QuOptuna**, an open-source AutoML framework that performs a unified hyperparameter search over 21 quantum and classical classifiers within a single optimization study. QuOptuna implements 17 quantum classifiers on PennyLane — including data-reuploading, circuit-centric, quantum-kernel, quantum-kitchen-sink, tree-tensor-network, and quanvolutional models — alongside classical baselines, and searches them jointly using Optuna with TPE sampling and ASHA/Hyperband pruning. Circuit evaluation is vectorized with JAX to make the search tractable. Distinctively, QuOptuna incorporates fairness directly into optimization, supporting both constrained (feasibility-threshold) and multi-objective (accuracy–fairness Pareto) formulations via standard group-fairness metrics, and produces SHAP-based explanations and automated reports for the selected model. The framework is accessible through a web wizard, a REST API, and a command-line interface, and installs with zero configuration. We describe the architecture and design decisions, and discuss QuOptuna as reproducible infrastructure for honest quantum-versus-classical model comparison under fairness and explainability constraints.

## Links to include in the paper
- Code: https://github.com/Qentora/quoptuna
- Docs: https://Qentora.github.io/quoptuna
- Package: https://pypi.org/project/quoptuna/

## After posting
- Register the arXiv ID on **Papers with Code**, linking the repo and tagging tasks: *AutoML*, *Hyperparameter Optimization*, *Quantum Machine Learning*, *Fairness*.
- Add the arXiv badge to the README.
