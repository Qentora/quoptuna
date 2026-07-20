# r/MachineLearning

**Flair:** `[Project]` (required — posts without it get removed)
**Rules note:** r/ML is strict. No hype, no emoji-spam, no "please star." Lead with what it does and what's technically interesting. Be ready to discuss method in comments.

---

**Title:**
`[Project] QuOptuna: one Optuna search over 21 quantum + classical classifiers, with fairness constraints and SHAP built in`

**Body:**

I've been building **QuOptuna**, an open-source AutoML tool that runs a single hyperparameter search across both quantum and classical ML classifiers and returns the best one for your dataset. It's also my PhD project (Western Michigan University), so I'm especially interested in method-level feedback.

**What's technically interesting:**

- **Unified search space over 21 models** — 17 quantum classifiers (data-reuploading, circuit-centric, IQP & projected quantum kernels, quantum kitchen sinks, quantum metric learner, tree tensor networks, quanvolutional NNs, WeiNet, separable/dressed variants) + classical baselines (SVC, LinearSVC, MLP, Perceptron), with one-vs-rest multiclass. Each model has its own conditional Optuna search space.
- **Optimization:** Optuna TPE/random/grid samplers, ASHA & Hyperband pruning to kill weak configs early. Quantum circuits are evaluated with JAX `vmap` for vectorized batching (the main thing that makes searching quantum models tractable).
- **Fairness inside the objective**, not post-hoc: either a hard constraint (disparity feasibility threshold) or multi-objective, returning an accuracy–fairness Pareto front. Metrics via fairlearn (demographic parity, equalized odds, equal opportunity).
- **Explainability:** SHAP (bar/beeswarm/violin/heatmap/waterfall), ROC/PR, confusion matrices for the selected model.
- Optional two-agent LLM pipeline (analyst + reviewer) that drafts a report.

Built on PennyLane + Optuna; quantum model implementations extend Xanadu's `qml-benchmarks`.

**Try it:**
```bash
uvx quoptuna    # full web wizard, no install
# or
pip install quoptuna
quoptuna optimize --uci-id 267 --trials 25 --sampler tpe
```

Apache-2.0, Beta (0.1.4). Quantum models run on simulators (JAX/CPU) — this is a research/prototyping tool, not production quantum hardware. Written up in three IEEE papers if you want the details.

Repo: https://github.com/Qentora/quoptuna

I'd love feedback on two things specifically: (1) the conditional search-space design across such heterogeneous models, and (2) whether the constrained-vs-multi-objective fairness framing matches how people actually want to use it. Happy to answer method questions.
