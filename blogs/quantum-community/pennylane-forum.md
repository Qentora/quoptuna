# PennyLane forum + Xanadu Discord

**Why this converts:** QuOptuna is built on PennyLane and extends Xanadu's `qml-benchmarks`. This is the most on-target audience you have — people already writing quantum circuits who feel the exact pain QuOptuna solves. Lead with respect for the ecosystem, not a sales pitch.

**Forum:** https://discuss.pennylane.ai/ (post under "Projects / Show & Tell" or the general category)
**Discord:** Xanadu/PennyLane community server — share in the projects/showcase channel.

---

**Title:**
`Built an AutoML tool on PennyLane: automated search + tuning over 17 quantum classifiers (with fairness & SHAP)`

**Body:**

Hi everyone 👋 — I've been building **QuOptuna** on top of PennyLane as part of my PhD (Western Michigan University), and I'd love this community's feedback since you're the people who'd actually use it.

It automates the tedious part of quantum ML: instead of hand-writing a circuit and guessing hyperparameters, you point it at a dataset and it runs an Optuna search across **17 quantum classifiers** (data-reuploading, circuit-centric, IQP & projected quantum kernels, quantum kitchen sinks, quantum metric learner, tree tensor networks, quanvolutional NNs, WeiNet, separable/dressed variants) plus classical baselines, and returns the best model — honestly, so classical often wins and it says so.

PennyLane-specific notes:
- The quantum classifier implementations extend Xanadu's `qml-benchmarks` (Apache-2.0) — huge thanks for that foundation.
- Circuits are evaluated with **JAX `vmap`** to vectorize across the batch, which is what makes a full hyperparameter search over quantum models tractable on a simulator.
- Fairness is optimized *in the loop* (constraint or Pareto), and SHAP explanations + reports come out for the winner.

Try it:
```bash
uvx quoptuna    # full web wizard, no install
pip install quoptuna
```

It's Apache-2.0, Beta, simulator-only for now. Repo: https://github.com/Qentora/quoptuna

Two questions for the community:
1. Which additional quantum classifiers or ansätze would you most want to see supported?
2. For a first hardware backend, which PennyLane device(s) would be most valuable to prioritize?

Genuinely grateful for any feedback — and if it's useful to your work, a ⭐ helps others find it.
