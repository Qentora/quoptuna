# r/QuantumComputing

**Rules note:** This is your core niche. Genuine, discussion-first tone. It's fine to share your own project here if it's substantive and you engage.

---

**Title:**
`I built an open-source AutoML tool that searches 17 quantum ML classifiers (PennyLane + Optuna) and benchmarks them against classical baselines`

**Body:**

Quantum ML tooling is thin, so I built **QuOptuna** to scratch my own itch during my PhD (WMU). It automates the part of quantum ML that's the most tedious: choosing a model, writing the circuit, and tuning hyperparameters — then it honestly benchmarks the result against classical models on the same data.

**The quantum side:**
- 17 quantum classifiers implemented on **PennyLane** — data-reuploading, circuit-centric, IQP & projected quantum kernels, quantum kitchen sinks, quantum metric learner, tree tensor networks, quanvolutional NNs, WeiNet, plus separable/dressed variants. Implementations extend Xanadu's `qml-benchmarks`.
- Circuits are evaluated with **JAX `vmap`** for vectorized batching, which is what makes running a full Optuna search over quantum models actually feasible on a simulator.
- **Optuna** drives the search (TPE/random/grid) with ASHA/Hyperband pruning, and each model gets its own conditional search space.

**Why it might be useful to this sub:**
- It runs quantum *and* classical models in one search, so you get an honest apples-to-apples comparison instead of hand-tuning a circuit and hoping. Often classical wins on tabular data — the tool says so rather than hiding it.
- Fairness auditing (fairlearn) and SHAP explainability are built in, which I haven't seen bundled into quantum ML tooling before.

**Try it:**
```bash
uvx quoptuna    # boots a web wizard, no install
# or
pip install quoptuna
```

Everything runs on simulators (JAX/CPU) — no hardware backend yet. Apache-2.0, Beta. Repo: https://github.com/Qentora/quoptuna

Curious what this sub thinks: which quantum classifiers would you most want added, and would a hardware backend (e.g. via PennyLane devices) be worth prioritizing over more simulator models?
