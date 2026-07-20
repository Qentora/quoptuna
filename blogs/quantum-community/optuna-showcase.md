# Optuna community — GitHub Discussions & "Made with Optuna" showcase

**Why:** QuOptuna is a genuinely interesting Optuna use case (conditional search spaces over quantum circuits, multi-objective fairness, ASHA/Hyperband on quantum models). The Optuna community likes seeing novel applications, and a mention/showcase link is a high-quality backlink from a large project.

**Where:**
- Optuna **GitHub Discussions** → "Show and tell" category: https://github.com/optuna/optuna/discussions
- Ask maintainers about the **"Made with Optuna"** / showcase listing (see `../academic/outreach-emails.md` §C).

---

**Title:**
`Show & tell: QuOptuna — using Optuna to search 21 quantum + classical ML models jointly`

**Body:**

Wanted to share a project that leans heavily on Optuna: **QuOptuna**, an AutoML tool that searches quantum and classical ML classifiers together in one study. It's my PhD project (WMU), Apache-2.0.

How it uses Optuna:
- **Conditional per-model search spaces** for 21 heterogeneous models — a single study can sample a quantum circuit's depth on one trial and an SVC kernel on the next.
- **Samplers:** TPE / random / grid.
- **Pruning:** ASHA and Hyperband to kill weak configs early — especially valuable because quantum circuit evaluation is expensive.
- **Multi-objective optimization** for the accuracy-vs-fairness Pareto front (fairlearn metrics), plus a constrained mode using a feasibility threshold.
- Circuit evaluation is JAX-`vmap`-vectorized so trials run at a reasonable speed on a simulator.

Try it:
```bash
uvx quoptuna    # web wizard, no install
quoptuna optimize --uci-id 267 --trials 25 --sampler tpe
```

Repo: https://github.com/Qentora/quoptuna · Docs: https://Qentora.github.io/quoptuna

Thanks for building Optuna — the pruning + multi-objective support made the quantum search practical. Would this be a fit for the showcase? And I'd welcome any suggestions on structuring conditional search spaces across such heterogeneous model families.
