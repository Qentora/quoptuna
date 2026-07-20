# r/opensource

**Rules note:** Focus on the *open-source* story — license, self-hosting, how to contribute — not a feature dump. This community rewards genuine openness.

---

**Title:**
`QuOptuna (Apache-2.0): fairness-aware AutoML for quantum + classical ML, fully self-hostable, looking for contributors`

**Body:**

I've open-sourced **QuOptuna**, an AutoML tool that searches 21 quantum and classical ML classifiers in one Optuna run, audits models for fairness, and explains them with SHAP. It's Apache-2.0 and also my PhD project (Western Michigan University).

**Why it's a good open-source citizen:**
- **License:** Apache-2.0 (permissive, patent grant included).
- **Fully self-hosted / local-first:** `uvx quoptuna` runs the whole app on your machine; nothing leaves it unless you opt into the (bring-your-own-key) LLM report feature. No SaaS, no telemetry, no account.
- **Built on open shoulders:** PennyLane, Optuna, JAX, scikit-learn, fairlearn, SHAP, FastAPI — and extends Xanadu's `qml-benchmarks`.
- **Runs anywhere Python 3.11+ runs;** the Next.js UI is bundled into the wheel so there's no Node.js dependency.

**Where I'd love help (good first issues welcome):**
- Adding new quantum classifiers / hardware backends via PennyLane devices.
- Docs, examples, and dataset connectors.
- Testing on more platforms and reporting rough edges.

```bash
uvx quoptuna            # try it, zero install
pip install quoptuna    # or install it
```

Repo: https://github.com/Qentora/quoptuna
Docs: https://Qentora.github.io/quoptuna

It's Beta (0.1.4), quantum models run on simulators. If quantum ML, AutoML, or fairness/explainability tooling is your thing, I'd genuinely welcome contributors — happy to mentor first-timers through a PR.
