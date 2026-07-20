# r/Python

**Rules note:** r/Python favors "Showcase" posts that explain **what it does, target audience, and comparison to alternatives** (the mods often ask for this structure). Weekends get more Showcase traffic. Keep it Python-flavored — lead with the packaging/DX story.

---

**Title:**
`QuOptuna: a zero-install AutoML tool (uvx quoptuna) that tunes quantum + classical ML models with Optuna`

**Body:**

**What My Project Does**
QuOptuna is an open-source AutoML library. You point it at a dataset and it runs a single Optuna search across 21 quantum and classical classifiers, prunes weak configs early, audits the winner for fairness, explains it with SHAP, and can even draft the report. There's a 6-step web wizard, a REST API, and a headless CLI.

The Python/packaging bit I'm proud of: the Next.js frontend is **statically exported and bundled into the wheel**, so:
```bash
uvx quoptuna
```
…boots the *entire* app (UI + API on localhost:8000) with **no Node.js and no install step**. Or the classic way:
```bash
pip install quoptuna
quoptuna optimize --uci-id 267 --trials 25 --sampler tpe
```
CLI is Typer + Rich; backend is FastAPI + Pydantic v2 + SQLModel (restart-safe persistence); optimization is Optuna + PennyLane with JAX-vectorized circuit evaluation.

**Target Audience**
ML researchers, data scientists, and Python devs curious about quantum ML — anyone who wants automated model selection with fairness + explainability. It's Beta (0.1.4) and quantum models run on simulators, so it's for research/prototyping, not production.

**Comparison**
- vs. **plain Optuna:** Optuna is the engine; QuOptuna pre-wires 21 conditional search spaces (quantum circuits included), fairness constraints, SHAP, and the UI/CLI so you don't hand-roll them.
- vs. **classical AutoML** (auto-sklearn, FLAML, etc.): those don't support quantum models at all; QuOptuna searches quantum + classical together and reports an honest winner.
- vs. **raw PennyLane:** PennyLane gives you the circuits; QuOptuna gives you the automated search, tuning, and governance around them.

Apache-2.0. Repo: https://github.com/Qentora/quoptuna · PyPI: https://pypi.org/project/quoptuna/

Feedback on the CLI ergonomics and the `uvx` packaging approach very welcome.
