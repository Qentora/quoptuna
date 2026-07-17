<div align="center">

<img src="https://raw.githubusercontent.com/Qentora/quoptuna/main/assets/branding/logo-full.png" alt="QuOptuna — Quantum ML Optimization" width="380" />

**Fairness-aware, explainable AutoML for quantum + classical machine learning — powered by [Optuna](https://optuna.org) & [PennyLane](https://pennylane.ai).**

[![PyPI](https://img.shields.io/pypi/v/quoptuna?color=5b6cf9&label=PyPI)](https://pypi.org/project/quoptuna/)
[![Downloads](https://img.shields.io/pepy/dt/quoptuna?color=8e44ad&label=downloads)](https://pepy.tech/project/quoptuna)
[![Tests](https://img.shields.io/github/actions/workflow/status/Qentora/quoptuna/test.yml?branch=main&label=tests)](https://github.com/Qentora/quoptuna/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://pypi.org/project/quoptuna/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/Qentora/quoptuna/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-teal)](https://Qentora.github.io/quoptuna)

<!-- DEMO GIF PLACEHOLDER: capture the 6-step wizard (Dataset → Optimize → Analyze)
     and save as assets/branding/demo.gif (< 8 MB), then uncomment:
<img src="https://raw.githubusercontent.com/Qentora/quoptuna/main/assets/branding/demo.gif"
     alt="QuOptuna wizard: dataset → features → configure → optimize → analyze → report" width="760" />
-->

</div>

---

## Why QuOptuna?

Training a good quantum machine-learning model today means hand-writing circuits, guessing hyperparameters, and hoping the result is trustworthy. **QuOptuna runs one automated search across 21 quantum and classical classifiers**, prunes hopeless configurations early, audits every model for fairness, explains the winner with SHAP, and writes the research report for you — all from a point-and-click web wizard or a single CLI command.

The framework is peer-reviewed: it is published in the *IEEE Systems, Man, and Cybernetics Magazine* (2026) and two IEEE conference proceedings — see [Publications](#-publications--citation).

|  | Plain Optuna | Classical AutoML | **QuOptuna** |
| --- | :---: | :---: | :---: |
| Quantum models (PennyLane/JAX) | manual wiring | ✗ | ✅ 17 built in |
| Quantum **and** classical in one search | manual | ✗ | ✅ |
| Fairness-aware search (constrained / multi-objective) | manual | rare | ✅ built in |
| SHAP explainability + AI-written reports | ✗ | partial | ✅ built in |
| Zero-install web UI | dashboard only | varies | ✅ `uvx quoptuna` |

## ✨ Features

- **21 models, one search** — 17 quantum classifiers (Data Reuploading, Circuit-Centric, IQP & Projected Quantum Kernels, Quantum Kitchen Sinks, Quantum Metric Learner, Quantum Boltzmann Machines, Tree Tensor, Quanvolutional NN, WeiNet, separable & dressed variants) alongside classical baselines (SVC, LinearSVC, MLP, Perceptron), with automatic one-vs-rest multiclass support.
- **Smart optimization** — Optuna TPE / random / grid samplers with ASHA & Hyperband pruning, conditional per-model search spaces, and vectorized (JAX `vmap`) circuit evaluation for fast trials.
- **Fairness in the loop** — don't just measure bias, *search under it*: constrained mode (feasibility threshold on disparity) or multi-objective mode (accuracy-vs-fairness Pareto front), using demographic parity, equalized odds, or equal-opportunity metrics via [fairlearn](https://fairlearn.org).
- **Explainability built in** — SHAP bar / beeswarm / violin / heatmap / waterfall plots, ROC & PR curves, confusion matrices for every trained model.
- **AI-written reports** — a two-agent analyst + reviewer pipeline turns your run into a readable research report (works with OpenAI, Anthropic, and Google Gemini keys).
- **6-step web wizard** — Dataset → Features → Configure → Optimize → Analyze → Report. A Next.js UI served by a FastAPI backend on a single port, with live trial monitoring and restart-safe run persistence.
- **REST API & headless CLI** — automate everything (`/api/v1/...`), or run `quoptuna optimize` in CI. A legacy Streamlit dashboard remains available via `quoptuna run --streamlit`.

## 📦 Installation

Requires **Python 3.11 or 3.12**.

```bash
# Zero-install: run the full app straight from PyPI
uvx quoptuna

# Or install into your environment
pip install quoptuna        # or: uv pip install quoptuna
```

## 🚀 Quick Start

**Web UI** — one command boots the bundled app (FastAPI + pre-built UI on one port, no Node.js needed) and opens your browser:

```bash
uvx quoptuna
```

| URL | What |
| --- | --- |
| `http://localhost:8000` | Web UI (6-step wizard) |
| `http://localhost:8000/api/v1/...` | JSON API |
| `http://localhost:8000/api/docs` | Interactive API docs |

**Headless CLI** — optimize a UCI dataset without touching a browser:

```bash
quoptuna optimize --uci-id 267 --trials 25 --sampler tpe
```

**Python API** — drive the search from your own code:

```python
from quoptuna import DataPreparation, Optimizer

data_prep = DataPreparation(
    file_path="your_data.csv",
    x_cols=["feature_1", "feature_2", "feature_3"],
    y_col="target",
)
data_dict = data_prep.get_data(output_type="2")

optimizer = Optimizer(db_name="experiment", study_name="trial_1", data=data_dict)
study, best_trials = optimizer.optimize(n_trials=25)

print(f"Best F1 score: {best_trials[0].value:.4f}")
print(f"Best model:    {best_trials[0].params['model_type']}")
```

## 📖 Documentation

Full documentation lives at **[Qentora.github.io/quoptuna](https://Qentora.github.io/quoptuna)**:

- [Quickstart](https://Qentora.github.io/quoptuna/getting-started/quickstart/) — first optimization in five minutes
- [Samplers & pruning](https://Qentora.github.io/quoptuna/how-to/choose-samplers-and-pruners/) — TPE vs. random vs. grid, ASHA/Hyperband
- [Fairness-aware search](https://Qentora.github.io/quoptuna/how-to/run-fairness-aware-search/) — constrained & multi-objective modes
- [Tuning for speed & quality](https://Qentora.github.io/quoptuna/how-to/tune-for-speed-and-quality/) — vectorization, devices, validation splits

## 📄 Publications & Citation

QuOptuna is described in three peer-reviewed IEEE publications:

- E. Jose, A. C. Fong, C. S. Lai, B. Fong, L. L. Lai, ["**Quoptuna: Automated Optimization and Governance for Quantum Machine Learning**,"](https://ieeexplore.ieee.org/abstract/document/11603915) *IEEE Systems, Man, and Cybernetics Magazine*, vol. 12, no. 3, pp. 116–121, 2026. doi:[10.1109/MSMC.2025.3613072](https://doi.org/10.1109/MSMC.2025.3613072)
- E. Jose, A. C. Fong, ["**Quoptuna: An Open-Source Framework for Explainable Quantum Machine Learning**,"](https://ieeexplore.ieee.org/abstract/document/11256089) *2025 Interdisciplinary Conference on Electrics and Computer (INTCEC)*, pp. 1–5. doi:[10.1109/INTCEC65580.2025.11256089](https://doi.org/10.1109/INTCEC65580.2025.11256089)
- E. Jose, A. C. Fong, ["**Design and Development of an Open-Source Toolkit for Quantum Machine Learning**,"](https://ieeexplore.ieee.org/abstract/document/11315342) *2025 IEEE Int. Conf. on Recent Advances in Systems Science and Engineering (RASSE)*, pp. 1–6. doi:[10.1109/RASSE64831.2025.11315342](https://doi.org/10.1109/RASSE64831.2025.11315342)

If you use QuOptuna in your research, please cite the magazine article (GitHub's *Cite this repository* button uses [`CITATION.cff`](CITATION.cff)):

```bibtex
@article{jose2026quoptuna,
  author  = {Jose, Edwin and Fong, Alvis C. and Lai, Chun Sing and Fong, Bernard and Lai, Loi Lei},
  journal = {IEEE Systems, Man, and Cybernetics Magazine},
  title   = {Quoptuna: Automated Optimization and Governance for Quantum Machine Learning},
  year    = {2026},
  volume  = {12},
  number  = {3},
  pages   = {116--121},
  doi     = {10.1109/MSMC.2025.3613072}
}
```

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup and workflow, and the [contributor docs](https://Qentora.github.io/quoptuna/contributing/) for the long-form guide. Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).

```bash
git clone https://github.com/Qentora/quoptuna.git && cd quoptuna
uv sync                # install dependencies
uv run pytest          # run the test suite
```

## 📜 License

Apache License 2.0 — see [LICENSE](https://github.com/Qentora/quoptuna/blob/main/LICENSE).

## 🙏 Acknowledgments

Built on the shoulders of [Optuna](https://optuna.org), [PennyLane](https://pennylane.ai), [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) (Apache-2.0), [fairlearn](https://fairlearn.org), [SHAP](https://github.com/shap/shap), [FastAPI](https://fastapi.tiangolo.com), and [Next.js](https://nextjs.org). Project scaffolding from the [Wolt Python Package Cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter). Developed at Western Michigan University. Thanks to all [contributors](https://github.com/Qentora/quoptuna/graphs/contributors)!

---

<div align="center">

⭐ If QuOptuna helps your research, consider starring the repo — it helps others find it.

[![Star History Chart](https://api.star-history.com/svg?repos=Qentora/quoptuna&type=Date)](https://star-history.com/#Qentora/quoptuna&Date)

**[Documentation](https://Qentora.github.io/quoptuna)** • **[Report Bug](https://github.com/Qentora/quoptuna/issues)** • **[Request Feature](https://github.com/Qentora/quoptuna/issues)**

</div>
