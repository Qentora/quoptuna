# X / Twitter — launch thread

**Tips:** Post the thread all at once (or use a scheduler). Tag @PennyLaneAI and @OptunaAutoML in a reply (not the first tweet — tagging in tweet 1 can suppress reach). Attach the social preview image to tweet 1, and a SHAP plot GIF to a later tweet. Pin the thread.

---

**1/ (hook — attach `assets/branding/social-preview.png`)**

I open-sourced QuOptuna today.

It's AutoML for quantum + classical machine learning — one automated search across 21 models, with fairness and explainability built in.

And it installs in zero steps:

`uvx quoptuna`

🧵👇

**2/**

The problem: training a quantum ML model means hand-writing circuits, guessing hyperparameters, and having no way to know if the result is trustworthy.

Classical AutoML ignores quantum models. Plain Optuna makes you wire everything by hand.

QuOptuna automates all of it.

**3/**

It searches 21 classifiers in ONE run:

• 17 quantum (data-reuploading, quantum kernels, tree tensor nets, quanvolutional NNs…)
• 4 classical baselines (SVC, MLP…)

Optuna TPE + ASHA/Hyperband pruning, circuits vectorized with JAX vmap. It reports the honest winner — often classical, and it says so.

**4/**

The part I care about most: **fairness is inside the search loop.**

Not measured afterward. Either a hard constraint (disparity threshold) or a multi-objective Pareto front (accuracy vs. fairness), via fairlearn.

**5/ (attach a SHAP plot image)**

Every winning model comes explained — SHAP plots, ROC/PR, confusion matrices — and an optional 2-agent LLM pipeline drafts the report for you.

No more selecting a model and *then* scrambling to justify it.

**6/**

Nerdy detail: the Next.js UI is bundled into the Python wheel. So `uvx quoptuna` serves a full web wizard with zero Node.js and zero install. 🤯

**7/ (close)**

Apache-2.0 · Beta · runs on simulators · it's my PhD project (WMU), written up in 3 IEEE papers.

If quantum ML / AutoML / responsible AI is your thing, I'd love feedback — and a ⭐ helps a ton.

Repo → https://github.com/Qentora/quoptuna

---

## Standalone tweet hooks (for reuse / later)

- "Classical AutoML won't touch quantum models. QuOptuna searches quantum AND classical in one run — and tells you honestly which wins. `uvx quoptuna`, zero install. Apache-2.0 🧵"
- "Hot take: fairness should be a constraint *during* model search, not a report you read after. Here's an open-source AutoML tool that does exactly that (for quantum + classical ML) 👇"
- "I bundled a Next.js app into a Python wheel so `uvx quoptuna` boots a full AutoML web wizard with no Node.js. Here's why + how 🧵"

**Hashtags:** #QuantumML #AutoML #MachineLearning #OpenSource #Python
