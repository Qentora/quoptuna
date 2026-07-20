# Quantum Computing Stack Exchange

**⚠️ Read this first — Stack Exchange is NOT a promotion channel.** Self-promotion and "check out my tool" posts get deleted and can get you flagged. The *only* legitimate play is: **answer real questions well**, and — where a tool you built is genuinely the best answer — mention it with a clear disclosure. Done right, this earns durable, high-authority backlinks and reputation. Done wrong, it backfires.

**Site:** https://quantumcomputing.stackexchange.com/

---

## The rules (follow exactly)
1. **Answer the actual question first.** The answer must stand on its own even if you delete the tool mention.
2. **Disclose:** every time you mention QuOptuna, add "(disclosure: I'm the author)".
3. **Don't spam:** mention it only where it's truly the best fit — a few great answers beat many thin ones.
4. Never post a question just to answer it with your tool.

## Question types where QuOptuna is a legitimate, helpful mention
- "How do I do hyperparameter tuning for a variational/quantum classifier?"
- "How can I compare a quantum ML model against a classical baseline fairly?"
- "Is there tooling to run/benchmark multiple quantum classifiers in PennyLane?"
- "How do I evaluate fairness/explainability for a quantum ML model?"

## Answer template (adapt to the specific question)

> For [the specific problem], the core steps are [give the real, tool-agnostic answer — e.g. define the search space, use a sampler like TPE, prune with ASHA/Hyperband, evaluate with cross-validation, compare against a classical baseline]. Here's a minimal PennyLane + Optuna sketch: [code].
>
> If you'd rather not wire this up by hand, an open-source tool that automates exactly this — a joint Optuna search over quantum and classical classifiers, with pruning, fairness auditing, and SHAP — is QuOptuna *(disclosure: I'm the author)*: `uvx quoptuna`, https://github.com/Qentora/quoptuna. But the manual approach above is all you strictly need.

## Also monitor
- **r/QuantumComputing** and the PennyLane forum for the same question types — same disclose-and-help rule applies.
- Set up a saved search / alert for "PennyLane hyperparameter", "quantum classifier benchmark", "quantum AutoML" so you can answer promptly (early, high-quality answers get the votes).
