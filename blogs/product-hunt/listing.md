# Product Hunt — listing kit

**Timing:** PH days run 12:01am PT → 11:59pm PT. Launch Tue–Thu. Line up a few friends to be early supporters (no vote-begging in public — against PH rules). Be online all day to reply to every comment.

---

## Name
QuOptuna

## Tagline (≤60 chars — pick one)
- `AutoML for quantum + classical ML, zero install`
- `Fairness-aware AutoML for quantum machine learning`

## Topics / tags
Artificial Intelligence · Open Source · Developer Tools · Machine Learning · Python

## Links
- Website / demo: `https://Qentora.github.io/quoptuna` (or hosted demo if deployed)
- GitHub: `https://github.com/Qentora/quoptuna`

## Description (≤260 chars)
QuOptuna runs one automated search across 21 quantum & classical ML classifiers, audits every model for fairness, explains the winner with SHAP, and drafts the report — from a web wizard or one command. `uvx quoptuna`, no install. Open source (Apache-2.0).

## Gallery
1. `assets/branding/social-preview.png` (hero)
2. Screenshot of the 6-step web wizard *(TODO: capture)*
3. A SHAP beeswarm plot (`experiments/basic_dataset_test/blood/beeswarm.png`)
4. Demo GIF of a search running *(TODO: record)*

## Maker's first comment (post at launch)

Hey Product Hunt 👋 I'm Edwin, the maker.

QuOptuna started as my PhD project: I was tired of hand-writing quantum ML circuits, guessing hyperparameters, and having no way to check if a model was fair or trustworthy. So I built one tool that searches quantum **and** classical models together and returns a governable result.

What makes it different:
🔭 **21 models, one search** — 17 quantum + 4 classical, powered by Optuna + PennyLane
⚖️ **Fairness in the search loop** — a constraint or a Pareto objective, not an afterthought
🔍 **SHAP explainability + an auto-written report** for the winner
⚡ **Zero install** — `uvx quoptuna` boots the whole web app; no Node, no setup

It's Apache-2.0 and Beta — quantum models run on simulators, so it's a research/prototyping tool for now. On tabular data classical models often win, and QuOptuna tells you so honestly.

I'll be here all day — I'd love your feedback, feature requests, and honest criticism. What would make this genuinely useful for your work?

## Reply snippets
- **"Does it need my data in the cloud?"** → No — everything runs locally. The only optional external call is the LLM report writer, and only if you add your own API key.
- **"Is quantum ML actually better?"** → Usually not yet, on tabular data — and the tool says so. The value is honest, automated comparison + governance, so you're ready when quantum advantage does appear on your data.
