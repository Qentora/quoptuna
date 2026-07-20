# Outreach templates — newsletters, awesome-lists, communities, press

Keep everything short, specific, and value-first. Personalize the first line every time.

---

## A. Awesome-list PRs

Target lists (open a PR adding QuOptuna in the right section):
- `krishnakumarsekar/awesome-quantum-machine-learning`
- `windmaple/awesome-AutoML` and `hibayesian/awesome-automl-papers`
- `vinta/awesome-python` (ML section — high bar, only if it clearly fits)
- `qosf/awesome-quantum-software` (great fit — quantum software)
- Any "awesome-fairness" / responsible-AI lists

**PR entry format (one line, match the list's style):**
```markdown
- [QuOptuna](https://github.com/Qentora/quoptuna) - Fairness-aware, explainable AutoML that searches 21 quantum and classical classifiers in one Optuna run (PennyLane + Optuna). Apache-2.0.
```
**PR description:** "Adds QuOptuna, an open-source (Apache-2.0) AutoML tool for quantum + classical ML with fairness-in-loop optimization and SHAP explainability. Described in three IEEE papers; happy to adjust the wording/section placement."

---

## B. Newsletter pitch (Data Elixir, Deep Learning Weekly, quantum newsletters, TLDR AI)

**Subject:** `Open-source AutoML for quantum + classical ML (fairness built in)`

> Hi [name],
>
> I'm Edwin, a PhD researcher at Western Michigan University. I just open-sourced **QuOptuna**, an AutoML tool that searches 21 quantum and classical ML classifiers in one Optuna run and — unusually — builds fairness auditing and SHAP explainability directly into the search loop.
>
> It might fit [newsletter] because it's a genuinely new angle (quantum + classical + responsible-ML in one tool) and it's dead simple to try: `uvx quoptuna` boots the whole web app with no install.
>
> Repo: https://github.com/Qentora/quoptuna · it's Apache-2.0 and described in three IEEE papers.
>
> Happy to send a summary blurb or screenshots if useful. Thanks for considering it!
>
> Edwin

---

## C. Community intros (Unitary Fund / QOSF, Optuna maintainers, PennyLane team)

**Unitary Fund / QOSF Slack/Discord:**
> 👋 I built QuOptuna — open-source AutoML that automates model selection + hyperparameter tuning across 17 quantum classifiers (on PennyLane) and classical baselines, with fairness and SHAP built in. It's my PhD project, Apache-2.0. Would love feedback from this community, and I'm exploring a hardware backend next. Repo: https://github.com/Qentora/quoptuna

**Optuna team (kindly ask about the "Made with Optuna" showcase / mention):**
> QuOptuna uses Optuna to drive a search across 21 quantum + classical models (conditional search spaces, ASHA/Hyperband pruning, multi-objective fairness). Would it be a fit for the Optuna showcase / "Made with Optuna"? Repo: https://github.com/Qentora/quoptuna

---

## D. Academic amplification
- **ResearchGate / Google Scholar:** link the software repo from your IEEE paper pages; add QuOptuna as a "software" research item.
- **University news / dept comms (WMU):** pitch a short "PhD student open-sources quantum ML tool described in IEEE" story — university press drives credible backlinks.
- **Lab / advisor networks:** ask co-authors and your advisor to share the launch post (their reach ≠ yours).
- **Conference/workshop channels:** mention it where you present (e.g. AutoML or quantum ML workshops) — a slide with the `uvx quoptuna` command and repo URL.

---

## E. Papers with Code
Once the arXiv note is live: create the paper entry, link the repo, and attach tasks *AutoML*, *Hyperparameter Optimization*, *Quantum Machine Learning*, *Fairness*.
