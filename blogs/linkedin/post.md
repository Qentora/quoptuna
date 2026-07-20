# LinkedIn — short post (feed)

**Voice:** first-person researcher. LinkedIn rewards a personal story + one clear link. Post natively; put the link in the first comment if you want max reach (LinkedIn suppresses posts with external links in the body — a common workaround is link-in-comment).

**Best time:** Tue–Thu morning. Add a hook line, whitespace between lines, 3–5 hashtags.

---

For the last stretch of my PhD at Western Michigan University, I've been building something I couldn't find anywhere else — and today I'm open-sourcing it.

**QuOptuna** is AutoML for quantum *and* classical machine learning.

Point it at a dataset and it runs one automated search across 21 classifiers — 17 quantum, 4 classical — and tells you, honestly, which one wins. But the part I'm proudest of isn't the quantum models. It's what surrounds them:

→ Fairness is a constraint *inside* the search, not a report you read afterward.
→ Every winning model comes explained with SHAP — no scrambling to justify it later.
→ It even drafts the research report for you.

And it installs in exactly zero steps:

`uvx quoptuna`

That boots the whole web app on your machine. No Node, no circuits to hand-write, no account.

It's Apache-2.0 and Beta — the quantum models run on simulators, so it's a research tool, not production quantum hardware (yet). The approach is written up in three IEEE papers.

If you work in ML, responsible AI, or quantum computing, I'd genuinely love your feedback. And if you find it useful, a ⭐ on GitHub means a lot to a solo project.

Repo link in the comments 👇

#MachineLearning #QuantumComputing #AutoML #ResponsibleAI #OpenSource

---

**First comment (post immediately after):**
> Repo: https://github.com/Qentora/quoptuna · Docs: https://Qentora.github.io/quoptuna · try it with `uvx quoptuna`
