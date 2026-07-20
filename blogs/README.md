<div align="center">

# 📣 QuOptuna Promotion Playbook

**A phased, platform-specific launch kit to grow GitHub stars & PyPI downloads for QuOptuna.**

</div>

---

## 0. What we're promoting (the pitch)

**QuOptuna** — *Fairness-aware, explainable AutoML for quantum + classical machine learning, powered by Optuna & PennyLane.*

> Training a good quantum ML model today means hand-writing circuits, guessing hyperparameters, and hoping the result is trustworthy. QuOptuna runs **one automated search across 21 quantum and classical classifiers**, prunes hopeless configs early, **audits every model for fairness**, **explains the winner with SHAP**, and **writes the research report for you** — all from a point-and-click web wizard or a single CLI command.

**Coordinates**
- Repo: `https://github.com/Qentora/quoptuna`
- PyPI: `https://pypi.org/project/quoptuna/` (`pip install quoptuna`)
- Docs: `https://Qentora.github.io/quoptuna`
- Zero-install demo: `uvx quoptuna` → opens the full app on `http://localhost:8000`
- License: Apache-2.0 · Version: 0.1.4 (Beta)
- Backed by **3 IEEE publications** (incl. *IEEE Systems, Man & Cybernetics Magazine*, 2026) — developed at Western Michigan University.

**Three sentences that do the heavy lifting** (reuse everywhere):
1. *The only AutoML that searches quantum **and** classical models in one Optuna run.*
2. *Fairness isn't measured after the fact — it's a constraint (or a Pareto objective) inside the search loop.*
3. *Zero install: `uvx quoptuna` boots a full web wizard; no Node, no circuits to write.*

---

## 📅 Phase 1 — Pre-launch assets (the foundation)

Make the repo convert before you send a single visitor.

| Asset | Status | Action |
|---|---|---|
| Open-graph / social preview image | ✅ Exists → `assets/branding/social-preview.png` | Set it in **GitHub → Settings → Social preview** |
| Hero logo in README | ✅ `assets/branding/logo-full.png` | — |
| Zero-install snippet above the fold | ✅ `uvx quoptuna` | Keep it first in README |
| One-command install | ✅ `pip install quoptuna` | — |
| `CONTRIBUTING.md` | ✅ Present in repo root | — |
| **Hosted live demo** (one-click, no install) | ❌ Missing | *TODO:* deploy a read-only demo (HF Spaces / Railway) — biggest single conversion lever |
| **Demo GIF / 60-sec video** | ❌ Placeholder only | *TODO:* record the 6-step wizard → embed in README + every post |
| `CITATION.cff` | ⚠️ Verify | Add so researchers cite you (feeds the academic tier) |

> **Do the two ❌ items before launch day if you can.** A live demo link and a demo GIF roughly double click-through everywhere below.

---

## 📌 Phase 2 — The multi-platform blast list

Grouped by tier. Ready-to-post copy for each lives in the matching subfolder.

### Tier 1 — Tech-core launch (the spike)

| Platform | Why it fits QuOptuna | Format | Copy |
|---|---|---|---|
| **Hacker News — Show HN** | Technical crowd rewards genuinely novel infra; "quantum + classical AutoML" is a strong hook | `Show HN: QuOptuna – …` text post, reply to first comments fast | [`hacker-news/show-hn.md`](hacker-news/show-hn.md) |
| **Reddit r/MachineLearning** | Use the `[Project]` flair; fairness+explainability resonates | `[Project]` post, no hype, link to repo/demo | [`reddit/r-machinelearning.md`](reddit/r-machinelearning.md) |
| **Reddit r/QuantumComputing** | Core niche audience — quantum ML tooling is rare | Discussion post | [`reddit/r-quantumcomputing.md`](reddit/r-quantumcomputing.md) |
| **Reddit r/Python** | `uvx quoptuna` zero-install + Typer CLI is very r/Python | Show-and-tell | [`reddit/r-python.md`](reddit/r-python.md) |
| **Reddit r/opensource** | License, self-hosting, contribution angle | Project post | [`reddit/r-opensource.md`](reddit/r-opensource.md) |
| **Product Hunt** | Broad reach; front page → hundreds of stars in 24h | Listing + maker's comment | [`product-hunt/listing.md`](product-hunt/listing.md) |
| **Lobsters** | Smaller, high-signal; needs invite | Tag `ml`, `python` | (reuse Show HN body) |

### Tier 2 — Deep-dive blogs (the long tail / SEO)

| Platform | Angle | Copy |
|---|---|---|
| **DEV Community (dev.to)** | "How I built AutoML for quantum ML" — architecture + code | [`dev-to/article.md`](dev-to/article.md) |
| **Hashnode** | Canonical cross-post of the dev.to piece | [`hashnode/article.md`](hashnode/article.md) |
| **Medium / Towards Data Science** | Fairness + explainability in ML (broader DS audience) | [`medium-tds/article.md`](medium-tds/article.md) |
| **LinkedIn** | Researcher/PhD voice — credibility + reach to academia & industry | [`linkedin/post.md`](linkedin/post.md), [`linkedin/article.md`](linkedin/article.md) |

### Tier 3 — Academic / research credibility (the PhD moat)

| Channel | Why it matters | Copy |
|---|---|---|
| **JOSS (Journal of Open Source Software)** | A **citable, peer-reviewed paper for the software itself** — durable credibility + citations for the PhD, and a badge that converts researchers | [`joss/submission-notes.md`](joss/submission-notes.md) |
| **arXiv (software/tool note)** | Discoverable by the exact researchers who'd star it; pairs with the IEEE papers | [`academic/arxiv-abstract.md`](academic/arxiv-abstract.md) |
| **Papers with Code** | Link tool to tasks (AutoML, quantum ML, fairness) | see `academic/outreach-emails.md` |
| **ResearchGate / university news** | WMU press + your networks | see `academic/outreach-emails.md` |

### Tier 4 — Quantum & AutoML niche communities (highest-converting per view)

| Channel | Copy |
|---|---|
| **PennyLane forum + Xanadu Discord** (built on PennyLane) | [`quantum-community/pennylane-forum.md`](quantum-community/pennylane-forum.md) |
| **Optuna — GitHub Discussions / "Made with Optuna" showcase** | [`quantum-community/optuna-showcase.md`](quantum-community/optuna-showcase.md) |
| **Quantum Computing Stack Exchange** (answer relevant Qs, mention tool) | [`quantum-community/qc-stackexchange.md`](quantum-community/qc-stackexchange.md) |
| **Unitary Fund / QOSF community** | see `academic/outreach-emails.md` |
| **Awesome-list PRs** — `awesome-quantum-machine-learning`, `awesome-automl`, `awesome-python` | see `academic/outreach-emails.md` |

### Tier 5 — Amplification

| Channel | Copy |
|---|---|
| **X / Twitter** — launch thread + `#QuantumML #AutoML #OpenSource` | [`twitter-x/launch-thread.md`](twitter-x/launch-thread.md) |
| **Mastodon / Bluesky** | reuse the thread hooks |
| **YouTube / Loom** — 60-sec wizard walkthrough | *TODO once GIF/video exists* |
| **Newsletters to pitch** — Data Elixir, Deep Learning Weekly, quantum newsletters | see `academic/outreach-emails.md` |

---

## 🗓️ Phase 3 — The 48-hour launch schedule (stagger to dodge spam filters)

Pick a **Tuesday–Thursday, ~8–10am ET** (best HN/PH windows).

| Time | Action |
|---|---|
| **T-1 week** | Submit JOSS + arXiv; open awesome-list PRs; deploy live demo + record GIF |
| **Day 0, 8am ET** | **Show HN** goes live. Reply to first 3 comments within minutes. |
| **Day 0, 8am ET** | **Product Hunt** listing goes live (same morning). |
| **Day 0, 10am** | Post **r/QuantumComputing** (niche, warms up). |
| **Day 0, 1pm** | Publish **dev.to** deep dive; cross-post to Hashnode w/ canonical URL. |
| **Day 0, 3pm** | **X/Twitter** launch thread; tag PennyLane, Optuna. |
| **Day 0, evening** | **LinkedIn** post (researcher voice). |
| **Day 1, morning** | **r/MachineLearning** `[Project]`. |
| **Day 1, midday** | **r/Python** show-and-tell. |
| **Day 1, afternoon** | **PennyLane forum** + **Optuna showcase** post. |
| **Day 2** | **r/opensource**; **Medium/TDS** article; pitch newsletters. |

> **Rule:** never fire two Reddit posts in the same hour — space them ≥6h apart and vary the wording per subreddit's rules.

---

## 📊 Phase 4 — Post-blast maintenance

- **Issue triage:** check GitHub notifications hourly for the first 24h; label + reply fast.
- **Star management:** thank early stargazers and anyone opening a high-quality issue/PR.
- **Analytics:** watch **Insights → Traffic** to see which platform converts; double down on the winner.
- **Momentum:** a good first-day spike can hit **GitHub Trending** — keep replies flowing to sustain it.

## 📈 Tracking (which channel actually converts)

Add `?utm_source=<platform>` to repo/docs links per post, then log results here:

| Platform | Posted (date) | Referral clicks | Stars gained | Notes |
|---|---|---|---|---|
| Show HN | | | | |
| Product Hunt | | | | |
| r/QuantumComputing | | | | |
| r/MachineLearning | | | | |
| dev.to | | | | |
| X/Twitter | | | | |
| PennyLane forum | | | | |
| JOSS/arXiv | | | | |

---

## Golden rules for every post
1. **Lead with value, not "please star."** Ask for feedback; stars follow.
2. **Never fabricate** benchmarks, users, or a demo link that doesn't exist. Every claim here is verifiable in the repo.
3. **Be present.** Reply to every comment on launch day — active maintainers convert lurkers.
4. **One honest weakness disclosed** (e.g. "Beta, quantum sims run on CPU") earns more trust than a flawless pitch.
