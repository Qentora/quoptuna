# Hashnode cross-post

**Strategy:** Publish the dev.to article first, then cross-post here. In Hashnode's post settings, set the **Canonical URL** to your dev.to URL (or the repo) so you don't split SEO. Hashnode lets you import from a URL — you can paste the dev.to article directly.

**Recommended tags:** `opensource`, `machine-learning`, `python`, `quantum-computing`, `automl`

**Body:** Use the exact content of [`../dev-to/article.md`](../dev-to/article.md) — it's written to work verbatim on both platforms.

**One tweak for Hashnode:** Hashnode's audience skews slightly more toward web/backend devs, so in the intro you can lead a touch harder on the packaging trick (the no-Node.js `uvx quoptuna` bundling) since that tends to land well here. Optional swap for the first line:

> **TL;DR** — `pip install quoptuna` ships a full AutoML web app *with the Next.js UI bundled into the wheel* — no Node.js, no build step. It searches 21 quantum + classical ML models in one Optuna run, with fairness and SHAP built in. Here's how it's put together.

**CTA (same as dev.to):** ask for a ⭐ and feedback on the search-space + fairness design. Link the repo: https://github.com/Qentora/quoptuna
