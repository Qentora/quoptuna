---
title: Quickstart
description: Launch QuOptuna and run your first optimization on a built-in UCI dataset in about five minutes.
---

Already installed QuOptuna? This five-minute path gets you from a cold start to
your first best model using a built-in UCI dataset. If you haven't installed
yet, see [Installation](/getting-started/installation/).

## 1. Launch the app

```bash
uvx quoptuna
```

A gradient ASCII banner prints, and your browser opens to the wizard at
[http://localhost:8000](http://localhost:8000).

## 2. Pick a dataset

On **Step 1 — Dataset**, choose a built-in UCI dataset (for example, **Iris**,
id 53) instead of uploading a CSV.

## 3. Choose features and target

On **Step 2 — Features**, select your feature columns and the target column.
The defaults from the dataset work fine for a first run.

## 4. Configure a short run

On **Step 3 — Configure**, keep the default model (SVC), leave the sampler and
pruner as-is, and set **number of trials** to a small value like **3** so it
finishes quickly.

## 5. Optimize and see the best model

On **Step 4 — Optimize**, click run. Watch trials complete live. When it
finishes, the best trial and its model are highlighted.

:::tip
Continue to **Step 5 — Analyze** for metrics and SHAP plots, and **Step 6 —
Report** for an AI-generated summary.
:::

## Next steps

- [First optimization in the UI](/tutorials/first-optimization-ui/)
- [Optimize from the CLI](/tutorials/optimize-from-cli/)
