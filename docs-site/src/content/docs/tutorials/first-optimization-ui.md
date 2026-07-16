---
title: First optimization in the UI
description: A guided walkthrough of all six wizard steps, from dataset selection to an AI-generated report.
---

In this tutorial you'll run a complete optimization from start to finish using
the QuOptuna web wizard and a built-in UCI dataset. You'll learn what each of the
six steps does and what to expect at the end: a best trial, metrics, a SHAP
plot, and a written report. Plan on 10–15 minutes.

Before you begin, launch the app and open the wizard:

```bash
uvx quoptuna
```

Your browser opens to [http://localhost:8000](http://localhost:8000). The
wizard's state persists across navigation and restarts, so you can leave and
come back without losing progress.

## Step 1 — Dataset

Choose your data source. You can **upload a CSV** or **pick a UCI dataset**. For
this walkthrough, select the **Iris** dataset (UCI id 53). Iris is small,
multiclass (three species), and runs quickly.

You'll see a preview of the loaded rows and columns.

## Step 2 — Features

Now choose which columns feed the model:

- Select the **feature** columns (the four Iris measurements).
- Select the **target** column (the species).

This step also lets you remap or label-encode values and pick the categorical
encoding — **ordinal** or **onehot**. Iris features are numeric, so you can
accept the defaults.

:::note
QuOptuna encodes targets automatically. Binary targets use the {-1, +1}
convention; multiclass targets use codes 0..K-1 and are scored with macro-F1.
:::

## Step 3 — Configure

Set up the search:

| Option | What it controls | Suggested first value |
| --- | --- | --- |
| Models | Which classifiers to search over | SVC |
| Sampler | How trials are proposed | TPE |
| Pruner | Early-stopping of weak trials | None |
| Fairness | Fairness-aware search | Off |
| Number of trials | How many configurations to try | 5 |

Keeping a single classical model (SVC) and a handful of trials makes this first
run fast. You can add quantum models like `IQPKernelClassifier` later.

## Step 4 — Optimize

Click run. The **Optimize** step shows **live trial progress** as each
configuration is evaluated. When the run completes, QuOptuna reports the **best
trial** — its score and the winning model configuration.

Expected result: a best trial with a high macro-F1 (Iris is an easy dataset,
so scores near the top of the range are normal).

## Step 5 — Analyze

Explore why the model performs as it does:

- **SHAP** explanations show which features drive predictions.
- **Metrics, curves, and a confusion matrix** summarize classification quality.

Expect a SHAP plot (for example, a bar or beeswarm plot) ranking the four Iris
features, plus a confusion matrix that is nearly diagonal for a well-fit model.

## Step 6 — Report

Finally, generate an **AI-written summary** of the run. The report ties together
the best model, its metrics, and the SHAP findings into readable prose.

:::note
Report generation uses an LLM and needs internet access.
:::

## What you accomplished

You loaded a dataset, chose features and a target, configured and ran a search,
analyzed the winning model with SHAP and metrics, and produced a report — the
full QuOptuna loop.

## Next steps

- [Optimize from the CLI](/tutorials/optimize-from-cli/)
- [Python API tutorial](/tutorials/python-api/)
