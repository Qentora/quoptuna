---
title: Run fairness-aware search
description: Search for models that stay accurate and equitable across a protected attribute.
---

QuOptuna can search for models that stay accurate AND equitable across a protected attribute. You supply a categorical `sensitive_feature` column; multiclass problems also need a `favorable_class`.

## Modes

Set `fairness_mode`:

| Mode | Behavior |
| --- | --- |
| `off` (default) | No fairness handling. |
| `constrained` | TPE feasibility constraint — every trial must satisfy a disparity threshold `fairness_threshold`. |
| `multi_objective` | Pareto front of F1 vs disparity. Pruning is disabled in this mode. |

## Metrics

Set `fairness_metric`:

| Metric | Notes |
| --- | --- |
| `equal_opportunity_difference` (default) | Difference in true-positive rate across groups. |
| disparate impact / four-fifths rule | Ratio-based; the classic 80% rule. |
| demographic-parity difference | Difference in positive-prediction rate across groups. |

Fairness search is available via `POST /api/v1/optimize`, the `Optimizer`, and the CLI (`--sensitive-feature`, `--fairness-mode`, `--fairness-metric`, `--fairness-threshold`, `--favorable-class`).

## Constrained search (CLI)

Keep every kept trial within a disparity budget:

```bash
quoptuna optimize \
  --sensitive-feature sex \
  --fairness-mode constrained \
  --fairness-metric equal_opportunity_difference \
  --fairness-threshold 0.1
```

Only trials whose disparity is within `0.1` are treated as feasible; TPE steers toward the feasible region and `best_trial` comes from it.

## Multi-objective search (CLI)

Explore the trade-off instead of committing to one threshold:

```bash
quoptuna optimize \
  --sensitive-feature sex \
  --fairness-mode multi_objective \
  --fairness-metric equal_opportunity_difference
```

:::note[Multiclass fairness]
For multiclass targets you must also pass `--favorable-class <code>` so the metric knows which outcome is the favorable one.
:::

## Read a Pareto front

`multi_objective` produces a set of non-dominated trials — each is optimal for some balance of F1 vs disparity, and none strictly beats another on both. There is no single "best": **you pick the operating point**. Plot F1 against disparity across the front and choose the trial whose trade-off matches your governance requirements (for example, the highest-F1 trial whose disparity stays under your acceptable ceiling).

:::caution
`multi_objective` disables pruning, so every trial trains to completion. Budget trials accordingly.
:::

## Next steps

- [Optimize a multiclass problem](/how-to/use-multiclass/)
- [Choose samplers and pruners](/how-to/choose-samplers-and-pruners/)
- [CLI reference](/reference/cli/)
