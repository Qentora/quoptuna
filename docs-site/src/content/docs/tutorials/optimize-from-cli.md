---
title: Optimize from the CLI
description: Run the full QuOptuna pipeline headlessly with quoptuna optimize on a UCI dataset and a local CSV.
---

The `quoptuna optimize` command runs the exact same optimization pipeline as the
web wizard, without a browser. In this tutorial you'll run it two ways — on a
built-in UCI dataset and on your own CSV — read the JSON summary it prints, and
re-open the saved study with Optuna. Plan on about 10 minutes.

## Run on a UCI dataset

Optimize over the Iris dataset (UCI id 53). Provide exactly one data source —
either `--uci-id` or `--csv`, never both.

```bash
quoptuna optimize \
  --uci-id 53 \
  --target class \
  --trials 5 \
  --models SVC \
  --sampler tpe \
  --pruner none \
  --study-name iris_demo
```

## Run on a local CSV

Point at a CSV and name the feature and target columns:

```bash
quoptuna optimize \
  --csv ./your_data.csv \
  --target label \
  --features f1,f2,f3 \
  --trials 5 \
  --models SVC,IQPKernelClassifier \
  --sampler tpe \
  --study-name csv_demo
```

Here are the representative flags used above:

| Flag | Purpose |
| --- | --- |
| `--uci-id` / `--csv` | Data source (choose exactly one) |
| `--target` | Target column name |
| `--features` | Comma-separated feature columns |
| `--trials` | Number of trials (default 3) |
| `--models` | Comma-separated models (default SVC) |
| `--sampler` | `tpe`, `random`, or `grid` |
| `--pruner` | `none`, `asha`, or `hyperband` |
| `--study-name` | Name for the Optuna study |
| `--db-name` | SQLite database name (default `cli_runs`) |

There are more flags — including fairness options like `--sensitive-feature` and
`--fairness-mode`. See the full [CLI reference](/reference/cli/).

## Read the JSON summary

When the run finishes, `quoptuna optimize` prints a JSON summary of the results —
including the best trial's score and its winning model parameters. Redirect it to
a file if you want to keep it:

```bash
quoptuna optimize --uci-id 53 --target class --trials 5 > summary.json
```

## Where the study is saved

Results persist to an Optuna **SQLite** database under `db/`, keyed by the
`--db-name` you chose (default `cli_runs`) and stored under the `--study-name`.
Because studies persist, you can inspect them long after the run finishes.

## Re-open the study with Optuna

Load the saved study directly with Optuna to inspect trials programmatically:

```python
import optuna

study = optuna.load_study(
    study_name="iris_demo",
    storage="sqlite:///db/cli_runs.db",
)

best = study.best_trial
print(f"Best value: {best.value:.4f}")
print(f"Best params: {best.params}")
```

:::note
Adjust the `study_name` and the `.db` filename to match the `--study-name` and
`--db-name` you passed to `quoptuna optimize`.
:::

## Next steps

- [CLI reference](/reference/cli/)
- [Python API tutorial](/tutorials/python-api/)
