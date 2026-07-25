---
title: CLI reference
description: The quoptuna command-line interface — the run and optimize subcommands and their options.
---

The `quoptuna` command is a [Typer](https://typer.tiangolo.com/) application. Invoking `quoptuna` with no subcommand is equivalent to `quoptuna run`.

## `quoptuna migrate-supabase`

Copy application metadata from the legacy SQLite `quoptuna_app.db` into the
configured SQLModel database. The source file is not deleted.

```bash
quoptuna migrate-supabase \
  --source-db db/quoptuna_app.db \
  --database-url "$DATABASE_URL" \
  --dry-run

quoptuna migrate-supabase \
  --source-db db/quoptuna_app.db \
  --database-url "$DATABASE_URL"
```

`--dry-run` reports the records that would be copied. The command is safe to
repeat and uses stable IDs when writing the destination.

## `quoptuna migrate-optuna`

Copy all Optuna studies from one local SQLite database into the PostgreSQL
database configured by `OPTUNA_DATABASE_URL`:

```bash
quoptuna migrate-optuna db/results-trial-june15.db
```

Copy only one study:

```bash
quoptuna migrate-optuna \
  db/results-trial-june15.db \
  --study-name heart-june16-trial3
```

Override the configured destination:

```bash
quoptuna migrate-optuna \
  db/results-trial-june15.db \
  --target-url "$OPTUNA_DATABASE_URL"
```

The command copies study metadata and trials, does not delete the source file,
and reports each migrated study. Existing studies with the same name can cause
Optuna to reject a repeat; verify the destination before rerunning a partially
completed migration.

## `quoptuna run`

Launch the QuOptuna application. Serves the Web UI, the JSON API, and interactive docs on a single port. API docs are available at `/api/docs`.

| Option | Default | Description |
| --- | --- | --- |
| `--streamlit` | off | Launch the legacy Streamlit dashboard instead of the full stack |
| `--host <h>` | access host | Access host used in generated URLs and links |
| `--port <n>` | `8000` | Port for the QuOptuna server (auto-increments if busy) |
| `--no-browser` | off | Do not auto-open the browser |

## `quoptuna optimize`

Run a single optimization headless through the exact UI pipeline and print a JSON summary to stdout.

| Option | Default | Description |
| --- | --- | --- |
| `--uci-id <n>` | — | UCI dataset id (e.g. `53` for Iris). Mutually exclusive with `--csv` |
| `--csv <path>` | — | Path to a local CSV dataset. Provide exactly one of `--uci-id` / `--csv` |
| `--target <col>` | dataset target / last column | Target column |
| `--features a,b,c` | all non-target | Comma-separated feature columns |
| `--trials <n>` | `3` | Number of Optuna trials (min 1) |
| `--models <list>` | `SVC` | Comma-separated model types, e.g. `SVC,IQPKernelClassifier` |
| `--label-neg <v>` | — | Binary targets: label mapped to `-1` |
| `--label-pos <v>` | — | Binary targets: label mapped to `+1` |
| `--favorable-class <v>` | — | Multiclass: favorable outcome for fairness auditing (only with fairness) |
| `--sensitive-feature <col>` | — | Protected-attribute column for fairness auditing |
| `--fairness-mode <m>` | `off` | `off` \| `constrained` \| `multi_objective` |
| `--fairness-metric <m>` | `equal_opportunity_difference` | Disparity metric |
| `--fairness-threshold <f>` | — | Constrained-mode disparity threshold |
| `--sampler <s>` | `random` | `tpe` \| `random` \| `grid` |
| `--seed <n>` | `0` | Sampler seed for reproducible runs |
| `--pruner <p>` | `none` | `none` \| `asha` \| `hyperband` |
| `--max-steps <n>` | `20` | Training-step cap for iterative quantum models |
| `--convergence-interval <n>` | `5` | Flat-loss convergence window |
| `--max-vmap <n>` | — | Circuit vectorization width |
| `--categorical-encoding <e>` | `ordinal` | `ordinal` \| `onehot` |
| `--study-name <s>` | — | Optuna study name |
| `--db-name <s>` | `cli_runs` | Optuna storage database name |
| `--subset-size <n>` | `30` | Analysis subset size |
| `--no-analyze` | off | Skip the post-run analysis summary |

### Examples

```bash
quoptuna optimize --uci-id 53 --trials 2 --models SVC,IQPKernelClassifier
```

```bash
quoptuna optimize --csv data.csv --target label --trials 3
```

:::note
The `optimize` defaults (trials=3, sampler=random, pruner=none, max-steps=20) are tuned for a fast smoke-test run, not a production search — raise them for real optimization.
:::

## See also

- [REST API reference](/reference/rest-api/)
- [Python API reference](/reference/python-api/)
- [Model catalog](/reference/models/)
- [Configuration reference](/reference/configuration/)
