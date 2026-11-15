# Smart-thermostat outreach: a fairness audit with QuOptuna

Utilities and thermostat makers run demand-response programs that reward
owners of Internet-connected thermostats for letting the device ease its
setting during demand peaks. A program that recruits devices households
already own must first guess who owns one: it scores customers with a model
and invites the top of the list within an offer budget. This example asks
whether lower-income owners are invited as often as higher-income ones when
that model is chosen by a QuOptuna search.

It is the code behind the IEEE Consumer Electronics Magazine submission
"Who Gets the Smart-Thermostat Invitation? Fair Targeting for Demand
Response". The paper's numbers came from QuOptuna 1.0.1 with the decision
threshold sweep disabled by a subclass; from 1.1.0 the same behaviour is the
`tune_decision_threshold=False` option used here.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `prepare_data.py` | Downloads RECS 2020 microdata (EIA, version 7), builds the target (`TYPETHERM`: smart / Internet-connected thermostat) and twelve inputs (income and race are not inputs), and writes three disjoint splits: a survey-weighted audit half, a 2,500-household calibration sample, and a 2,000-household search sample enriched to 30% owners |
| 2 | `run_search.py` | One QuOptuna search per arm and seed: `off` (accuracy-only, ASHA), `constrained` (equal-opportunity gap <= 0.10, ASHA), `pareto` (F1 vs gap, multi-objective, no pruning). All arms set `tune_decision_threshold=False` so F1 and the gap describe the same classifier |
| 3 | `audit.py` | Selects one model per run by fixed rules, sets a cutoff that invites 15% of calibration households, and audits the held-out half: owners invited per income band, equal-opportunity gap, yield, 95% stratified-bootstrap intervals, random and logistic-regression baselines, scoring cost, and a two-cutoff release-gate fix that uses only a lower-income flag |
| - | `run_all.sh` | Runs steps 1-3 end to end (six searches in parallel, two threads each) |

## Run

```bash
pip install "quoptuna>=1.1.0"
cd examples/smart_thermostat_outreach
PYTHON=python ./run_all.sh            # ~3 h on 12 cores with 60 trials x 2 seeds
TRIALS=5 SEEDS=42 PYTHON=python ./run_all.sh   # quick smoke run
```

Outputs go to `results/` (ignored by git): `data/` (splits; the 54 MB raw
RECS file is downloaded on first run), `runs/` (per-trial JSON exports,
Optuna SQLite studies, logs) and `summary.json` (every audit metric).

Re-running `audit.py` on stored runs reproduces the audit metrics exactly;
search wall-clock times depend on hardware and load.

## Data

U.S. Energy Information Administration, 2020 Residential Energy Consumption
Survey public-use microdata, version 7:
<https://www.eia.gov/consumption/residential/data/2020/>. The paper's runs used
the file with SHA-256
`8d5f6d3120d2cbc2535151c7d455e7a536ca9768f773e15edd45932ec38b3592`; if EIA
republishes it, the seeded splits may change.
