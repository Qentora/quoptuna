"""Run one arm of the connected-thermostat outreach experiment with Quoptuna.

Arms (same data, same sampler seed, same training budget):
  off          accuracy-only search, ASHA pruning on
  constrained  F1 subject to equal-opportunity gap <= --threshold, ASHA on
  pareto       F1 vs equal-opportunity gap, multi-objective (pruning disabled)

The Optuna SQLite file (db/) and the JSON trial export are written to
results/runs/. Numerical libraries are limited to two
threads per process so the six runs share a 12-core machine (see run_all.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from prepare_data import DATA, FEATURES, RESULTS_DIR
from sklearn.preprocessing import StandardScaler

RESULTS = RESULTS_DIR / "runs"
SENSITIVE = "income_group"
TARGET = "smart_thermostat"


def fit_scaler() -> StandardScaler:
    s = pd.read_csv(DATA / "search.csv")
    return StandardScaler().fit(s.loc[s["split"] == "train", FEATURES].to_numpy(float))


def encode(frame: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    """Scaled features and {-1, +1} labels (+1 = owns a connected thermostat)."""
    return scaler.transform(frame[FEATURES].to_numpy(float)), np.where(
        frame[TARGET] == "yes", 1, -1
    )


def load_splits() -> tuple[dict, np.ndarray]:
    s = pd.read_csv(DATA / "search.csv")
    scaler = fit_scaler()
    data = {}
    for name in ("train", "val", "test"):
        data[f"{name}_x"], data[f"{name}_y"] = encode(s[s["split"] == name], scaler)
    return data, s.loc[s["split"] == "val", SENSITIVE].to_numpy()


def pin_mlp_seed() -> None:
    """Give the MLP a fixed random_state so a refit reproduces the trial."""
    from quoptuna.backend import models  # noqa: PLC0415

    models.MODEL_FIXED_KWARGS.setdefault("MLPClassifier", {})["random_state"] = 0


def export(study, path: Path, meta: dict) -> None:
    rows = [
        {
            "number": t.number,
            "state": t.state.name,
            "values": t.values,
            "params": t.params,
            "user_attrs": t.user_attrs,
            "start": t.datetime_start.isoformat() if t.datetime_start else None,
            "complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "duration_s": t.duration.total_seconds() if t.duration else None,
        }
        for t in study.trials
    ]
    path.write_text(json.dumps({"meta": meta, "trials": rows}, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["off", "constrained", "pareto"], required=True)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--convergence-interval", type=int, default=25)
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--model-types", nargs="*", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data, sensitive_val = load_splits()
    RESULTS.mkdir(parents=True, exist_ok=True)
    os.chdir(RESULTS)  # Quoptuna writes its SQLite study under ./db/
    pin_mlp_seed()
    from quoptuna import Optimizer  # noqa: PLC0415

    name = f"thermostat_{args.arm}_s{args.seed}{args.tag}"
    mode = {"off": "off", "constrained": "constrained", "pareto": "multi_objective"}[args.arm]
    opt = Optimizer(
        db_name=name,
        data=data,
        study_name=name,
        model_types=args.model_types,
        sampler="tpe",
        sampler_seed=args.seed,
        pruner="none" if mode == "multi_objective" else "asha",
        max_steps=args.max_steps,
        convergence_interval=args.convergence_interval,
        fairness_mode=mode,
        fairness_metric="equal_opportunity_difference",
        fairness_threshold=args.threshold if mode != "off" else None,
        sensitive_val=sensitive_val if mode != "off" else None,
        # The disparity is measured at the default cutoff; keep F1 there too so
        # both objectives describe the same classifier.
        tune_decision_threshold=False,
    )
    t0 = time.perf_counter()
    opt.optimize(n_trials=args.trials)
    wall = time.perf_counter() - t0
    meta = {**vars(args), "mode": mode, "wall_clock_s": wall, "study": name}
    export(opt.study, RESULTS / f"{name}.json", meta)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
