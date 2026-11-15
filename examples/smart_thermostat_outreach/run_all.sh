#!/usr/bin/env bash
# Reproduce the connected-thermostat experiment end to end.
#   PYTHON=/path/to/venv/bin/python examples/smart_thermostat_outreach/run_all.sh
# The venv needs `pip install "quoptuna>=1.1.0"`. Each search process is pinned to
# two CPU threads so the six arm x seed runs share a 12-core machine without
# contending for cores; wall-clock times are then comparable across arms.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
TRIALS="${TRIALS:-60}"
SEEDS="${SEEDS:-42 7}"

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=2"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2

"$PY" prepare_data.py
LOGS=results/runs/logs
mkdir -p "$LOGS"
pids=()
for seed in $SEEDS; do
  for arm in off constrained pareto; do
    "$PY" run_search.py --arm "$arm" --seed "$seed" --trials "$TRIALS" \
      >"$LOGS/${arm}_s${seed}.log" 2>&1 &
    pids+=($!)
  done
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" audit.py
