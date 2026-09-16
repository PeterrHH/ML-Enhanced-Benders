#!/usr/bin/env bash
# Launch the three variants concurrently, each with a capped thread count.
# Extra arguments are passed to every run, e.g.
#     flowfirst/run_all.sh --loss-norm opt --tag norm-opt
# Console lines are prefixed with the variant; the clean per-run log is
# flowfirst/runs/<variant>[-tag]/train.log.
# Restrict the variants with the VARIANTS environment variable, e.g.
#     VARIANTS="old-prioritized flowfirst" flowfirst/run_all.sh --epochs 500
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
VARIANTS="${VARIANTS:-old-prioritized old-nocompletion flowfirst}"

pids=()
for v in $VARIANTS; do
  .venv/bin/python -m flowfirst.train --variant "$v" "$@" 2>&1 | sed -u "s/^/[$v] /" &
  pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [ "$status" -eq 0 ]; then
  echo "all runs finished. inspect with: tensorboard --logdir flowfirst/runs"
else
  echo "at least one run failed, check the output above"
fi
exit "$status"
