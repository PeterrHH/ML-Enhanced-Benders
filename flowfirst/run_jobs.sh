#!/usr/bin/env bash
# Run several training jobs concurrently: one job per line of a jobs file
# (or stdin), each line being the arguments for `python -m flowfirst.train`.
# Lines that are empty or start with # are skipped. Console output is
# prefixed with the job's --tag (or its line number); the clean per-run log
# is flowfirst/runs/<variant>[-tag]/train.log.
#     flowfirst/run_jobs.sh flowfirst/jobs/jobs-long.txt
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
jobs_file="${1:-/dev/stdin}"

pids=(); n=0
while IFS= read -r line || [ -n "$line" ]; do
  [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
  n=$((n + 1))
  tag=$(sed -n 's/.*--tag[= ]\([^ ]*\).*/\1/p' <<< "$line"); [ -z "$tag" ] && tag="job$n"
  # shellcheck disable=SC2086
  # PYTHON=... to point at the interpreter, e.g. a cluster module's python instead of the local .venv
  "${PYTHON:-.venv/bin/python}" -m flowfirst.train $line 2>&1 | sed -u "s/^/[$tag] /" &
  pids+=($!)
done < "$jobs_file"

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if [ "$status" -eq 0 ]; then
  echo "all $n jobs finished. inspect with: tensorboard --logdir flowfirst/runs"
else
  echo "at least one job failed, check the output above"
fi
exit "$status"
