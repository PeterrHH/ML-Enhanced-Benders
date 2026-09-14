#!/usr/bin/env bash
# Pull the run directories from the Modal volume into flowfirst/runs every few minutes, so a local
# TensorBoard (tensorboard --logdir flowfirst/runs) follows jobs while they run. The container commits
# the volume every 5 minutes, so the curves lag by at most that. Ctrl-C to stop.
#     flowfirst/modal/sync.sh            # everything on the volume
#     flowfirst/modal/sync.sh <run-dir>  # one run directory only, e.g. flowfirst-gnn-20node-x8-b128-gnn128r6ln-step-clip-ema-b200
cd "$(dirname "$0")/../.."
remote="/${1:-}"
while true; do
  .venv/bin/modal volume get --force flowfirst-runs "$remote" flowfirst/runs/ >/dev/null 2>&1 && echo "$(date +%H:%M:%S) synced $remote" || echo "$(date +%H:%M:%S) sync failed (volume empty yet?)"
  sleep 300
done
