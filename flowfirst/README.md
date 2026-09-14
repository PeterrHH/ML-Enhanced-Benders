# flowfirst

Ben's flow-first experiments, kept in one folder so they are separable from
the rest of the repository. Everything reuses the existing problem class
(`GEPOperationalProblemSet`) for instances, constraint matrices and Gurobi
labels, and nothing here touches the PDL trainer.

| File | Purpose |
|---|---|
| `config-3node.json` | copy of `configs/config.json` with 3 nodes, 2 generators each, six distinct variable costs, node-budget capacity sampler |
| `config-3node-3gen.json` | same, with 3 generators per node (one renewable, two dispatchable), nine distinct costs; FRA Gas set to 0.07 via `var_cost_override` |
| `config-6node-x8.json`, `config-20node-x8.json` | six nodes with all generators, and the full instance (20 countries, 107 generators, 44 lines); `-x8` is the 262k-instance sampler |
| `dataset.py` | build or load the labelled ED dataset (`python -m flowfirst.dataset`) |
| `fill.py` | merit-order fill: production and unmet demand from the flows, plus explicit nodal prices |
| `test_fill.py` | checks of the fill against Gurobi (`pytest flowfirst`) |
| `train.py` | primal-only training of the three variants with TensorBoard logging |
| `dual.py` | `PrimalDual`: input to primal, nodal prices, box multipliers and certificate in one call (`exact=True` finishes with the min-cost flow); `GraphedPrimalDual` replays the non-exact pipeline as one CUDA graph at a fixed batch; `DualRecovery`, `Polish`, `PathPolish`, `load_run` |
| `test_dual.py` | checks of the dual, the certificate and the polish against Gurobi (`pytest flowfirst`) |
| `dual_analysis.py` | prints the tables of F26 to F29 for a saved run |
| `polish_compare.py` | raw against polished gap and exact-step augmentations for several saved runs (F32) |
| `modal_bench.py` | times the deployed pipeline of a saved run on a Modal GPU: eager, CUDA graph, compiled network, both, per precision (float64, TF32, network in float32 or half precision with the rest in float64; `--precisions`), batch 1024 and 8192, with the accuracy of each precision against the Gurobi labels (`--stages` adds the per-stage table) |
| `run_all.sh` | launch the three variants concurrently |
| `modal_train.py`, `modal_sync.sh` | run a jobs file on Modal GPUs, one container per job; pull the run directories back while they run |
| `FINDINGS.md` | numbered findings with evidence tables |
| `IDEAS.md` | prioritized backlog of training and architecture changes |
| `RELATED_WORK.md` | prior art per topic, checked against primary sources, with what appears to be new and what must be cited |
| `run_jobs.sh`, `jobs-*.txt` | concurrent job runner and the job lists used so far |
| `datasets/`, `runs/` | generated, git-ignored |

## Setup (once)

From the repo root `ML-Enhanced-Benders/`:

```
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install "setuptools<81" pytest     # tensorboard 2.18 needs pkg_resources
```

Gurobi is only needed to label a dataset. The academic license in
`~/gurobi.lic` expired 2025-11-20; the size-limited license bundled with
gurobipy is enough for the 12-variable ED instances:

```
export GRB_LICENSE_FILE=.venv/lib/python3.12/site-packages/gurobipy/.libs/gurobi.lic
.venv/bin/python -m flowfirst.dataset
```

## Run

```
.venv/bin/pytest flowfirst                                   # fill checks, both datasets
.venv/bin/python -m flowfirst.train --variant flowfirst      # one run, see --help
flowfirst/run_all.sh [--epochs 200 --tag long]               # all three variants concurrently
flowfirst/run_all.sh --config flowfirst/config-3node-3gen.json --tag 3gen   # three generators per node
.venv/bin/tensorboard --logdir flowfirst/runs                # http://localhost:6006
```

Variants: `old-prioritized` (Peter's primal net with bounds repair,
generation-prioritized rescale and completion), `old-nocompletion` (p, f and
md all predicted and box-repaired, L1 balance penalty), `flowfirst` (flows
predicted, merit-order fill). Each run writes `train.log`, `metrics.csv`,
`args.json`, `model.pt` (final), `model_best.pt` (best validation gap),
`summary.json` and the TensorBoard events to `runs/<variant>[-tag]/`.

`--loss-norm opt` divides each instance's objective by Gurobi's optimum so
the training loss equals the mean relative gap and shortage instances no
longer dominate by size; `--loss-norm log` does the same without labels via
the log of the objective; default `none` is the raw mean objective.

Why two datasets: with two units per node the prioritized rescale reproduces
merit order once the production head saturates the cheap unit, so the old
architecture and flow-first compute the same function. With three units the
rescale splits the remainder across two unsaturated units by headroom, not by
cost, and the production head has to learn the residual itself.

## Running on Modal GPUs

`modal_train.py` runs any jobs file on Modal, one GPU container per job. Once:
`.venv/bin/modal setup`, create the volumes with `modal volume create flowfirst-datasets`
and `modal volume create flowfirst-runs`, then upload the dataset to the first one
with `modal volume put` (see the file's docstring). Then

```
FLOWFIRST_GPU=A100 .venv/bin/modal run --detach flowfirst/modal_train.py --jobs flowfirst/jobs-20node-modal.txt
.venv/bin/modal volume get --force flowfirst-runs / flowfirst/runs/      # fetch run directories, also mid-run
```

`--device cuda` is added automatically; the GPU type is `FLOWFIRST_GPU`
(default A10G; use A100, the training is float64). Always `modal run --detach`.
Only the `flowfirst-gnn` variant runs on CUDA.
