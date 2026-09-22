# flowfirst

Ben's flow-first experiments, kept in one folder so they are separable from
the rest of the repository. Everything reuses the existing problem class
(`GEPOperationalProblemSet`) for instances, constraint matrices and Gurobi
labels, and nothing here touches the PDL trainer.

| Path | Purpose |
|---|---|
| `dataset.py` | build or load the labelled ED dataset (`python -m flowfirst.dataset`) |
| `fill.py` | re-export only: the merit-order fill and the flow-first networks now live in `networks.py`, next to the thesis primal, so `gep_benders.py` can import them |
| `train.py` | primal-only training of the four variants with TensorBoard logging |
| `dual.py` | `PrimalDual`: input to primal, nodal prices, box multipliers and certificate in one call (`exact=True` finishes with the min-cost flow); `GraphedPrimalDual` replays the non-exact pipeline as one CUDA graph at a fixed batch; `DualRecovery`, `Polish`, `PathPolish`, `cast_net`, `load_run` |
| `dual_analysis.py` | prints the tables of F26 to F29 for a saved run |
| `polish_compare.py` | raw against polished gap and exact-step augmentations for several saved runs (F32) |
| `configs/` | dataset definitions: 3 nodes with 2 or 3 generators per node, 6 nodes with all generators, and the full instance (20 countries, 107 generators, 44 lines); `-x8` is the 262k-instance sampler |
| `jobs/`, `run_jobs.sh`, `run_all.sh` | the job lists used so far, the concurrent job runner, and a launcher for three variants at once |
| `modal/train.py`, `modal/sync.sh` | run a jobs file on Modal GPUs, one container per job; pull the run directories back while they run |
| `modal/bench.py` | times the deployed pipeline of a saved run on a Modal GPU: eager, CUDA graph, compiled network, both, per precision (float64, TF32, network in float32 or half precision with the rest in float64; `--precisions`), batch 1024 and 8192, with the accuracy of each precision against the Gurobi labels (`--stages` adds the per-stage table) |
| `tests/` | checks of the fill, the dual, the certificate and the polish against Gurobi (`pytest flowfirst`) |
| `docs/OVERVIEW.md` | what flowfirst does and why: the major method changes, each with the finding behind it, and how they map onto the main pipeline — start here |
| `docs/BENCHMARKING.md` | how to make the Gurobi baseline in the Benders benchmark realistic (handoff note for Peter) |
| `docs/FINDINGS.md` | numbered findings with evidence tables |
| `docs/IDEAS.md` | prioritized backlog of training and architecture changes |
| `docs/RELATED_WORK.md` | prior art per topic, checked against primary sources, with what appears to be new and what must be cited |
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
flowfirst/run_all.sh --config flowfirst/configs/config-3node-3gen.json --tag 3gen   # three generators per node
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

## Running on a cluster

`train.py` and `dataset.py` take the same root flags as `main.py`
(`--home-path`, or `--data-root` / `--output-root` separately; `PDL_HOME`,
`DATA_ROOT` and `OUTPUT_ROOT` work too). With a root, the dataset goes to
`<data-root>/data/flowfirst/` and the run directories to
`<output-root>/outputs/FlowFirst/ED/N<nodes>_G<generators>/`, beside the PDL
runs. Without one, everything stays inside the package as before.

Run from the repo root, inside your usual sbatch script:

```
python -m flowfirst.dataset flowfirst/configs/config-6node-x8.json --home-path $SCRATCH/<project>
python -m flowfirst.train --variant flowfirst-gnn --config flowfirst/configs/config-6node-x8.json \
    --gnn-hidden 96 --gnn-rounds 4 --batch-size 128 --lr-schedule step --lr-decay 0.7 --clip-grad 8000 \
    --epochs 250 --home-path $SCRATCH/<project> --device auto
```

`--device auto` picks CUDA when the node has one and the CPU otherwise. Only
the `flowfirst-gnn` variant runs on CUDA. For several jobs at once,
`PYTHON=python flowfirst/run_jobs.sh flowfirst/jobs/<file>.txt`.

## Running on Modal GPUs

`modal/train.py` runs any jobs file on Modal, one GPU container per job. Once:
`.venv/bin/modal setup`, create the volumes with `modal volume create flowfirst-datasets`
and `modal volume create flowfirst-runs`, then upload the dataset to the first one
with `modal volume put` (see the file's docstring). Then

```
FLOWFIRST_GPU=A100 .venv/bin/modal run --detach flowfirst/modal/train.py --jobs flowfirst/jobs/jobs-20node-modal.txt
.venv/bin/modal volume get --force flowfirst-runs / flowfirst/runs/      # fetch run directories, also mid-run
```

`--device cuda` is added automatically; the GPU type is `FLOWFIRST_GPU`
(default A10G; use A100, the training is float64). Always `modal run --detach`.
Only the `flowfirst-gnn` variant runs on CUDA.

## Training on Benders-harvested instances

Every dataset behind F1 to F32 comes from the node-budget capacity sampler. In deployment
the capacities are whatever the master proposes, starting at `u = 0` (IDEAS I9). With
`ED_args.use_direct_data` — the same key `main.py` uses — `dataset.py` samples nothing and
loads `direct_data_path` as is, which is how the `gen_GEP/` harvests enter: 81 perturbed
3-node GEP instances solved by exact Benders, the master's investment trajectory paired with
representative hours, deduplicated to 48k Gurobi-labelled ED states.

```
.venv/bin/python -m flowfirst.dataset flowfirst/configs/config-3node-harvest.json
.venv/bin/python -m flowfirst.train --variant flowfirst       --config flowfirst/configs/config-3node-harvest.json --epochs 250 --tag harvest
.venv/bin/python -m flowfirst.train --variant old-prioritized --config flowfirst/configs/config-3node-harvest.json --epochs 250 --tag harvest
```

Defaults throughout, matching `runs/flowfirst-3` and `runs/old-prioritized`, so the dataset
is the only variable. `compare_harvest.ipynb` compares them: curves, per-instance gaps,
the dispatch/VOLL split, the polish ladder and the certificate.

Validation (4800 instances, seed 0, one run each), flow-first first: ratio of totals 0.00895
against 0.00877, median gap 0.0188 against 0.0145, within 1 % 0.431 against 0.454, trimmed
mean 0.62 against 0.58. The two are within 3 % of each other on every metric and the thesis
primal is marginally ahead — the separation of F17 is gone, as F1 predicts at two generators
per node.

Ignore the mean per-instance gap here (1.86 against 2.62, the one metric flow-first leads).
The harvest's optima span 2.7 to 1.9e6, so F5's denominator problem is extreme: mean absolute
error is ~650 for both models, which is 2 % of a median optimum and 250x the cheapest one.
Ten instances out of 4800 make up half the mean, the top 1 % make up two thirds, and that is
what the jagged curve in the notebook's second panel is tracking. Read the ratio of totals,
the median and the share within 1 %.

The level is well below F17/F18 (43 % within 1 %) on a harder distribution: 23 % of instances
short, every one with a saturated line. One polish sweep divides the mean gap by 70 to 75 and takes the median to
zero, against 8 to 15 in F32 — on Benders states the network routes and the exact local
moves supply nearly all the precision. The warm start saves little (2.24 augmentations
against 2.81 cold), but three nodes leave almost nothing to route.

This does not yet say whether harvest-training beats sampler-training for deployment: the
harvest is a different 3-node system from `config-3node.json` (GER Gas/SunPV, FRA
Nuclear/SunPV, two cost ties), so neither model can be scored on the other's data. The 2x2
needs a sampler dataset on the harvest's own system; notebook section 7 has the recipe and a
`cross_evaluate` that asserts on mismatched generators instead of returning a meaningless
number.
