# Accelerating Generation Expansion Planning with ML-Enhanced Benders Decomposition

The master thesis can be accessed at:

https://repository.tudelft.nl/record/uuid:18f3745d-38ba-4aba-855b-a60f27d03c39

---

## Overview

We solve a **Generation Expansion Planning (GEP)** problem with **Benders decomposition**:

- **Master:** chooses integer investments: the number of units built per candidate generator.
- **Subproblems:** price the operation of those investments, as economic-dispatch (**ED**) linear programs: one per hour, or one per ramping block (see below).

A learned **surrogate** answers the subproblems quickly in the first iterations, and exact LP solves finish the run (*exact refinement*).

```
Stage 1  Train the surrogate
  configs/config*.json  ──>  main.py              (primal-dual learning, PDL)
                        ──>  flowfirst/train.py   (flow-first GNN, optional)
                             └─> trained nets in outputs/

Stage 1b (optional) Build training data from Benders trajectories
  gen_GEP/create_GEP_for_training.py  ──>  perturbed GEP instances
  gen_GEP/solve_for_train.py          ──>  exact Benders per instance, harvest

Stage 2  Solve the GEP
  configs/config*.json  ──>  gep_benders.py
                             ├─> Exact Benders            (all subproblems by LP)
                             ├─> Inexact Benders + refine (surrogate, then LP)
                             └─> Direct solve             (the full MIP, no decomposition)
```

The repository also contains **QP** benchmarks inherited from earlier work. They are supported in `main.py` but are not the focus of this thesis.

### Current direction: GEP with ramping

Ramp limits are being added to the GEP: within each **block of B consecutive hours**, a generator may change its output by at most `rate_g · UnitCap_g · u_g` per hour. Each generator's ramp rate is the `ramping_rate` column of `inputs/iGEP_data_generation.csv`. With ramping, one subproblem covers a block of B hours instead of a single hour.

| Part | Status |
|---|---|
| Exact Benders and the direct solve with ramping (`--ramp-block-hours B`) | done; checked against the direct MIP within 1e-4 on every cut mode |
| Training instances for ramping (`gen_GEP`, `--ramp-block-hours B`) | done: instance generation and the Benders solves (trajectories) |
| Block harvest (labelled block data for training) | next |
| Ramp-aware surrogate (time-expanded graph network) | in design |

Design notes: `docs/ramping-blocks-plan.md`.

---

## Prerequisites

### Python environment

```bash
conda create -n {env_name} python=3.9.20      # 3.9.20 was used for the thesis experiments
conda activate {env_name}
pip install -r requirements.txt
```

### Solvers

Every optimisation model is built in **Pyomo** (`solver_backend.py`) and solved by **Gurobi** or **HiGHS** (`--solver gurobi|highs`, default `gurobi`).
- **Gurobi** needs a licence. The size-limited licence bundled with `pip install gurobipy` only handles the smallest models.
- **HiGHS** needs no licence.

---

## Configuration files

Two configuration layers control each experiment, both in `configs/`.

### `config.toml`: raw inputs
Points to the energy-system data in `inputs/` (demand, availability, generators, lines, scalars). You normally do not edit it.
- `inputs/scalars.toml` holds VOLL and `ramping_value`, the fallback ramp rate for generator data without a `ramping_rate` column.
- `config.toml`'s `ramping` switch only affects `gep_exact_solver.py`, which ramps over the whole horizon. The Benders pipeline ramps only when asked with `--ramp-block-hours`.

### `config*.json`: experiment settings, one per network

| File | Network |
|---|---|
| `config.json` | 3 nodes (BEL, GER, FRA) |
| `config-4node.json`, `config-5node.json`, `config-6node.json` | 4, 5, 6 nodes |
| `config-6node-pdlmatch.json`, `config-12node-pdlmatch.json`, `config-20node.json` | 6, 12, 20 nodes |

| Section | Controls |
|---|---|
| `ED_args` | nodes, generators and lines of the dispatch problem; dataset generation; training repeats |
| `Benders_args` | the GEP topology; Benders mode, cut selection, solver; trained-net paths; output directory |
| top-level keys | PDL hyper-parameters (`rho`, `alpha`, `tau`, learning rates, batch size, …), data split, Optuna |

Every `Benders_args` key below can be overridden on the command line.

---

## Stage 1: train the surrogate (`main.py`)

1. Reads the JSON config given by `-c/--config` (default `configs/config.json`).
2. Builds (or loads) the dataset under `data/`. With `ED_args.use_direct_data`, it loads a harvested dataset instead (Stage 1b).
3. Trains the primal and dual networks with primal-dual learning (PDL).
4. Writes weights, metrics and TensorBoard logs to `outputs/PDL/<problem_type>/<run_name>/repeat:<n>/`:
   ```
   args.json            config of the run
   primal_weights.pth   primal network
   dual_weights.pth     dual network
   train_time.txt       wall-clock training time
   events.out.*         TensorBoard log
   ```

```bash
python main.py                                  # 3 nodes (default config)
python main.py -c configs/config-5node.json     # 5 nodes
```

### Where data and outputs go
With no flags, both `data/` and `outputs/` are in the current directory:

```bash
python main.py --home-path "$SCRATCH/thesis"                                  # both under scratch
python main.py --data-root . --output-root "$SCRATCH/runs/$SLURM_JOB_ID"      # shared data, per-job outputs
```

`configs/` and `inputs/` always resolve against the repository, so these commands work from any directory. Order of precedence, highest first: `--data-root`/`--output-root`, `--home-path`, the config's `data_root`/`output_root`, `$DATA_ROOT`/`$OUTPUT_ROOT`, the config's `home_path`, `$PDL_HOME`, then `.`.

### Device
The configs ship with `"device": "auto"`:

| Machine | `auto` resolves to | Precision |
|---|---|---|
| Laptop without CUDA | `cpu` | float64 |
| GPU node | `cuda` | float32 |
| CPU-only cluster node | `cpu` | float64 |

GPU and CPU results are therefore not bit-comparable.
- **`auto` never picks MPS** (Apple GPU), because that would silently drop a local run to float32. Ask for it with `--device mps`.
- **An unavailable device raises an error** instead of falling back to the CPU.
- **`args.json` records** the device actually used.

On a node without network access, log W&B offline and sync later:

```bash
python main.py --home-path "$SCRATCH/thesis" --wandb-mode offline --wandb-group "$SLURM_JOB_ID"
wandb sync "$SCRATCH/thesis/wandb/offline-run-"*      # later, from a login node
```

### Key training options

| Key | Effect |
|---|---|
| `learn_primal`, `learn_dual` | train the primal / dual network |
| `ED_args.repeats` | number of independent training runs |
| `use_heuristic_lambda_loss`, `heuristic_lambda_weight` | SLA in dual training, and its weight β |

### Flow-first networks (`flowfirst/`)
A second surrogate family. The network predicts only the line flows, the dispatch is computed exactly by merit order, and the dual is built from the primal. See `flowfirst/README.md`.

If `primal_net_directory` points at a flow-first run (one with `model.pt`), `gep_benders.py` detects it and builds the dual from the flows; no dual checkpoint is read. Related flags: `--flowfirst-weights model.pt|model_best.pt|model_ema.pt` and `--flowfirst-polish-sweeps N` (default 1).

---

## Stage 2: Benders decomposition (`gep_benders.py`)

### Modes

| Mode | How to select | What solves the subproblems |
|---|---|---|
| Exact Benders | `--benders-setup Exact` | the LP solver, every iteration |
| Inexact Benders with refinement | `--benders-setup Inexact_Refine` | the surrogate until the gap switch (see below), then the LP solver |
| Inexact Benders | `--benders-setup Inexact` | the surrogate only |
| All three | `--benders-setup All` | runs Exact, Inexact and Inexact_Refine one after another |
| Direct solve | `-s` / `--solve-direct` | the full GEP as one MIP, no decomposition |

### The algorithm
- The year is split into samples of `sample_duration` hours. Each sample is its own GEP, solved independently.
- **Master:** integer investments u plus one recourse variable per cut group, with the cuts collected so far.
- **Subproblems:**
  - Without ramping: one dispatch LP per hour.
  - With ramping: one LP per block of B hours.
  - One persistent model is re-solved with new right-hand sides.
- **Stopping rule:** the run stops when `(UB* − LB) / max(1, |UB*|) ≤ 1e-4`, the same criterion as the direct solve's MIP gap.
  - LB is the master's bound.
  - UB* is the best integer, exactly evaluated plan so far (investment cost plus subproblem costs).
  - Results report UB* and its investment.
- **Thesis algorithm:** the code follows "Inexact Benders with exact refinement, LP warm start and gap switch" in the thesis. Master gap ε_M ≤ ε; no LP-iteration cap.
- **Limits:** one hour of wall-clock time per sample. The master's own time limit is the time remaining.

### LP phase (on by default)
For the first iterations the master's investments are **continuous**, so each master is a cheap LP. When the relaxed problem's gap drops below **1e-3**, the investments become integer and **all cuts are kept**.
- This is safe: every cut is valid for any u, and the LP master's bound is a valid lower bound.
- During the LP phase, the stopping tests wait for the integer phase, because an upper bound at a fractional u is not a buildable investment. The gap switch to exact refinement does not wait (see below).
- The phase also ends when the LP master repeats an investment. There is no iteration cap.
- On 3 nodes it cut master solver time roughly in half for `full` cuts and by about 70 % for `kmeans` cuts. `single` cuts converge slowly even as an LP.

| Flag / key | Effect |
|---|---|
| `--lp-phase-tol 1e-3` / `lp_phase_tol` | gap of the relaxed problem at which the master turns integer (default 1e-3) |
| `--no-lp-phase` / `"lp_phase_tol": null` | integer master from the first iteration |

The iteration log marks LP-phase iterations in the `lp_phase` column. `gen_GEP/solve_for_train.py` does not use the LP phase, so harvested trajectories stay integer.

### When the inexact phase ends
**Default: on a gap tolerance.** Exact refinement starts as soon as the inexact UB−LB gap is below **1e-2** (`--refine-gap-tol`, key `refine_gap_tol`; `0` turns it off).
- **Why:** surrogate cuts cannot close the gap below the surrogate's own error. On 12 nodes that error is 0.4–0.8 %, above the LP-phase tolerance, so the inexact gap stalls there. Waiting for a repeated investment instead spent integer masters of 50–110 s each that raised the lower bound by only 0.002 %, plus one more master just to detect the repeat.
- **The check also runs during the LP phase.** After the switch the LP phase continues on exact cuts and ends as usual.
- **Fallback:** if the LP phase would end while still inexact (its gap below 1e-3 or a repeated investment), the run switches to exact first. So the integer master never runs on surrogate cuts.
- **Without the tolerance**, exact refinement starts only when the master proposes **exactly** the same investment twice. With surrogate cuts the master often keeps moving a few units back and forth, so that can take very long.

Two more optional triggers are off by default and inactive during the LP phase:

| Flag | Ends the inexact phase when |
|---|---|
| `--refine-gap-tol 1e-2` (default) | the inexact UB−LB gap is below 1 % (also in the LP phase) |
| `--refine-change-tol 0.01` | the investment moved by less than 1 % of built capacity (MW, summed over generators) since the last iteration |
| `--refine-stall-iters 5` (`--refine-stall-tol 1e-4`) | the lower bound rose by less than 1e-4 over the last 5 iterations, counted only once the inexact UB−LB gap is below 1 % |

Every trigger switches to exact refinement. The change and stall triggers stop the run instead in pure `Inexact` mode; the gap tolerance does nothing there. The result rows record what ended the phase in `refine_reason`, and the tolerance in `refine_gap_tol`.

### One solver profile for every model
Every model in a run uses the same solver and the same parameters, so the direct solve and Benders compare like for like:
- **Applies to:** the direct solve, the master, the subproblem LPs, and the gap-gate re-solves.
- **One solver per run:** `--solver gurobi|highs`.
- **Parameters:** 1 thread, seed 0, primal and dual feasibility tolerances 1e-9.
- **Gaps:**
  - Direct solve: MIP gap **1e-4**, the same tolerance Benders certifies.
  - Benders master: MIP gap **1e-5** (`--master-mip-gap`; at most 1e-4).
  - Why up to 1e-4 is enough: the master only has to be ε-optimal (Geoffrion & Graves 1974), and LB is its bound, valid at any gap. At the master's plan x, UB − LB = cut error at x + master gap.
  - The default (1e-4 vs 5e-5) is to be set from the 12-node test.
- **Precision:** all matrices are built in float64.

### Timing
- **`total_time`** is time spent *inside the solver's own optimize/run calls*. For Benders that is master plus subproblems, plus surrogate inference on inexact iterations; for the direct solve, its MIP solve.
- **`wall_time`** covers the whole call, including Pyomo model building, cut building and clustering.
- **`build_time`** (direct solve only) is the Pyomo model build.

Compare `total_time` with `total_time` and `wall_time` with `wall_time`.

### Cut selection

| `cut_selection` | Cuts per iteration | Notes |
|---|---|---|
| `single` | one cut over all hours or blocks | small master, slow convergence |
| `full` | one cut per hour or block | most information; the master grows fastest |
| `kmeans` (`cut_selection_k`) | one cut per cluster of hours or blocks, clustered once on demand and available capacity | balances the two |
| `kmeans_dynamic` | re-clustered every iteration; one recourse variable per hour or block | features from `dynamic_cluster_features` |
| `kmeans_dynamic_shared` | re-clustered every iteration; one recourse variable per cluster, all past cuts re-aggregated | features from `dynamic_cluster_features` |

`dynamic_cluster_features` is `price` (nodal prices), `slope` (cut slopes) or `capacity` (demand and installed capacity at the current investment).

**Gap gate** (inexact iterations only, off by default): `--gap-gate-threshold 0.01` flags subproblems whose surrogate duality gap exceeds 1 %. `--gap-gate-action resolve` solves them exactly; `drop` leaves them out of the cut.

### Ramping inside blocks

```bash
python gep_benders.py -c configs/config.json --benders-setup Exact --sample-duration 876 --ramp-block-hours 4
python gep_benders.py -c configs/config.json -s --sample-duration 876 --ramp-block-hours 4     # direct reference
```

- **`--ramp-block-hours B`** (or `Benders_args.ramp_block_hours`) turns on ramp limits between consecutive hours inside blocks of B hours. There are never ramp limits across blocks, so blocks remain independent subproblems.
- **Ramp rates** come from the `ramping_rate` column of `inputs/iGEP_data_generation.csv`, per technology and country (falling back to `ramping_value` in `inputs/scalars.toml`). They are re-read on every run, so cached datasets never hold stale rates. A generator without a rate stops a ramping run with an error that names it.
- **The sample length** must divide 8760 and be a multiple of B: 24, 120, 876 (B = 4 or 6), 1752 or 8760 h.
- **Ramping works only with `Exact` and the direct solve.** The current surrogates predict single hours.
- **Results** go to `outputs/Benders/<N>Node_Block<B>/`, laid out like `<N>Node/` with the same file names.

### Running

```bash
python gep_benders.py -c configs/config.json                                     # mode and nets from Benders_args
python gep_benders.py -c configs/config.json --benders-setup Exact --cut-selection kmeans --cut-selection-k 8
python gep_benders.py -c configs/config.json --solver highs                      # HiGHS instead of Gurobi
python gep_benders.py -c configs/config.json -s                                  # direct MIP
```

Main overrides:
- `--sample-duration`, `--cut-selection`, `--cut-selection-k`
- `--specific-name` (the label in output names)
- `--primal-net-dir` / `--dual-net-dir` (trained nets, resolved under the output root)
- `--device` (inference device, independent of where the nets were trained)
- the same `--home-path` / `--data-root` / `--output-root` flags as `main.py`

`python gep_benders.py --help` lists everything.

### Outputs

| Path | Contents |
|---|---|
| `outputs/Benders/<N>Node[_Block<B>]/Sample_<dur>/experiment_data_*_<specific_name>.csv` | one row per sample: bounds, iterations, `total_time` and its parts, `wall_time`, investments, and the tolerance profile used |
| `outputs/Benders/<N>Node[_Block<B>]/Sample_<dur>/iter_logs_<setup>_<specific_name>/iterlog_*.csv` | one row per iteration: UB, LB, gap, times, master status/bound/gap/nodes, cut statistics, `cut_tight_rel`, `lp_phase`, investment |
| `outputs/Benders/<N>Node[_Block<B>]/Sample_<dur>/Gurobi_Solution.csv` (`highs_Solution.csv`) | the direct solve |

HiGHS runs get a `_highs` suffix, so they never overwrite Gurobi results.

### Debugging a slow master

```bash
python gep_benders.py -c configs/config-12node-pdlmatch.json --solver-log-dir solver_logs --dump-master
```

- `solver_logs/<setup>/master_sample<s>_iter<i>.log` is the solver's own log for each master solve, written live. A stalled master is the last file.
- `--dump-master` writes each master as an `.lp` file to replay offline with `gurobi_cl` or `highs`.
- `--solver-tee` also prints the solver log to the console.
- The iteration log's `cut_coef_min/max` and `cut_rhs_max` show the cuts' coefficient range; a widening range is a common reason a master slows down.
- `cut_tight_rel` checks the exact cuts: each cut at the current investment against the subproblem optimum. It should stay near 1e-14.

### Trained nets
`Benders_args.primal_net_directory` and `dual_net_directory` point at a `main.py` run folder, `outputs/PDL/ED/<run_name>/repeat:<n>`. Both may be the same folder: weights are loaded by file name.
- **Nets are tied to their network:** a 3-node model cannot be used for a 6-node run.
- **The thesis models** are in `experiment-output/ch7/`, e.g. `experiment-output/ch7/3nodes/primal_model` and `…/dual_model`.

---

## Stage 1b: training data from Benders trajectories (`gen_GEP/`)

Instead of sampling investments from a fixed range, this pipeline harvests the investments that **exact Benders actually visits**, so the training data matches what the deployed solver will ask.

```
gen_GEP/create_GEP_for_training.py  →  data/GEP_perturb_mix_<topology>_H<h>[_B<B>]/  perturbed GEP instances
gen_GEP/solve_for_train.py          →  data/ED_data_gen/                             trajectories, or an hourly ED dataset
```

1. **`create_GEP_for_training.py`** builds perturbed GEP instances from real demand and availability, perturbing demand level, investment costs and (lightly) availability. A class mixture covers renewable-heavy, high-demand and middle optima (`USE_CLASS_MIXTURE`, `RENEWABLE_FRAC`).
2. **`solve_for_train.py`** solves each instance with exact Benders, keeps the master's investment trajectory, and pairs investments with the instance's hours to form labelled dispatch problems.

| Script | Flag | Effect |
|---|---|---|
| `create_GEP_for_training.py` | `-n`, `--horizon`, `--seed` | number of instances, hours per instance, seed |
| | `--ramp-block-hours B` | draw hours as whole windows of B consecutive hours, with one noise draw per window; the horizon must be a multiple of B (e.g. 216 or 876) |
| `solve_for_train.py` | `-j/--workers`, `--threads` | worker processes (default: the job's CPUs) and threads per worker |
| | `--check-direct` | also solve each instance as one MIP and print both objectives |
| | `--cut-selection`, `--cut-selection-k` | cut mode of the harvesting solves (default `single`: slow convergence, widely spread trajectories) |
| | `--ramp-block-hours B --solve-only` | solve ramping instances and save their trajectories to `data/ED_data_gen/GEP_trajectories_<topology>_H<h>_B<B>_<cut>.pkl`; the block harvest is not built yet |

```bash
# hourly (as before)
python gen_GEP/create_GEP_for_training.py
python gen_GEP/solve_for_train.py

# ramping: instances in 4-hour windows, then exact Benders with ramping
python gen_GEP/create_GEP_for_training.py --horizon 876 --ramp-block-hours 4 -n 80
python gen_GEP/solve_for_train.py --ramp-block-hours 4 --solve-only --check-direct
```

To train on a harvested hourly dataset, point `main.py` at it:

```jsonc
"ED_args": { "use_direct_data": true, "direct_data_path": "data/ED_data_gen/<harvested_dataset>.pkl" }
```

> **Leakage:** with `POOL_TIMES = None` (the default in `create_GEP_for_training.py`), instances draw from every hour of the year, including any hours you later evaluate on. Restrict `POOL_TIMES` if you need held-out hours.

---

## Other scripts

- **`Cut_selection_experiment.py`** compares cut-aggregation strategies on one GEP sample (calls gurobipy directly):
  ```bash
  python Cut_selection_experiment.py --gep-data-path <gep.pkl> --ed-data-path <ed.pkl> --sample 0 --k 6
  ```
- **`gep_exact_solver.py`** is the original Pyomo GEP model over the full horizon. Its `ramping` switch in `config.toml` ramps across all hours.

---

## Notes

- `main.py`, `gep_benders.py` and the `gen_GEP` scripts can run from any working directory. Other scripts expect the repository root.
- Dataset and output names encode the experiment settings, so different configurations never overwrite each other.
- **Datasets are cached.** A cached dataset is reused when its file name matches, even if `inputs/` changed since (ramp rates excepted: they are re-read every run). Delete the cache to rebuild it.
- This is research code: test with a short `sample_duration` (e.g. 24) before a full 8760-hour run.
