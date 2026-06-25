# Accelerating Generation Expansion Planning with ML-Enhanced Benders Decomposition

The master thesis can be accessed at: 

https://repository.tudelft.nl/record/uuid:18f3745d-38ba-4aba-855b-a60f27d03c39

---

## Overview

The pipeline has two sequential stages:

```
Stage 1 — Train PDL models
  configs/config*.json + config.toml
       └─> main.py
            ├─> auto-generates dataset (data/)
            └─> trains primal & dual networks → outputs/PDL/

Stage 2 — Run Benders decomposition
  configs/config*.json  (with model paths set under Benders_args)
       └─> gep_benders.py
            ├─> Inexact Benders  (neural warm-start)
            ├─> Exact Benders    (Gurobi only)
            └─> Direct solve     (full MIP via Gurobi)
```

The two problem families this thesis contributes to are:
- **ED** — Economic Dispatch (operational, short-term)
- **GEP** — Generation Expansion Planning (investment, long-term)

The codebase also includes **QP** quadratic-programming benchmarks inherited
from prior work by Ben Jacobs (linked below). QP support is present in
`main.py` and the supporting files but is not the focus of this thesis.

---

## Prerequisites

### Python environment

Python 3.9.20 was used for all experiments.

```bash
conda create -n {env_name} python=3.9.20
conda activate {env_name}
pip install -r requirements.txt
```

### Gurobi licence

To fully reproduce experience. You need a working
[Gurobi](https://www.gurobi.com/) installation with an active licence.

---

## Configuration files

Two configuration layers control each experiment. All files are in the 'configs' folder.

### `config.toml` — solver and raw-input settings

Points to the energy-system CSV/TOML files in `inputs/` and selects the
optimisation solver. You normally do not edit this unless you change the solver
or the raw input files.

### `config*.json` — experiment settings (one per network size)

| File | Network |
|---|---|
| `config.json` | 3-node system (BEL / GER / FRA) |
| `config-4node.json` | 4-node system |
| `config-5node.json` | 5-node system |
| `config-6node.json` | 6-node system |

Each JSON file has three main sections:

| Section | Controls |
|---|---|
| `ED_args` | Nodes, generators, lines, dataset generation, training repeats |
| `Benders_args` | Nodes, generators, lines for the GEP problem; Benders method; **trained model paths**; output directory |
| Top-level keys | PDL hyperparameters (`rho`, `alpha`, `tau`, LR, batch size, …), training split, Optuna settings |

---

## Stage 1 — Train PDL models (`main.py`)

### What it does

1. Reads the JSON config specified by `ARGS_FILE_NAME` at the top of the file.
2. Builds (or loads) the ED/GEP/QP dataset and saves it under `data/`.
3. Trains the primal network and dual network using the Primal-Dual Learning
   (PDL) algorithm.
4. Saves model weights, training metrics, and TensorBoard logs to:

```
outputs/PDL/<problem_type>/<run_name>/repeat:<n>/
    args.json           ← copy of the config used
    primal_weights.pth  ← primal network weights
    dual_weights.pth    ← dual network weights
    train_time.txt      ← wall-clock training time
    events.out.*        ← TensorBoard log
```

### How to run

**Step 1.** Open [main.py](main.py) and set `ARGS_FILE_NAME` near the top to
the config file you want:

```python
# main.py, line ~20
ARGS_FILE_NAME = "configs/config.json"        # 3-node  (default)
```

**Step 2.** Run from the repository root:

```bash
python main.py
```

If the dataset for the chosen configuration does not yet exist, `main.py`
generates it automatically before training begins (this can take several
minutes for large configurations).


### Key training options (in `config*.json`)

| Key | Effect |
|---|---|
| `learn_primal` | Train the primal (investment + dispatch) network |
| `learn_dual` | Train the dual (Lagrange multiplier) network |
| `ED_args.repeats` | Number of independent training runs |
| `use_heuristic_lambda_loss` |Whether to use **SLA** in dual training|
|`heuristic_lambda_weight`|Value of $\beta$ in SLA training|


---

## Stage 2 — Benders decomposition (`gep_benders.py`)


Runs Benders decomposition on the GEP problem. 

### Benders Mode
Three modes are available:
| Mode | Flag / config | Subproblem solver |
|---|---|---|
| Inexact_Refine | `benders_setup: "Inexact_Refine"` | ML-enhanced Benderes |
| Exact Benders | `benders_setup: "Exact"` | Use Benders Decomposition but subproblems always solve with Gurobi |
| Direct solve | `--solve-direct` or `-s` CLI flag | Solve GEP with Gurobi, no decomposition |



### Cut Management

The Benders cut aggregation strategy is set via `Benders_args.cut_selection`.
Each Benders iteration solves one ED subproblem per timestep; the cut
strategy controls how those per-timestep duals are combined into cuts added
to the master problem.

| `cut_selection` | `cut_selection_k` used? | Description |
|---|---|---|
| `"single"` | No | All timesteps aggregated into **one** Benders cut per iteration. Fastest master solve, but least information per cut. |
| `"kmeans"` | Yes (`k` clusters) | Timesteps clustered by demand + available-capacity features (k-means); one cut per cluster. Balances cut richness and master-problem size. |
| `"full"` | No | One cut per timestep. Most informative, but master problem grows fastest. |

`cut_selection_k` sets the number of groups for `kmeans` ; it is
ignored for `single` and `full`.

### Other key `Benders_args` settings

| Key | Description |
|---|---|
| `primal_net_directory` | Path to the folder containing `primal_weights.pth` and `args.json` for the trained primal network. Must match the node/generator configuration of the current config file. |
| `dual_net_directory` | Path to the folder containing `dual_weights.pth` and `args.json` for the trained dual network. Can point to the same folder as `primal_net_directory` (weights are loaded by filename). |
| `sample_duration` | Length (in hours) of one GEP sample. The full year (8 760 h) is split into `8760 / sample_duration` non-overlapping samples, each solved as an independent Benders problem. Longer samples capture richer seasonal patterns but take more time per solve. |



### How to run

```bash
# Benders decomposition — uses model paths from Benders_args in the JSON
python gep_benders.py --config config.json
python gep_benders.py --config config-4node.json
python gep_benders.py --config config-6node.json

# Solve the full GEP as a single MIP (no Benders, no neural net needed)
python gep_benders.py --config config.json -s
```

All commands must be run from the **repository root**.

### Outputs

| Path | Contents |
|---|---|
| `outputs/Benders/<N>Node/Sample_<dur>/experiment_data_*.csv` | Per-sample summary (iterations, times, bounds, investments) |
| `outputs/Benders/<N>Node/Sample_<dur>/iter_logs_*/iterlog_*.csv` | Per-iteration UB / LB / gap / investment history |

---



### Where to find / put your trained models

After running `main.py`, your trained models land at:

```
outputs/PDL/<problem_type>/<run_name>/repeat:<n>/
    primal_weights.pth
    dual_weights.pth
    args.json
```

**Recommended approach for a fresh training run:**

1. Train with `python main.py` (Stage 1).
2. Identify the output folder, e.g.:
   ```
   outputs/PDL/ED/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-<timestamp>/repeat:0
   ```
3. Update `Benders_args` in your config:
   ```jsonc
   "primal_net_directory": "outputs/PDL/ED/<run_name>/repeat:<repeat_number>",
   "dual_net_directory":   "outputs/PDL/ED/<run_name>/repeat:<repeat_number>"
   ```
   (Primal and dual weights are stored in the **same** repeat folder; the
   loader picks up `primal_weights.pth` and `dual_weights.pth` separately.)

### Suggestion — a dedicated `models/` folder

To avoid editing config paths every time you retrain, consider creating a
`models/` folder at the repository root and copying or symlinking the best
weights there:

```
models/
├── 3node/
│   ├── primal_weights.pth
│   ├── dual_weights.pth
│   └── args.json
├── 4node/
│   └── ...
└── 6node/
    └── ...
```

Then set `Benders_args` once per config file and never change it again:

```jsonc
"primal_net_directory": "models/3node",
"dual_net_directory":   "models/3node"
```

### Archived thesis models

The thesis experiments used models stored in `experiment-output/ch7/`:

```
experiment-output/ch7/
├── 3nodes/
│   ├── primal_model/    ← primal_weights.pth + args.json
│   └── dual_model/      ← dual_weights.pth  + args.json
├── 2nodes-2gens/
│   ├── primal_model/
│   └── dual_model/
└── ...
```

These paths are the defaults already set in `config.json`
(`"experiment-output/ch7/3nodes/primal_model"` and
`"experiment-output/ch7/3nodes/dual_model"`). Use them to reproduce thesis
results without retraining.

---

## Complete workflow example (3-node system from scratch)

```bash
# 1. Create and activate the environment
conda create -n {env_name} python=3.9.20
conda activate {env_name}
pip install -r requirements.txt

# 2. Train the PDL models (generates dataset if needed, then trains)
#    ARGS_FILE_NAME = "config.json" must be set in main.py
python main.py

# 3. Find the run folder (printed at the start of training)
#    e.g. outputs/PDL/ED/learn_primal:True_.../repeat:0/

# 4. Update config.json → Benders_args:
#    "primal_net_directory": "outputs/PDL/ED/learn_primal:True_.../repeat:0"
#    "dual_net_directory":   "outputs/PDL/ED/learn_primal:True_.../repeat:0"

# 5. Run Benders (Inexact with exact refinement, as set in config.json)
python gep_benders.py --config config.json

# 6. Or run the exact Gurobi baseline
python gep_benders.py --config config.json --solve-direct
```

---

### Cut-selection trial experiment

`Cut_selection_experiment.py` compares different Benders cut-aggregation
strategies (single, full, k-means, stress-bin) on a single GEP sample.
Run it after generating the datasets (Stage 1 without training):

```bash
python Cut_selection_experiment.py \
  --gep-data-path "<path-to-gep-dataset.pkl>" \
  --ed-data-path  "<path-to-ed-dataset.pkl>" \
  --sample 0 \
  --k 6
```

---

## Notes

- Always run commands from the **repository root**: several scripts use
  relative paths to `data/`, `outputs/`, and `inputs/`.
- `main.py` selects its config via the `ARGS_FILE_NAME` constant (not a
  CLI argument). Edit the constant directly before running.
- `gep_benders.py` selects its config via `--config` on the command line.
- Dataset and model filenames encode experiment settings and can be long; this
  is intentional so that different configurations do not overwrite each other.
- Model weights from different node counts are **not interchangeable**: a
  3-node model cannot be used with a 6-node Benders run.
- This is research code. Test with a small `sample_duration` (e.g. 24) before
  launching a full 8 760-hour run.

