# Running flowfirst: datasets, recipes and flags

What to type to reproduce or extend the experiments. The recipes are taken
flag for flag from `flowfirst/jobs/*.txt`, the files that produced the runs
behind F17 to F29; `OVERVIEW.md` explains what they do and `FINDINGS.md` has
the evidence.

Everything runs **from the repo root**. Locally `python` means
`venv/bin/python`; on the cluster it is whatever your module provides.

---

## 1. Paths and device

Every command takes the same root flags as `main.py`:

| Flag | Effect |
|---|---|
| `--home-path $SCRATCH/<project>` | both roots at once: datasets under `<root>/data/`, runs under `<root>/outputs/FlowFirst/ED/N<nodes>_G<gens>/` |
| `--data-root` / `--output-root` | the two roots separately |
| `--runs-dir <path>` | overrides the run location completely |
| `--device auto` | CUDA when the node has one, else CPU. **Only `flowfirst-gnn` runs on CUDA**; the MLP variants are CPU (or `mps`) |
| `--tag <name>` | the run folder is `<variant>-<tag>`, with `-2`, `-3`, … if it exists |

`PDL_HOME`, `DATA_ROOT` and `OUTPUT_ROOT` work as environment variables
instead. With no root flag and no variable set, everything stays inside
`flowfirst/` as it always has: datasets in `flowfirst/datasets/`, runs in
`flowfirst/runs/`.

**Weights & Biases.** The trainer reads the same keys as the PDL runs
(`use_wandb`, `wandb_project`, `wandb_entity`, `wandb_mode`, `wandb_group`), so
a run on a PDL config lands in the same project (`MLBenders`) next to them,
tagged `flowfirst` and the variant. `--wandb` forces it on for a config without
the key, `--no-wandb` off, and `--wandb-mode offline` is what compute nodes
without outbound network need — sync afterwards with
`wandb sync <output-root>/wandb/offline-run-*`. Everything that goes to
TensorBoard is logged per epoch, and `summary.json` becomes the run summary.

## 2. Which dataset a config selects

`--config` is the only dataset selector, the same role `-c` plays in
`main.py`. Both go through `create_gep_ed_dataset`; only the file naming
differs.

| | `main.py` | flowfirst |
|---|---|---|
| name from | `build_ed_data_save_path`: topology plus sampler flags | `dataset_path`: `ED_args.specific_name` + `2n_synthetic_samples` |
| example | `data/ED_data/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15_Label.pkl` | `flowfirst/datasets/flowfirst-6node-allgen_smp18.pkl` |
| under a root | `<data-root>/data/ED_data/…` | `<data-root>/data/flowfirst/…` |

**`specific_name` is the cache key.** Change the topology or the sampler
settings without changing it and the old pickle is silently reused; change the
name and the dataset is rebuilt and relabelled.

**To point at one exact file**, in `ED_args`:

```json
"use_direct_data": true,
"direct_data_path": "data/ED_data_gen/GEP_train_data_N3_G6_L3_H219.pkl"
```

the same two keys `main.py` reads. Nothing is sampled or labelled, and
`specific_name` becomes a label only. The path is relative to the repo root,
or to `--data-root` when one is given.

### Training on the same data as the PDL runs

The PDL configs and the flowfirst configs have identical `ED_args` schemas, so
**`configs/config.json` and `configs/config-6node.json` can be passed straight
to `--config`** — no copy to keep in sync, and the dataset is guaranteed to be
the one `main.py` trains on. The CLI overwrites `hidden_size_factor`,
`n_layers` and `body`, so the config's PDL values never reach the network.

Two things this does not make identical by itself:

- **The rows.** By default flowfirst takes the first 80 % for training and the
  next 10 % for validation, while PDL truncates training to `train_count` and
  validates on the rows right after it. On the 48000-instance file that is
  0–38400 / 38400–43200 against 0–26142 / 26142–37071: the same file, a
  different experiment. `--split pdl --train-size 26142` makes them identical,
  including the 37071–48000 rows neither trainer touches. The config's own
  `train_count` is **ignored** (every config here carries 26142, which would
  silently cut the 262k-instance runs); `train.log` prints a note when it is
  present.
- **What the comparison can show.** `configs/config.json` is three nodes with
  two generators per node, the F1 case where the thesis primal and flow-first
  provably compute the same function. Expect a tie; the six-node systems are
  where a difference appears.

### What an epoch means here, and matching the PDL budget

They are different units. `--epochs` is plain passes over the training set,
one loop. `PrimalDualTrainer` is `outer_iterations × inner_iterations`, and
**one inner iteration is a full pass** (`primal_dual.py`, the `train_loader`
loop inside the inner loop), so `configs/config.json` (200 × 10) is 2000
passes for the primal and 2000 more for the dual.

On that dataset, with `--split pdl --train-size 26142`:

| | PDL | flowfirst, 250 epochs |
|---|---|---|
| batch | 2000 | 128 |
| steps per pass | 14 | 205 |
| passes | 2000 primal + 2000 dual | 250 |
| primal gradient steps | ~28,000 | ~51,000 |
| instances seen | ~52M | ~6.5M |

So flowfirst does more updates on fewer passes. 250 epochs is a time budget,
not convergence: FINDINGS records flow-first still improving there, and F21
measured 250 → 500 epochs taking the 6-node no-shortage total from 3.4 % to
2.7 %. Matching PDL's 2000 passes costs about five minutes for the MLP at this
size, so run both and say which budget a reported number used:

```bash
python -m flowfirst.train --variant flowfirst --config configs/config.json \
  --split pdl --train-size 26142 --input-scale zscore --layers 3 \
  --batch-size 128 --epochs 250 --eval-every 5 --tag pdlmatch-250

# the same at PDL's 2000 passes
python -m flowfirst.train --variant flowfirst --config configs/config.json \
  --split pdl --train-size 26142 --input-scale zscore --layers 3 \
  --batch-size 128 --epochs 2000 --eval-every 20 --tag pdlmatch-2000
```

Run each with `--variant old-prioritized` as well for the baseline on the same
rows.

## 3. Building a sampled dataset

Only needed for configs with `use_direct_data: false`:

```bash
python -m flowfirst.dataset flowfirst/configs/config-6node-x8.json --home-path $SCRATCH/<project>
```

It prints a summary (generator costs, capacity/demand quantiles, shortage
share, nodal price alphabet) and caches the pickle. Training would build it on
demand too; doing it separately keeps Gurobi out of the training job.

| Config | Size | Instances | Note |
|---|---|---|---|
| `config-3node.json` | N3 G6 L3 | 32k | dataset A: two generators per node (F1) |
| `config-3node-3gen.json` | N3 G9 L3 | 32k | dataset B: the comparison that separates the architectures (F2, F17) |
| `config-3node-3gen-x8.json` | N3 G9 L3 | 262k | dataset B at 8× data (F18) |
| `config-6node-x8.json` | N6 G30 L8 | 262k | the main GNN testbed (F20 to F24) |
| `config-20node-x8.json` | N20 G107 L44 | 262k | the full instance (F26 to F31) |
| `config-3node-harvest.json` | N3 G6 L3 | 48k | loads the Benders harvest directly, no sampling |

- **Gurobi licence.** The size-limited licence bundled with `gurobipy` only
  covers the small 3-node LPs. Build the 6- and 20-node sets where the real
  licence is.
- **Time and size.** 262k instances is 10 to 15 minutes of Gurobi at 20 nodes
  plus the sampler's Python loop, and a 1 to 2 GB pickle in float64.

## 4. The recipes

### 4a. Flow-first MLP, 3 nodes (F18)
```bash
python -m flowfirst.train --variant flowfirst --config flowfirst/configs/config-3node-3gen-x8.json \
  --input-scale zscore --layers 3 --batch-size 128 --epochs 250 \
  --eval-every 5 --log-every 10 --tag 3gen-x8-b128-l3-fixed
```
Add `--lr-schedule step --lr-decay 0.7 --lr-step 50` for the step-decay
variant; `--variant old-prioritized` with the same flags is the thesis-primal
baseline.

### 4b. Flow-first MLP, 6 nodes (F20, ~35 min)
```bash
python -m flowfirst.train --variant flowfirst --config flowfirst/configs/config-6node-x8.json \
  --input-scale zscore --layers 3 --hidden-factor 14 --batch-size 128 --epochs 250 \
  --eval-every 1 --log-every 10 --tag 6node-x8-b128-l3-w504
```
`--hidden-factor 14` on the 36-dimensional input is width 504 (532k
parameters); 28 gives width 1008 and about 80 minutes.

### 4c. GNN, 6 nodes, quick (F22, F23, 10 to 15 min)
```bash
python -m flowfirst.train --variant flowfirst-gnn --config flowfirst/configs/config-6node-x8.json \
  --gnn-hidden 96 --gnn-rounds 2 --train-size 65536 --batch-size 128 --epochs 100 \
  --eval-every 1 --log-every 10 --tag 6node-quick-gnn96r2
```
`--train-size 65536` is what makes it quick. Vary `--gnn-hidden {64,96,128}`
and `--gnn-rounds {2,3,4}`; rounds buy more than width.

### 4d. GNN, 6 nodes, the full recipe (F24, the current default, ~1.8 h)
```bash
python -m flowfirst.train --variant flowfirst-gnn --config flowfirst/configs/config-6node-x8.json \
  --gnn-hidden 96 --gnn-rounds 4 --batch-size 128 --epochs 250 \
  --lr-schedule step --lr-decay 0.7 --lr-step 50 --clip-grad 8000 \
  --eval-every 1 --log-every 10 --device auto --tag 6node-x8-b128-gnn96r4-step-clip
```
**The decay is not optional here:** the same run at a fixed rate ends 3 to 4
times worse (1.4 % against 0.36 % no-shortage total). Drop the schedule and
the clip only to reproduce that control run.

### 4e. GNN, 20 nodes (the F26 to F31 model, ~4 h on an A100)
```bash
python -m flowfirst.train --variant flowfirst-gnn --config flowfirst/configs/config-20node-x8.json \
  --gnn-hidden 128 --gnn-rounds 6 --gnn-layernorm --batch-size 128 --epochs 250 \
  --lr-schedule step --lr-decay 0.7 --lr-step 50 --clip-grad 180000 \
  --valid-size 8192 --census-size 256 --eval-every 2 --log-every 5 \
  --device auto --tag 20node-x8-b128-gnn128r6ln-step-clip
```
What changes with size: rounds follow the graph diameter (5 here),
`--gnn-layernorm` becomes necessary at that depth, the clip threshold scales
with the gradient norms (8000 at 6 nodes, 180000 at 20), and `--valid-size`
with `--eval-every 2` keeps evaluation from dominating.

### 4f. On the PDL datasets
```bash
# 3 nodes, the configs/config.json dataset
python -m flowfirst.train --variant flowfirst --config configs/config.json \
  --input-scale zscore --layers 3 --batch-size 128 --epochs 250 \
  --eval-every 5 --log-every 10 --tag main3node

# 6 nodes (22 generators), the configs/config-6node.json dataset
python -m flowfirst.train --variant flowfirst-gnn --config configs/config-6node.json \
  --gnn-hidden 96 --gnn-rounds 4 --batch-size 128 --epochs 250 \
  --lr-schedule step --lr-decay 0.7 --lr-step 50 --clip-grad 8000 \
  --eval-every 1 --log-every 10 --device auto --home-path $SCRATCH/<project> --tag main6node-gnn
```
Run each with `--variant old-prioritized` as well for the baseline on the same
data. Note that `configs/config-6node.json` is the 22-generator harvested
system, a different problem from `flowfirst/configs/config-6node-x8.json`
(30 generators, sampled); their numbers are not comparable.

### 4g. Several jobs at once
```bash
PYTHON=python flowfirst/run_jobs.sh flowfirst/jobs/jobs-6node-gnn.txt
```
One job per line, each line exactly the arguments above. The published runs
are in `jobs-x8.txt`, `jobs-6node.txt`, `jobs-6node-gnn-quick.txt`,
`jobs-6node-gnn-rounds.txt`, `jobs-6node-gnn.txt` and `jobs-20node-modal.txt`.

## 5. Flag reference by role

**Architecture**
- `--variant`: `flowfirst` (MLP), `flowfirst-gnn` (message passing),
  `old-prioritized` (thesis primal), `old-nocompletion` (the failing
  ablation, F3).
- MLP: `--layers`, `--hidden-factor` (width = factor × input dim),
  `--body residual` with `--small-output-init` beyond four layers, which
  otherwise hits the sigmoid trap (F21).
- GNN: `--gnn-hidden`, `--gnn-rounds`, `--gnn-layernorm`,
  `--gnn-no-id-embed` (drop the fixed-topology embeddings, for transfer),
  `--gnn-antisym`.

**Data:** `--config`, `--train-size N`, `--train-subset shortage|noshortage`,
`--valid-size`, `--split flowfirst|pdl` (see §2).

**Optimization:** `--batch-size 128` (the recipe; 2048 was the old
under-trained setting), `--lr 5e-4`, `--epochs 250`,
`--lr-schedule step --lr-decay 0.7 --lr-step 50`, `--clip-grad`, `--ema`,
`--flow-reg`. `plateau` and `cosine` exist but did not help (F15);
`--loss-norm` and `--train-voll` are measured and closed (F7, F10).

**Inputs and precision:** `--input-scale zscore` for the MLP variants — **the
GNN ignores it**, it z-scores its own node features. `--dtype` is float64 by
default on CPU and CUDA, float32 forced on `mps`.

**Logging:** `--eval-every`, `--log-every`, `--census-size`, `--seed`,
`--tag`, `--runs-dir`, `--wandb` / `--no-wandb` / `--wandb-mode` /
`--wandb-project` / `--wandb-entity` / `--wandb-group`.

## 6. Reading the results

Each run writes `train.log`, `metrics.csv`, `args.json`, `model.pt`,
`model_best.pt`, `summary.json` and TensorBoard events.

```bash
tensorboard --logdir flowfirst/runs        # or <output-root>/outputs/FlowFirst
python -m flowfirst.dual_analysis <run_dir>                    # dual, certificate, polish, exact step (F26 to F29)
python -m flowfirst.polish_compare <run_dir> [<run_dir> …]     # raw against polished gap (F32)
```

Both analysis scripts take the root flags too, and otherwise read the run's
own `data_root` from its `args.json`, so a run trained under `$SCRATCH` finds
its dataset again instead of relabelling a fresh copy. They also read
`valid_start`, so a `--split pdl` run is scored on the rows it actually
validated on; `test_range` in the same file is what neither trainer saw, for a
final held-out table.

With W&B on, the curves are in the project as well, and the run's summary
carries the final and best validation gap.

Read **`val/gap_total`** (the ratio of summed objectives) and
`val/gap_total_noshortage` first, not the mean gap — F5 explains why the mean
measures one imprecision against two yardsticks. In `train.log` the
`true0 / act0 / miss / sign / sat` fields are the gradient census: `sat`
climbing toward 0.6 with `miss` above zero is the sigmoid trap (F21), and a
nonzero `sign` on a flow-first run would mean the fill is not what produced
the gradient.
