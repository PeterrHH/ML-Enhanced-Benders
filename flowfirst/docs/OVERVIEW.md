# What flowfirst does: the major changes and the experiments behind them

This is the orientation document for `flowfirst/`. It explains the method
changes that matter, each with the experiment that justifies it and the code
that implements it, so the big ones can be moved into the main pipeline first.

Detail levels: this file is the overview, `FINDINGS.md` has the numbered
findings with full evidence tables (F1 to F32), `IDEAS.md` is the backlog of
what has not been tried. Every claim here cites its finding, so it can be
checked there.

The baseline it is compared against throughout is the thesis primal:
`PrimalNetEndToEnd` with bounds repair, the generation-prioritized rescale and
completion, trained by `PrimalDualTrainer` with a dual network and the
augmented Lagrangian.

---

## TL;DR

1. **Predict only the flows.** The network outputs one number per line. The
   dispatch at every node is then computed exactly by merit order instead of
   learned (`fill.py`).
2. **Train the primal alone on the objective.** No dual network, no augmented
   Lagrangian, no ρ. The loss is the ED cost itself.
3. **Use a GNN over the grid graph.** Node and line states, a few rounds of
   message passing, a flow read off each edge. Far more sample-efficient than
   the MLP (F22, F24).
4. **Get the dual from the primal in closed form.** Prices come from a
   construction, not a second network, and they carry a certificate: an upper
   bound on each prediction's gap (F26).
5. **Finish with exact local moves.** One sweep of per-line exact line search
   divides any network's gap by about ten (F27, F32); a batched min-cost flow
   makes it exactly optimal (F29).
6. **Where it lands.** Six nodes, 30 generators: 0.016 % ratio of totals and
   96 % of instances within 1 % (F24). Twenty nodes, 107 generators: exact and
   certified on every validation instance (F29), 0.0040 ms per instance on an
   A100 (F31).

---

## 1. The problem, and what it reused

The subproblem is the economic dispatch LP that Benders solves for every
representative hour. Variables per instance:

- `p` generation per unit, in `[0, cap_g]`, where the capacities come from the
  master's investment,
- `f` flow per line, in `[-imp_l, exp_l]`,
- `md` unmet demand per node, in `[0, D_n]`.

Objective `Σ_g c_g p_g + VOLL · Σ_n md_n`, subject to the nodal balance
`p at n + net inflow at n + md_n = D_n`.

flowfirst reuses the existing problem class (`GEPOperationalProblemSet`) for the
instances, the constraint matrices and the Gurobi labels. `dataset.py` is a thin
wrapper around `create_gep_ed_dataset`, the same function `main.py` calls. So
the data, the objective and the labels are identical to the thesis pipeline's;
only the model and the training change.

**One sign convention to remember.** The problem class stores λ negated: the
nodal *price* is `-opt_targets["lamb_operational"]`. flowfirst works with
positive prices internally and negates at the boundary. This matters when
wiring the dual into `gep_benders.py`.

---

## 2. Major change 1: the flow-first parameterization

**This is the core idea; everything else follows from it.**

### What it is

The network predicts the flows `f` and nothing else. A sigmoid maps its raw
output into the line limits. Then `MeritOrderFill` (`fill.py`) computes the rest
of the solution:

1. **Residual.** `r_n = D_n - (net inflow at n)`. What the node must cover
   locally once the flows are fixed.
2. **Dispatch.** Each node's units are dispatched in increasing cost order up to
   their available capacity, filling `r_n`.
3. **Remainder.** Whatever is left is unmet demand `e_n`, negative when the node
   is over-supplied.

### Why this is exact, and why the gradient is right

Once the flows are fixed, the nodes decouple: each one is a one-row box LP
(cover `r_n` from units with costs and capacities, the rest at VOLL), whose
exact solution is the merit-order sort. So the dispatch is not approximated —
it is computed.

The resulting cost as a function of the flows,

```
U(f) = Σ_g c_g p_g(f) + VOLL · Σ_n |e_n(f)|,
```

is convex and piecewise-linear, with

```
dU / df_l = price(from-node) - price(to-node),
```

where a node's price is its marginal generator's cost, VOLL when it is short and
−VOLL when it is over-supplied. Autograd through the fill produces exactly this,
so **the training gradient on a flow is the true price difference across that
line** — the economic signal, at full size, on every line, in every instance.
`price_interval` returns the same thing explicitly, including the left/right
pair at a kink.

Three consequences worth stating:

- The network's whole job is the **routing**: which lines carry power where. The
  local, combinatorial part is computed.
- The output is feasible by construction for the boxes and the balance. Unmet
  demand is not a violation, it is a priced decision.
- A zero gradient is informative rather than a failure. If both ends of a line
  are marginal on equally expensive units, the objective really is flat there,
  so the flow is already optimal (F19).

### Against the thesis primal

| | thesis primal (`old-prioritized`) | flow-first |
|---|---|---|
| network predicts | `p` and `f` | `f` only |
| dispatch | sigmoid repair, then the generation-prioritized rescale moves every unit at a node by the same fraction toward its bound | merit-order fill, exact |
| unmet demand | completion from the balance | from the fill |
| flow gradient | headroom-weighted average cost | the true price difference |

The other rung of the ladder, `old-nocompletion`, predicts `p`, `f` and `md`,
repairs all three into their boxes and penalizes the balance with an L1 term.

### The evidence

**F1 — with two generators per node, the thesis primal *is* flow-first.** Its
production head learns to saturate the cheap unit and zero the expensive one;
the prioritized rescale then fills the expensive unit to the residual, which is
merit order. A node with at most one unsaturated unit makes the two identical.
Per-instance gap correlation 0.998, and 87 % of the worst 5 % of instances are
shared. **This is why the 3-node/2-generator comparisons look like a tie: they
are measuring the same function.**

**F2, F17 — with three generators per node the equivalence breaks.** The rescale
then splits the remainder across two unsaturated units by headroom rather than
by cost, so the production head must learn the residual dependence itself.
Properly trained (batch 128, 3 layers, 51k steps):

| validation, dataset B | flow-first | thesis primal |
|---|---|---|
| mean gap | 1.06 % | 4.04 % |
| ratio of totals | 0.11 % | 0.22 % |
| within 1 % | 87 % | 74 % |
| dispatch part, no-shortage | 0.81 % | 4.76 % |
| still improving at epoch 250? | yes | flattened |

The split of the baseline's excess shows *where* it loses: its flows are almost
as good as flow-first's, and 40 to 46 % of its excess is its own dispatch being
worse than merit order **given its own flows**. On its training instances that
share is 57 %, so it is a fitting failure, not a generalization one.

**F3 — bounds repair without completion does not learn** (77 to 89 % mean gap).
The L1 balance penalty gives a flow the difference of the *signs* of the
imbalances at its two ends: zero whenever both ends are imbalanced the same way,
which is 42 to 49 % of flows, and the wrong sign on 21 %. This is the failure
mode the thesis summary predicted, reproduced.

**F32 — the fill also rescues the baseline.** Take the thesis primal's flows and
dispatch them by merit order, and its gap falls to 1.6× flow-first's. So F17's
separation was mostly the production head, which the fill replaces.

**Code:** `fill.py` (`MeritOrderFill`), `train.py` (`FlowFirst`,
`OldPrioritized`, `OldNoCompletion`).

---

## 3. Major change 2: train the primal alone, on the objective

The loss is the objective itself:

```
mean_i [ Σ_g c_g p_g + VOLL · Σ_n |md_n| ]
```

with plain Adam. No dual network, no augmented Lagrangian, no ρ or ρ_max, no
penalty schedule, no completion classification. The fill is what makes this
possible: every output already satisfies the boxes and the balance, so there is
nothing left for a penalty to enforce and nothing for a dual network to price
during training.

Knobs that were tried on the loss and closed:

- `--loss-norm opt|log` divides each instance by its optimum, or takes a log, so
  that expensive shortage instances stop dominating. **No real effect** (F7).
- `--train-voll` lowers VOLL in the training loss only. A value of 0.4 doubles
  precision but doubles the mean gap at the true VOLL (F10). Not adopted.
- Training on no-shortage instances only does not make them more precise (F8).

The reason these do so little is F5 below: they change the denominator, not the
megawatts.

**Code:** `train.py` (`loss_terms`, `main`).

---

## 4. Major change 3: the training recipe

The same architecture at a different budget moves the result by more than the
architecture change did, so the recipe is part of the method.

| Setting | Value | Why | Finding |
|---|---|---|---|
| input scaling | z-score per feature | Peter's input LayerNorm is scale-invariant: it cannot tell an instance from the same one with all demands and capacities doubled | F9 |
| batch size | 128, not 2048 | the floor at 250 epochs was optimizer budget, not representation: 3250 steps vs 51k | F16, F17 |
| layers | 3 | 3 beats 2 on every metric | F16 |
| schedule | fixed 5e-4 for the MLP; step ×0.7 every 50 epochs for the GNN | plateau does nothing, cosine hurts; but for the GNN on 6 nodes step decay is decisive (1.4 % → 0.36 %) | F15, F18, F24 |
| gradient clipping | 8000 for the GNN | smaller gain on top of the decay, smoother tail | F24 |
| data | 262k instances (the `-x8` configs) | closes the generalization gap: training and validation within 0.05 pp | F18 |
| precision | float64 | the fill and the polish need exact arithmetic (F31) | F31 |

Two failure modes worth knowing before changing anything:

**The subgradient loss circles.** The gradient on a flow is a price difference
of roughly constant size that flips sign at a kink. It does not shrink as the
fit improves, so a fixed step never settles. That is what the decay fixes, and
why weight averaging is the first item in `IDEAS.md`.

**The sigmoid trap (F21).** With a plain deep stack, a 5-layer body drove 62 %
of its flows against a line limit within 25 epochs, where the sigmoid has no
gradient, a quarter of them at the *wrong* limit, and it sat there for 425
epochs (40 % gap) before escaping. 4 layers equals 3. The remedy for depth is a
residual body plus a small output initialization, both in `train.py`
(`--body residual`, `--small-output-init`).

---

## 5. Major change 4: the GNN over the grid graph

`FlowFirstGNN` replaces the MLP on the flat input with message passing over the
physical grid:

- **Node features:** demand, plus the node's units in merit order as
  (capacity, cost) slots, padded to the largest node. z-scored with statistics
  shared across nodes.
- **Edge features:** the line's export and import limits.
- **Rounds:** each round updates every edge from its own state and its two end
  nodes, then every node from its own state and the sums of its incoming and
  outgoing edge states. Residual updates.
- **Readout:** per edge, from the edge state and its two end nodes, into the
  sigmoid and the line limits — and then the same fill as before.
- **Identity embeddings:** learnable per-node and per-line vectors, valid
  because the topology is fixed. `--gnn-no-id-embed` turns them off, which is
  what one would do for topology transfer.

**F22 — sample efficiency.** On 6 nodes, at equal data and steps, with a third
of the parameters:

| epoch 100, 65k instances | MLP 3 × 504 | GNN 96, 2 rounds |
|---|---|---|
| parameters | 532k | 198k |
| ratio of totals, no-shortage | 10.3 % | 3.3 % |
| mean gap | 6.6 % | 2.3 % |

The MLP needed 3× the data and 10× the steps to reach that 3.3 %. No sigmoid
trap in the GNN.

**F23 — rounds beat width.** At width 96, the best mean gap goes 2.1 → 1.7 →
1.4 % for 2 → 3 → 4 rounds, and a parameter-matched 3-round network beats the
2-round one clearly. The rule of thumb: rounds should cover the graph diameter
(3 on six nodes, 5 on twenty).

**F24 — six nodes solved to the 3-node level.** GNN width 96, 4 rounds, 346k
parameters, full data, step decay and clipping: 0.016 % ratio of totals, 0.36 %
on no-shortage instances, 96 % of instances within 1 %, training and validation
aligned. The same run at a fixed rate ends 3 to 4× worse.

The 20-node model of F26 to F31 is the same architecture at width 128 with 6
rounds and LayerNorm.

**Code:** `train.py` (`FlowFirstGNN`).

---

## 6. Major change 5: the dual comes from the primal, not a second network

This replaces the thesis's dual network entirely.

**The construction.** Every price vector in `[0, VOLL]` completes to a feasible
dual solution in closed form: the box multipliers are the positive parts of
`c_g - λ` and `λ - c_g` for generators, of the price difference for lines, and
`VOLL - λ` for unmet demand (`DualRecovery.completion`). So **the dual objective
of any price vector is a valid lower bound on the optimum**, whatever the
network predicted. Only tightness depends on the primal.

Three steps to make it tight:

1. **Raw fill prices are useless as a certificate** (17.6 % of instances within
   1 %). Each node prices its own market, and the bound then charges every line
   its full capacity times the price difference across it — which uncongested
   lines pay for nothing.
2. **Quotient fill.** Contract every uncongested line, run the merit-order fill
   on each resulting region's aggregate residual, and price the whole region at
   once. This is the same construction one level up. 91 % certified within 1 %.
3. **Node-wise ascent.** The dual objective is concave and piecewise-linear in
   each price with kinks only at the cost alphabet, so a node's best price is an
   exact coordinate maximization over about ten values. Three sweeps. 94.5 %
   certified within 1 %, exact dual on 91 %.

Both steps are needed in that order: ascent alone stalls at corners (85 % at ten
sweeps).

| 20-node validation (F26) | certified gap, mean | within 1 % | dual exact |
|---|---|---|---|
| raw fill prices | 29.8 % | 17.6 % | 1.7 % |
| quotient fill | 0.68 % | 91.1 % | 79.9 % |
| + 3 ascent sweeps | 0.31 % | 94.5 % | 91.4 % |

The primal's true gap is 0.24 %, so after the last row the certificate is
dominated by primal error, not dual slack. Note what this says about the thesis
dual net: it was asked to learn a fixed point that turns out to be **one sort
per region**, given the primal's congestion pattern.

**Why this matters for Benders.** The cut reads the same rows the master already
consumes (the capacity duals and λ for the constant), and it is valid for every
investment by weak duality — so a cut built from these prices is always legal,
even when the network is wrong. The certificate `(primal - dual) / dual` then
says per instance how wrong.

**Code:** `dual.py` (`DualRecovery`), analysis in `dual_analysis.py`.

---

## 7. Major change 6: exact local moves at inference

**The polish (F27).** The fill cost is linear in a single flow until the
residual at either end hits a merit-order breakpoint, and the breakpoints are
the cumulative capacities. So the best move along one line is closed-form: shift
flow toward the pricier end by the distance to the nearest breakpoint, keep it
if the cost drops. One sweep over the lines:

| 20 nodes | mean gap | median | within 1 % | cost |
|---|---|---|---|---|
| network | 0.241 % | 0.038 % | 95.6 % | |
| + 1 sweep | 0.046 % | 0.003 % | 99.2 % | 0.05 ms/instance |
| + 3 sweeps | 0.041 % | 0.001 % | 99.3 % | |

Snapping flows to their limits does nothing, which identifies the remaining
error: not a wrong congestion pattern, but interior flows sitting a little off a
breakpoint — the hedge of F14.

**F32 — the sweep is a near-constant factor, not a floor.** Applied to every
saved 6-node run it removes 85 to 93 % of the gap and **preserves the ranking**:
the quick MLP 5.8 % → 0.66 %, the thesis primal 3.5 % → 0.37 %, the full-recipe
GNN 0.27 % → 0.033 %. So a cheaper network plus the sweep lands where the
stronger network was raw, not where the stronger polished one is.

**F28 — but local moves cannot replace the network.** Run the same exact line
search from zero flows and it stalls at a 72 % mean gap after 5000 sweeps, at
160× Gurobi's time. Routing power two lines away needs both flows to move
together; each single move alone raises the cost, so coordinate descent stops at
a corner. **This is the division of labour in one experiment: the network
supplies the global routing, exact local moves supply the precision.**

**F29 — the exact finish.** The whole dispatch is a min-cost flow (a source
node, one arc per generator, an unmet-demand arc at VOLL, lines as zero-cost
box arcs), so max-gain cycle cancelling, batched in torch (`PathPolish`),
reaches the optimum from any start. What the network buys is the *number of
augmentations*:

| start | augmentations, mean | at the optimum |
|---|---|---|
| network flows + 1 polish sweep | 4.8 | 100 % |
| zero flows | 46.5 | 100 % |

With exact flows the quotient-fill dual is exact on every instance, so one call
returns a certified optimal primal-dual pair.

**Be honest about speed.** On CPU at 20 nodes the exact pipeline is 0.59 ms per
instance against 0.16 ms for a persistent Gurobi model. The defensible claim at
this size is the tenfold cut in solver work, not wall-clock. The speed argument
needs either the GPU (below) or a larger subproblem.

**Code:** `dual.py` (`Polish`, `PathPolish`, `PrimalDual`, `PrimalDualSolution`).

---

## 8. Later: the deployed pipeline on a GPU

One paragraph, because none of this needs to move early.

`PrimalDual(exact=False)` — network, one polish sweep, quotient-fill prices with
node ascent, completion, certificate — was made to have fixed shapes, no host
syncs and no data-dependent control flow, so `GraphedPrimalDual` replays it as a
single CUDA graph, bitwise equal to eager (F30). On an A100 at batch 8192 that
is 0.0101 ms per instance in float64. Casting **only the network's parameters**
to float16, with features, fill, polish, dual and certificate left in float64,
gives 0.0040 ms with float64's accuracy — 40× one-core Gurobi (F31). An
all-float32 pipeline is slightly faster still but wrecks the glue (2.1 % mean
gap): the fill and the polish need exact arithmetic.

---

## 9. What flowfirst measures, and why

The metric choices are themselves a finding, and worth adopting in the main
pipeline.

- **Ratio of totals** (summed objectives / summed optima), not the mean relative
  gap. **F5:** the flow error in MW is the same on shortage and no-shortage
  instances, but their optimal costs differ by ~40×, so the mean relative gap
  measures one imprecision against two yardsticks. The ratio of totals weights
  every MWh by its cost, is what the raw loss encodes, and is what Benders sees.
- **The no-shortage split.** Where the hard, precision-limited instances live.
- **Dispatch part vs VOLL part** of the excess. This is what exposed the thesis
  primal's production head (F17).
- **Residual error, not flow error.** On a meshed grid a circulating flow
  changes nothing, so `|f - f*|` is misleading (F12); and **half of all 6-node
  optima admit a free circulation** with a median range of 236 MW (F25), so the
  flow target is a face, not a point. Line-saturation accuracy against Gurobi's
  vertex is unreliable for the same reason.
- **Gradient census.** Per epoch: the share of flows with a true zero gradient,
  a zero after the repair, a missing gradient, a sign mismatch, or sitting at a
  bound. This is how the sigmoid trap (F21) and the `old-nocompletion` failure
  (F3) were diagnosed.
- **The kink decomposition (F14).** The excess splits exactly into a price part
  (flow error on priced lines) and a kink part (a Bregman divergence from a
  residual crossing a breakpoint). On no-shortage instances the kink part is
  91 %, so the remaining problem is precision at breakpoints, not routing.
- **F25, the regime count.** 95k distinct active sets over 262k 6-node
  instances, 28 % of validation instances with a regime never seen in training.
  **Global active-set classification is not viable on this distribution;** any
  discrete structure has to be per node.

---

## 10. Data

`dataset.py` wraps `create_gep_ed_dataset` and caches the labelled pickle, so
the fill tests, training and analysis all see the same instances.

- **`var_cost_override`** in a config replaces the variable cost of named
  generators before the instances are built. It exists to make every
  dispatchable cost distinct, which is what dataset B needs.
- **The `-x8` configs** are the 262k-instance samplers (F18).
- **`use_direct_data` / `direct_data_path`** — the same keys `main.py` uses —
  skip sampling and load a pickle as is. That is how the Benders harvests from
  `gen_GEP/` enter: 81 perturbed 3-node GEP instances solved by exact Benders,
  deduplicated to 48k labelled ED states.

On that harvest dataset flow-first and the thesis primal come out within 3 % of
each other on every metric — exactly as F1 predicts, since that system has two
generators per node. What it did show: one polish sweep divides the mean gap by
70 to 75 there, against 8 to 15 on the sampler data. On Benders states the
network routes and the exact local moves supply nearly all the precision.
See `compare_harvest.ipynb` and the README section.

---

## 11. Tried and closed

| Change | Result | Finding |
|---|---|---|
| loss normalization by the optimum, or log | no real effect | F7 |
| training on no-shortage instances only | no gain in precision | F8 |
| `ReduceLROnPlateau` (thesis setting) | rate barely moves, changes nothing | F15 |
| cosine annealing | worse: best at epoch ~145, then degrades | F15 |
| low training VOLL (0.4) | 2× precision, 2× mean gap at the true VOLL | F10 |
| plain 5-layer body | sigmoid trap, 40 % gap for 425 epochs | F21 |
| snapping flows to their limits at inference | no gain | F27 |
| node-wise price ascent without the quotient fill | stalls at corners | F26 |

Open items live in `IDEAS.md`. The nearest ones: weight averaging (I1),
breakpoint features per node (I2), prices fed back into a second round (I3), and
the quadratic flow regularizer that makes the degenerate flow target unique
(I16).

---

## 12. How this fits the main pipeline

**Done.** The fill and the flow-first networks (`MeritOrderFill`, `FlowFirst`,
`FlowFirstGNN` and their helpers) now live in `networks.py`, beside
`PrimalNetEndToEnd`, so `gep_benders.py` and any future PDL experiment can
import them the same way. `flowfirst/fill.py` re-exports for older scripts, and
saved checkpoints load unchanged.

Training keeps its own entry point, `python -m flowfirst.train`, rather than
being merged into `main.py` or `PrimalDualTrainer` — the loops have nothing in
common (one network and plain Adam against two networks and the augmented
Lagrangian), and `flowfirst/train.py` already is that entry point, with its CLI,
jobs runner and logging. It now takes the same `--home-path` / `--data-root` /
`--output-root` flags and `--device auto` as `main.py`, so a cluster job writes
its dataset and runs under `$SCRATCH` (see the README).

Order after that, largest first:

1. **`DualRecovery` behind the Benders dual interface.** `gep_benders.py:698`
   calls `dual_net(X) -> (mu, lamb)`; a thin adapter can return the constructed
   prices and multipliers, with the sign flip of §1 and the row order of
   `split_ineq_constraints`. This is where the certificate enters the Benders
   loop.
2. **Polish at inference**, then `PathPolish` where an exact subproblem is
   wanted.
3. **GPU and precision**, only once a size is in play where it pays.

---

## 13. File index

| Path | Purpose |
|---|---|
| `fill.py` | `MeritOrderFill`: dispatch and unmet demand from the flows, plus explicit nodal prices |
| `dataset.py` | build or load the labelled ED dataset (`python -m flowfirst.dataset`) |
| `train.py` | the four variants, the training loop, evaluation, gradient census |
| `dual.py` | `DualRecovery`, `Polish`, `PathPolish`, `PrimalDual`, `GraphedPrimalDual`, `cast_net`, `load_run` |
| `dual_analysis.py` | the tables of F26 to F29 for a saved run |
| `polish_compare.py` | raw vs polished gap across saved runs (F32) |
| `configs/` | dataset definitions: 3 nodes (2 or 3 generators), 6 nodes, 20 countries; `-x8` is the 262k sampler; `-harvest` is the Benders-derived set |
| `jobs/`, `run_jobs.sh`, `run_all.sh` | job lists and the concurrent runner |
| `modal/` | run a jobs file on Modal GPUs (`train.py`), benchmark a saved run (`bench.py`) |
| `tests/` | fill, dual, certificate and polish checked against Gurobi (`pytest flowfirst`) |
| `docs/FINDINGS.md` | F1 to F32 with evidence tables |
| `docs/IDEAS.md` | prioritized backlog |
| `docs/RELATED_WORK.md` | prior art per topic, what appears new, what must be cited |
| `docs/BENCHMARKING.md` | how to make the Gurobi baseline in the Benders benchmark realistic |
| `compare_harvest.ipynb` | flow-first vs the thesis primal on the Benders harvest |
| `runs/`, `datasets/` | generated, git-ignored |
