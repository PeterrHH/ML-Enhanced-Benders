# Findings: flow-first vs the thesis primal on economic dispatch

Running record of what the experiments in this folder have shown. Append-only:
new findings get the next number, a superseded finding keeps its number and
gets a note pointing at the one that replaced it. Every finding has the same
four parts so the file stays readable as it grows.

- **Severity.** *Major* changes what we believe about the method or what we
  should do next. *Minor* is a useful fact, a knob that did or did not matter,
  or a measurement detail.
- **Status.** *Confirmed* when measured on more than one run or verified
  analytically; *lead* when it rests on a single seed or a diagnostic run.
- **Evidence.** Tables with the numbers, and the runs they come from. Run
  names refer to `flowfirst/runs/<name>`; diagnostic runs lived in the
  scratch directory and are marked as such.
- **Source.** What produced the evidence: run tag, command, or analysis.

All numbers are on the validation split unless stated otherwise. "Gap" is
`(objective - Gurobi optimum) / Gurobi optimum` per instance; "gap total" is
the ratio of the summed objectives; the "dispatch part" and "VOLL part" of a
gap split the excess into generator cost and lost-load charges. Seed 0 unless
stated otherwise. Dates are when the runs were made.

## Index

| ID | Severity | Status | Claim |
|---|---|---|---|
| F1 | major | confirmed | With two generators per node the thesis primal with the generation-prioritized layer *is* flow-first: its production head saturates the cheap unit, the rescale does the rest. |
| F2 | major | confirmed | With three generators per node the equivalence breaks; flow-first leads on dispatch error. (Margin was small at 3250 steps; see F17 for the properly trained comparison.) |
| F3 | major | confirmed | Bounds repair without completion (balance penalized) does not learn: half the flows receive no gradient. |
| F4 | major | confirmed | The remaining error is imprecision at the capacity kink of tight nodes: a few instances overshoot into VOLL, the rest hedge inside capacity. |
| F5 | major | confirmed | Flow error in MW is the same on shortage and no-shortage instances; the relative-gap difference is the denominator. The ratio of totals is the honest metric, and by it every model is within 0.5 %. |
| F6 | minor | confirmed | Interior flows exist only at kink nodes; the datasets are dominated by saturated lines. |
| F7 | minor | confirmed | Loss normalization by the optimum (or by log) changes little. |
| F8 | minor | confirmed | Training on no-shortage instances only does not make them more precise. |
| F9 | minor | lead | Input scaling: fixed-constant scaling helps slightly; z-scoring helped flow-first noticeably and hurt the baseline. |
| F10 | minor | lead | A training-time VOLL of 0.4 doubles precision but doubles the mean gap at the true VOLL. |
| F11 | minor | confirmed | Long runs overfit on no-shortage instances; shortage instances do not overfit. |
| F12 | minor | confirmed | Flow error `|f - f*|` is misleading on a triangle; nodal residual error is the right flow metric. |
| F13 | minor | confirmed | Runs are deterministic given the seed. |
| F14 | major | confirmed | Exact decomposition: 91 % of the no-shortage excess is kink crossing at nodes, split about evenly between capacity kinks and generator breakpoints; only 9 % is flow error on priced lines. Half of it is a generalization gap. |
| F15 | minor | confirmed | Learning-rate decay does not help: the thesis-style plateau barely moves the rate, cosine makes validation worse once the rate is small. The kink imprecision is not a step-size sawtooth. |
| F16 | major | confirmed | The architecture can sit on the kinks: on 1024 training instances with 24k steps the fit reaches 0.03 % ratio of totals and 0.01 % median gap, VOLL part zero. The full-set floor was optimizer budget (3250 steps); the remaining problem is generalization. Depth helps. |
| F17 | major | confirmed | With an adequate optimizer budget (batch 128, 3 layers, 51k steps) flow-first separates clearly: 1.1 % mean gap vs 4.0 %, 0.11 % vs 0.22 % ratio of totals, still improving while the baseline stalls. The baseline's stall is its production head: half its excess is dispatch worse than merit order given its own flows. Supersedes the "small margin" of F2. |
| F18 | major | confirmed | Eight times the data (262k instances) closes the generalization gap: validation ratio of totals 0.030 %, no-shortage 0.30 %, 96 % of instances within 1 %, training and validation within 0.05 pp of each other. Still improving at 250 epochs. |
| F19 | minor | analytical | Cost ties across nodes give a zero flow gradient, but with flow-first that zero is exact: the objective is flat there and the flows are optimal. No perturbation is needed on the primal side. To be checked on the 6-node dataset, which has many ties. |
| F21 | major | confirmed | Depth without skip connections fails: a plain 5-layer body falls into the sigmoid trap, 62 % of flows pinned at a bound from epoch 25 with dead gradients, a quarter of them at the wrong bound, and never recovers (40 % gap). 4 layers equals 3. More steps helped: 500 epochs took the 3-layer no-shortage total from 3.4 % to 2.7 %. |
| F22 | major | lead | A message-passing GNN over the grid graph is far more sample-efficient than the MLP on 6 nodes: with 198k parameters, 65k training instances and 41k steps it reaches the no-shortage total (3.3 %) the 532k MLP needed 210k instances and 410k steps for; the MLP on the same budget is at 10.3 %. Two rounds, no sigmoid trap. |
| F23 | minor | lead | GNN rounds beat width: at width 96 the best mean gap goes 2.1 → 1.7 → 1.4 % for 2 → 3 → 4 rounds, and a parameter-matched 3-round network (width 80) beats the 2-round one clearly. Width 96 → 128 at 2 rounds gains less than one extra round. |
| F24 | major | lead | Six nodes solved to the same level as three: the 4-round GNN with step decay and gradient clipping reaches 0.016 % ratio of totals, 0.36 % on no-shortage instances, 96 % of instances within 1 %, training and validation aligned. Decay was decisive here (fixed rate: 1.4 % no-shortage); clipping helped on top. |
| F25 | major | confirmed | The optimal regime space is huge: the 6-node labels fall into 95k distinct active sets over 262k instances, 28 % of validation instances have a regime never seen in training, and even the line-saturation pattern alone has 4016 variants. Global active-set classification is out; any discrete structure must be per node. Half the optima also admit a free circulation (median 236 MW), so the flow target is a face, not a point. |
| F26 | major | confirmed | The dual follows from the primal in closed form once prices are made consistent: raw fill prices certify 18 % of 20-node validation instances within 1 %; one merit-order fill per region of uncongested lines certifies 91 %, plus node-wise ascent 94.5 %, with the exact dual on 91 % of instances, for 0.1 ms per instance. |
| F27 | major | confirmed | One sweep of exact line search on the predicted flows cuts the 20-node gap five times (0.24 % to 0.046 % mean, 95.6 % to 99.2 % within 1 %) for 0.05 ms per instance; snapping flows to their limits does nothing, so the network's residual error is interior flows off a breakpoint. |
| F28 | major | confirmed | The flows are the one global part: the same exact line search from zero flows stalls at a 72 % mean gap after 5000 sweeps (94 % of instances above 1 %, 0.1 % at the optimum, 160 times Gurobi's time), because routing power along a path needs two flows to move together. The network supplies the routing, exact local moves the precision. |
| F29 | major | confirmed | The exact min-cost flow (max-gain cycle cancelling, batched) warm-started from the network's polished flows reaches Gurobi's optimum on all 8192 validation instances with 4.8 augmentations on average against 46.5 from zero flows; the whole primal-dual solution is then exact and certified at 0.59 ms per instance on CPU. A persistent Gurobi model solves the same LP in 0.16 ms, so at 20 nodes the speed argument is not won on CPU; the learned start's tenfold cut in solver work is the claim. |
| F30 | major | confirmed | The deployed pipeline (network, one polish sweep, quotient-fill dual with node ascent, completion, certificate) has fixed shapes, no host syncs and no data-dependent control flow; `GraphedPrimalDual` replays it as one CUDA graph at a fixed batch, bitwise equal to eager, with `torch.compile` of the network as a separate switch. A100, float64, batch 8192: 0.025 (F29) → 0.0187 ms per instance from a leaner node ascent, 0.0173 with the graph, 0.0114 compiled, 0.0101 compiled + graph: 2.9 times the roofline floor, 16 times faster per hour than one-core Gurobi, 2 times faster than eight cores. The graph pays at batch 1024 (2.5 times), at 8192 the launches were already hidden behind GPU work. The 0.005 target is out of reach in float64: the compiled forward pass alone is about 0.0085. |
| F31 | major | confirmed | Reduced precision belongs in the network only. With the GNN's parameters cast to float16 (or bfloat16) and features, flows, fill, polish, dual and certificate in float64, the compiled CUDA graph runs at 0.0040 ms per instance at batch 8192 on the A100 (float64: 0.0111; F29: 0.025) with every gap metric of the float64 pipeline (0.047 % mean, 99.3 % within 1 %), exact bounds (2.6e-14) and a balanced dispatch (3e-11 MW): 40 times faster per hour than one-core Gurobi, 5 times faster than eight cores. TF32 float32 in the network gives 0.0057. An all-float32 pipeline with TF32 on is 0.0052 but wrecks the glue (2.1 % mean gap, 6 MW balance error): the fill and the polish need exact arithmetic. |
| F32 | major | confirmed | One polish sweep divides every 6-node network's gap by 8 to 15 and preserves the ranking: the quick MLP goes 5.8 % → 0.66 %, the full MLP 2.2 % → 0.23 %, the thesis primal 3.5 % → 0.37 %, the quick GNNs 1.5 to 2.9 % → 0.17 to 0.31 %, the full-recipe GNN 0.27 % → 0.033 %. A weaker or shorter-trained network plus the sweep lands where the stronger raw network was, not where the stronger polished one is; the exact path step needs about one augmentation from any of them. |
| F20 | major | lead | Six nodes, all 30 generators: flow-first 0.17 % ratio of totals and 2.3 % mean gap against 0.52 % and 14 % for the baseline, a four- to sixfold separation. But flow-first's no-shortage total is 3.4 %, training and validation alike, so it is a fit floor: 95 % of it is kink crossing, 69 % at capacity kinks, concentrated at FRA and GER. Residual error is 6 % of demand against 0.8 % on 3 nodes. |

## Setup (for reference)

- Three nodes BEL, GER, FRA with the real demand and availability series, real
  line limits, VOLL 10 kEUR/MWh. Capacities from the node-budget sampler.
  32768 instances per dataset, Gurobi labels for all of them.
- Dataset A (`config-3node.json`): two generators per node, six distinct costs.
  BEL WindOff 0.005 / Gas 0.05; GER SunPV 0.0001 / Lignite 0.1; FRA Nuclear
  0.01 / Oil 0.2.
- Dataset B (`config-3node-3gen.json`): three per node, nine distinct costs.
  BEL WindOff / Gas / Oil; GER SunPV / Lignite / Coal; FRA WindOn / Nuclear /
  Gas at 0.07 (edited from 0.05 so that no dispatchable cost is shared between
  nodes).
- Variants, all with the same feed-forward body, Adam 5e-4, batch 2048:
  `old-prioritized` (thesis primal: bounds repair, generation-prioritized
  rescale, completion of unmet demand), `old-nocompletion` (p, f, md all
  predicted and box-repaired, L1 balance penalty of 2 VOLL), `flowfirst`
  (flows predicted, merit-order fill).
- The loss is the objective itself, generator cost plus VOLL times absolute
  unmet demand. No dual network, no augmented Lagrangian.

---

## F1 (major, confirmed): with two generators per node the thesis primal is flow-first in disguise

**Claim.** On dataset A, `old-prioritized` and `flowfirst` compute the same
function once trained. The baseline's production head learns to put the
cheap unit at full capacity and the expensive one at zero; the prioritized
rescale then fills the expensive unit to the residual, which is merit order.
A static production strategy plus the rescale is exactly optimal whenever a
node has at most one unsaturated unit.

**Evidence.** 100 epochs, dataset A, 2026-09-04.

| | flowfirst | old-prioritized |
|---|---|---|
| gap mean | 9.91 % | 9.90 % |
| gap median | 0.23 % | 0.18 % |
| instances within 1 % | 67.3 % | 67.9 % |
| flow gradient sign mismatches (census) | 0 % | 0 % |
| flows with missing gradient (census) | 0 % | 0 % |
| per-instance gap correlation between the two | 0.998 | |
| overlap of the worst 5 % of instances | 87 % | |

Baseline production head before the rescale, share of node-instances with the
sigmoid output at 99 % or more of capacity (cheap unit) or 1 % or less
(expensive unit): 100 % for all six generators.

Static strategies under the prioritized rescale, evaluated on Gurobi's optimal
flows (so only the production side is tested):

| Strategy | dataset A gap | dataset B gap |
|---|---|---|
| cheapest at max, rest 0 | 0.00 % | 16.2 % |
| cheapest and second at max, rest 0 | 45.7 % | 3.3 % mean, 28 % of instances above 1 % |

**Interpretation.** The sign-only gradient failure described in the summary
applies to completion *without* the prioritized rescale (see F3). With the
rescale and two units per node the old architecture receives the correct
price gradient by accident of the count. The number of generators per node,
not the distinctness of costs, decides whether the equivalence holds.

**Source.** Runs `flowfirst`, `old-prioritized`; head-saturation and
static-strategy analyses on the saved models.

## F2 (major, confirmed): with three generators per node the equivalence breaks, flow-first leads by a small margin

**Claim.** On dataset B the baseline's flow gradient is a headroom-weighted
average cost and its production head must learn the residual itself.
Flow-first is ahead on dispatch error and on the fraction of instances solved
to within 1 %, in every matched pair of runs. The margin on the ratio of
totals is about a tenth of a percent.

**Evidence.** Dataset B, 2026-09-04.

| Setting | variant | gap total | gap mean | gap median | within 1 % | dispatch part | VOLL part | sign mismatch |
|---|---|---|---|---|---|---|---|---|
| 100 ep | flowfirst | 0.45 % | 3.90 % | 0.82 % | 53 % | 2.48 % | 1.42 % | 0 % |
| 100 ep | old-prioritized | 0.52 % | 4.88 % | 1.90 % | 35 % | 3.50 % | 1.37 % | 4.5 % |
| 500 ep | flowfirst | 0.38 % | 4.66 % | 0.34 % | 71 % | 1.36 % | 3.30 % | 0 % |
| 500 ep | old-prioritized | 0.39 % | 5.02 % | 0.89 % | 53 % | 2.05 % | 2.97 % | 2.5 % |
| 250 ep, z-score inputs | flowfirst | 0.35 % | 2.53 % | 0.28 % | 70 % | 1.36 % | 1.17 % | 0 % |
| 250 ep, z-score inputs | old-prioritized | 0.44 % | 5.73 % | 0.86 % | 54 % | 4.60 % | 1.13 % | 2.9 % |

The baseline's dispatch part at 100 epochs (3.5 %) sits at the 3.3 % floor of
the best static strategy (F1); by 500 epochs it is below it (2.05 %), so it
does learn some residual dependence, but stays behind flow-first (1.36 %).

**Interpretation.** The architectural claim holds where the theory says it
should, but at this problem size the difference is small in absolute cost.
The shared VOLL part (F4, F5) dominates both.

**Source.** Runs `*-3gen`, `*-3gen-500`, `*-3gen-zscore`.

## F3 (major, confirmed): bounds repair without completion does not learn

**Claim.** Predicting production, flows and unmet demand, box-repairing all
three and penalizing the balance with an L1 penalty of twice VOLL leaves the
flows with a sign-only gradient: the penalty gradient on a flow is the
difference of the signs of the imbalances at its two ends, zero whenever
both ends are imbalanced the same way.

**Evidence.**

| | dataset A, 100 ep | dataset B, 100 ep |
|---|---|---|
| gap mean | 77 % | 89 % |
| gap median | 35 % | 55 % |
| balance violation (relative to demand) | 2.8 % | 2.7 % |
| flows with zero gradient although the true price difference is nonzero | 42 % | 49 % |
| flow gradient sign mismatches | 21 % | 21 % |

**Interpretation.** This is the configuration the thesis summary lists as the
untried ablation. It reproduces the sign-only failure with the penalty in
place of the completion. Kept as the floor of the ladder.

**Source.** Runs `old-nocompletion`, `old-nocompletion-3gen`.

## F4 (major, confirmed): the remaining error is imprecision at the capacity kink

**Claim.** In no-shortage instances the optimum often has a node importing
exactly enough to sit on its capacity limit. The network places that flow
with a precision of a few GW. It hedges inside capacity, paying the
neighbour's higher price for the surplus import, and still overshoots into
VOLL in about one case in ten. The overshoots are a small set of instances
that carry most of the excess cost.

**Evidence.** Dataset A, flow-first, 100 epochs, validation set.

| Group | instances | share of all excess cost | gap mean |
|---|---|---|---|
| shortage instances | 1310 | small | 1.4 % |
| no shortage, with a VOLL charge | 63 | 53 % | 365 % |
| no shortage, dispatch error only | 462 | 12 % | 13 % |

Of the 63 VOLL-charged instances, 98 % have Gurobi's residual exactly at the
node's total capacity; the network overshoots it by 1253 MW at the median.
Over all 563 kink nodes in no-shortage instances the network's residual minus
capacity has quantiles (5/25/50/75/95 %) of −5159 / −2976 / −1938 / −931 /
+1490 MW; 11 % land above capacity.

**Interpretation.** The loss landscape at such a node has slope VOLL on one
side of the kink and a generator cost difference on the other. A network
trained with a fixed step cannot place a sigmoid output on that kink
precisely, so it learns to hedge. The same mechanism produces the 2 GW hedge
in the baseline. This is the "regression errors concentrate at the steps"
point of the summary, and it is independent of the gradient path.

**Source.** Tail analysis on run `flowfirst` (dataset A); the same pattern on
dataset B in F5.

## F5 (major, confirmed): the flow error is the same everywhere; the relative gap only differs by its denominator

**Claim.** The network's nodal residual error, in MW, and its excess cost, in
money, are about the same on shortage and no-shortage instances. The optimal
cost differs by a factor of about 40, so the same error is a fraction of a
percent on one group and several percent on the other. The mean relative
gap measures one imprecision against two yardsticks; the ratio of totals
weights every MWh by its cost and is the metric the raw loss encodes and the
one Benders sees.

**Evidence.** Dataset B, 250 epochs, z-score inputs, validation set.

| | shortage instances | no-shortage instances |
|---|---|---|
| number of instances | 844 | 2432 |
| share of the validation objective | 93 % | 7 % |
| optimal cost, mean | 176,673 | 4,684 |
| residual error, MW, mean (flowfirst) | 3,131 | 3,125 |
| residual error, MW, median (flowfirst) | 1,013 | 1,906 |
| excess cost, mean (flowfirst) | 199 | 160 |
| relative gap, mean (flowfirst) | 0.6 % | 3.2 % |
| ratio of totals (flowfirst) | 0.1 % | 3.4 % |
| residual error, MW, mean (old-prioritized) | 3,165 | 3,373 |
| excess cost, mean (old-prioritized) | 218 | 212 |
| relative gap, mean (old-prioritized) | 0.8 % | 7.4 % |

Ratio of totals, all dataset-B runs, final weights: 0.35 to 0.52 % over the
whole validation set, 0.08 to 0.17 % on shortage instances, 3.4 to 5.4 % on
no-shortage instances.

**Interpretation.** Because the loss is absolute cost, an instance pulls on
the weights in proportion to its absolute excess, and the largest absolute
excess is always unmet demand at VOLL. The network learns to avoid shedding
to high precision on every instance; dispatch cost, fifty times cheaper per
MWh, only shapes the flows once shedding is under control. Reweighting and
subsetting (F7, F8) change the denominator, not the megawatts. Raising
precision is the target; the remaining levers are optimization (learning
rate schedule), data size (F11), and the certificate-and-re-solve fallback
of the summary.

**Source.** Runs `*-3gen-zscore`, `*-3gen-zscore-noshort`; residual-error
analysis on the saved models.

## F6 (minor, confirmed): interior flows exist only at kink nodes; the datasets are saturation-heavy

**Claim.** At an LP vertex only N of the variables are strictly interior, so
a line carries an interior flow only when one end node has no marginal
generator, its residual sitting exactly on a capacity breakpoint. Most
lines are saturated at the optimum.

**Evidence.**

| | dataset A | dataset B |
|---|---|---|
| instances with unmet demand at optimum | 41 % | 26 % |
| lines saturated at optimum, BEL-GER / BEL-FRA / GER-FRA | 90 / 59 / 89 % | 84 / 52 / 94 % |
| instances with 0 / 1 / 2 interior flows | 44 / 50 / 6 % | |
| node-instances priced at VOLL | 15 % | 12 % |
| GER hours with zero solar | 48 % | |
| middle-cost unit marginal, BEL / GER / FRA | | 12.5 / 13.7 / 21.7 % |
| SunPV marginal | 0 % | 0 % |

**Interpretation.** The hard learning content is the interior flows, roughly
half the instances, where the flow must land on a kink. Saturated flows are
easy for a sigmoid output.

**Source.** `python -m flowfirst.dataset` summaries; vertex analysis of the
Gurobi labels.

## F7 (minor, confirmed): loss normalization changes little

**Claim.** Dividing each instance's loss by its optimum, or taking the log of
the objective, equalizes the weight of cheap and expensive instances but does
not improve the no-shortage gap much, because the error is a precision floor
(F5), not a weighting effect.

**Evidence.** Dataset B, 250 epochs, LayerNorm inputs.

| variant | loss norm | gap total | gap mean (final / best) | no-shortage gap | no-shortage dispatch part |
|---|---|---|---|---|---|
| flowfirst | none (500 ep) | 0.38 % | 4.66 % | 6.1 % | 1.8 % |
| flowfirst | opt | 0.39 % | 3.69 / 2.94 % | 4.7 % | 1.9 % |
| flowfirst | log | 0.38 % | 4.33 / 2.95 % | 5.6 % | 1.7 % |
| old-prioritized | none (500 ep) | 0.39 % | 5.02 % | 6.6 % | 2.7 % |
| old-prioritized | opt | 0.47 % | 4.17 / 3.35 % | 5.3 % | 2.8 % |
| old-prioritized | log | 0.44 % | 4.72 / 3.36 % | 6.2 % | 2.5 % |

**Source.** Runs `*-3gen-opt`, `*-3gen-log`.

## F8 (minor, confirmed): training on no-shortage instances only does not make them more precise

**Claim.** A network trained on the 19262 training instances without load
shedding is not more accurate on the no-shortage validation instances than
one trained on everything; its residual error drops somewhat but its excess
cost on them rises, with a larger share landing on VOLL.

**Evidence.** Dataset B, 250 epochs, z-score inputs, no-shortage validation
instances.

| trained on | variant | residual error MW, mean / median | excess cost, mean | share of excess from VOLL | relative gap, mean |
|---|---|---|---|---|---|
| all | flowfirst | 3,125 / 1,906 | 160 | 54 % | 3.2 % |
| no shortage | flowfirst | 2,454 / 1,442 | 201 | 74 % | 4.5 % |
| all | old-prioritized | 3,373 / 2,030 | 212 | 34 % | 7.4 % |
| no shortage | old-prioritized | 2,618 / 1,446 | 227 | 47 % | 7.6 % |

**Source.** Runs `*-3gen-zscore-noshort`.

## F9 (minor, lead): input scaling

**Claim.** The shared body applies LayerNorm to the raw inputs, which is
scale-invariant and so cannot tell an instance from the same instance with
all demands and capacities doubled. Replacing it with a fixed division helps
a little; per-feature z-scoring helped flow-first clearly and hurt the
baseline. One seed each.

**Evidence.** Dataset B.

| variant | input scaling | epochs | gap total | gap mean | no-shortage dispatch part |
|---|---|---|---|---|---|
| flowfirst | LayerNorm | 100 | 0.45 % | 3.90 % | 3.3 % |
| flowfirst | fixed constant (scratch) | 100 | | 3.71 % | 2.7 % |
| flowfirst | z-score | 250 | 0.35 % | 2.53 % | 1.4 % (all instances) |
| old-prioritized | LayerNorm | 250 (opt norm) | 0.47 % | 4.17 % | |
| old-prioritized | z-score | 250 | 0.44 % | 5.73 % | 4.6 % (all instances) |

**Source.** Runs `*-3gen-zscore`; scratch diagnostic for the fixed constant.

## F10 (minor, lead): a low training-time VOLL trades mean gap for precision

**Claim.** Any lost-load price above the most expensive generator leaves the
optimal dispatch unchanged, so the training VOLL is a free conditioning
parameter. With 0.4 (twice the most expensive unit) the network gets much
closer to the kinks but overshoots more often, and each overshoot is charged
at the true VOLL of 10 in evaluation.

**Evidence.** Dataset B, flow-first, 100 epochs, fixed-constant input scaling,
scratch diagnostic.

| training VOLL | gap mean | gap median | within 1 % | no-shortage dispatch part | VOLL part | residual error (relative) |
|---|---|---|---|---|---|---|
| 10 | 3.71 % | 0.65 % | 56 % | 2.7 % | 1.7 % | 6.6 % |
| 0.4 | 7.54 % | 0.32 % | 69 % | 1.3 % | 6.6 % | 3.9 % |

**Interpretation.** Whether this is good depends on the metric. Benders uses
the true VOLL, so the mean at VOLL 10 is the honest number. An intermediate
value, or a schedule from low to high, has not been tried.

## F11 (minor, confirmed): long runs overfit on no-shortage instances

**Evidence.** Dataset B, 500 epochs, LayerNorm inputs, training subset of the
same size as validation.

| variant | gap mean train / val | no-shortage train / val | shortage train / val |
|---|---|---|---|
| flowfirst | 1.6 / 4.7 % | 1.9 / 6.1 % | 0.7 / 0.6 % |
| old-prioritized | 2.0 / 5.0 % | 2.6 / 6.6 % | 0.6 / 0.5 % |

The 500-epoch flow-first run is worse on no-shortage instances than the
100-epoch one (6.1 % vs 5.0 %). Runs now save the best-validation checkpoint
and default to 250 epochs. A larger dataset (65k instances costs about half
a minute of Gurobi time) has not been tried.

## F12 (minor, confirmed): use nodal residual error, not flow error

On a three-node triangle the incidence matrix has a one-dimensional null
space, a circulating flow, which changes no residual and no cost. `|f − f*|`
therefore overstates errors; the evaluation logs `val/residual_err_rel`
instead.

## F13 (minor, confirmed): runs are deterministic

Duplicate launches of the same variant and seed produced bit-identical
metrics on dataset A.

---

## F14 (major, confirmed): the no-shortage excess is kink crossing, and half of it is generalization

**Claim.** Because the dispatch cost is convex and piecewise-linear in the
nodal residuals, the excess over the optimum splits exactly into a *price
part*, the optimal price difference times the flow error on each line, and a
*kink part*, the extra cost from a residual crossing a breakpoint at a node
(a Bregman divergence, nonnegative by convexity). On no-shortage instances
the kink part is 91 % of the excess. It is split between nodes whose optimum
sits on the capacity kink (the VOLL cliff) and nodes whose optimum sits on a
generator breakpoint inside capacity (for example wind fully used, gas off,
imports covering the rest exactly). Under-saturating priced lines is a minor
contributor. The same decomposition on training instances gives half the
excess, so the imprecision is partly a fitting floor and partly a
generalization gap.

**Evidence.** Dataset B, flow-first, 250 epochs, z-score inputs, 2026-09-05.

| No-shortage instances | validation | training subset |
|---|---|---|
| ratio of totals (flow-attributable excess) | 3.42 % | 1.67 % |
| price part: flow error on priced lines | 9 % | 17 % |
| kink part, total | 91 % | 83 % |
| of which at nodes on the capacity kink (224 / 226 node-instances) | 44 % | 38 % |
| of which at nodes on a generator breakpoint inside capacity (1591 / 1511) | 40 % | 43 % |
| of which at nodes with a marginal generator (5480 / 5399) | 7 % | 2 % |
| residual minus capacity at capacity-kink nodes, MW, median | −3289 | −2747 |
| share of capacity-kink nodes overshot into VOLL | 6 % | 2 % |

The baseline (`old-prioritized`, same setting) has the same split of its
flow-attributable excess (90 % kink, 10 % price), plus its own dispatch
excess over merit order given its flows, 40 % on top on validation and 72 %
on training instances, which is the residual-dependence it has to learn (F2).

**Interpretation.** The hedge at capacity kinks, about 3 GW inside capacity
even on training instances, is the rational response of an imprecise
function approximator to an asymmetric loss: overshooting costs VOLL,
undershooting costs a price difference. Raising precision at breakpoints is
the whole remaining problem for flow-first; the breakpoint locations are
linear functions of the inputs, so this is representational and
data-related rather than a weighting issue.

**Source.** Decomposition script on the saved models of runs `*-3gen-zscore`.

## F15 (minor, confirmed): learning-rate decay does not help

**Claim.** The imprecision at kinks is not a fixed-step sawtooth. The thesis
trainer's `ReduceLROnPlateau` (factor 0.99, patience 10) moves the rate from
5e-4 to 4.6e-4 in 250 epochs and changes nothing. Cosine annealing reaches
its best at about epoch 145 and validation then gets worse as the rate
drops below 1e-4, while the fixed-rate run keeps improving through epoch
250. That is the overfitting pattern of F11, not damped oscillation.

**Evidence.** Dataset B, z-score inputs, 250 epochs, 2026-09-05.

| variant | schedule | rate at epoch 250 | gap mean, final / best (epoch) | no-shortage dispatch part |
|---|---|---|---|---|
| flowfirst | none | 5.0e-4 | 2.53 % / 2.50 % (248) | 1.82 % |
| flowfirst | plateau | 4.6e-4 | 2.93 % / 2.55 % (198) | 1.82 % |
| flowfirst | cosine | 5.0e-6 | 3.53 % / 3.17 % (144) | 2.17 % |
| old-prioritized | none | 5.0e-4 | 5.73 % / 5.69 % (242) | 6.17 % |
| old-prioritized | plateau | 4.6e-4 | 6.06 % / 5.69 % (208) | 5.95 % |
| old-prioritized | cosine | 5.0e-6 | 6.54 % / 6.33 % (155) | 6.67 % |

**Source.** Runs `*-3gen-zscore-plateau`, `*-3gen-zscore-cosine`.

## F16 (major, confirmed): the network can represent the kinks; the full-set floor was optimizer budget

**Claim.** Trained on 1024 instances at batch 128 for 3000 epochs (24k Adam
steps, fixed rate 5e-4, z-score inputs), flow-first fits its training
instances to a ratio of totals of 0.03 to 0.05 % and a median gap of 0.01 to
0.05 %, with the VOLL part essentially zero: it places residuals on the
capacity kinks without overshooting on instances it has seen. The full
training set at 250 epochs had only 3250 steps and a training-set floor of
1.7 %, so that floor was optimizer budget, not representation. Three hidden
layers fit better than two on every metric. The fit was still improving
slowly at 3000 epochs.

**Evidence.** Dataset B, flow-first, 1024 training instances, 2026-09-05.

| training-set metric at epoch 3000 | 2 layers | 3 layers |
|---|---|---|
| ratio of totals | 0.051 % | 0.030 % |
| ratio of totals, no-shortage instances | 0.67 % | 0.46 % |
| gap mean | 0.60 % | 0.44 % |
| gap median | 0.05 % | 0.01 % |
| instances within 1 % | 90 % | 93 % |
| dispatch part, no-shortage | 0.80 % | 0.61 % |
| VOLL part | 0.02 % | 0.002 % |
| lines saturated correctly | 82 % | 88 % |
| change of gap mean over epochs 2000 to 3000 | −0.19 pp | −0.17 pp |
| validation ratio of totals (1024 training instances only) | 2.4 % | 3.5 % |

Remaining training-set excess, 3 layers: 93 % kink part, of which 58 % at
capacity kinks and 31 % at generator breakpoints; at generator-breakpoint
nodes the median residual error is 0 MW and 36 % are within 10 MW. The
worst five instances carry 15 % of the excess.

**Interpretation.** Representation and optimization are sufficient given
enough steps per instance. The open problem is generalization: precision at
kink locations on unseen instances. Next: the full training set with many
more steps (smaller batches), then more data.

**Source.** Runs `flowfirst-overfit-1024`, `flowfirst-overfit-1024-l3`;
decomposition on the saved models.

## F17 (major, confirmed): properly trained, flow-first separates clearly and the baseline stalls in its production head

**Claim.** With the optimizer budget the overfit test (F16) showed to be
necessary, batch 128 (205 steps per epoch, 51k steps in 250 epochs), three
hidden layers and z-scored inputs, flow-first reaches a mean gap of 1.1 %
and a ratio of totals of 0.11 % on validation, four and two times better
than the baseline, and is still improving at 250 epochs while the baseline
has flattened. The baseline's flows are almost as good as flow-first's; what
stalls is its production head, which dispatches worse than merit order given
its own flows. On its own training instances that head accounts for two
thirds of its excess, so it is a fitting failure of the residual-dependent
dispatch, exactly the coupling the three-generator dataset was built to
expose (F1, F2).

**Evidence.** Dataset B, batch 128, 3 layers, z-score inputs, fixed rate
5e-4, 250 epochs, 2026-09-05.

| validation, epoch 250 | flowfirst | old-prioritized |
|---|---|---|
| ratio of totals | 0.11 % | 0.22 % |
| ratio of totals, no-shortage instances | 1.12 % | 2.62 % |
| gap mean | 1.06 % | 4.04 % |
| gap median | 0.05 % | 0.29 % |
| instances within 1 % | 87 % | 74 % |
| dispatch part, no-shortage | 0.81 % | 4.76 % |
| VOLL part | 0.46 % | 0.50 % |
| residual error (relative) | 1.95 % | 2.15 % |
| change of no-shortage total over epochs 200 to 250 | −0.33 pp | +0.09 pp |
| training-set gap mean | 0.56 % | 2.96 % |
| training-set ratio of totals, no-shortage | 0.54 % | 1.68 % |

Split of the excess into flow error (merit-order dispatch of the model's own
flows versus the optimum) and production-head error (the model's dispatch
versus merit order of its own flows):

| old-prioritized | flows | production head | ratio of totals if its flows were dispatched by merit order |
|---|---|---|---|
| validation, no-shortage | 54 % | 46 % | |
| validation, all | 60 % | 40 % | 0.13 % (vs 0.22 % actual; flowfirst 0.11 %) |
| training subset, all | 43 % | 57 % | 0.06 % (vs 0.14 % actual; flowfirst 0.06 %) |

For comparison with the earlier budget: the same variants at batch 2048 and
two layers (3250 steps) stood at 2.53 % and 5.73 % mean gap, 0.35 % and
0.44 % ratio of totals (F9).

**Interpretation.** This is the paper's claim in one table: the coupling
variables are learned equally well by both architectures, and the difference
is entirely in what is *computed* versus *learned* downstream of them. Merit
order is exact and free; the prioritized rescale needs a residual-dependent
head that the network fails to fit even on its training data.

**Source.** Runs `flowfirst-3gen-b128-l3`, `old-prioritized-3gen-b128-l3`;
flow/head split on the saved models.

## F18 (major, confirmed): eight times the data closes the generalization gap

**Claim.** On 262,144 instances (2^18, same sampler and generator set as
dataset B) with batch 128, three layers, z-score inputs and a fixed rate of
5e-4 for 250 epochs (410k steps), flow-first reaches a validation ratio of
totals of 0.030 % and a mean gap of 0.36 %, with 96 % of instances within
1 %. Training-set and validation metrics now agree to within 0.05
percentage points, so the memorization of kink locations seen on 26k
training instances (F11, F14) is gone. The run was still improving at 250
epochs. A conservative plateau rule (halve after 50 epochs without
improvement) never fired, so that run is identical.

**Evidence.** Dataset B at 2^18 instances (`config-3node-3gen-x8.json`),
2026-09-05.

| flow-first, epoch 250 | validation | training subset | F17 run (26k instances), validation |
|---|---|---|---|
| ratio of totals | 0.030 % | 0.023 % | 0.11 % |
| ratio of totals, no-shortage | 0.30 % | 0.24 % | 1.12 % |
| gap mean | 0.36 % | 0.31 % | 1.06 % |
| gap median | 0.014 % | | 0.05 % |
| instances within 1 % | 96.2 % | | 87 % |
| VOLL part | 0.09 % | | 0.46 % |
| dispatch part, no-shortage | 0.36 % | | 0.81 % |
| residual error (relative) | 0.84 % | | 1.95 % |
| lines saturated correctly | 93.5 % | | |

Trajectory of the validation ratio of totals at epochs 50 / 100 / 150 /
200 / 250: 0.065 / 0.054 / 0.042 / 0.035 / 0.030 %; instances within 1 %:
90.4 / 91.2 / 94.1 / 95.3 / 96.2 %.

Step decay on the same data (rate times 0.7 every 50 epochs, 5e-4 to
8.4e-5): final ratio of totals 0.032 % vs 0.030 % fixed, no-shortage 0.27 %
vs 0.30 %, best mean gap 0.31 % vs 0.32 %, instances within 1 % 97.5 % vs
96.2 %, training-set total 0.016 % vs 0.023 %. The training loss oscillates
three times less in the last 50 epochs, and the validation total ticked up
from 0.027 % at epoch 200 to 0.032 % at 250 as the rate fell. Net: a small
gain on the tail, none on the total, and the first sign of the F15 pattern.
Runs `flowfirst-3gen-x8-b128-l3-fixed`, `flowfirst-3gen-x8-b128-l3-step`.

**Interpretation.** With training and validation aligned, the remaining
0.3 % on no-shortage instances is a fitting floor again, to be attacked with
capacity and steps rather than data. For the Benders use the ratio of totals
is already below typical convergence tolerances; the 4 % tail above 1 % is
what a certificate-and-re-solve fallback would handle.

**Source.** Runs `flowfirst-3gen-x8-b128-l3`, `flowfirst-3gen-x8-b128-l3-plateau`.

## F19 (minor, analytical): cost ties across nodes are harmless for flow-first

**Claim.** When both ends of a line are marginal on units of equal cost, the
price difference and hence the flow gradient is zero. With flow-first that
gradient is the exact subgradient of a convex cost, so a zero means the
flows are optimal for that instance: the objective is flat along the line
until a tied unit saturates, at which point the price and the gradient
return. This differs from the thesis architecture's zero gradient, which
appeared at balanced nodes regardless of cost and hid a suboptimal dispatch
(F3). The evenly spaced cost perturbation of the summary is therefore not
needed for the primal; it remains relevant for making the dual unique.

**Evidence.** Stated, not yet measured. The 6-node dataset with all
generators has gas, oil and the renewables at identical costs in every node;
the census category `grad/true_zero_cost_tie` counts these zeros separately
from the VOLL and over-supply zeros. Prediction: that share is substantial
and the gap does not suffer.

## F20 (major, lead): six nodes, all generators: the separation grows, and flow-first hits a fit floor at 3.4 % on no-shortage instances

**Claim.** On Peter's six nodes with every generator the data has for them
(30 units, 8 lines, 262k instances, 57 % of them with shortage at the
optimum, 2.7 interior flows per instance, 48 % of line-instances on a cost
tie), the same setting as F18 but width 504 separates the architectures by
a factor four to six. Flow-first itself, however, plateaus with training and
validation aligned at about 3 % no-shortage total, so the limit is the fit,
not the data. The excess is 95 % kink crossing, 69 % of it at capacity
kinks, and sits at the large nodes FRA (47 %) and GER (29 %). The residual
error is 6.2 % of demand, seven times the 3-node figure, and interior flows
are off by 8 % of the line range at the median.

**Evidence.** `config-6node-x8.json`, batch 128, 3 layers, width 14 × 36 =
504, z-score inputs, fixed rate 5e-4, 250 epochs, 2026-09-05.

| validation, epoch 250 | flowfirst | old-prioritized |
|---|---|---|
| ratio of totals | 0.17 % | 0.52 % |
| ratio of totals, no-shortage | 3.4 % | 13.0 % |
| gap mean | 2.3 % | 14.0 % |
| gap median | 0.38 % | 1.5 % |
| instances within 1 % | 60 % | 45 % |
| dispatch part, no-shortage | 3.5 % | 28 % |
| VOLL part | 0.73 % | 1.45 % |
| residual error (relative) | 6.2 % | 6.6 % |
| lines saturated correctly | 73 % | 71 % |
| training-set ratio of totals, no-shortage | 3.0 % | 12.8 % |
| flows with a true zero gradient / of which cost ties (census) | 30 % / 18 % | 29 % / 17 % |

Trajectory of flow-first's no-shortage validation total at epochs 50 / 100 /
150 / 200 / 250: 6.0 / 4.6 / 3.8 / 4.3 / 3.4 %; training loss still
decreasing slowly.

Decomposition of flow-first's no-shortage excess (validation; training
subset in brackets): price part 5 % (6 %); kink part 95 % (94 %), of which
capacity kinks 69 % (71 %), generator breakpoints 20 % (19 %), marginal
nodes 5 % (4 %). By node: FRA 47 %, GER 29 %, SWI 19 %, BEL 5 %, SPA and
NED about 0 %. Nodes whose optimum sits on a breakpoint: NED 89 %, SWI 86 %,
BEL 66 %, FRA 37 %, SPA 37 %, GER 17 %.

Residual error by node type on no-shortage validation instances (signed
r − r*, quantiles 5/25/50/75/95, MW): capacity-kink nodes −1029 / −449 /
−274 / −131 / −35 (a hedge inside capacity, as on 3 nodes); generator
breakpoints −4047 / −997 / −167 / +640 / +4000; nodes with a marginal
generator, where the cost is locally linear and nothing is charged for
imprecision, −3869 / −65 / +64 / +830 / +4858, median absolute error
522 MW. The 3-node model's median absolute residual error on the same
statistic is 7 MW. By node, median absolute error and its share of demand:
FRA 1407 MW (4.7 %), GER 600 MW (1.6 %), BEL 513 MW (9.9 %), SPA 386 MW,
NED 342 MW, SWI 128 MW.

**Interpretation.** The flows themselves are imprecise everywhere, by two
orders of magnitude more than on 3 nodes, including at nodes where the loss
gives no kink signal at all. The kinks are only where that imprecision is
charged. So this is a flow-regression floor of a capacity-limited network,
not the fine hedging pattern of F4. The importers NED and SWI are almost
always kink nodes but cheap to get wrong; the money is at FRA and GER, whose capacity kinks
the network places with 400 MW median and 1.1 GW mean error. The regime
count grows combinatorially with lines and units, and a 3-layer, 532k
parameter network is short of capacity for it. Since training and
validation agree, the levers are depth, width and steps (F16), and after
those the relative-flow parameterization, whose whole point is the capacity
kink. On ties (F19): 18 % of flows have a tie zero and the baseline shows
the same share, so ties are not what separates the two.

**Source.** Runs `flowfirst-6node-x8-b128-l3-w504`,
`old-prioritized-6node-x8-b128-l3-w504`; decomposition on the saved model.

## F21 (major, confirmed): depth without skip connections falls into the sigmoid trap

**Claim.** On the 6-node data a plain 4-layer body matches the 3-layer one
and a plain 5-layer body fails outright. The failure is not a plateau but a
trap: within 25 epochs 55 to 62 % of its flows sit at a line limit, where the
sigmoid repair has no gradient, and a quarter of those are at the wrong
place, so the network can never move them. Doubling the epochs of the
3-layer run, on the other hand, did help. The remedy for depth is a residual
body (skip connections) with a small initial output scale so flows start
mid-range; both are in `train.py` (`--body residual`, `--small-output-init`).

**Evidence.** 6 nodes, width 504, batch 128, z-score inputs, fixed rate,
500 epochs, 2026-09-05. The 5-layer numbers are from its best checkpoint
around epoch 300 while the run was still going, flat since epoch 25.

| validation | 3 layers | 4 layers | 5 layers |
|---|---|---|---|
| ratio of totals | 0.13 % | 0.14 % | 7.8 % |
| ratio of totals, no-shortage | 2.7 % | 2.8 % | |
| gap mean, final / best | 2.3 % / 1.7 % | 2.0 % / 1.7 % | 42 % / 42 % |
| training-set ratio of totals, no-shortage | 2.3 % | 2.4 % | 7.9 % |
| flows at a line limit (census) | 46 % | 46 % | 61 % |

Saturated flows on 4096 validation instances, 3-layer vs 5-layer model:
raw output magnitude median 4.9 vs 15.2 and 95th percentile 44 vs 84 (the
sigmoid's slope at 10 is 4.5e-5, at 20 it is 2e-9); share of saturated
flows at the same bound as Gurobi 93 % vs 75 %, at the opposite bound 0.6 %
vs 6 %, where Gurobi is interior 6 % vs 19 %; mean gradient magnitude at the
raw output on saturated flows 0.06 vs 0.02, against about 200 on interior
flows. The 5-layer model has both lines into NED pinned in 100 % of
instances.

For comparison, 250 vs 500 epochs of the 3-layer run: no-shortage total
3.4 % → 2.7 %, mean gap 2.3 % → 1.7 % at best, instances within 1 % 60 % →
66 %.

Addendum: the 5-layer run escaped at epoch 452, after 425 flat epochs.
Saturation dropped from 62 % to 46 % within ten epochs and the run then
recovered to 2.9 % mean gap and 0.20 % ratio of totals by epoch 500, close
to the 3-layer run. Adam divides each step by the running gradient
magnitude, so even a nearly dead sigmoid input keeps moving at about the
learning rate per step; the trap is metastable, not permanent, but escaping
it cost 450 epochs.

**Interpretation.** A deeper plain ReLU stack has larger and faster-moving
initial outputs; with a sigmoid repair that is enough to drive flows into
the flat region of the sigmoid in the first epochs, after which the census
shows a healthy gradient at the repaired flow and none at the raw output.
This is a property of the sigmoid box repair as much as of depth, and the
F16 result that three layers beat two does not extend to plain deeper
stacks. Depth needs skip connections and a small output initialization.

**Source.** Runs `flowfirst-6node-x8-b128-l{3,4,5}-w504-e500`; saturation
analysis on the saved weights.

## F22 (major, lead): the GNN is far more sample-efficient on 6 nodes

**Claim.** Flow-first with a message-passing network over the grid graph
(node features: demand and merit-ordered unit capacities and costs; edge
features: line limits; two rounds of edge and node updates; edge readout
through the sigmoid into the line limits; then the fill) beats the MLP by a
wide margin at equal data and steps, with a third of the parameters. On the
same budget it reaches what the MLP needed ten times the steps and three
times the data for (F20, F21).

**Evidence.** 6 nodes, first 65k training instances, batch 128, 100 epochs
(41k steps), fixed rate 5e-4, validation on the full 26k split, 2026-09-05.

| epoch 100 | MLP 3 × 504 | GNN hidden 64, 2 rounds | GNN hidden 96, 2 rounds |
|---|---|---|---|
| parameters | 532k | 89k | 198k |
| ratio of totals | 0.60 % | 0.20 % | 0.15 % |
| ratio of totals, no-shortage | 10.3 % | 4.3 % | 3.3 % |
| gap mean, final / best | 6.6 % / 6.6 % | 2.9 % / 2.1 % | 2.3 % / 2.1 % |
| gap median | 1.2 % | 0.46 % | 0.42 % |
| instances within 1 % | 48 % | 59 % | 59 % |
| residual error (relative) | 7.6 % | 6.5 % | 6.3 % |
| lines saturated correctly | 67 % | 74 % | 74 % |
| flows at a limit / raw gradient zero (census) | 39 % / 35 % | 47 % / 47 % | 46 % / 50 % |
| training time | 6 min | 9 min | 12 min |

The MLP with 210k instances and 250 epochs (F20) stood at 3.4 % no-shortage
total and 2.3 % mean gap. The width-64 GNN is noisy at a fixed rate (a spike
to 11 % at epoch 60); width 96 is smooth.

**Interpretation.** Weight sharing across nodes and lines and the locality of
message passing fit the problem: a flow is determined by the states of its
end nodes and their neighbourhoods. The GNN's raw-gradient-zero share equals
its saturation share, so no sigmoid trap. Per step it costs two to three
times the MLP, but per unit of accuracy it is much cheaper. It is also the
architecture that transfers to 20 nodes.

**Source.** Runs `flowfirst-6node-quick-mlp504`,
`flowfirst-gnn-6node-quick-gnn64r2`, `flowfirst-gnn-6node-quick-gnn96r2`.

## F23 (minor, lead): more message-passing rounds help more than width

**Claim.** In the quick 6-node setting, adding rounds improves the GNN
monotonically and more cheaply than adding width. Single seed, endpoints at
a fixed rate are noisy, so the table gives the mean over epochs 80 to 100
and the best checkpoint as well.

**Evidence.** 65k training instances, 100 epochs, batch 128, 2026-09-05.

| hidden / rounds | params | no-shortage total, epochs 80-100 | best mean gap (epoch) | within 1 % at 100 |
|---|---|---|---|---|
| 64 / 2 | 89k | 3.97 % | 2.10 % (94) | 59 % |
| 96 / 2 | 198k | 3.74 % | 2.09 % (96) | 59 % |
| 80 / 3 | 190k | 3.37 % | 1.53 % (100) | 69 % |
| 96 / 3 | 272k | 3.40 % | 1.72 % (96) | 58 % |
| 96 / 4 | 346k | 2.67 % | 1.44 % (88) | 61 % |
| 128 / 2 | 350k | 3.47 % | 1.83 % (100) | 62 % |
| MLP 3 × 504 | 532k | 10.8 % | 6.6 % | 48 % |

**Interpretation.** The graph's diameter is 3, so two rounds already give
every edge readout information from the whole grid; the gain from further
rounds is iterative refinement, one round per hop of price propagation.
Rounds also cost less than width per step: 96 / 4 runs at 15 ms per step
against 11 ms for 128 / 2, for a better result. At 20 nodes the diameter
grows and rounds will have to grow with it.

**Source.** Runs `flowfirst-gnn-6node-quick-*`.

## F24 (major, lead): six nodes with all generators solved to the 3-node level

**Claim.** The GNN (hidden 96, 4 rounds, 346k parameters) on the full 210k
training instances for 250 epochs, with the step schedule (rate × 0.7 every
50 epochs) and gradient-norm clipping at 8000, brings the 6-node,
30-generator problem to the level the MLP reached on 3 nodes (F18):
0.016 % ratio of totals, 0.36 % on no-shortage instances, 96 % of instances
within 1 %, with training and validation aligned. On this problem the decay
is not optional: the identical run at a fixed rate ends at 1.4 %
no-shortage total and 77 % within 1 %, three to four times worse. Clipping
adds a smaller gain and a smoother tail.

**Evidence.** `config-6node-x8.json`, batch 128, z-score node features,
250 epochs, evaluated every epoch, 2026-09-06.

| validation, epoch 250 | fixed rate | step decay | step decay + clip 8000 |
|---|---|---|---|
| ratio of totals | 0.064 % | 0.019 % | 0.016 % |
| ratio of totals, no-shortage | 1.37 % (last-20 mean 1.47 %) | 0.51 % (0.49 %) | 0.36 % (0.52 %) |
| gap mean, final / best | 1.10 % / 0.60 % | 0.36 % / 0.30 % | 0.29 % / 0.27 % |
| instances within 1 % | 77 % | 94 % | 96 % |
| training-set ratio of totals, no-shortage | 1.34 % | 0.49 % | 0.32 % |

For reference: the best MLP on this dataset (3 layers, width 504, 500
epochs) stood at 2.7 % no-shortage total and 66 % within 1 % (F21); the
prioritized baseline at 13 % and 45 % (F20).

**Interpretation.** With the fill exact, the GNN's locality prior, enough
data and steps, and a decaying step for the subgradient loss, the 6-node
ED primal is at the same precision as the 3-node one. The oscillation seen
during training was the raw iterate circling at a fixed step (F15, F23);
the decay is what settles it, and weight averaging (IDEAS I1) is the
untested way to get the same effect without slowing the descent.

**Source.** Runs `flowfirst-gnn-6node-x8-b128-gnn96r4-{fixed,step,step-clip}`.

## F25 (major, confirmed): the regime space is huge, and half the optima are degenerate

**Claim.** The active-set learning route of the OPF literature (predict
which constraints bind, then solve the affine system) relies on a small
number of active sets covering the distribution. On our sampler it does
not hold: the 6-node labels fall into 95,446 distinct regimes (line states,
unit states, shortage flags) over 262,144 instances, 66,713 of them seen
once, and 28 % of validation instances have a regime that never occurs in
the training split. Even the line-saturation pattern alone takes 4016 of
the 6561 possible values and needs 919 patterns to cover 90 % of
instances. Separately, 52 % of the optima admit a nonzero circulation
around a cycle, with a free range of 236 MW at the median and 3.6 GW at the
75th percentile: the optimal flow is a face, not a point, and Gurobi's
vertex is one arbitrary corner of it.

**Evidence.** `config-6node-x8.json` labels, 2026-09-07.

| | value |
|---|---|
| distinct full regimes / instances | 95,446 / 262,144 |
| regimes needed for 50 / 90 / 99 % coverage | 6,799 / 69,232 / 92,825 |
| validation instances with an unseen regime | 27.6 % |
| distinct line-saturation patterns (of 6561) | 4,016; 919 for 90 % coverage |
| instances with a feasible circulation at the optimum | 51.7 % |
| free circulation range, MW, median / 75th / 95th percentile | 236 / 3,576 / 4,697 |

**Interpretation.** Two consequences. A discrete "which regime" head can
only work per node, where the class set is the node's few breakpoints and
regimes compose, not globally. And the network's flow target is set-valued
in half the instances with nothing in the loss to pick a point; a small
quadratic flow regularizer (IDEAS I16) makes the selection unique and
continuous at no cost to the objective. Metrics that compare flows or line
saturation with Gurobi's vertex are meaningless on degenerate instances;
residual error and cost are the right reads (F12).

**Source.** Regime and circulation analysis of the Gurobi labels.

## F26 (major, confirmed): the dual follows from the primal once prices are made consistent

**Claim.** Section 2 of the summary holds on the trained 20-node model. Every
price vector in [0, VOLL] completes to a feasible dual in closed form, so the
dual objective is a valid lower bound for any prediction; only tightness
depends on the primal. The raw fill prices are useless as a certificate
(each node prices its own market; the bound charges every line its full
capacity times the price difference across it, which uncongested lines with
different fill prices pay for nothing). Making prices consistent fixes it:
contract every uncongested line, run the merit-order fill on each region's
aggregate residual (the "quotient fill", the same construction one level
up), then let single nodes move by exact coordinate ascent over the
ten-value price alphabet. Both steps are needed in that order: ascent alone
stalls at corners of the piecewise-linear dual objective.

**Evidence.** 8192 validation instances of `config-20node-x8.json`, model
`flowfirst-gnn-20node-x8-b128-gnn128r6ln-step-clip-modal` at epoch 250
(F24-level: 0.48 % no-shortage total, 95.6 % within 1 %), CPU float64,
2026-09-08. Gurobi's own prices reproduce its optimum to 2e-14, weak duality
held on every instance. A line within 1 % of its limit counts as congested
(1e-3 was worse on every metric; the hedge of F14 leaves congested lines
just inside).

| prices from | certified gap, mean | instances certified within 1 % | dual exact (slack < 1e-6) | node prices equal to Gurobi's | per 8192 |
|---|---|---|---|---|---|
| raw fill prices | 29.8 % | 17.6 % | 1.7 % | 72.5 % | 8 ms |
| quotient fill, no iteration | 0.68 % | 91.1 % | 79.9 % | | 40 to 60 ms |
| quotient fill + 3 node-wise ascent sweeps | 0.31 % | 94.5 % | 91.4 % | 97.6 % | 460 to 680 ms |
| node-wise ascent only, 10 sweeps | 1.48 % | 85.1 % | 61.6 % | | 1.9 s |

The true primal gap is 0.24 % mean, so after the last row the certificate
is dominated by the primal error (dual slack 0.075 %). At a 1 % threshold
5.5 % of instances would be re-solved, 4.4 % truly exceed it, none are
missed. The Benders cut reads the same rows the master already consumes
(capacity duals in rows G to 2G, λ for the constant) and is valid for every
investment by weak duality.

**Interpretation.** "Exact as soon as the price regime is right" is measured:
91 % of instances end at the exact dual while only 65 % have a primal gap
below 0.1 %. The thesis's dual network was asked to learn this fixed point
from the instance alone (§1b of the summary); given the primal's congestion
pattern it is one sort per region. No dual network, no cost perturbation
(Benders needs a valid dual, not a unique one).

**Source.** `flowfirst/dual_analysis.py` on the run above.

## F27 (major, confirmed): one sweep of exact line search removes most of the kink hedge

**Claim.** The fill cost is linear in a single flow until the residual at
either end reaches a merit-order breakpoint, and the breakpoints are the
cumulative capacities, so a per-line line search is closed-form: move flow
toward the pricier end by the distance to the nearest breakpoint, keep the
move if the cost drops. One sweep over the 44 lines cuts the mean gap five
times. Snapping flows to their limits along price differences does nothing
(thousands of accepted moves, gap unchanged), which says the network's
remaining error is not a wrong congestion pattern but interior flows sitting
a little off a breakpoint, the hedge of F14. This is IDEAS I11, measured.

**Evidence.** Same model and instances as F26.

| flows | gap mean | gap median | no-shortage gap mean | within 1 % | within 0.1 % | certified within 1 % | per 8192 |
|---|---|---|---|---|---|---|---|
| network | 0.241 % | 0.038 % | 0.729 % | 95.6 % | 65.1 % | 94.5 % | |
| + line search, 1 sweep | 0.046 % | 0.003 % | 0.132 % | 99.2 % | 89.9 % | 96.3 % | 40 ms |
| + line search, 3 sweeps | 0.041 % | 0.001 % | 0.118 % | 99.3 % | 91.9 % | 96.5 % | 120 ms |

Every polished point is still a valid upper bound (min gap −4e-15), line
limits hold by construction, and all over-supply violations disappear
(max e < 0 from 172 MW to 3e-11). What remains: 0.7 % of instances with
unmet demand above the node's demand (an over-exporting node, up to 2.9 GW),
the relaxation the fill allows by design; they carry large certificates and
go to the fallback. Re-solve rate at a 1 % certificate: 3.5 %.

Timings are for the linear implementation: a step updates only its two end
nodes, and lines sharing no node move in the same round (11 rounds for the
44 lines), so a sweep costs O(batch × lines × units per node) with a
sequential depth near the maximum degree. The first version recomputed the
whole fill through the dense G × G product at every step, O(batch × N³),
and took 450 ms per sweep at this size (measured 3-node 8 ms, 6-node 38 ms,
20-node 527 ms per sweep, batch 8192). Coordinate descent has one slow mode:
rerouting power through a node that sits exactly on a breakpoint advances
by that node's marginal unit per sweep, since each single-line move is
capped by the kink. From the network's flows it affects 1 % of instances
(24 % still have a single-line descent after one sweep, 1.6 % after three,
0.9 % after thirty) and the mean gap is flat from three sweeps on. Path
moves through a node (two lines at once) would remove it if it ever
mattered.

**Interpretation.** The network's job reduces to getting the regime right;
precision comes from exact local moves that are cheap because the structure
is closed-form, on both sides (line search on the primal, quotient fill on
the dual). Inference pipeline: network, polish, quotient fill, certificate,
fallback, about 0.5 ms per hour on CPU. That is not faster than a
properly used solver at this size: a persistent Gurobi model that only
takes new right-hand sides solves this LP in 0.16 ms, a fresh model in
0.32 ms (F29); the 2.4 ms of the labelling loop was mostly Python.

**Source.** `flowfirst/dual_analysis.py` on the run of F26.

## F28 (major, confirmed): the flows are the one global part; local exact moves cannot find them

**Claim.** The line search of F27 is an exact solver for the flows in the
sense that every step is optimal along its line, so it can be run from zero
flows instead of from the network's. It stalls. Sending power from a cheap
node to an expensive one two lines away needs both flows to move together:
moving the first alone dumps power at the middle node (over-supply, charged
at VOLL), moving the second alone creates a shortage there, so each single
move raises the cost and coordinate descent stops at a corner of the
piecewise-linear objective. Minimizing the fill cost over flows is a
convex-cost network flow problem; the algorithms that solve it (network
simplex, successive shortest paths, or the LP solver) move flow along whole
paths and cycles, and cost what a solver costs. The network supplies that
global routing amortized over instances; exact local moves supply the
precision (F27).

**Evidence.** 1024 validation instances of the F26 model, CPU float64,
2026-09-08. The labelling loop took 2.38 ms per LP, mostly Python
overhead; Gurobi itself solves this LP in 0.16 to 0.32 ms (F29).

| start | sweeps | gap mean | gap median | instances above 1 % | at the optimum (gap < 1e-6) | instances still moving in the last 1000 sweeps | time per instance |
|---|---|---|---|---|---|---|---|
| zero flows | 1 | 172 % | 57 % | 98.8 % | | | 0.09 ms |
| zero flows | 30 | 79 % | 26 % | 94.3 % | | | 2.3 ms |
| zero flows | 1000 | 75 % | 25 % | 94.1 % | | 15.7 % | 77 ms |
| zero flows | 5000 | 72 % | 25 % | 94.0 % | 0.1 % | 9.5 % | 385 ms |
| network flows | 0 | 0.23 % | | 4.4 % | | | 0.3 ms (forward) |
| network flows | 1 | 0.046 % | 0.003 % | 0.8 % | | | + 0.05 ms |

After 5000 sweeps the cold start has spent a thousand times Gurobi's
solve time, is
still creeping (a tenth of the instances accept a move every 1000 sweeps)
and has reached the optimum on one instance in a thousand.

Measured with the first, dense implementation of the line search. The
linear implementation of F27 (two-node updates, coloured rounds, a
different Gauss-Seidel order) stalls the same way: from zero flows 56 %
mean gap after 30 sweeps, 52 % after 1000, 89 % of instances above 1 %,
0.3 % at the optimum, 10.5 ms per instance at 1000 sweeps (four times
Gurobi). The stall is the algorithm's, not the implementation's.

**Interpretation.** This is the division of labour in one experiment.
Everything local and combinatorial, the fill, the quotient fill, the line
search, is one sort or one breakpoint away and is computed exactly. The one
thing that is global, which lines carry power to where, is what the network
learns, and it is not reachable by local moves. It also settles "why train a
network if the pieces are closed-form": the exact alternative for the flows
is a solver, at solver cost per instance, while the network lands inside the
right regime for 0.3 ms and the polish finishes for 0.05 ms.

**Source.** `flowfirst/dual_analysis.py --valid-size 1024 --cold-sweeps 5000`
on the run of F26 (the 30-sweep row is the script's default).

## F29 (major, confirmed): the exact min-cost flow, warm-started by the network, needs a tenth of the work

**Claim.** The dispatch with transport flows is a min-cost flow: a source
node, one arc per generator with its capacity and cost, an unmet-demand arc
at VOLL, and the lines as zero-cost arcs with box capacities. A feasible
flow is optimal iff its residual graph has no negative cycle, and since
lines cost nothing every negative cycle passes through the source: a path
from a node whose marginal unit is cheap to a node whose marginal unit is
dear, over lines with spare capacity, gaining lo_b − hi_a per MW. Max-gain
cycle cancelling, batched in torch (`PathPolish`), reaches Gurobi's optimum
on every validation instance from any start. What the network buys is the
number of augmentations: 4.8 from its polished flows, 46.5 from zero flows.
With exact flows the quotient-fill dual is exact on every instance (the 1 %
congestion tolerance of F26 was for hedged flows; 1e-6 is right for exact
ones), so one call returns a certified optimal primal-dual pair.

**Evidence.** Same model and 8192 instances as F26, CPU float64, 2026-09-08.

| start | augmentations mean / median / p95 / max | instances at the optimum (gap < 1e-9) | ms per instance (batch 8192) |
|---|---|---|---|
| network flows | 14.6 / 15 / 26 / 38 | 100 % | 0.60 |
| network flows + 1 polish sweep | 4.8 / 4 / 12 / 24 | 100 % | 0.23 (+ 0.005 for the sweep) |
| zero flows | 46.5 / 47 / 56 / 66 | 100 % | 1.80 |
| Gurobi, fresh model per instance | 28 simplex iterations | | 0.32 |
| Gurobi, persistent model, previous basis kept | 38 iterations | | 0.16 |

Dual after the exact primal: slack ≤ 1e-6 on 99.5 % of instances with the
1e-2 tolerance, 100 % (max slack 1e-14) with 1e-6. `PrimalDual(exact=True)`:
0.59 ms per instance at batch 8192, true gap 8e-15, certificate ≤ 1e-6 on
100 %; `exact=False` (one polish sweep): 0.4 to 0.5 ms, gap 0.047 %. Gurobi
solves the same LP in 0.16 ms from a persistent model (0.32 ms building
the model each time); the 2.38 ms of the labelling loop was Python
overhead. Warm-starting Gurobi from us: passed as primal-dual start vectors
our exact solution cuts its dual simplex iterations from 28 to 21 but needs
a reset and gains no wall-clock (0.24 ms); passed as an optimal basis
(VBasis/CBasis after a reset, measured with Gurobi's own basis as the
oracle) it takes 0 iterations and 0.05 ms per LP, three times faster than
Gurobi's own persistent warm start. Building that basis from our active set
is the remaining step if the solver is kept in the loop. The polish sweep
before the path step pays
for itself three times over (14.6 → 4.8 augmentations).

**Interpretation.** This is the cleanest form of the division of labour:
the exact solver's work, counted in augmentations, drops by 90 % when it
starts from the network. At this size that does not translate into
wall-clock against a well-used solver: Gurobi with a persistent model is
0.16 ms per LP, our exact pipeline 0.59 ms on CPU. The augmentation loop,
not the network, is the cost (each iteration rebuilds a dense node-pair
matrix in a Python loop over lines); the forward pass on a GPU and a leaner
loop are where the ratio would move, and the scale-up notes stand: the
speed claim needs a subproblem where the solver's cost grows.
The fallback of IDEAS I10 is no longer needed for the primal at this size:
every instance is optimal, and the certificate reports it.

**On the GPU** (A100 via Modal, `modal/bench.py`, ms per instance, median of
5, same model and instances; CPU column is the M2 Pro at batch 8192):

| stage | batch 128 | 1024 | 8192 | CPU, 8192 |
|---|---|---|---|---|
| forward, GNN + fill | 0.043 | 0.018 | 0.017 | 0.32 |
| polish, 1 sweep | 0.20 | 0.025 | 0.003 | 0.005 |
| dual: prices, objective, multipliers | 0.22 | 0.033 | 0.005 | 0.06 |
| exact path step, from polished flows | 3.2 | 0.51 | 0.078 | 0.23 |
| `PrimalDual`, one polish sweep | 0.57 | 0.074 | 0.025 | 0.4 to 0.5 |
| `PrimalDual`, exact | 3.9 | 0.59 | 0.109 | 0.59 |

Correctness on the GPU: max gap 8e-13, weak duality on all 8192. Against
Gurobi's 0.16 ms (persistent model) the certified 0.047 %-gap pipeline is
six times faster at batch 8192 and the exact one 1.5 times; against the
0.05 ms basis-warm-start floor, two times and 0.5 times. The path step is
now 70 % of the exact pipeline and launch-bound: about 20 ms per
augmentation whatever the batch (a Python loop over lines rebuilds the
node-pair matrix, then closure, search and walk, several hundred small
kernels), and a batch pays for its slowest instance's 24 augmentations.
Vectorizing that loop and capturing the iteration as a CUDA graph are the
levers; a five-fold cut would put the exact pipeline at about 0.03 ms.

**Source.** `flowfirst/dual.py` (`PathPolish`, `PrimalDual(exact=True)`),
`flowfirst/dual_analysis.py` section F29, `flowfirst/modal/bench.py`,
tests in `flowfirst/tests/test_dual.py`.

## F30 (major, confirmed): the deployed pipeline is one CUDA graph at 0.0101 ms per instance; what is left is the network's float64 arithmetic

**Claim.** The non-exact pipeline of `PrimalDual` (network forward, one
polish sweep, quotient-fill prices with three node-ascent sweeps, completion,
dual objective, certificate) now has fixed shapes, no host-device
synchronization and no data-dependent control flow, so
`GraphedPrimalDual(pipeline, batch)` records it once as a CUDA graph on a
static input buffer and replays it: a call copies its inputs in, replays,
copies the outputs out, and the per-batch cost is GPU time alone. The replay
is bitwise the eager pipeline, on the CPU and on the A100. Against F29's
0.025 ms per instance at batch 8192 the pipeline is now 0.0101 in float64,
2.5 times faster, and each of the three changes has its own share: the
leaner node ascent (0.0187), `torch.compile` of the network, kept as a
separate switch (`PrimalDual(data, torch.compile(net, dynamic=False))`,
0.0114), and the graph on top (0.0101). F29's reading that half of the 205
ms per batch was Python dispatch was wrong for that batch size: kernel
launches are asynchronous and at 8192 they hide behind the GPU work, so the
graph saves 11 of 153 ms there; at batch 1024, where the eager pipeline is
launch-bound at about 40 ms per batch whatever the GPU does, the graph
saves 19 ms of 42 and, with the compiled network, 24 of 39, a factor 2.5.
What remains at 8192 is GPU time in the network's float64 matrix products,
so the 0.005 target is not reachable in float64 with this network.

Two places had to change to make the pipeline capturable, and one was also a
memory hog:

- `DualRecovery.regions` labelled nodes by min-label propagation until the
  labels stopped changing (`torch.equal`, a host sync per iteration, up to N
  iterations). It now takes the transitive closure of the uncongested-line
  adjacency by ceil(log2(N − 1)) squarings of the [B, N, N] matrix, five on
  the 20-node grid, and reads the smallest reachable index. Same labels.
- `DualRecovery.node_ascent` evaluated every node's update over all 44 lines
  and masked, [B, 44, 10] temporaries per node for 60 node updates (three
  sweeps of 20 nodes). It now works on the node's incident lines only,
  padded to the maximum degree of 11: the same sums, a quarter of the
  traffic; on the CPU the ascent went from 376 to 83 ms per 8192 instances
  and the whole eager pipeline from 6.3 to 3.7 s.
- `one_hot` became a comparison against `arange` (it syncs on the CPU; on
  CUDA it was already sync-free with `num_classes` given).

`PathPolish` stays eager: its augmentation loop runs until no negative cycle
is left, which no graph can record, and it is not in the deployed pipeline.

A partial final batch is not sent through an eager fallback. It fills the
static buffer partly, the unused rows keep whatever the previous chunk left
there (rows are independent and every input is valid), the graph replays
and the first rows are copied out. An eager pass would cost at least the 110
ms of dispatch whatever its size; a replay costs at most one full batch of
GPU time, less than that. Without CUDA the same chunked path runs eagerly,
so the class works on the CPU and MPS and its chunking is tested there.

**Evidence, local (2026-09-09).** `pytest flowfirst`: 40 passed (6 new: the
pipeline runs under `FakeTensorMode`, which raises on every `.item()`,
`torch.equal`, `nonzero` or tensor-valued `if`, the previous `regions`
failed it; the chunked `GraphedPrimalDual` equals eager to 1e-12 on 512, 200
and 37 instances with chunk 200; the network traces as one graph under
`torch.compile(fullgraph=True)` with the trace-only backend and gives the
same flows). On the trained 20-node model, CPU float64, 8192 validation
instances: the new pipeline against the previous `dual.py`, maximum
absolute difference 0 in y, λ, μ, primal, dual and certificate, no instance
differs; `GraphedPrimalDual` with chunk 4096 on 9192 instances (two chunks
and a partial one of 1000) against eager, difference 0.

Roofline of the forward pass, counted from the network (20 nodes, 44 lines,
hidden 128, 6 rounds, 877k parameters): 57 MFLOP per instance and 2.8 MB of
activations that must round-trip memory in float64 (each linear layer's
input and output rows). On an A100 that is 0.0029 ms per instance at the
19.5 TFLOPS fp64 tensor-core peak and 0.0018 ms at 1.56 TB/s, so the float64
floor is 0.003 to 0.0047 ms depending on how much compute and traffic
overlap; the benchmark uses 0.0035. In float32 without TF32 (PyTorch's
default, kept) the compute peak is the same 19.5 TFLOPS, so the float32
floor is again 0.0029 ms unless TF32 tensor cores are enabled (156 TFLOPS,
0.0004 ms compute, 0.0009 ms memory); the benchmark's 0.0018 for float32 is
the memory floor and is not reachable without TF32. Gurobi's marks for the
same LP (F29): 0.16 ms per hour on one core from a persistent model, about
0.02 on eight cores, 0.05 with the optimal basis set after a reset.

**Evidence, A100** (SXM4 80 GB via Modal, torch 2.5.1+cu124, TF32 off,
`modal/bench.py`, 2026-09-09; ms per instance, median of 5; same model and
validation instances as F26 to F29):

| pipeline | float64, 1024 | float64, 8192 | float32, 1024 | float32, 8192 |
|---|---|---|---|---|
| eager (F29: 0.074 / 0.025) | 0.0413 | 0.0187 | 0.0402 | 0.0117 |
| eager + CUDA graph | 0.0230 | 0.0173 | 0.0160 | 0.0098 |
| compiled network | 0.0386 | 0.0114 | 0.0385 | 0.0088 |
| compiled network + CUDA graph | 0.0154 | 0.0101 | 0.0125 | 0.0070 |

Best float64 at 8192, 0.0101: 2.9 times the benchmark's floor of 0.0035
(3.5 times the compute floor of 0.0029); per hour 15.9 times faster than
one-core persistent Gurobi (0.16), 2.0 times faster than eight cores
(0.02), 5.0 times faster than the optimal-basis oracle (0.05). Best float32,
0.0070: 22.9, 2.9 and 7.2 times. The float32 rows ran without TF32
(inductor itself warns about it), so their matrix products have the same
19.5 TFLOPS peak as float64 and the gain is halved traffic; TF32 is the
untried lever.

Deviation from the eager pipeline, 8192 + 1000 instances so that the graph
also sees a partial chunk: the float64 graph rows are exactly 0 in every
field. The compiled rows differ at rounding level, y by 9.5e-11 MW and the
primal by 3.5e-10 kEUR, except on 2 of 8192 instances where a discrete
choice flipped on a tie, a congestion tolerance or a breakpoint comparison
and one node's price moved by one alphabet step (0.05): the dual changes by
180 kEUR there and the certificate by 0.57 percentage points, both prices
being valid lower bounds. In float32 the graph rows differ from eager on 1
instance (a polish breakpoint decided the other way, 2 MW), the compiled
rows on 5 instances with a price jumping between a cost and VOLL; float32
is speed-only.

Reading the rows. Eager → graph is the launch overhead that did not overlap
the GPU: 7 % at 8192, 44 % at 1024. Eager → compiled is the fusion of the
network's elementwise work (LayerNorm, concatenations, ReLU, sigmoid,
residual adds), 60 ms of 153 per batch at 8192 and nothing at 1024 where the
launches, not the kernels, set the time. Compiled → compiled + graph is the
launch overhead left in the glue, 10 ms per batch. The leaner ascent alone
took eager from 0.025 to 0.0187 at 8192 and from 0.074 to 0.041 at 1024:
the old ascent was about 35 ms of GPU time per batch and a good share of
the launches.

**Interpretation.** At batch 8192 the pipeline is GPU-bound: 83 ms per
batch, of which the compiled forward pass is roughly 69 (0.0084 per
instance, about three times the 24 ms the 467 GFLOP would take at the fp64
tensor-core peak; cuBLAS does not reach that peak on [360k, 384] × [384,
128] products and the fused elementwise kernels still move the 2.8 MB per
instance) and polish plus dual about 14. So 0.005 ms per instance, 41 ms per
batch, is below what this network's float64 matrix products cost on an A100,
and the levers now are the network's arithmetic, not its dispatch: TF32 (or
bf16) products in the network with the fill, polish and dual kept in
float64, which cuts the matmul share up to eightfold and needs a check that
the certified gap survives the 10-bit mantissa (the polish removes kink
error, F27, so it may); or a smaller network (hidden 96, four to five
rounds) if F24-level accuracy holds. A larger batch buys nothing. Batch 1024
is the operating point for latency: 15.8 ms per batch with the compiled
graph, so a year of 8760 hours in nine replays, 142 ms, against 166 ms in
two replays of 8192.

The Gurobi comparison stands as F29 framed it: per hour of the 20-node ED
LP the certified 0.047 %-gap solution now costs a sixteenth of a persistent
single-core solve and half of an eight-core one, with the GPU at about the
price of those cores; the exact pipeline (`exact=True`, 0.109 in F29) is
untouched by this finding, its path step stays eager. F31 measures the
levers named above.

**Source.** `flowfirst/dual.py` (`GraphedPrimalDual`,
`DualRecovery.regions`, `DualRecovery.node_ascent`, `load_run(dtype=)`),
`flowfirst/tests/test_dual.py`, `flowfirst/modal/bench.py` (`benchmark`,
`--stages` for the F29 table); A100 log of 2026-09-09, Modal app
ap-HQk3gsVkycqKQ7qPrpxkyN.

## F31 (major, confirmed): reduced precision belongs in the network only; float16 there beats the target with float64's accuracy

**Claim.** F30 left the pipeline GPU-bound in the network's float64 matrix
products, at the same 19.5 TFLOPS peak the A100 gives float32 without TF32.
The cheap arithmetic is TF32 (10-bit mantissa inputs, float32 accumulation,
156 TFLOPS) or half precision (312 TFLOPS), and it belongs in the network
alone. `cast_net` copies the GNN with its parameters (and the 0/1 incidence
buffers of the message passing) in the cheaper dtype; the forward pass
standardizes the features in the input's dtype, casts them, runs encoders,
rounds and readout in the parameter dtype, and squashes the logit back in
the input's dtype against the float64 line limits. Raw inputs in MW would
overflow float16 (65504), standardized features never do. So the fill, the
polish, the prices, the dual objective and the certificate are computed in
float64 as before: the dispatch balances to float64 precision and the dual
is a valid lower bound exactly, whatever the network did; the only effect
of the cheaper arithmetic is on the flows, and the polish sweep removes
most of what that costs (F27). Casting the whole pipeline to float32
instead keeps the speed but loses the certificate: dual objective and cost
are then rounded at the 1e-6 level on values of 1e6 kEUR, so weak duality
and `certificate >= gap` hold only up to about 1e-5, and the dispatch
balances to about 1e-2 MW.

**Evidence, local (2026-09-09).** `pytest flowfirst`: 46 passed (6 new:
with the small network cast to float32, float16 and bfloat16 the outputs
are float64, the balance residual below 1e-6 MW, the primal equals `obj_fn`
of the returned dispatch, the dual stays below the optimum and the
certificate above the true gap on both 3-node datasets). The float64 path
is unchanged: flows equal the previous formula bitwise on both 3-node
datasets and the 20-node validation set matches the pre-F30 code in every
field. On the way `line_bounds` was found to return int64 on the 3-node
datasets (integer line limits); it now returns the default dtype.

CPU smoke run of `modal/bench.py` on 128 validation instances of the
20-node model, scored in float64 (no TF32 or graphs on the CPU, so these
rows only show what precision does to the outputs):

| precision | gap mean | median | within 1 % / 0.1 % | certificate mean | max (dual − opt)/opt | max (gap − certificate) | max balance error |
|---|---|---|---|---|---|---|---|
| float64 | 0.050 % | 0.003 % | 0.992 / 0.883 | 0.312 % | 2e-15 | 2e-15 | 2e-11 MW |
| float32 throughout | 0.050 % | 0.004 % | 0.992 / 0.883 | 0.313 % | 5e-6 | 5e-6 | 9e-3 MW |
| net-fp32: network float32, rest float64 | 0.050 % | 0.003 % | 0.992 / 0.883 | 0.312 % | 2e-15 | 2e-15 | 2e-11 MW |
| net-fp16: network float16, rest float64 | 0.050 % | 0.004 % | 0.992 / 0.898 | 0.313 % | 2e-15 | 2e-15 | 2e-11 MW |
| net-bf16: network bfloat16, rest float64 | 0.049 % | 0.003 % | 0.992 / 0.883 | 0.288 % | 2e-15 | 2e-15 | 3e-11 MW |

**Evidence, A100** (SXM4 80 GB via Modal, torch 2.5.1+cu124, `modal/bench.py
--precisions float64,tf32,net-fp32,net-fp16,net-bf16`, 2026-09-09; ms per
instance, median of 5; same model and validation instances as F26 to F30).
`tf32` is everything in float32 with TF32 products; `net-*` is the network
cast, everything else float64:

| pipeline | float64 1024 | float64 8192 | tf32 1024 | tf32 8192 | net-fp32 1024 | net-fp32 8192 | net-fp16 1024 | net-fp16 8192 | net-bf16 1024 | net-bf16 8192 |
|---|---|---|---|---|---|---|---|---|---|---|
| eager | 0.0410 | 0.0202 | 0.0393 | 0.0088 | 0.0394 | 0.0090 | 0.0397 | 0.0075 | 0.0404 | 0.0072 |
| eager + CUDA graph | 0.0242 | 0.0189 | 0.0120 | 0.0071 | 0.0125 | 0.0076 | 0.0107 | 0.0060 | 0.0107 | 0.0058 |
| compiled network | 0.0374 | 0.0125 | 0.0377 | 0.0069 | 0.0379 | 0.0070 | 0.0384 | 0.0055 | 0.0384 | 0.0054 |
| compiled + CUDA graph | 0.0162 | 0.0111 | 0.0102 | 0.0052 | 0.0106 | 0.0057 | 0.0089 | 0.0040 | 0.0089 | 0.0041 |

The float64 column is 8 to 10 % slower than in F30's run (0.0101 there):
that is the run-to-run spread between two A100 instances, so differences
below a tenth between rows mean nothing. Per hour of the ED LP, the float16
network's compiled graph at 8192 is 39.9 times faster than one-core
persistent Gurobi (0.16 ms), 5.0 times faster than eight cores (0.02), 12.5
times faster than the optimal-basis oracle (0.05); the TF32 network 28.2,
3.5 and 8.8 times.

Accuracy of the compiled graph on the 8192 validation instances, the
returned dispatch scored in float64 against the Gurobi labels:

| precision | gap mean | median | no-shortage mean | ratio of totals | within 1 % / 0.1 % | certificate mean | re-solve at 1 % | max (dual − opt)/opt | max (gap − certificate) | max balance error |
|---|---|---|---|---|---|---|---|---|---|---|
| float64 | 0.047 % | 0.003 % | 0.135 % | 0.007 % | 0.993 / 0.899 | 0.624 % | 6.7 % | 2.6e-14 | 2.6e-14 | 2.9e-11 MW |
| tf32, everything float32 | 2.06 % | 0.383 % | 6.31 % | 0.208 % | 0.655 / 0.161 | 2.68 % | 36.2 % | 1.4e-5 | 1.4e-5 | 6.1 MW |
| net-fp32 | 0.047 % | 0.003 % | 0.135 % | 0.007 % | 0.993 / 0.899 | 0.639 % | 6.7 % | 2.6e-14 | 2.6e-14 | 2.9e-11 MW |
| net-fp16 | 0.047 % | 0.003 % | 0.135 % | 0.007 % | 0.993 / 0.900 | 0.625 % | 6.7 % | 2.6e-14 | 2.6e-14 | 2.9e-11 MW |
| net-bf16 | 0.047 % | 0.003 % | 0.135 % | 0.008 % | 0.993 / 0.898 | 0.563 % | 6.5 % | 2.6e-14 | 2.6e-14 | 2.9e-11 MW |

The three cast networks reproduce every gap metric of the float64 pipeline
to the third digit, with exact bounds and a balanced dispatch. The
all-float32 row is not the 1e-5 certificate loss of the CPU preview: with
TF32 on, the fill's and the polish's own matrix products (residual `D − f
Aᵀ`, cumulative capacities) run with 10-bit mantissas on values of 1e4 to
1e5 MW, so residuals are off by tens of MW, the exact line search moves
flows to wrong breakpoints, and the gap goes from 0.047 % to 2.06 % with
a 6 MW balance error. Float32 without TF32 was not scored at this size (the
CPU preview on 128 instances put it at float64's gap with a 5e-6 certificate
loss and 9e-3 MW imbalance); either way the fill and the polish want
float64.

Deviation lines (the graph and the compiled network against each
precision's eager pipeline, 8192 + 1000 instances): the graph replays are
bitwise for the float16 and bfloat16 networks, and differ on one instance
for the TF32 network, where cuBLAS picks a different kernel for the batch
of 9192 than for 8192 and one polish breakpoint decides the other way (2.2
MW). The compiled network differs from eager on 0.9 % (TF32), 0.7 %
(float16) and 3.9 % (bfloat16) of instances by one flipped discrete choice,
a price jumping between a cost level and VOLL: inductor keeps fused
intermediates in float32 where eager rounds every op to the network's dtype,
so the two are different roundings of the same network, and the accuracy
table, measured on the compiled graph, shows neither loses anything.

**Interpretation.** The target of F30, 0.005 ms per instance at batch 8192,
falls with the network in float16 and everything else in float64: 0.0040,
2.8 times faster than the float64 graph and 6 times faster than F29's
eager pipeline, at float64's accuracy. TF32 alone gives 0.0057, half the
float64 time, also at float64's accuracy; float16 and bfloat16 are equal
within noise, so float16 (three more mantissa bits) is the choice. Per
batch of 8192 the float16 pipeline takes 33 ms of which the network is
roughly 19 (against 77 in float64 and 33 with TF32) and the float64 fill,
polish and dual about 14, so the glue is now 40 % of the time and the next
lever, if one is needed: the ascent's 60 node updates and the region
closure are the candidates. The network's own floor in float16 is well
below a millisecond per batch; what it spends is elementwise work
(LayerNorm, concatenations, casts) in inductor's kernels, not the products.
The division of labour is now the precise one: cheap arithmetic where the
result is a prediction anyway, exact arithmetic where the result is a
bound. Deployment: `cast_net(net, torch.float16)`, `PrimalDual` on the
float64 data, `torch.compile(dynamic=False)` on the network,
`GraphedPrimalDual` at batch 8192 (throughput) or 1024 (latency: 9.1 ms
per batch, a year of hours in 82 ms).

**Source.** `flowfirst/dual.py` (`cast_net`, `cast_data`),
`flowfirst/train.py` (`FlowFirstGNN.forward` dtype boundary, `line_bounds`),
`flowfirst/tests/test_dual.py`, `flowfirst/modal/bench.py` (`PRECISIONS`,
`accuracy`); A100 log of 2026-09-09, Modal app ap-LZMBBJBJB5J1FtUIqsq7Ne.


## F32 (major, confirmed): the sweep divides every network's gap by ten and keeps the ranking

**Claim.** Applied to the flows of every saved 6-node run, one exact
line-search sweep (F27) removes 85 to 93 % of the gap whatever the network,
and three sweeps a little more. The ranking of the networks survives it:
after the sweep the full-recipe GNN is still six times better than the
full MLP and the quick GNNs, and the fixed-rate GNN four times worse than
the step-decay one, so what remains after the sweep scales with the raw
quality rather than vanishing. A weaker or shorter-trained network plus the
sweep reaches what the stronger network reached raw (the full MLP at 0.18
to 0.23 % and 95 to 96 % within 1 % is F24's raw GNN), not what the stronger
polished network reaches (0.03 %). The routing deficit is negligible on six
nodes for every network: the exact path step (F29) needs 0.7 to 1.2
augmentations on average and at most 8 from any of them. The sweep also
rescues the thesis primal: its flows polished by merit order are only 1.6
times worse than flow-first's, so F17's separation was mostly the production
head, which the fill replaces.

**Evidence.** `polish_compare.py` on the 8192 validation instances of the
6-node dataset, CPU float64, 2026-09-10; gap mean, no-shortage mean, share
within 1 %:

| run | raw | 1 sweep | 3 sweeps | augmentations mean / max |
|---|---|---|---|---|
| MLP 3 layers, 100 epochs, 65k instances (quick) | 5.80 % / 11.8 % / 49 % | 0.66 % / 1.31 % / 87 % | 0.49 % / 0.97 % / 90 % | 1.2 / 8 |
| MLP 3 layers, 250 epochs, full data | 2.20 % / 4.64 % / 60 % | 0.23 % / 0.49 % / 94 % | 0.19 % / 0.39 % / 95 % | 1.0 / 7 |
| MLP 3 layers, 500 epochs | 3.20 % / 7.15 % / 67 % | 0.18 % / 0.39 % / 96 % | 0.15 % / 0.31 % / 96 % | 0.9 / 7 |
| MLP 4 layers, 500 epochs | 2.15 % / 4.57 % / 64 % | 0.49 % / 1.12 % / 96 % | 0.36 % / 0.81 % / 97 % | 0.9 / 7 |
| thesis primal (old-prioritized), 250 epochs | 3.45 % / 7.31 % / 58 % | 0.37 % / 0.75 % / 94 % | 0.26 % / 0.50 % / 95 % | 1.0 / 7 |
| GNN 64 wide, 2 rounds, quick | 2.89 % / 6.18 % / 60 % | 0.31 % / 0.64 % / 94 % | 0.25 % / 0.52 % / 95 % | 1.0 / 7 |
| GNN 128 wide, 2 rounds, quick | 1.69 % / 3.55 % / 62 % | 0.21 % / 0.45 % / 95 % | 0.18 % / 0.38 % / 95 % | 0.9 / 7 |
| GNN 96 wide, 3 rounds, quick | 1.98 % / 4.23 % / 59 % | 0.17 % / 0.37 % / 96 % | 0.14 % / 0.30 % / 96 % | 0.9 / 7 |
| GNN 96 wide, 4 rounds, quick | 1.80 % / 3.87 % / 61 % | 0.19 % / 0.40 % / 96 % | 0.16 % / 0.33 % / 97 % | 0.9 / 8 |
| GNN 96 / 4, full data, fixed rate | 0.98 % / 2.07 % / 77 % | 0.13 % / 0.25 % / 98 % | 0.11 % / 0.21 % / 98 % | 0.7 / 7 |
| GNN 96 / 4, full data, step decay + clip (F24) | 0.27 % / 0.58 % / 96 % | 0.033 % / 0.071 % / 99.4 % | 0.029 % / 0.063 % / 99.4 % | 0.7 / 6 |

Quick runs: 100 epochs on 65k instances; full runs: 250 or 500 epochs on
210k. The median gap after one sweep is below 1e-5 for every run: the
sweep puts the typical hour on the optimum, and the mean is the tail.

**Interpretation.** The sweep is a near-constant factor, not a floor: it
removes the kink hedge (F14) from any network, and what it leaves is the
routing error the network made, in proportion to how well it was trained.
So the trade Ben asked about, a smaller or shorter-trained network plus the
sweep, is available at a price: the quick GNN plus one sweep (0.17 to 0.21
%, 95 to 96 % within 1 %) costs a quarter of the data and 40 % of the
epochs of the full recipe and lands six times above it. Depth is not the
lever on six nodes (three rounds suffice, diameter three), the training
recipe is: step decay alone takes the polished gap from 0.13 % to 0.03 %.
These networks were trained on the raw cost; a network trained through the
polish would spend its capacity on the routing the sweep cannot do, which
is the open experiment. The augmentation counts say that on six nodes there
is almost no routing left to learn; the 20-node grid (4.8 augmentations,
F29) is where that question has content.

**Source.** `flowfirst/polish_compare.py` over `flowfirst/runs/*6node*`;
`load_run` now rebuilds the MLP variants too.

## Open questions and next steps

The backlog of training and architecture changes, with rationale and status,
lives in `IDEAS.md`.

1. Learning-rate schedules are answered (F15): keep the fixed rate.
2. Optimizer budget is answered (F17): batch 128 and three layers are the
   new default setting for comparisons. Flow-first was still improving at
   250 epochs; run longer.
3. Data is answered (F18): 262k instances close the gap. Next lever for the
   last 0.3 % is capacity (width, depth) and steps.
4. Seeds: F17 and F18 rest on one seed each.
5. Six nodes are done (F24). The comparison default from here: GNN, 4
   rounds, step decay 0.7 / 50, clip 8000, batch 128, 250 epochs. Seeds on
   F24 are still open, and the IDEAS backlog (I1 weight averaging, I2
   breakpoint features, I3 prices in the loop) is where further precision
   would come from.
7. The full instance: 20 nodes, 107 generators, 44 lines, graph diameter 5,
   NED and NOR without dispatchable units. See the scale-up notes at the end
   of this file.
   If the no-shortage floor stays above 1 %, the relative-flow
   parameterization, now naturally per node inside the GNN.
6. Beyond the ED primal: duals from the primal with the closed-form gap (the
   certificate and the Benders cut), robustness to the investment
   distribution the master produces, and the 20-country scale, the full dataset.
3. Seeds: F2 and F9 rest on one seed each.
4. Certificate and re-solve: answered on 20 nodes (F26, F27, F29): after
   the polish 3.5 % of instances exceed a 1 % certified gap; with the exact
   path step every instance is optimal and certified, no re-solve needed.
5. Duals from the primal and the closed-form gap (summary §2): done,
   `dual.py` (F26, F27); wiring `PrimalDual` into `gep_benders.py` is the
   next step (the problem class wants the price negated, see the module
   docstring).

---

## Scale-up notes: the full instance (20 nodes, 107 generators, 44 lines)

Written 2026-09-06 before any run at that size.

- **Size.** Input dimension 127 (20 demands, 107 capacities); LP per instance
  171 variables, still milliseconds for Gurobi, so 262k labels cost about
  10 to 15 minutes; the node-budget sampler's Python loop is the slower part.
  The pickle with all labels is 1 to 2 GB in float64; drop the inequality
  duals or store labels in float32 if that hurts.
- **Graph.** Diameter 5 against 3 on six nodes, so the GNN needs 5 to 6
  rounds for every edge readout to see the whole grid, or fewer rounds plus
  a global term; pre-LayerNorm residual blocks (IDEAS I8) become relevant at
  that depth. Rounds scale the step time linearly: expect 30 to 40 ms per
  step at hidden 96 and 6 rounds, about 4 hours for 250 epochs at 262k
  instances.
- **Regimes.** More interior flows, more ties (gas, oil and the renewables
  at identical costs in most nodes; harmless, F19), two pure importers (NED
  and NOR have only renewables), likely a high shortage share. Read the
  ratio of totals and the no-shortage split, as before.
- **What carries over unchanged.** The fill (verified exact on every 6-node
  label), the census, the metrics, the job runner, z-scored node features
  shared across nodes, the step schedule and clipping. The identity
  embeddings stay valid because the topology is fixed.
- **What the paper needs from this size.** The speed argument: a batch of
  thousands of hours through the GNN against thousands of LP solves. Measure
  wall time per hour for both, on the same machine, before claiming it.
- **After the primal.** Duals from the primal with the closed-form gap
  (summary §2) for the Benders cut and the certificate; robustness to the
  investment distribution the master produces; then the Benders loop itself
  through the existing `primal_net` / `dual_net` interface in
  `gep_benders.py`, minding its sign convention (stored λ is minus the
  price).
