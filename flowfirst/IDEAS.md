# Ideas: what we could still change in how the flow network is trained

A prioritized backlog, kept next to `FINDINGS.md`. Each entry says what the
change is, why it should help *this* problem, what it costs to build, and its
status. Move entries to `FINDINGS.md` once they have been measured. Written
2026-09-06 after F1 to F23.

Three properties of the problem drive most of the list:

1. **The loss is a subgradient loss.** Its gradient on a flow is a price
   difference of constant size until a kink, then it flips. It never shrinks
   as the fit improves, so a fixed-step optimizer circles the optimum
   (F15, F18, the GNN oscillation).
2. **The target is piecewise-linear with a cliff.** Flows must land on
   breakpoints; the VOLL cliff makes hedging the rational response to any
   imprecision (F4, F14, F20).
3. **The box repair is a sigmoid**, which has no gradient at the bounds
   (F21: the trap).

| # | Idea | Why here | Cost | Status |
|---|---|---|---|---|
| I1 | **Weight averaging for evaluation** (EMA of the weights, evaluate the average) | The averaged iterate is how subgradient methods converge; removes the circling without lowering the rate; stabilizes the reported curve | small | open, first |
| I2 | **Breakpoint features** per node: cumulative merit-order capacities | Hands the network the kink locations it keeps getting wrong (F20); "compute what is combinatorial" | trivial | open |
| I3 | **Prices in the loop** (preferred over a discrete breakpoint head: F25 shows 95k global regimes and 28 % unseen, so any discrete choice must stay per node, and the continuous route avoids the softmax-mixture pitfall of the thesis's classification layer): predict flows, run the fill, feed nodal prices and unmet demand back as node features, predict a correction, 2 to 3 rounds, loss summed over rounds | Makes message passing the price propagation of the summary; graph neural solvers for power flow work this way; gives the network which node is at a breakpoint and on which side | medium | open |
| I4 | **Smooth fill during training**: softplus clamps with a temperature annealed to zero; exact fill at evaluation | Gradients vanish near kinks late in training instead of flipping; the summary's perturbation done continuously | medium | open |
| I5 | **VOLL annealing**: training VOLL from about 0.4 up to 10 over the run | F10: a low VOLL gives precision but overshoots under the true metric; a schedule gets precision first, the right incentive last | small | open |
| I6 | **Penalty box instead of sigmoid**: unconstrained flow output, exact L1 penalty outside the line limits during training, clip at inference | No dead gradient anywhere; any penalty weight above the largest price difference keeps the optimum; the DC3 / E2ELR way of handling inequalities | small | open |
| I7 | **GNN hyperparameter sweep**: learning rate (5e-4 was tuned for the MLP), short warmup, AdamW weight decay, batch 256 with a scaled rate | Never done for the GNN; due diligence | small | open |
| I8 | **Pre-LayerNorm residual blocks in the GNN** | The standard for deep message passing; needed when rounds grow with the diameter at 20 nodes; provably prevents oversmoothing | small | open, for 20 nodes |
| I9 | **Targeted data**: oversample instances with interior flows and breakpoint nodes; add instances from Benders investment trajectories | F18: data is the strongest lever; the master will query capacities the sampler never drew | medium | open |
| I10 | **Certificate and fallback**: bound each prediction's gap through the dual, re-solve above a threshold | The tail (4 % of instances) is what this is for; needed for Benders anyway (summary §2, Klamkin et al. 2025) | large | measured: F26, F27, F29 (`PrimalDual(exact=True)` is optimal and certified on every instance, no fallback needed at 20 nodes); Benders wiring still to do |
| I11 | **Inference polish**: a few batched projected subgradient steps on the flows with the exact fill as objective, or a snap to the nearest breakpoint when it lowers the fill cost | Cheap with a learned warm start; targets the kink error directly | small | measured: F27 (gap 0.24 % to 0.046 % in one sweep) |
| I16 | **Quadratic flow regularizer** ε Σ_l (f_l / F_l)² in the training loss, ε below Δc_min·F_min / (2·diameter) so no non-degenerate optimum changes (0.54 on the 6-node set; ≤ 0.007 % of the objective) | Half of the 6-node instances have a feasible circulation at the optimum (median free range 236 MW, 75th percentile 3.6 GW), plus tie plateaus: the target is a face, not a point. The regularizer selects the minimum-norm flow, a unique target continuous in the inputs, and gives the flat directions curvature. Ben's proposal, 2026-09-07 | trivial | open |
| I12 | Smooth activations (GELU) in the message-passing body, ReLU kept on the readout | Stabler training; the target is piecewise-linear so ReLU stays on the output path | trivial | low priority |
| I13 | Drop identity embeddings when topology transfer matters | CANOS gets N-1 robustness by having none | trivial | for later |
| I14 | Learning-rate decay (step, cosine), gradient clipping | Done: step decay helps a little (F18), cosine hurts (F15), clipping added and under test on 6 nodes | | measured / running |
| I15 | Loss normalization by the optimum or log objective; training on no-shortage instances only | Done: no effect on the metric (F7, F8) | | closed |

Things not to change without a reason: ReLU on the readout path, z-scored
inputs (F9), float64, the merit-order fill itself, batch 128 (or scale the
rate with the batch).

Reading list: Chen, Tanneau & Van Hentenryck, *End-to-End Feasible
Optimization Proxies* (arXiv 2304.11726); Klamkin, Tanneau & Van Hentenryck,
*Self-Certifying Primal-Dual Optimization Proxies* (arXiv 2510.15850);
Donti, Rolnick & Kolter, *DC3* (arXiv 2104.12225); Donon et al., *Neural
networks for power flow: graph neural solver* (2020); PowerFlowNet (arXiv
2311.03415); CANOS (arXiv 2403.17660); *Residual connections provably
mitigate oversmoothing* (arXiv 2501.00762); *When, where and why to average
weights?* (arXiv 2502.06761).
