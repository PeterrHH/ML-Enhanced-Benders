# Related work: where the flow-first certified proxy sits in the literature

**Written 2026-09-10.** This file places the method described in `FINDINGS.md`
(F17, F25–F32) and `flow_first_summary.md` against the published literature.

**Method in one paragraph, for the reader of this file.** The economic-dispatch
(ED) LP subproblem of a Benders decomposition for generation expansion planning
is replaced by a proxy. A graph neural network over the grid predicts *only the
line flows* (an edge readout, squashed by a sigmoid so line limits hold by
construction). Given the flows, each node's residual is covered by a closed-form
**merit-order fill**: generators sorted by marginal cost, filled greedily, the
remainder charged as unmet demand at the value of lost load. That completion is
*optimal*, not merely feasible, so the dispatch cost is exact and the training
loss is the cost itself (self-supervised, no solved labels in the loss). By the
envelope theorem the gradient of the cost with respect to a line flow is the
nodal price difference across the line. The nodal prices are then recovered
**from the primal's own congestion pattern** in closed form — contract the
uncongested lines into regions, run one merit-order fill per region on its
aggregate residual, then node-wise exact coordinate ascent over the finite price
alphabet — and completed to a dual-feasible point, giving a valid lower bound and
a per-instance optimality certificate. An **exact polish** removes the residual
error: closed-form coordinate descent along single lines, then max-gain cycle
cancelling, since the transport-model dispatch is a min-cost flow. The deployed
pipeline runs as one CUDA graph with the network in float16 and every exact step
(fill, polish, dual, certificate) in float64. A further idea under discussion:
for subproblems with no closed-form completion (ramping, storage), learn a
per-node value function ("critic") of the node's residual from small offline
solves, input-convex in the residual, and train the flow network through the sum
of those critics, actor-critic style.

**Verification standard.** Every entry below was checked against a primary
source: the arXiv abstract page or arXiv HTML/PDF full text, the PMLR/NeurIPS
proceedings page, the publisher page, or the Crossref publisher record for the
DOI. No entry rests on a blog, a Semantic Scholar summary, or recollection.
Where a bibliographic detail could not be confirmed from a primary source it is
marked and listed again under **Unverified** at the end. Several citations that
circulate informally in our notes turned out to be **wrong**; those are flagged
inline with ⚠️.

---
## 1. Learned subproblem value functions used to optimize coupling variables

This is the literature the "certified critic" extension (`flow_first_summary.md`
§5) would join. The short answer of the search: the **ingredients all exist
separately**, and the **combination we would use does not appear**. Details and
near-misses at the end of this section.

### Neur2SP

Justin Dumouchelle\*, Rahul Patel\*, Elias B. Khalil, Merve Bodur,
*Neur2SP: Neural Two-Stage Stochastic Programming*, Advances in Neural
Information Processing Systems 35 (NeurIPS 2022). arXiv:2205.12006.
(\* equal contribution. The NeurIPS proceedings web page lists the authors in a
different order — Patel first — than the PDF and arXiv; cite the PDF order.)

Two architectures. NN-E maps a first-stage decision `x` together with a *set* of
scenarios to the **expected second-stage value**, embedding the scenario set
DeepSets-style (one shared network per scenario, mean-aggregated) and
concatenating that embedding to `x`. NN-P maps `(x, ξ)` to the single-scenario
value and the expectation is taken outside. Either trained network is then
**"MIPified"**: its ReLUs become big-M mixed-integer constraints, and a solver
(Gurobi) optimizes over the surrogate to pick `x`. Training is **supervised** —
each label costs one solved second-stage problem.

*Relation.* Same high-level move as our extension: replace an expensive
subproblem by a learned value function and then optimize the master/coupling
variable against it. Three differences. (i) Their value function is **global** —
one network over the whole first-stage vector and the whole scenario set — where
ours would be **per node**, of that node's scalar residual, summed. (ii) They
optimize the coupling variable with a **MIP solver over the embedded network**;
we would optimize it by **gradient descent through the critic**, which is what
makes the critic's *input derivative* (not its value) the quantity that must be
right. (iii) They are supervised on solved subproblems; our loss today is the
exact cost itself.

### Neur2RO

Justin Dumouchelle, Esther Julien, **Jannis Kurtz**, Elias B. Khalil,
*Neur2RO: Neural Two-Stage Robust Optimization*, ICLR 2024. Preprint
arXiv:2310.04345.

⚠️ **Two citation traps.** The third author is **Jannis Kurtz** — our notes had
"Michael", which is wrong. And arXiv:2310.04345 has been **retitled**: v3
(1 Nov 2024) is *"Deep Learning for Two-Stage Robust Integer Optimization"*.
Cite the ICLR 2024 proceedings entry; if the arXiv id is given, mark it as the
preprint.

Two-stage robust optimization is a nested min-max-min. Neur2RO learns a network
estimating the second-stage **value (and, in the newer version, feasibility)**
and embeds it inside **column-and-constraint generation**, so the learned value
function accelerates an otherwise exact decomposition. The newer abstract adds
approximation guarantees that scale with the network's prediction error.
Benchmarks: knapsack, capital budgeting, facility location.

*Relation.* The guarantee structure is instructive and is the opposite of ours:
theirs degrades with prediction error, ours (the dual certificate, F26) is valid
for **any** prediction and only its *tightness* depends on quality. Otherwise the
same three differences as Neur2SP apply.

### Neur2BiLO (the third member of the family)

Justin Dumouchelle, Esther Julien, Jannis Kurtz, Elias B. Khalil,
*Neur2BiLO: Neural Bilevel Optimization*, NeurIPS 2024. arXiv:2402.02552.
"embeds a neural network approximation of the leader's or follower's value
function, trained via supervised regression, into an easy-to-solve mixed-integer
program." An arXiv author sweep confirms the family is exactly these three.

*Relation.* Same pattern, bilevel instead of two-stage. Confirms that "supervised
global value surrogate, embedded and solved by a MIP" is the established recipe
in this line, and that a *summed, per-block, gradient-trained* critic is not.

### SDDP as the exact, cut-based version of a learned value function

M. V. F. Pereira, L. M. V. G. Pinto, *Multi-stage stochastic optimization applied
to energy planning*, Mathematical Programming **52**(1–3):359–375, May 1991.
DOI 10.1007/BF01582895 (volume, pages, date and authors confirmed via Crossref).

From the abstract, verbatim: the method rests on "the approximation of the
expected-cost-to-go functions of stochastic dynamic programming by piecewise
linear functions… The piecewise functions are obtained from the dual solutions of
the optimization problem at each stage and **correspond to Benders cuts in a
stochastic, multistage decomposition framework**." Case study: a 39-reservoir
system.

Modern analysis: Alexander Shapiro, *Analysis of stochastic dual dynamic
programming method*, European Journal of Operational Research **209**(1):63–72,
2011, DOI 10.1016/j.ejor.2010.08.007. Shapiro states plainly that the current
approximation of the cost-to-go is "given by the maximum of a collection of
cutting planes", that each such plane satisfies `Q(x) ≥ ℓ(x)`, that "the above
'backward step' procedure is the standard cutting plane algorithm (Kelley's
cutting plane algorithm) applied to the SAA problem", and that these "give lower
bounds for the optimal value."

*Relation.* SDDP is the exact instance of the pattern our critic would
approximate: a convex value function of a state, built as a polyhedral *lower*
approximation from duals, and its slope is the price. Two contrasts worth making
in the paper. (i) SDDP's value function is **built by cuts and is therefore a
valid bound at every iteration**; a learned critic is not, which is why our
extension keeps the certificate on a *completed predicted dual* rather than on
the critic. (ii) SDDP recomputes the cuts inside the algorithm; a critic
amortizes across instances. The natural comparison point for the storage
extension (`flow_first_summary.md` §3) is SDDP, not a proxy.

⚠️ **"Water values" is our gloss, not the source's.** The Pereira–Pinto abstract
says "expected-cost-to-go functions"; it does not use the term. The identity
*water value = marginal value of stored water = slope of the cost-to-go in
storage* is standard hydro-scheduling vocabulary but was **not** verifiable from
either primary source read here. Use it as framing, not as a quotation. Also
"outer approximation" is our phrasing; Shapiro says "cutting planes", "maximum of
a collection of cutting planes", "lower bounds". The substance is supported; the
words are ours.

### Input-convex neural networks

Brandon Amos, Lei Xu, J. Zico Kolter, *Input Convex Neural Networks*,
Proceedings of the 34th ICML, PMLR **70**:146–155, 2017. arXiv:1609.07152.

The fully input-convex net stacks `z_{i+1} = g_i(W_i^{(z)} z_i + W_i^{(y)} y +
b_i)`, and Proposition 1 states the constraint exactly: `f` is convex in `y`
"provided that all `W_{1:k−1}^{(z)}` are non-negative, and all functions `g_i`
are convex and non-decreasing." Bias terms and the `W^{(y)}` pass-through weights
may be negative — hence the "passthrough" layers, which exist because the
non-negativity constraint otherwise stops hidden units reproducing an identity
map. From the abstract: the networks "allow for efficient inference via
**optimization over some inputs to the network given others**."

*Relation.* This is the tool the critic plan names, and the property it needs is
exactly the one Amos et al. advertise: because the network is convex in the
residual, minimizing the summed critics over the flows is a convex problem, and
the critic's input-gradient is a subgradient of a convex function — i.e. a price.
What ICNNs do **not** give is any guarantee that that input-gradient matches the
true multiplier, which is the failure mode `flow_first_summary.md` §5 diagnoses
(the leak `Σ_g J_g = 1 − ε`).

### Convex learned models in energy control

Yize Chen\*, Yuanyuan Shi\*, Baosen Zhang, *Optimal Control Via Neural Networks:
A Convex Approach*, ICLR 2019 (poster; OpenReview forum `H1MW72AcK7`).
arXiv:1805.11835.

⚠️ **What is learned as convex here is the *dynamics model*, not a cost-to-go.**
From the abstract: "we design **input convex recurrent neural networks** to
capture temporal behavior of dynamical systems. Then optimal controllers can be
achieved via solving a **convex model predictive control** problem." The energy
application is **building HVAC control** (up to 20 % energy reduction versus
classic linear models), plus MuJoCo locomotion. Not a battery.

*Relation.* Cite it as precedent for "make the learned object convex so the
downstream optimization stays tractable", which is our reason too. Do **not**
cite it as precedent for a convex *value function*; that is Zhang et al. (§3) and
Rosemberg et al. (below).

### Learned convex value functions for OPF

Andrew Rosemberg, Mathieu Tanneau, Bruno Fanzeres, Joaquim Garcia, Pascal Van
Hentenryck, *Learning Optimal Power Flow value functions with input-convex neural
networks*, Electric Power Systems Research **235**:110643, October 2024
(DOI 10.1016/j.epsr.2024.110643). Preprint arXiv:2310.04605.

An ICNN with skip connections takes `x⁰ = (p^d, q^d)` — the **system-wide**
active and reactive demand vector — and outputs the **optimal objective value**
of an OPF, for AC-OPF, its SOC relaxation, and DC-OPF, on PGLib cases up to 6468
buses. Training is supervised on 50,000 solved instances per system
(PowerModels.jl with Mosek or Ipopt). The stated motive is that "the neural
network must be embedded in a larger optimization", and ICNNs avoid the discrete
variables a general DNN's ReLU encoding would need. The paper also contributes
generalization bounds for ICNNs.

*Relation.* **This is the closest published thing to our critic, and it is the
one to cite when we claim the critic is a decomposed version of a known idea.**
Same tool (ICNN), same object (a dispatch value function), same reason (embed in
a larger optimization). Different in exactly the way that matters: theirs is
**one global value function of the whole demand vector**, ours would be **one
shared per-node critic of that node's scalar residual, summed over nodes**, which
is what makes the flow gradient a difference of two node prices. Theirs is
supervised on full solves; ours would be trained on tiny per-node LPs. And they
motivate but do not demonstrate use inside a decomposition.

### Deterministic policy gradient and the critic's action gradient

David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin
Riedmiller, *Deterministic Policy Gradient Algorithms*, Proceedings of the 31st
ICML, PMLR **32**(1):387–395, 2014.

Theorem 1 is exactly the form the summary invokes:
`∇_θ J(µ_θ) = ∫_S ρ^µ(s) ∇_θ µ_θ(s) ∇_a Q^µ(s,a)|_{a=µ_θ(s)} ds`.
Assumptions A.1 explicitly require `∇_a Q^µ(s,a)` to exist. So the actor is
trained on the **critic's action-gradient**, chained with the actor's Jacobian.

Timothy P. Lillicrap\*, Jonathan J. Hunt\*, Alexander Pritzel, Nicolas Heess, Tom
Erez, Yuval Tassa, David Silver, Daan Wierstra, *Continuous control with deep
reinforcement learning*, ICLR 2016. arXiv:1509.02971. (The arXiv abstract page
carries no venue field; the PDF's first line reads "Published as a conference
paper at ICLR 2016.") "an actor-critic, model-free algorithm **based on the
deterministic policy gradient** that can operate over continuous action spaces."

*Relation.* This is the name for our proposed training loop: flow network =
actor, node residual = action, node critic = critic, and the flow gradient is the
critic's action-gradient. Stating it this way buys the diagnosis for free — see
next.

### MAGE: value learning does not constrain the action gradient

Pierluca D'Oro, **Wojciech Jaśkowski**, *How to Learn a Useful Critic?
**Model-based Action-Gradient-Estimator** Policy Optimization*, NeurIPS 2020.
arXiv:2004.14309. Code: `github.com/nnaisense/MAGE`.

⚠️ **Our notes had the acronym expanded wrongly.** It is **A**ction-Gradient-
Estimator, not "Actor-Gradient-Estimator".

The abstract states our §5 diagnosis in the general case, verbatim:
"Deterministic-policy actor-critic algorithms … improve the actor by plugging its
actions into the critic and ascending the action-value gradient … **However,
instead of gradients, the critic is, typically, only trained to accurately
predict expected returns, which, on their own, are useless for policy
optimization.**" MAGE backpropagates through learned dynamics to build
TD *gradient* targets.

*Relation.* Direct support for the summary's claim that value training does not
constrain `J = ∂p̃/∂r`. Our setting is strictly easier than MAGE's: the correct
slope is a known multiplier from a tiny LP, so we can supply the derivative
target directly rather than estimating it through learned dynamics.

Wojciech Marian Czarnecki, Simon Osindero, Max Jaderberg, Grzegorz Świrszcz,
Razvan Pascanu, *Sobolev Training for Neural Networks*, NIPS 2017.
arXiv:1706.04859. Train a network to match "not only … the function's outputs but
also the function's derivatives" when target derivatives are available. This is
the name for the summary's fix 1 (the `β(λ̂_n − λ*_n)²` term).

Andrew W. Rosemberg, Joaquim Dias Garcia, Russell Bent, Pascal Van Hentenryck,
*Sobolev Training of End-to-End Optimization Proxies*, arXiv:2505.11342 (16 May
2025; preprint, no venue found). Sobolev training applied to optimization proxies
in both supervised and self-supervised settings, injecting **solver
sensitivities** as directional-derivative targets; up to 56 % MSE reduction on
three AC-OPF benchmarks with optimality gap below 0.22 %, and halved optimality
gap in the medium-risk region of a self-supervised portfolio task.

*Relation.* This is Sobolev training already applied inside our exact literature,
and it is the natural baseline for the critic's Sobolev term. Their sensitivities
come from differentiating a solver; ours would come free from the per-node LP's
multiplier.

### Did anyone learn a per-block value function and sum it to train a coupling network?

**No — searched hard, found nothing that does all three of (a) per-node/per-block
value function of that block's local residual, (b) summed, (c) differentiated
through to train the producer of the coupling variables.** Search angles included
learned value function + Benders, neural value function + decomposition, ICNN +
value function + decomposition, amortized + Lagrangian decomposition, tie-line /
interface flows + regional cost functions, and an arXiv author sweep of the
Neur2\* corpus. The nearest misses, and precisely how each differs:

- **Minas Chatzos, Terrence W. K. Mak, Pascal Van Hentenryck, *Spatial Network
  Decomposition for Fast and Scalable AC-OPF Learning*, IEEE Transactions on
  Power Systems **37**(4):2601–2612, July 2022 (DOI 10.1109/TPWRS.2021.3124726;
  preprint arXiv:2101.06768).** Structurally the closest architecture found: "The
  first stage learns to predict the flows and voltages on the buses and lines
  **coupling the regions**, and the second stage trains, in parallel, the
  machine-learning models for each region." *Differs:* both stages are
  **supervised on solved AC-OPF solutions**; the regional models predict
  *solutions*, not value functions; nothing is summed; no regional cost gradient
  flows back into the coupling predictor; no bound or certificate. It is a fast
  approximator, not an optimizer over the coupling variables. **This is the
  citation that most nearly pre-empts "predict the coupling variables" as a
  slogan, so name it and draw the line explicitly.**
- **Rosemberg et al. (above)** — right tool, right object, but global rather than
  per-node and not summed.
- **Yu Liu, Fabricio Oliveira, Jan Kronqvist,
  *ICNN-enhanced 2SP: Leveraging input convex neural networks for solving
  two-stage stochastic programming*, arXiv:2505.05261 (preprint, no venue
  found).** ICNN as recourse surrogate so the embedding is an LP rather than a
  MIP; up to 100× speedups. Abstract only was read. Still one aggregate surrogate
  over the first-stage decision, supervised, solved by a solver.
- **Hanjun Dai, Yuan Xue, Zia Syed, Dale Schuurmans, Bo Dai, *Neural Stochastic
  Dual Dynamic Programming*, ICLR 2022 (venue via OpenReview id `aisKPsMM3fg`;
  the arXiv abs page states only "24 pages"). arXiv:2112.00874.** "a trainable
  neural model that learns to map problem instances to a piece-wise linear value
  function … architected specifically to interact with a base SDDP solver."
  *Differs:* amortizes the *stage* value function across instances to warm-start
  SDDP; per-stage, not per-node; no summation; no coupling-variable predictor.
- **Maxime Bouton, Kyle Julian, Alireza Nakhaei, Kikuo Fujimura, Mykel J.
  Kochenderfer, *Decomposition Methods with Deep Corrections for Reinforcement
  Learning*, arXiv:1802.01772; JAAMAS 2019.** "utility decomposition … separate
  the global objective into local tasks considering each individual entity
  independently. An arbitrator is then responsible for combining the individual
  utilities … this paper proposes … a correction term represented by a neural
  network." The closest "sum of per-entity values plus a learned correction"
  found — but RL over entities, the arbitrator picks actions not coupling
  variables, and there is no optimization decomposition or certificate.
- **Peter Sunehag et al., *Value-Decomposition Networks For Cooperative
  Multi-Agent Learning*, arXiv:1706.05296.** "learns to decompose the team value
  function into agent-wise value functions." Note the direction: it **decomposes
  a learned joint value**; we would **compose known local values** to optimize a
  coupling vector. (Only the abstract was read; the additive form is in the body.)
- **Kengy Barty, Pierre Carpentier, Pierre Girardeau, *Decomposition of
  large-scale stochastic optimal control problems*, arXiv:0903.1148 (2009).** The
  classical, non-learned ancestor: subsystems linked by an almost-sure coupling
  constraint at each time step, "production/portfolio management where subsystems
  are, for instance, power units", coordinated by an Uzawa-type dual-price scheme.
  No learning; coordination by iterative prices rather than a one-shot predictor.

---

## 2. Completion-based optimization proxies

The question this section answers: **does a closed-form merit-order completion
appear anywhere?** Answer: **no.** Every completion in this literature restores
*feasibility*; none of them solves the local problem to *optimality*. That is the
distinction `flow_first_summary.md` §4 rule 2 makes ("never complete for
feasibility what you could complete for optimality"), and it survives the search.

### DC3

Priya L. Donti, David Rolnick, J. Zico Kolter, *DC3: A learning method for
optimization with hard constraints*, ICLR 2021. arXiv:2104.12225.
"enforces feasibility via a differentiable procedure, which implicitly completes
partial solutions to satisfy equality constraints and unrolls gradient-based
corrections to satisfy inequality constraints."

Read from §3 of the paper. **Equality completion** is variable elimination: the
network outputs `z ∈ R^m`, and the remaining `n − m` entries are `φ_x(z)`, found
either "explicitly (e.g. in a linear system)" or by "Newton's Method", with
gradients via the implicit function theorem —
`∂φ_x(z)/∂z = −(J^h_{:,m:n})^{-1} J^h_{:,0:m}`. **Inequality correction** takes
`t` gradient steps in `z` on the inequality-violation norm, "along the manifold of
points satisfying the equalities", `t_train` small enough to backpropagate
through. Training uses the **soft loss**: objective plus penalties.

*Relation.* DC3's completion is the same *idea* as ours — predict a subset,
determine the rest — but it is chosen so the equalities **hold**, with no
reference to the objective. Ours picks, among all completions of a node's
residual, the **cheapest** one; that is what makes the returned cost exact, makes
the loss the true objective with no penalty term, and makes the coupling gradient
the multiplier rather than a penalty coefficient. DC3 also needs a linear solve
or Newton iterations per instance and 200 unrolled correction steps at inference
(measured as the slow baseline in E2ELR); our fill is a cumulative sum and a clip.
IDEAS I6 ("penalty box instead of sigmoid") is explicitly the DC3/E2ELR way of
handling inequalities and should cite DC3 here.

### DeepOPF

Xiang Pan, Tianyu Zhao, Minghua Chen, Shengyu Zhang, *DeepOPF: A Deep Neural
Network Approach for Security-Constrained DC Optimal Power Flow*, IEEE
Transactions on Power Systems **36**(3):1725–1735, May 2021
(DOI 10.1109/TPWRS.2020.3026379). Preprint arXiv:1910.14448. Conference version:
Pan, Zhao, Chen, *DeepOPF: Deep Neural Network for DC Optimal Power Flow*, IEEE
SmartGridComm 2019, pp. 1–6, DOI 10.1109/SmartGridComm.2019.8909795 (three
authors there).

"We first train a DNN to learn the mapping and predict the generations from the
load inputs. We then **directly reconstruct the phase angles from the generations
and loads by using the power flow equations**. Such a predict-and-reconstruct
approach reduces the dimension of the mapping to learn." A post-processing step
based on ℓ₁-projection restores feasibility. Reported: under 0.2 % optimality
loss, up to two orders of magnitude speedup.

*Relation.* **This is the origin of the dimensionality argument we make** — predict
a subset, derive the rest exactly — and it predates DC3. But the derivation is of
*dependent physical quantities* from an equality system, cost-blind; the objective
plays no part in choosing what is reconstructed, and there is no dual. Our
inversion of it is the point: we predict the *zero-cost coupling* variables
(flows) and compute the *priced* ones (generation, unmet demand) by optimizing,
which is the choice that routes the price gradient into the prediction
(FINDINGS F3, F17).

### E2ELR — the nearest neighbour on the primal side

Wenbo Chen, Mathieu Tanneau, Pascal Van Hentenryck, *End-to-End Feasible
Optimization Proxies for Large-Scale Economic Dispatch*, IEEE Transactions on
Power Systems **39**(2):4723–4734, March 2024 (DOI 10.1109/TPWRS.2023.3317352;
volume/issue/pages confirmed via Crossref). Preprint arXiv:2304.11726.

Read from the PDF, since the details matter. The problem (their Eq. 1) is an ED
with reserves whose balance constraint is **system-wide**, `eᵀp = eᵀd`, with
thermal limits `f − ξ_th ≤ Φ(p − d) ≤ f̄ + ξ_th` handled as **soft** constraints
penalized by `M_th`. The network emits `z ∈ [0,1]^n` through a sigmoid and sets
`p̃ = z · p̄`, so generation bounds hold by construction. Then the **power balance
repair layer** `P` is a *proportional response*:

```
P(p) = (1 − η↑) p + η↑ p̄   if eᵀp < D ,      η↑ = (eᵀd − eᵀp) / (eᵀp̄ − eᵀp)
P(p) = (1 − η↓) p + η↓ · 0  if eᵀp ≥ D ,      η↓ = (eᵀp − eᵀd) / (eᵀp − 0)
```

"if the initial dispatch `p` has an energy shortage … the output of each
generator is increased by **a fraction η↑ of its upwards headroom**." A second
**reserve repair layer** (their Algorithm 1) splits generators into two groups and
moves dispatch proportionally to headroom in each. Both are closed-form and
differentiable. Training is self-supervised: `L_SSL(p̂) = c(p̂) + M_th ξ_th(p̂)`,
"the objective value of the predicted solution", and "the constraint penalty term
is zero when training end-to-end feasible models". No dual variables anywhere.

*Relation.* This is the closest published method to ours, and the comparison is
sharp on one axis. **Same:** self-supervised on the objective, closed-form
differentiable completion, generation bounds by sigmoid, economic dispatch.
**Different:** their repair is **proportional to headroom** — cost-blind — so the
completed point is feasible but not the cheapest completion, and the loss is the
cost of a repaired approximation rather than the exact optimum given the
prediction. Ours is a merit-order fill, which is provably the optimal completion
of a one-row box LP, so the returned cost *is* `U_t(f)` and the gradient is the
envelope gradient. That is precisely the difference FINDINGS F17 measures against
the thesis baseline's production head, and F32 quantifies again (the fill replaces
the head and one polish sweep recovers most of the gap for *any* network).
**Second difference:** their model has a single system-wide balance and *soft*
thermal limits; ours has a **per-node** balance and hard line limits, which is
where flows become the coupling variables at all. **Third:** no duals, so no
bound, no certificate, no Benders cut.

⚠️ An automated read of the E2ELR PDF produced a fabricated quotation claiming the
repair layer "uses a merit-order dispatch strategy … cost-ordering generators".
**It does not.** The equations above are from the paper's own text. Do not repeat
that claim.

### Gauge mapping / LOOP-LC

Meiyi Li, Javad Mohammadi, *Toward Rapid, Optimal, and Feasible Power Dispatch
through Generalized Neural Mapping*, arXiv:2311.04838 (8 Nov 2023; preprint, no
venue found).

LOOP-LC 2.0 uses a "generalized gauge map method, capable of **mapping any
infeasible solution to a feasible point** within the linearly-constrained domain",
avoiding post-processing and iterations. Benchmarked on IEEE-200. E2ELR's own
comparison (their "LOOP" baseline) notes the gauge map is non-convex.

*Relation.* Another *feasibility* mapping, this time onto a general polytope. It
makes the same trade as E2ELR: any point in the feasible set will do, and the
objective decides nothing. It is the strongest form of the thing our
"optimality completion" is contrasted against.

### Predicting only the coupling variables, elsewhere

Searched for any proxy that predicts line flows as the *sole* output with
generation computed downstream. **Nothing found in the completion literature**;
see §5 for the GNN side of the same search. The closest architectural relative is
Chatzos, Mak, Van Hentenryck (§1), who predict the inter-region coupling flows and
voltages — but supervised, and the regional models are themselves learned
approximators rather than exact completions.

**Verdict on the section question: a closed-form merit-order (cost-ordered
greedy) completion used as a differentiable layer does not appear in this
literature.** Repairs are proportional (E2ELR), projective (DC3's ℓ₂ steps,
DeepOPF's ℓ₁ projection, Euclidean projection), or gauge maps. None is an
optimizing completion.

---

## 3. Dual proxies and closed-form dual completion for valid bounds

Question: **does anyone derive the dual from the primal's active set?** Answer:
**no.** Every dual proxy found *predicts* the dual with a second network and then
completes it. Two papers derive duals from the **gradient of a learned convex
value function**, which is a third route and the closest relative to ours.

### Dual Lagrangian Learning

Mathieu Tanneau, Pascal Van Hentenryck, *Dual Lagrangian Learning for Conic
Optimization*, Advances in Neural Information Processing Systems 37 (NeurIPS
2024). arXiv:2402.03086.

⚠️ Note the full title: it is a **conic optimization** paper, not a power-systems
one. "DLL leverages conic duality and the representation power of ML models to
provide high-duality, dual-feasible solutions, and therefore **valid Lagrangian
dual bounds** … introduces a systematic dual completion procedure, differentiable
conic projection layers, and a self-supervised learning framework based on
Lagrangian duality … closed-form dual completion formulae for broad classes of
conic problems, which eliminate the need for costly implicit layers."

**Their Example 1 (bounded variables) is our box completion.** For
`min_x {cᵀx : Ax ⪰_K b, l ≤ x ≤ u}`, given any `ŷ ∈ K*` the optimal completion is
`ẑ^l = |c − Aᵀŷ|⁺, ẑ^u = |c − Aᵀŷ|⁻`, proved by eliminating `z^l` and observing
the objective coefficient of `z^u` is negative. They add: "The resulting
completion procedure is a generalization of that used in [Klamkin, Tanneau, Van
Hentenryck, *Dual Interior Point Optimization Learning*] for linear programming."

*Relation.* **Our closed-form dual completion is this formula.** `μ^u_g = (λ_n −
C_g)⁺`, `μ^l_g = (C_g − λ_n)⁺`, `μ^u_l = (λ_to − λ_from)⁺` and so on are exactly
`ẑ^l = |c − Aᵀŷ|⁺, ẑ^u = |c − Aᵀŷ|⁻` written out for our constraint matrix. We
should cite DLL for it and **not** claim it as new. What is not in DLL is where
`ŷ` (our `λ`) comes from: DLL predicts it with a network trained on the Lagrangian
dual; we read it off the primal's congestion pattern.

Michael Klamkin, Mathieu Tanneau, Pascal Van Hentenryck, *Dual Interior Point
Optimization Learning*, arXiv:2402.02596 (v1 4 Feb 2024, v3 12 Feb 2025;
preprint). "a smoothed self-supervised loss function that augments the objective
function with a dual penalty term" plus "a novel dual completion strategy that
guarantees dual feasibility by solving a convex optimization problem", with
closed-form solutions for several dual penalties. Two schemes, DIPL (mimicking a
dual interior point algorithm) and DSL (mimicking dual supergradient ascent).
This is DLL's LP-case predecessor.

### Dual Conic Proxies for AC-OPF

Guancheng Qiu, Mathieu Tanneau, Pascal Van Hentenryck, *Dual conic proxies for AC
optimal power flow*, Electric Power Systems Research **236**:110661, November 2024
(DOI 10.1016/j.epsr.2024.110661); presented at PSCC 2024. Preprint
arXiv:2310.02969.

"no existing learning-based approach can provide valid dual bounds for AC-OPF.
This paper addresses this gap by training optimization proxies for a convex
relaxation of AC-OPF … a second-order cone (SOC) relaxation … and proposes a novel
architecture that **embeds a fast, differentiable (dual) feasibility recovery**,
thus providing valid dual bounds", trained self-supervised.

*Relation.* Same shape as ours — predict something, complete it to dual
feasibility, get a valid bound — with the same difference: the prediction is the
dual itself. Also relevant as the paper that established "valid dual bounds from
a proxy" as a goal in power systems.

### PDL

Seonho Park, Pascal Van Hentenryck, *Self-Supervised Primal-Dual Learning for
Constrained Optimization*, AAAI 2023, Vol. 37, pp. 4052–4060. arXiv:2208.09046.
"PDL **mimics the trajectory of an Augmented Lagrangian Method (ALM)** and jointly
trains primal and dual neural networks. Being a primal-dual method, PDL uses
instance-specific penalties of the constraint terms in the loss function used to
train the primal network."

*Relation.* This is the architecture the thesis's dual network came from, and the
one `flow_first_summary.md` §1b diagnoses: two coupled networks solving a fixed
point, on an LP whose primal and dual maps are piecewise constant. Our
construction removes both networks on the dual side. Note also that PDL's dual is
a *penalty multiplier*, not a certified bound — its duals need not be dual
feasible, so they cannot certify.

### Self-certifying primal-dual proxies

Michael Klamkin, Mathieu Tanneau, Pascal Van Hentenryck, *Self-Certifying
Primal-Dual Optimization Proxies for Large-Scale Batch Economic Dispatch*,
arXiv:2510.15850 (v1 17 Oct 2025, v2 10 Apr 2026; preprint, no venue found).
IDEAS.md lists this correctly.

**This is the direct competitor for the certificate-and-fall-back claim.** They
train primal proxy `p_α` and dual proxy `d_β` **jointly** on the duality gap
`Γ_θ(p_α(θ), d_β(θ))`, and Algorithm 1 is a hybrid solver: if `ĝ ≤ ε` return the
prediction, else call a classical solver. A dedicated loss `Γ^{1%}` targets the
tolerance directly by penalizing only `max(Γ − ε, 0)`. Primal feasibility uses
**E2ELR's proportional response layer** verbatim ("[10, Eq. 4]"); the dual is
completed by DLL-style element-wise max operations. On 9241\_pegase: "up to 925×
speedup … while guaranteeing worst-case optimality below 1 %, and over 1000× at 2 %." Their motivation is worth quoting: "worst-case analyses show that there
exist in-distribution queries that result in orders of magnitude higher optimality
gap, making it difficult to trust the predictions in practice."

*Relation, stated honestly.* The **idea** of per-instance certification with
solver fallback is theirs and predates our measurements (their v1 is October 2025;
F26–F31 are September 2026). What differs is the mechanism: (i) their dual comes
from a **second trained network**, ours from the **primal's congestion pattern in
closed form** — no dual network, no dual training, and the dual is *exact* on
91 % of instances (F26) rather than learned; (ii) their primal repair is
proportional, ours is an optimizing fill; (iii) they apply **no polish**, where our
one-sweep line search cuts the gap five-fold (F27) and the exact path step removes
the fallback entirely at 20 nodes (F29); (iv) their ED has a system-wide balance
and PTDF thermal limits, ours a per-node balance with flows as decision variables.
Our re-solve rate (3.5–6.7 %) is directly comparable to their `ε`-fallback rate,
which they do **not** report as a fraction.

### Duals from the gradient of a learned convex value function

Ling Zhang, Yize Chen, Baosen Zhang, *A Convex Neural Network Solver for DCOPF
With Generalization Guarantees*, IEEE Transactions on Control of Network Systems
**9**(2):719–730, June 2022 (DOI 10.1109/TCNS.2021.3124283). Preprint
arXiv:2009.09109. Companion: Yize Chen, Ling Zhang, Baosen Zhang, *Learning to
solve DCOPF: A duality approach*, Electric Power Systems Research
**213**:108595, December 2022 (DOI 10.1016/j.epsr.2022.108595).

Read from the preprint. They train an **ICNN `g_θ(ℓ)` to predict the DCOPF optimal
cost `J*(ℓ)`** as a function of the load, and use Theorem 3.1: "A vector `µ*` is an
optimal solution to the dual problem if and only if it is a (sub)gradient of the
optimal cost `J*(ℓ)` at the point `ℓ`, that is, `∇_ℓ J* = µ*`." So the LMPs are the
network's input gradient. Every other dual then follows **in closed form from the
prices**: `τ(µ) = [µ − c]⁺ − (µ − c)`, `τ̄(µ) = [µ − c]⁺`, with `λ, λ̄` from an
ℓ₁ problem; the training loss adds KKT-violation terms built from these
expressions, which is what buys the generalization guarantee. With the duals in
hand they identify the active set and recover `x*` by solving a linear system.

*Relation.* **This is the closest published construction to our dual recovery, and
it is closer than the dual-proxy papers.** The bound-multiplier completion
`τ̄(µ) = [µ − c]⁺` is literally our `μ^u_g = (λ_n − C^var_g)⁺`. The difference is
where the price comes from: theirs is `∇_ℓ` of a learned convex cost, ours is the
**primal's congestion pattern** — contract uncongested lines, merit-order fill per
region, node-wise ascent — which needs no second model and is exact when the
regime is right. Also, their route ends in an **active-set identification plus
linear solve**, which FINDINGS F25 rules out at our scale (95,446 distinct regimes
over 262,144 instances, 28 % of validation instances with an unseen regime).

### Active-set learning, for completeness

Sidhant Misra, Line Roald, Yeesian Ng, *Learning for Constrained Optimization:
Identifying Optimal Active Constraint Sets*, INFORMS Journal on Computing
**34**(1):463–480, 2022 (DOI 10.1287/ijoc.2020.1037). Preprint arXiv:1802.09639.
"the proposed method [is] based on learning relevant sets of active constraints,
from which the optimal solution can be obtained efficiently."

*Relation.* This is the route F25 measures and rejects for our sampler: the method
needs a small number of active sets to cover the distribution. Cite F25 against
it, and note that our per-node breakpoint structure is the surviving form of the
same idea (any discrete choice must be per node, where regimes compose).

**Verdict on the section question: no dual proxy in this literature derives the
dual from the primal prediction's active set.** They predict the dual (DLL, DIPL,
Dual Conic Proxies, PDL, Klamkin et al.) or differentiate a learned convex value
function (Zhang et al., Chen et al.). Our route — the primal's own congestion
pattern, contracted into price regions — was not found.

---

## 4. Proxy plus exact local search or warm start

Question: **does an exact min-cost-flow polish of a proxy appear anywhere?**
Answer: **not in power systems, and not for min-cost flow.** The pattern
"learned prediction, exact algorithm finishes, optimality preserved" is
well established under the name **algorithms with predictions**, and there it has
been done for max-flow, bipartite matching and the assignment problem — but not
for capacitated min-cost flow, and never with a self-supervised cost loss or a
dual certificate attached.

### Learned warm starts for OPF

Kyri Baker (sole author), *Learning Warm-Start Points for AC Optimal Power Flow*,
2019 IEEE 29th International Workshop on Machine Learning for Signal Processing
(MLSP), pp. 1–6 (DOI 10.1109/MLSP.2019.8918690). Preprint arXiv:1905.08860.
⚠️ The predictor is a **multi-target random forest**, not a neural network. Inputs
are bus loads only; outputs are approximate voltages and generation. What is
warm-started is an **interior-point solver** — MATPOWER's MIPS and MATLAB
`fmincon`'s interior-point — measured in iterations to convergence against a
DC-OPF warm start and a flat start. ⚠️ **There is no headline speedup number**;
the abstract says only "the benefit … is shown to be solver and network dependent,
but shows promise." Do not attribute a factor to this paper.

Kyri Baker (sole author), *A Learning-boosted Quasi-Newton Method for AC Optimal
Power Flow*, arXiv:2007.06074 (2020; **venue unverified, cite as preprint**). This
one does not warm-start anything — a DNN with feedback replaces the Newton step
"without having to calculate a Jacobian or approximate Jacobian matrix", and with
suitable weights the map is a contraction, so convergence can be guaranteed.

Frederik Diehl, *Warm-Starting AC Optimal Power Flow with Graph Neural Networks*,
NeurIPS 2019 Workshop on Tackling Climate Change with Machine Learning
(`climatechange.ai/papers/neurips2019/1`; no DOI). A GNN, supervised on solved
ACOPF results via PowerModels.jl, emits generator, bus **and branch** predictions
through separate heads; these initialize **IPOPT**. On the 2000-bus Texas case the
model alone is "faster than ACOPF by four orders of magnitude" but is a black box,
while as a warm start it "improves its convergence speed by **3.8x** and
guarantees a solution" — 243.9 s versus 862.6 s cold and 849.7 s from a DC→AC warm
start. The abstract and conclusion say **3.75x**. ⚠️ A "2.8" figure that appears on
the landing page occurs nowhere in the PDF.

*Relation.* Diehl is the cleanest precedent for our F29 framing and makes our
argument for us: the black-box speed number is not the deployable result, the
reduction in solver work is. His DC→AC warm-start baseline is structurally our
"46.5 augmentations from zero flows versus 4.8 from the network". The difference:
his warm start is supervised on solutions and feeds a general NLP solver, ours is
self-supervised on cost and feeds a **combinatorial** exact step whose work we can
count exactly (augmentations, not wall-clock heuristics), and we also warm-start
the *dual*.

Dhruv Suri, Helgi Hilmarsson, Shourya Bose, *WARP: A Benchmark for Primal-Dual
Warm-Starting of Interior-Point Solvers*, arXiv:2605.05728 (7 May 2026; preprint,
no venue). **A fairness audit of exactly this literature, and it should be cited
in our own methods section.** "A growing body of work uses machine learning to
predict primal warm-start iterates, reporting iteration reductions of 30-46 %. We
show that these reported gains rest on an inappropriate evaluation baseline: prior
methods benchmark against the flat start `V_m = 1, V_a = 0`, whereas the solver's
actual default — the variable-bound midpoint `(l+u)/2` — is near-optimal for
log-barrier centrality. Against this corrected baseline, **no primal-only
warm-start method reduces solver iterations**." They further show that supplying
`x*` without duals makes IPOPT diverge, and that the full primal-dual-barrier
state cuts IPOPT from 23 to 3 iterations. Their own model reaches 76 %.

*Relation.* Two lessons for us. (i) The baseline matters more than the method:
their corrected baseline is the analogue of our persistent-Gurobi and
optimal-basis-preset marks (F29). (ii) A primal-only warm start is the wrong
object; the useful one is primal **and** dual — which is what `PrimalDual` returns.
Our F29 measurement that passing an optimal *basis* takes Gurobi to 0 iterations
and 0.05 ms, where primal-dual start vectors gained no wall-clock, is the same
finding in simplex clothing and should be reported next to WARP.

### Neural Diving / Neural Branching

Vinod Nair, Sergey Bartunov, Felix Gimeno, Ingrid von Glehn, Pawel Lichocki, Ivan
Lobov, Brendan O'Donoghue, Nicolas Sonnerat, Christian Tjandraatmadja, Pengming
Wang, Ravichandra Addanki, Tharindi Hapuarachchi, Thomas Keck, James Keeling,
Pushmeet Kohli, Ira Ktena, Yujia Li, Oriol Vinyals, Yori Zwols, *Solving Mixed
Integer Programs Using Neural Networks*, arXiv:2012.13349 (v1 23 Dec 2020, v3 29
Jul 2021). ⚠️ **Never formally published** — cite as a preprint.

"**Neural Diving** learns a deep neural network to generate multiple partial
assignments for its integer variables, and the resulting smaller MIPs for
un-assigned variables are solved with SCIP to construct high quality joint
assignments." "**Neural Branching** learns a deep neural network to make variable
selection decisions in branch-and-bound to bound the objective value gap with a
small tree", by imitating a GPU-scalable variant of Full Strong Branching. Their
framing of the split — learning applied to "the two key sub-tasks of a MIP solver,
generating a high-quality joint variable assignment, and bounding the gap in
objective value between that assignment and an optimal one" — maps onto our primal
proxy plus dual certificate.

*Relation.* Same division of labour: the network fixes the part that is hard and
global, an exact procedure finishes the rest. The difference is what "finishes"
means. Neural Diving hands the residual to **another solver call**; our completion
is closed-form and provably optimal, so no solver appears in the forward pass at
all, and the exact step (cycle cancelling) is a fixed combinatorial routine whose
work we can count.

### The exact combinatorial repair: min-cost flow

Morton Klein, *A Primal Method for Minimal Cost Flows with Applications to the
Assignment and Transportation Problems*, Management Science **14**(3), Theory
Series, November 1967, pp. 205–220 (DOI 10.1287/mnsc.14.3.205; verified via
Crossref and the JSTOR scan). Abstract: "A simple procedure is given for solving
minimal cost flow problems in which **feasible flows are maintained throughout**.
It specializes to give primal algorithms for the assignment and transportation
problems. **Convex cost problems can also be handled**." Negative-cycle cancelling.

Andrew V. Goldberg, Robert E. Tarjan, *Finding minimum-cost circulations by
canceling negative cycles*, Journal of the ACM **36**(4), October 1989,
pp. 873–886 (DOI 10.1145/76359.76368; verified via Crossref). A judicious choice
of cycles to cancel (minimum-mean cycle) gives a polynomial iteration bound and
"a very simple strongly polynomial algorithm that uses no scaling".
⚠️ Whether the "cancel and tighten" variant is introduced under that name **in this
paper** could not be confirmed; cite it for minimum-mean cycle cancelling.

*Relation.* Klein's two clauses are our polish exactly: feasible flows maintained
throughout (every polished point stays a valid upper bound, F27: min gap −4e−15)
and convex costs handled (the fill cost `φ_n` is convex piecewise-linear). F29's
`PathPolish` is max-gain cycle cancelling with the source-node structure that
follows from lines being zero-cost. F28 is the empirical statement of why the
*network* is needed at all: cycle cancelling and coordinate descent are both local,
and from zero flows the line search stalls at 72 % mean gap after 5000 sweeps.

### Learned predictions finished by an exact flow algorithm

**No work applies min-cost flow, cycle cancelling, or network simplex as an exact
repair of a machine-learned solution.** Searched: learning + min-cost flow warm
start; neural network + network simplex; learned + cycle canceling; predict-then-
repair flow. The five nearest, in the *algorithms-with-predictions* literature
(surveyed by Michael Mitzenmacher, Sergei Vassilvitskii, *Algorithms with
Predictions*, arXiv:2006.09123, a chapter in *Beyond the Worst-Case Analysis of
Algorithms*, ed. Tim Roughgarden, CUP 2021):

- **Sami Davies, Benjamin Moseley, Sergei Vassilvitskii, Yuyan Wang, *Predictive
  Flows for Faster Ford-Fulkerson*, ICML 2023, PMLR **202**:7231–7248
  (arXiv:2303.00837).** "improve the performance of the widely used Ford-Fulkerson
  algorithm for computing maximum flows by **seeding Ford-Fulkerson with predicted
  flows**", with theory in terms of prediction quality, on image segmentation.
  *Differs:* **max**-flow, so no costs, no dispatch objective, no cycle cancelling;
  the prediction is not trained on a cost objective; no dual certificate.
  **This is the closest structural relative of F29 in the whole literature.**
- **Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, Sergei
  Vassilvitskii, *Faster Matchings via Learned Duals*, NeurIPS 2021
  (arXiv:2107.09770).** Learned **duals** warm-start a primal-dual algorithm for
  weighted bipartite matching. They face our two problems on the dual side:
  "predicted duals may be infeasible, so we give an algorithm that efficiently maps
  predicted infeasible duals to nearby feasible solutions", and once feasible "they
  may not be optimal, so we show that they can be used to quickly find an optimal
  solution." *Differs:* matching, not capacitated min-cost flow; the prediction is
  a dual, not a primal flow; the dual repair is an algorithm, not a closed form
  read off a congestion pattern.
- **Justin Y. Chen, Sandeep Silwal, Ali Vakilian, Fred Zhang, *Faster Fundamental
  Graph Algorithms via Learned Predictions*, ICML 2022, PMLR **162**:3583–3602
  (arXiv:2204.12055).** The only work found that names min-cost flow: a reduction
  framework giving "new algorithms for degree-constrained subgraph and **minimum-
  cost 0-1 flow**", plus PAC-learnability results. *Differs:* unit capacities, via
  reduction; predictions are duals; theory-first, no domain application.
- **Ilay Yavlovich, Jad Agbaria, Muhamed Mhamed, Nir Weinberger, Jose Yallouz,
  *Learning-Augmented Scalable Linear Assignment Problem Optimization via Neural
  Dual Warm-Starts*, ICML 2026 (per the arXiv comments field), arXiv:2605.09382.**
  Predicts dual potentials to warm-start the exact Jonker–Volgenant solver, with
  "feasibility … guaranteed by construction via the Min-Trick mechanism, completely
  eliminating the need for costly iterative projections" and strict optimality;
  2× / 1.25× / 1.5× end-to-end. *Differs:* duals, LAP (a min-cost-flow special
  case), no cost-based self-supervision, and deliberately not a GNN.
- **Eleanor Wiesler, Trace Baxley, *Graph Neural Network-Informed Predictive Flows
  for Faster Ford-Fulkerson and PAC-Learnability*, arXiv:2604.21175 (April 2026).**
  Superficially the closest of all — a message-passing GNN over a flow network,
  warm-starting from predicted flows, optimality preserved, and **the headline
  metric is the reduction in the number of augmentations**, the same metric as F29.
  ⚠️ **But it is a preprint with no experimental results**: the paper says the
  runtime proof "remains future research" and "We encourage experimental analysis
  of these results as a next step." Cite it only to note that the framing has been
  proposed, never as prior empirical work.

*Relation.* This body of work is the right home for F28/F29 and we should cite it
rather than presenting "learned warm start for an exact combinatorial algorithm"
as new. What is new relative to it: the exact algorithm is **capacitated min-cost
flow with convex piecewise-linear node costs**, the predictor is trained
**self-supervised on the true objective** rather than fit to solutions, and the
same forward pass also yields a **dual certificate**, so the exact step can be
skipped whenever the certificate is tight.

---

## 5. Graph neural networks for dispatch and power flow

### Donon et al., the graph neural solver

Balthazar Donon, Rémy Clément, Benjamin Donnot, Antoine Marot, Isabelle Guyon,
Marc Schoenauer, *Neural networks for power flow: Graph neural solver*, Electric
Power Systems Research **189**:106547, December 2020
(DOI 10.1016/j.epsr.2020.106547); this volume is the PSCC 2020 proceedings issue,
so "PSCC 2020" and "EPSR 189" name one paper.

**Self-supervised on a physics residual, and it says so plainly:** "It learns to
perform a power flow computation by **directly minimizing the violation of
Kirchhoff's law at each bus** during training. Unlike previous approaches, our
graph neural solver learns by itself and does not try to imitate the output of a
Newton-Raphson solver." And: "During training, our model is never told what the
actual solution is." The architecture performs `K` correction rounds — "These
'correction' updates are **analogous to the iterations performed by a
Newton-Raphson solver**" — with a discounted loss summed over rounds. **Readout is
on nodes**: "Our goal is to predict `v` and `θ` at each bus."

⚠️ Balthazar Donon, Benjamin Donnot, Isabelle Guyon, Antoine Marot, *Graph Neural
Solver for Power Systems*, IJCNN 2019, pp. 1–8 (DOI 10.1109/IJCNN.2019.8851855) is
a **different, four-author paper** — Clément and Schoenauer are on the 2020 one
only. Per the 2020 paper's description of its own predecessor, the 2019
architecture decoded messages "into **flows through lines**" — an edge readout —
but was **supervised** ("minimizing the distance between our neural network's
predictions and the results of a classical Newton-Raphson method") and assumed all
lines share identical physical characteristics.

Worth citing alongside: Balthazar Donon, Zhengying Liu, Wenzhuo Liu, Isabelle
Guyon, Antoine Marot, Marc Schoenauer, *Deep Statistical Solvers*, NeurIPS 2020,
which "uses the objective function of the problem as the loss function" so that no
solution dataset is needed. That is the general form of "the loss is the dispatch
cost".

*Relation.* Donon's self-supervision is the closest precedent for our loss design,
and the multi-round correction structure is IDEAS I3 ("prices in the loop"). The
differences: their residual is a *physics* violation, not an objective, so there is
no price gradient and no optimality notion; their readout is nodal; and there is no
feasibility guarantee, no dual and no certificate. The 2019 IJCNN paper is the only
verified GNN with an edge-to-flow readout, and it is supervised imitation of a
power-flow solver, not a dispatch.

### PowerFlowNet

Nan Lin, Stavros Orfanoudakis, Nathan Ordonez Cardenas, Juan S. Giraldo,
Pedro P. Vergara, *PowerFlowNet: Power flow approximation using message passing
Graph Neural Networks*, International Journal of Electrical Power & Energy Systems
**160**:110112, 2024 (DOI 10.1016/j.ijepes.2024.110112). Preprint arXiv:2311.03415.

⚠️ **Not Applied Energy** — IJEPES. Node-level output only: it "reconstructs every
node's full feature vector `x̂_i = (V_i^m, θ_i, P_i, Q_i)` given partial
information"; line resistance and reactance are **edge input features** and line
flows are computed afterwards from the predicted bus states. Supervised on ~30,000
Newton–Raphson solutions per case (PandaPower), MSE loss; the paper considers a
self-supervised "unbalance error" but reports that "using only physical model
losses is insufficient".
⚠️ **The speedup differs by version**: the published abstract says "4 times faster
in the IEEE 14-bus system and **48 times** faster in … 6470rte", the arXiv v3
abstract says **145 times**. State which you cite.

### CANOS

Luis Piloto, Sofia Liguori, Sephora Madjiheurem, Miha Zgubic, Sean Lovett, Hamish
Tomlinson, Sophie Elster, Chris Apps, Sims Witherspoon, *CANOS: A Fast and Scalable
Neural AC-OPF Solver Robust To N-1 Perturbations*, arXiv:2403.17660 (26 Mar 2024).
⚠️ **Preprint; no peer-reviewed version found.**

Node outputs `v_a, v_m, p_g, q_g`; branch quantities `(p_f, q_f, p_t, q_t)` are
then "derive[d] … according to the branch flow equations", so branch flows are a
*function* of predicted bus states, the opposite direction from our architecture.
Supervised on ~300k solved AC-OPF instances per grid (PowerModels.jl + Ipopt),
loss = L2 plus constraint-violation penalties. N-1 robustness is evaluated on a
"TopDrop" dataset dropping one generator or branch. Grids to ~10,000 buses,
33–65 ms, within 1 % of true AC-OPF cost.
**No feasibility guarantee** — bounds are enforced by **sigmoid squashing** (the
same trick we use on line limits) and the reference angle is fixed, but power
balance, angle differences and thermal limits are reported as *violation metrics*;
their own limitations section names "the inability to guarantee full
AC-feasibility."

*Relation.* CANOS is the scale-and-topology reference (IDEAS I13 cites its lack of
identity embeddings). The contrast to draw is feasibility: their guarantee is
statistical, ours is structural — the fill balances every node to float64 precision
(F31: 2.9e−11 MW) and the certificate bounds the cost.

### OPF-Learn

Trager Joswig-Jones, Kyri Baker, Ahmed S. Zamzam, *OPF-Learn: An Open-Source
Framework for Creating Representative AC Optimal Power Flow Datasets*, IEEE PES
ISGT 2022, pp. 1–5 (DOI 10.1109/ISGT50606.2022.9817509). Preprint arXiv:2111.01228.
A Julia/Python package for **generating datasets**, not a learning method: loads
are sampled from a convex set containing the AC-OPF feasible set, and infeasibility
certificates from a relaxation shrink that set.

*Relation.* The counterpoint to self-supervision: OPF-Learn exists because
label-supervised proxies need representative inputs *and* expensive solved labels.
Our loss uses no labels, so only the input distribution matters — which is exactly
what IDEAS I9 ("targeted data … instances from Benders investment trajectories")
is about, and OPF-Learn is the right citation for that concern.

### Edge readouts predicting line flows

**Nothing found that predicts line flows as the sole output of a GNN, with the
flows being the decision variables of a dispatch whose cost is the training
loss.** Searched: GNN + line/branch flow prediction, edge-level readout power grid,
line-graph GNNs, transport / net-transfer-capacity GNNs. The four nearest:

1. **Donon et al., IJCNN 2019** (above) — genuine edge-to-flow decoder, but
   supervised imitation of Newton–Raphson, AC power flow simulation not dispatch,
   no cost objective, identical line characteristics.
2. **Dekang Meng, Rabab Haider, Pascal Van Hentenryck, *Flow-Aware GNN for
   Transmission Network Reconfiguration via Substation Breaker Optimization*
   (OptiGridML), arXiv:2508.01951 (3 Aug 2025; preprint).** A **line-graph** neural
   network approximates DC power flows for a given topology — architecturally the
   closest analogue to our edge readout, since a line graph makes lines the nodes.
   *Differs:* the flows are stage 1 of 2, feeding a HeteroGNN that predicts breaker
   states; the flow objective is a Kirchhoff-consistency loss, not dispatch cost;
   no bounds by construction, no guarantee.
3. **CANOS** — branch flows derived from bus predictions, not predicted.
4. **Damian Owerko, Fernando Gama, Alejandro Ribeiro, *Unsupervised Optimal Power
   Flow Using Graph Neural Networks*, arXiv:2210.09277 (2022; the comments field
   says "Submitted to IEEE Transactions on Power Systems" — **verify status before
   citing as published**).** The self-supervision precedent rather than the
   edge-readout one: it learns "in an unsupervised manner, **minimizing the cost
   directly**", but predicts **generator power at nodes** and handles constraints
   with "a novel barrier method that is differentiable and works on initially
   infeasible points", with the honest result that it avoids "constraint violations
   most of the time". That gap — cost-minimizing but sometimes infeasible — is
   exactly what sigmoid-bounded flows plus an exact merit-order completion closes.

---

## 6. How this literature reports speed

The scale-up notes at the end of `FINDINGS.md` say "Measure wall time per hour for
both, on the same machine, before claiming it." This section records what the
comparable papers actually do, so the paper's own protocol can be stated as at
least as strict. **Summary: E2ELR and DeepOPF are honest about single-threading
and say so; DLL is the loosest; Klamkin et al. is the fairest and is the template
to match or beat. No power-systems paper normalizes by cost or energy.**

### E2ELR (Chen, Tanneau, Van Hentenryck 2024), §VII, Tables VI–VII

- Solver: "solved with **Gurobi 9.5 with a single CPU thread** and default
  parameter settings". No warm start, no persistent model.
- Hardware: dual Intel Xeon 6226 @ 2.7 GHz for the solver; **Tesla V100-PCIE** for
  the ML.
- Table footnote: "solution time per instance (single thread). **All ML inference
  times are for a batch of 256 instances.**"
- They state the asymmetry outright: "Recall that the Gurobi's solving times are
  for a single instance solved on a single CPU core, whereas the ML inference times
  are reported for **a batch of 256 instances on a GPU**." And they defend the
  headline number on throughput grounds: "about 25,000 instances per second, on a
  single GPU. Solving the same volume of instances with Gurobi would require more
  than a day on a single CPU. Getting this time down … would require thousands of
  CPUs, which comes at high financial and environmental costs."
- Table VI (ED), Gurobi vs E2ELR per instance: ieee300 12.1 vs 4.5 ms; pegase1k
  51.5 vs 5.3; rte6470 364.4 vs 6.6; pegase9k 913.5 vs 7.3; pegase13k 1481.3 vs
  8.3; goc30k 4566.9 vs 10.0.
- A separate table times the repair layers alone against a Euclidean projection
  solved as a QP: 0.13 µs vs 0.45 ms on ieee300, up to 3439× — "Median computing
  times as measured by BenchmarkTools", Julia, single thread.

### DeepOPF (Pan, Zhao, Chen, Zhang 2021), §VI, Tables II–III

- "we obtain the solution … by **Gurobi (version 8.1.1)**", and explicitly: "The
  Gurobi solver by default uses multi-threading … **For fair comparison, we use the
  single-threading setting** in our simulations."
- Hardware: "quad-core (i7-3770@3.40GHz) CPU workstation and 16GB RAM". **No GPU
  anywhere** — the DNN also runs on that CPU, which makes this the most
  apples-to-apples comparison in the set.
- Table II: Case30 0.72 vs 17 ms (×24); Case57 0.76 vs 102 (×133); Case118 2.48 vs
  698 (×281); Case300 81.4 vs 5766 (×318). Batching is not stated; per-instance is
  inferred, not asserted by the paper.

### DLL (Tanneau, Van Hentenryck, NeurIPS 2024), §5, Tables 3 and 5

- Hardware: Intel Xeon Gold 6226 @ 2.70 GHz, **12 CPU threads**, 64 GB; one V100.
- Baselines: **Gurobi v10** (linear), **Mosek** (nonlinear). ⚠️ **Solver thread
  count and warm start are not specified.**
- Timing is "Time to run inference on **all instances in the test set**, using one
  V100 GPU", test sets of **4,096** instances — whole-batch throughput, not
  latency. Gurobi 2.8–40.0 CPU **seconds** vs DLL 0.3–13.6 GPU **milliseconds**;
  headline "1000x speedups over commercial interior-point solvers".
- **The loosest protocol found**: total batch GPU time against unspecified-thread
  interior-point CPU time.

### Klamkin, Tanneau, Van Hentenryck (2510.15850), §IV-A, §IV-C, Tables I–II

**The fairest protocol found, and the one to match.**

- Solver: **HiGHS 1.11.0** via JuMP 1.28.0 with a lazy-PTDF formulation — and they
  tuned it: lazy thermal constraints are "approximately 15x faster than an
  equivalent sparse phase-angle formulation" on 1354\_pegase, 45× on 2869, more on
  9241. Model-building time is excluded from the solve time.
- **They give the CPU side idealized parallelism rather than a single core**:
  "perfect sample-wise parallelism is emulated with **24 CPUs**, i.e., the solve
  time for a batch of `N` samples each with solve times `t_i` is computed using the
  **ideal makespan bound** `max(1/24 Σ_i t_i, max_i t_i)`."
- GPU: **NVIDIA H200**, PyTorch 2.8.0, `torch.compile` over "the proxy inference,
  repair, and objective value calculation", and — importantly — "time spent on data
  movement (inputs to GPU and outputs to CPU) **is included** in the inference time
  measurements."
- Batch: 240,000 unseen test samples, "chosen to mimic performing a 1-day
  hourly-granularity simulation with 10,000 scenarios". Table II reports, for
  240k samples: 90.8 s (opt) vs 0.03 s (ML) on 1354\_pegase; 199.7 vs 0.09 on 2869;
  711.2 vs 0.64 on 9241. Headline: 925× at a guaranteed <1 % worst case, >1000× at
  2 %.
- They also position warm-starting honestly: "Prior works in this direction report
  speedups on the order of **2-25x**", and their scheme "allows to avoid the solver
  entirely for a majority of queries".

### The Proxy Benders Decomposition (2606.07403), §5.1

- "Optimization uses **Gurobi 12.0.3 in single-threaded mode (Threads = 1)** with
  default presolve, cuts, heuristics, and tolerances and a fixed seed"; exclusive
  Intel Cascade Lake Gold 6226 nodes, 64 GB, MIPGap 1e−4, one-hour wall-clock limit.
- Held-out evaluation compares the exact oracle against "the **proxy-only run**,
  which replaces every subproblem solve with the learned predict–project–complete
  proxy and **never falls back to an exact solve**. The two share the same master
  and Benders scheme and differ only in how each cut is separated." State-generation
  time is excluded (it is not counted in any reported runtime).

### Cost or energy normalization

**No paper in the power-systems optimization-proxy literature examined here
normalizes by hardware cost or energy.** E2ELR, DeepOPF, DLL and Klamkin et al. all
report wall-clock only; three of them compare a datacenter GPU (V100 / H200) against
CPU cores with no accounting for price or power. E2ELR gestures at "high financial
and environmental costs" of thousands of CPUs but does not quantify either side.
This is a clean negative finding and the paper can state it.

The one framework found that does this properly is in neural combinatorial
optimization, not power systems: **Sohaib Afifi, *An Amortized Efficiency Threshold
for Comparing Neural and Heuristic Solvers in Combinatorial Optimization*,
arXiv:2605.14624 (May 2026; preprint).** It defines the **Amortized Efficiency
Threshold** — the deployment volume above which a neural solver breaks even with a
heuristic in total energy or carbon, `AET_E = E_train / max(E_base^inst −
E_NN^inst, ε)`, with the neural per-instance energy defined as throughput-amortized
`E_NN^inst = P_GPU / (3600 · τ_NN(B))`, plus an embodied-carbon term amortized
symmetrically on both sides. Instantiated on CVRP `n = 50`: operational crossover
≈ 4.56e3 deployed instances, per-instance energy ratio 2.29e−3. Its two objections
are ours to pre-empt: "A single-threaded CPU metaheuristic solves instances in
series … The correct quantity to compare is not the wall time of a single instance,
but the per-instance throughput-amortized energy", and "Comparing a 700 W
datacenter GPU … against a single laptop CPU core is ill-posed." It recommends
reporting **both single-thread and multi-thread baselines** — which is what our 1-
core / 8-core / optimal-basis split already does (F30, F31).

### Explicit audits of ML-versus-solver comparisons

- **Luca Accorsi, Andrea Lodi, Daniele Vigo, *Guidelines for the computational
  testing of machine learning approaches to vehicle routing problems*, Operations
  Research Letters **50**(2):229–234, March 2022 (DOI 10.1016/j.orl.2022.01.018).
  Preprint arXiv:2109.13983.** §2.3: "It is thus commonly accepted to consider
  **single-threaded algorithms run on standard CPU architectures**." On GPUs: "one
  should consider **adding a measure of the speedup associated with the model
  inference when run on a GPU rather than on a CPU** together with the total
  algorithmic time." On normalization, they report the VRP community's use of "the
  single-thread rating defined by PassMark", while conceding "the comparison is
  still rough".
- **WARP (arXiv:2605.05728)**, §4 above — the baseline-correction audit for
  warm-start claims.
- ⚠️ **Bengio, Lodi, Prouvost, *Machine Learning for Combinatorial Optimization: a
  Methodological Tour d'Horizon* (arXiv:1811.06128) contains no benchmarking-
  fairness critique.** Do not cite it for that; cite Accorsi/Lodi/Vigo instead.

### What our protocol should say

F29–F31 already do most of what this literature does not: a **persistent** Gurobi
model (0.16 ms) rather than a fresh one (0.32 ms), an **eight-core** mark (0.02 ms),
an **optimal-basis oracle** (0.05 ms) as a floor, a roofline computation for the
network, and a per-instance figure taken from a fixed batch with the batch size
stated. Two gaps remain relative to Klamkin et al. and Afifi: the CPU side is a
single-core figure multiplied out rather than an ideal-makespan bound over `k`
cores measured on one machine, and host-device transfer is inside the CUDA graph's
copy but should be stated explicitly as included. Neither is hard to add.

---

## 7. Other closely related work found while reading

### The Proxy Benders Decomposition — the closest overall relative

Changkun Guan, El Mehdi Er Raqabi, Mathieu Tanneau, Pascal Van Hentenryck,
*The Proxy Benders Decomposition*, arXiv:2606.07403 (5 Jun 2026; preprint, no venue
found). This is the "Proxy-BD" of `flow_first_summary.md` §6, now a full paper, and
it is three months old.

"subproblem optimization is replaced by **certified optimization proxies** rather
than repeated exact solves. The proposed proxy follows a self-supervised
**predict–project–and–complete** mechanism that produces dual-feasible solutions
for generating provably valid Benders cuts. The framework preserves the theoretical
validity of the decomposition **independently of prediction quality**."

Mechanism, read from §2. The network predicts the linking-row duals `λ̃`, projects
`λ̂ = (λ̃)⁺`, and completes the rest by
`µ̂(λ̂) ∈ argmax_{µ≥0} {gᵀµ : Gᵀµ ≤ q − Bᵀλ̂}`. Theorem 1: the completed pair is
dual feasible, so the cut is valid by weak duality; Theorem 2: an optimizer of the
completion LP gives the strongest cut among all completions of that `λ̂`. §2.2: for
bounded recourse the completion is closed-form, `z^l = (q − Bᵀλ)⁺,
z^u = (Bᵀλ − q)⁺`, "similarly to the DLL-style completion framework". Feasibility
cuts use a slice normalization `rᵀλ ≤ 1` of the Farkas cone. Training is
self-supervised on the certified dual bound, motivated explicitly by dual
degeneracy: "since Benders subproblems are often dual-degenerate, multiple
dual-optimal certificates may generate identical cuts, making dual vectors ambiguous
supervision targets." Instances: CAP, UFL to 2000×2000, MCNDP; up to **161× median
speedup** and **>240× fewer cuts** on the largest UFL, median gaps below 0.5 %.

*Relation.* Same skeleton, opposite side. **Same:** self-supervised, closed-form
completion, validity independent of prediction quality, Benders. **Different:**
(i) they predict the **dual** border; we predict the **primal** coupling variables
and *derive* the dual. (ii) They obtain a valid **lower** bound per subproblem and
say so about the state of the art: "For UBs, there is currently **no systematic
approach to obtaining high-quality UBs from the SP**. Existing methods rely
primarily on problem-specific heuristics." Their remedy is an "optional deployment
refinement" that re-prices integer incumbents by their exact recourse cost after the
search, plus a problem-specific greedy repair when a strictly feasible design is
required. Our fill returns a feasible dispatch with **exact cost** on every call, so
we have a certified UB *and* LB per subproblem — this substantiates the summary's
claim that Proxy-BD lacks the upper bound, but note the claim must be stated as
"per-subproblem certified UB", not "no UB at all". (iii) No exact polish; their
repair is greedy and problem-specific. (iv) Their domain is facility location and
network design; the ED subproblem's zero-cost coupling variables — the thing that
makes the sign-only gradient failure possible — do not arise there.

### ML-accelerated Benders, cut selection and cut classification

- **Huiwen Jia, Siqian Shen, *Benders Cut Classification via Support Vector Machines
  for Solving Two-Stage Stochastic Programs*, INFORMS Journal on Optimization
  **3**(3):278–297, 2021 (DOI 10.1287/ijoo.2019.0050; preprint arXiv:1906.05994).**
  LearnBD adds a per-iteration cut-classification step: an SVM trained on cuts
  sampled from training instances decides which cuts are worth generating. Tested on
  capacitated facility location and multicommodity network design.
- ⚠️ **"Lee, Ma, … *Accelerating Benders decomposition via machine learning*" does
  not exist under that title.** The actual paper is **Mengyuan Lee, Ning Ma,
  Guanding Yu, Huaiyu Dai, *Accelerating Generalized Benders Decomposition for
  Wireless Resource Allocation*, arXiv:2003.01294, IEEE Transactions on Wireless
  Communications.** A cut classifier and a cut regressor separate useful from
  useless cuts; **generalized** Benders for MINLP, wireless, not power systems.
- **Fouad Hasan, Amin Kargarian, *Accelerating L-shaped Two-stage Stochastic SCUC
  with Learning Integrated Benders Decomposition*, arXiv:2311.10835 (2023;
  preprint).** In power systems and closest in spirit on the cut side: a regressor
  reads load-profile scenarios and predicts **subproblem objective proxy variables**
  to form tighter cuts; a usefulness criterion keeps or discards cuts.
- **Kyle Mana, Fernando Acero, Stephen Mak, Parisa Zehtabi, Michael Cashmore,
  Daniele Magazzeni, Manuela Veloso, *Accelerating Cutting-Plane Algorithms via
  Reinforcement Learning Surrogates*, arXiv:2307.08816 (v2 27 Feb 2024).** ⚠️ The
  title changed; search engines still serve the older "Towards Accelerating Benders
  Decomposition…". RL surrogates for NP-hard elements of cut generation while
  retaining optimality guarantees; up to 45 % faster average convergence.

### ML for expansion planning

- **Stefan Borozan, Spyros Giannelos, Paola Falugi, Alexandre Moreira, Goran Strbac,
  *Machine Learning-Enhanced Benders Decomposition Approach for the Multi-Stage
  Stochastic Transmission Expansion Planning Problem*, Electric Power Systems
  Research **237**:110985, December 2024 (DOI 10.1016/j.epsr.2024.110985; preprint
  arXiv:2304.07534).** ML-enhanced **multicut** Benders that identifies effective
  and ineffective optimality cuts by supervised learning; multi-stage TEP on IEEE24
  and IEEE118 with storage investment options. **This is the nearest published work
  by problem class.**
- **Taehyeon Kwon, Anirudh Subramanyam, *Machine Learning-Enabled Large-Scale
  Capacity Expansion Planning under Uncertainty*, arXiv:2603.13508 (Mar 2026;
  preprint).** AutoSCEP selects the minimum sufficient scenario count and horizon
  to estimate production costs to a given precision, then trains linear and neural
  **surrogates of expected production cost for arbitrary expansion plans** and
  embeds them in the planning model; continental-scale EMPIRE. Its comparison
  protocol is worth copying: it beats parallel progressive hedging "under **equal
  wall-clock budgets that include data generation, training, and solve times**".
- **Wanhong Yu, Boyung Jürgens, Leonard Göke, *Surrogate-based prioritization of
  sub-problems for Benders decomposition in energy planning*, arXiv:2607.05063 (Jul
  2026; preprint).** Surrogates estimate sub-problem objectives, assess the
  cutting-plane estimator's error, and **prioritize the sub-problem with the largest
  error**. Honest negative result worth noting: "geometric interpolation-based
  surrogates are more accurate than machine learning methods" in their case.
  Speed-ups 33 % / 55 % sequential, 19 % asynchronous.

*Relation.* Every one of these uses ML for **cut selection**, **sub-problem
prioritization**, or a **surrogate objective embedded in the master**. None replaces
the subproblem with a proxy returning an exactly feasible primal, an exact cost and
closed-form duals that are simultaneously the cut coefficients. Proxy-BD is the only
one that returns valid cuts from a proxy, and it does so from predicted duals.

### Predict-then-repair for unit commitment

**Álinson S. Xavier, Feng Qiu, Shabbir Ahmed, *Learning to Solve Large-Scale
Security-Constrained Unit Commitment Problems*, INFORMS Journal on Computing, 2021
(DOI 10.1287/ijoc.2020.0976; preprint arXiv:1902.01697).** From previously solved
instances it predicts redundant constraints, good initial feasible solutions
(warm starts), and affine subspaces where the optimum likely lies, then hands
everything to the MIP solver: "**4.3× faster with optimality guarantees, 10.2×
faster without**", with out-of-distribution robustness experiments. Also **Arun
Venkatesh Ramesh, Xingpeng Li, *Feasibility Layer Aided Machine Learning Approach
for Day-Ahead Operations*, arXiv:2208.06742**, where classifiers predict the
commitment schedule and "a feasibility layer plus post-processing" prevents
infeasible predictions.

### Worst-case verification, an orthogonal guarantee

**Wenbo Chen, Haoruo Zhao, Mathieu Tanneau, Pascal Van Hentenryck, *Compact
Optimality Verification for Optimization Proxies*, ICML 2024 (arXiv:2405.21023)**
determines "the worst-case optimality gap over the instance distribution", building
on **Rahul Nellikkath, Spyros Chatzivasileiadis** (*Physics-Informed Neural Networks
for Minimising Worst-Case Violations in DC Optimal Power Flow*, IEEE SmartGridComm
2021, DOI 10.1109/SmartGridComm51999.2021.9632308; and *Physics-Informed Neural
Networks for AC Optimal Power Flow*, Electric Power Systems Research
**212**:108412, 2022).

*Relation.* A different kind of guarantee: theirs is an expensive **offline**
distribution-wide worst case; ours is a cheap **per-instance online** bound, which
is what a Benders master needs (every cut must be valid, not most of them). Klamkin
et al. make exactly this argument in their §I-A.

### Degeneracy of the solution map

**Milad Hoseinpour, Vladimir Dvorkin, *DiffOPF: Diffusion Solver for Optimal Power
Flow*, arXiv:2510.14075 (v1 15 Oct 2025, v2 13 Mar 2026; preprint).** "The optimal
power flow is a **multi-valued**, non-convex mapping from loads to dispatch
setpoints … Existing deep learning OPF solvers are single-valued and thus fail to
capture the variability", so they treat OPF as conditional sampling and produce
"statistically credible warm starts with favorable cost and constraint satisfaction
trade-offs."

*Relation.* This is the closest published statement of FINDINGS F25's second half
(52 % of our optima admit a nonzero circulation; the flow target is a face, not a
point) — though their multi-valuedness comes from parameter variability rather than
LP degeneracy, and they respond by sampling where IDEAS I16 responds with a small
quadratic regularizer that selects the minimum-norm point. Worth citing when F25 is
written up, and worth noting the difference in remedy.

### Mixed precision where the result is a prediction, exact where it is a bound

**Erin Carson, Nicholas J. Higham, *Accelerating the Solution of Linear Systems by
Iterative Refinement in Three Precisions*, SIAM Journal on Scientific Computing
**40**(2):A817–A847, 2018 (DOI 10.1137/17M1140819).** The classical statement of
the principle F31 arrives at independently: do the approximate work in low
precision, compute the residual and correction in high precision, and the accuracy
of the result is set by the high-precision part. No power-systems proxy paper found
does mixed precision this way (Klamkin et al. use `torch.compile` on an H200 but
report no precision split), so F31's "cheap arithmetic where the result is a
prediction, exact arithmetic where the result is a bound" appears to be new **in
this literature** while being an old idea **in numerical linear algebra**. Cite
Carson & Higham for the principle.

---

## What appears to be new

Stated conservatively: each item below is something the search did **not** find in
the literature, with the nearest thing that was found named next to it. None of
these is a claim that no such work exists anywhere — only that a targeted search
against primary sources did not surface it.

1. **A closed-form merit-order fill as a differentiable completion layer.** Every
   completion found restores feasibility and ignores cost: E2ELR's proportional
   response, DC3's variable elimination plus projected gradient steps, DeepOPF's
   power-flow reconstruction plus ℓ₁ projection, LOOP-LC's gauge map. *Nearest:*
   E2ELR (same architecture slot, same self-supervised loss, cost-blind repair).
2. **Completing for optimality rather than feasibility as a stated design rule,
   with the gradient consequence.** The observation that a feasibility completion
   hands the coupling variables a penalty multiplier while an optimality completion
   hands them the true price (`flow_first_summary.md` §4 rule 2, measured in F3 and
   F17) was not found stated anywhere. *Nearest:* Amos's amortized-optimization
   tutorial (arXiv:2202.00665) discusses envelope/Danskin differentiation of
   objective values generally, but not this design rule.
3. **Predicting only the line flows, with generation and unmet demand computed.**
   No GNN found predicts flows as its sole output with the dispatch completed
   downstream. *Nearest:* Donon et al., IJCNN 2019 (edge-to-flow decoder, but
   supervised imitation of a power-flow solver, no dispatch, no cost);
   Chatzos/Mak/Van Hentenryck TPWRS 2022 (predicts inter-region coupling flows, but
   supervised and the regions are learned approximators too).
4. **Recovering the dual from the primal prediction's congestion pattern.** No dual
   proxy found derives the dual from the primal's active set; they predict it (DLL,
   DIPL, Dual Conic Proxies, PDL, Klamkin et al.) or differentiate a learned convex
   value function (Zhang et al. TCNS 2022). The specific construction — contract
   uncongested lines into regions, one merit-order fill per region's aggregate
   residual, node-wise exact ascent over the finite price alphabet — was not found.
   *Nearest:* Zhang, Chen, Zhang (TCNS 2022), which derives every other multiplier
   from the prices in closed form exactly as we do, but takes the prices from
   `∇_ℓ` of an ICNN.
5. **A certified upper bound *and* lower bound per Benders subproblem from one
   forward pass.** Proxy-BD gets a valid cut (a lower bound on `Q`) and states that
   "there is currently no systematic approach to obtaining high-quality UBs from the
   SP". Our fill returns a feasible dispatch with exact cost, so the UB is free.
   *Nearest:* Proxy-BD (LB only, plus post-hoc re-pricing of incumbents); Klamkin
   et al. (both bounds, but for standalone ED, not inside a decomposition, and the
   dual is learned).
6. **An exact min-cost-flow polish of a learned solution.** Cycle cancelling as the
   exact finisher of a proxy was not found. *Nearest:* Davies et al. (ICML 2023),
   predicted flows seeding Ford–Fulkerson — max-flow, no costs; Dinitz et al.
   (NeurIPS 2021) and Chen et al. (ICML 2022), learned duals warm-starting
   matching and min-cost 0-1 flow; Wiesler & Baxley (arXiv 2604.21175), a GNN
   warm-starting Ford–Fulkerson measured in augmentations, but with no experiments.
7. **Mixed precision split along the prediction/bound boundary** — network in
   float16, fill, polish, dual and certificate in float64 — inside a single CUDA
   graph. No power-systems proxy paper found reports a precision split at all.
   *Nearest:* Carson & Higham (2018) for the principle in numerical linear algebra.
8. **A per-node value function summed and differentiated through to train a
   coupling-variable network** (the critic extension). The three ingredients each
   exist — ICNN value functions (Amos et al.; Rosemberg et al.; Zhang et al.),
   value surrogates embedded in decompositions (Neur2SP / Neur2RO / Neur2BiLO,
   ν-SDDP, ICNN-enhanced 2SP), and actor-critic training on the critic's action
   gradient (Silver et al.; Lillicrap et al.; D'Oro & Jaśkowski) — but the
   combination was not found. *Nearest:* Rosemberg et al. (ICNN value function of a
   dispatch, but global and supervised); Bouton et al. (summed per-entity utilities
   plus a learned correction, but RL and no optimization decomposition).

---

## Claims to be careful about

Where the literature is closer than the notes assume.

1. **The dual completion formula is DLL's Example 1.** `μ^u_g = (λ_n − C_g)⁺`,
   `μ^l_g = (C_g − λ_n)⁺` and the rest are the bounded-variable case
   `ẑ^l = |c − Aᵀŷ|⁺, ẑ^u = |c − Aᵀŷ|⁻`, proved in Tanneau & Van Hentenryck
   (NeurIPS 2024) and, before it, in Klamkin et al.'s DIPL for LPs — and the same
   `τ̄(µ) = [µ − c]⁺` appears in Zhang, Chen, Zhang (TCNS 2022). **Cite it; do not
   present it as new.** What is ours is the *source of `λ`*.
2. **"Every `λ` completes, so the dual is always feasible" is the same observation
   DLL and Proxy-BD build on.** Proxy-BD's Theorem 1 is exactly "any projected
   prediction completes to a dual-feasible point, hence a valid cut". The phrasing
   "the primal decides tightness only" is a good one, but the fact is established.
3. **Certificate-plus-fallback is Klamkin et al.'s idea and predates our
   measurements.** Their v1 is 17 October 2025; F26–F31 are September 2026.
   `flow_first_summary.md` §7 says "Independent parallel development with Klamkin et
   al." — that is a claim about our own timeline which this file cannot verify, and
   it should not appear in a paper without a dated artefact behind it. Safer: cite
   them as prior work on the certification axis and differentiate on mechanism
   (derived vs learned dual, polish, exact path step, no fallback needed at 20
   nodes).
4. **Self-supervised training on the objective through a closed-form differentiable
   repair is E2ELR's contribution.** They state it in the abstract and implement it
   for economic dispatch. Our loss is the same construction; the difference is that
   the repair is optimal, not proportional. **Do not claim self-supervision or "the
   loss is the objective" as new.**
5. **Sigmoid-squashed outputs to enforce box constraints are standard.** E2ELR
   (`p̃ = z · p̄`), CANOS (bound enforcement), and others use it. Our use on line
   limits is the same trick applied to a different variable. Relatedly, the sigmoid
   trap of F21 is the known dead-gradient problem, and IDEAS I6's fix (exact L1
   penalty outside the box) is DC3/E2ELR-adjacent.
6. **"Predict a subset, derive the rest" originates with DeepOPF (2019/2021), not
   with DC3.** The dimensionality argument we make for predicting only the flows is
   their predict-and-reconstruct argument, one variable class over.
7. **The finite price alphabet is textbook LP duality plus textbook LMP theory.**
   That an uncongested LP dispatch prices at the marginal unit's cost, and that
   prices are equal across an uncongested cut, is standard nodal-pricing material
   (William W. Hogan, *Contract networks for electric power transmission*, Journal
   of Regulatory Economics **4**(3):211–242, 1992, DOI 10.1007/BF00133621). What can
   be claimed is the **use** of the alphabet — exact coordinate ascent over ten
   values instead of a softmax head — not the fact.
8. **The min-cost-flow structure of a transport-model dispatch is classical.** Klein
   (1967) already covers convex-cost network flows and names the transportation
   problem. F29's contribution is the warm start and the measurement, not the model.
9. **Proxy-BD does obtain upper bounds, just not per subproblem from the proxy.**
   Their "optional deployment refinement" re-prices integer incumbents by exact
   recourse cost, and their MCNDP tables report proxy incumbent versus oracle upper
   bound. State the claim precisely: they have no *certified per-subproblem* UB from
   the proxy.
10. **Warm-start speedup claims in this literature are contested.** WARP
    (arXiv:2605.05728) shows that primal-only warm-start gains for interior-point
    OPF vanish against the solver's real default. Our F29 result is on the right
    side of that critique (we compare against a persistent model *and* an
    optimal-basis oracle, and our start is primal **and** dual), but the paper
    should say so explicitly rather than leave it implicit.
11. **"Optimization proxy", "completion", "repair layer", "certificate" are all
    established vocabulary** in the Van Hentenryck group's line of work. Using their
    terms is right; presenting the terms as ours is not.

---

## Unverified

Listed so nothing here is mistaken for a checked fact.

- **"Water values" as the name for the slope of the cost-to-go in reservoir
  storage** — not present in the Pereira & Pinto abstract; the full text is
  paywalled and was not read. Standard vocabulary, but unsourced here.
- **"Outer approximation" as Shapiro's or Pereira & Pinto's phrasing** — Shapiro
  says "cutting planes", "maximum of a collection of cutting planes", "lower
  bounds". Our gloss is accurate but should not be quoted.
- **Goldberg & Tarjan's "cancel and tighten"** — the minimum-mean-cycle framing and
  the strongly-polynomial claim are confirmed from the abstract; that the
  cancel-and-tighten variant appears in *this* paper under *that* name was not
  confirmed (JACM full text inaccessible).
- **Xavier, Qiu, Ahmed volume/pages** — Crossref and OpenAlex both return the DOI
  (10.1287/ijoc.2020.0976) with no volume, issue or pages, and INFORMS PubsOnline
  blocks automated fetch. Secondary indexes say INFORMS Journal on Computing
  **33**(2):739–756, 2021; **not primary-verified**. The publication year is
  inconsistent across records (2020 online, 2021 in-issue).
- **E2ELR's own numbers beyond Tables VI/VII** — the table structure and the quoted
  sentences were read from the PDF; the surrounding discussion of Table VIII/IX
  (benefits of end-to-end training) was not.
- **DeepOPF's Case300 anomaly** — DeepOPF's 81.4 ms on Case300 breaks the trend from
  Case118's 2.48 ms and is most likely the ℓ₁-projection post-processing, but that
  attribution was **not** verified in the text.
- **Baker's *A Learning-boosted Quasi-Newton Method for AC Optimal Power Flow*
  (arXiv:2007.06074) publication venue** — none found; treat as a preprint.
- **Owerko, Gama, Ribeiro (arXiv:2210.09277) status** — the comments field says
  "Submitted to IEEE Transactions on Power Systems"; whether it appeared was not
  checked.
- **ν-SDDP's ICLR 2022 venue** (Dai, Xue, Syed, Schuurmans, Dai, arXiv:2112.00874) —
  confirmed only indirectly, through an OpenReview forum id cited elsewhere; the
  arXiv abs page states only "24 pages".
- **ICNN-enhanced 2SP (arXiv:2505.05261) internals** — abstract only was read;
  whether the surrogate is per-scenario or aggregate was not confirmed. No venue
  found.
- **Sunehag et al. (arXiv:1706.05296) additive form** — stated in the body, which
  was not read; only the abstract was.
- **Wang, Feng, You, *Non-Iterative Coordination of Interconnected Power Grids via
  Dimension-Decomposition-Based Flexibility Aggregation* (arXiv:2502.07226)** —
  abstract only. Structurally it composes per-region models to choose tie-line
  schedules, which would make it a near-miss for §1's key question, but whether the
  regional models are neural, and whether they are cost functions or feasibility
  sets, was **not** determined. Read before citing.
- **Older non-ML work on neural surrogates of coupling variables in
  decomposition-based multidisciplinary design optimization** (e.g. *Structural and
  Multidisciplinary Optimization*, DOI 10.1007/s00158-011-0636-9) — surfaced in
  search, **not read**; authors, dates and content unverified.
- **Located but not read, listed so they are not forgotten:** "Learning to Cut:
  Reinforcement Learning for Benders Decomposition" (arXiv:2605.06516);
  "Graph-Based Imitation and Reinforcement Learning for Efficient Benders
  Decomposition" (arXiv:2511.11870); "Feasibility-Aware Imitation Learning for
  Benders Decomposition" (arXiv:2604.04801); "Speeding Up Logic-Based Benders
  Decomposition by Strengthening Cuts with Graph Neural Networks"
  (DOI 10.1007/978-3-031-53969-5_3); "Is learning for the unit commitment problem a
  low-hanging fruit?" (arXiv:2106.11687 — the title suggests an audit angle useful
  for §6); RL4CO (arXiv:2306.17100 — reported to contain a CPU-versus-GPU fairness
  critique); "Compact Optimization Learning for AC Optimal Power Flow"
  (Park, Chen, Mak, Van Hentenryck, arXiv:2301.08840); "Learning Optimization
  Proxies for Large-Scale Security-Constrained Economic Dispatch" (Chen, Park,
  Tanneau, Van Hentenryck, EPSR **213**:108566, 2022, DOI
  10.1016/j.epsr.2022.108566).

---

## Corrections to citations circulating in our notes

| Where | Was | Is |
|---|---|---|
| §1 | Neur2RO authors "Dumouchelle, Julien, Michael, Khalil" | **Jannis Kurtz**, not "Michael" |
| §1 | arXiv:2310.04345 titled "Neur2RO" | retitled *Deep Learning for Two-Stage Robust Integer Optimization* (v3); cite the ICLR 2024 entry |
| §1 | "MAGE: Model-based **Actor**-Gradient-Estimator" | **Action**-Gradient-Estimator; D'Oro & **Jaśkowski** |
| §1 | Chen, Shi, Zhang learn a convex value function | they learn convex **dynamics** for convex MPC; application is **building HVAC** |
| §3 | "Dual Lagrangian Learning" as a power paper | *Dual Lagrangian Learning **for Conic Optimization***, NeurIPS 2024; Tanneau & Van Hentenryck only |
| §5 | PowerFlowNet in Applied Energy | **IJEPES 160**:110112, 2024; speedup 48× (published) vs 145× (arXiv v3) |
| §5 | Donon 2019 IJCNN, six authors | four: Donon, Donnot, Guyon, Marot |
| §5 | CANOS published | **preprint only**, arXiv:2403.17660 |
| §4 | Baker 2019 as a neural warm start with a speedup | **random forest**; no headline speedup exists |
| §4 | Diehl "2.8×" | **3.75×** (abstract/conclusion), 3.8× in the results paragraph |
| §4 | Nair et al. published | **preprint**, never formally published |
| §6 | Bengio, Lodi, Prouvost as a fairness critique | it contains none; cite **Accorsi, Lodi, Vigo**, ORL 50(2), 2022 |
| §7 | "Lee, Ma, … Accelerating Benders decomposition via machine learning" | *Accelerating **Generalized** Benders Decomposition for **Wireless Resource Allocation***, Lee, Ma, Yu, Dai, arXiv:2003.01294 |
| §7 | arXiv:2307.08816 "Towards Accelerating Benders Decomposition…" | retitled *Accelerating Cutting-Plane Algorithms via Reinforcement Learning Surrogates* |
| §6 | "Compact Optimization Learning for AC OPF" by Klamkin/Tanneau | **Park, Chen, Mak, Van Hentenryck**, arXiv:2301.08840 |
| §2 | E2ELR's repair layer described as merit order | it is a **proportional response** to headroom; the "merit-order" quotation attributed to it is fabricated |
