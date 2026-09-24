# Plan: ramping inside temporal blocks — LP, matrices, Benders cut, data

No code changes yet. This plan fixes the formulation and the design; every step is gated so the current no-ramping problem keeps working unchanged.

## Context
Ramping exists in the Pyomo path (`gep_exact_solver.py`, gated on `inputs["ramping"]`, `ramping_value = 0.2` in `inputs/scalars.toml`) but not in the matrix pipeline the ML and Benders code use. The extension adds, inside each block of B hours (4/6/8) and never across blocks:

```
p_{g,t} − p_{g,t−1} ≤ R_g · u_g ,      p_{g,t−1} − p_{g,t} ≤ R_g · u_g ,      R_g := pRamping · pUnitCap_g
```

Two facts drive everything: the indivisible unit becomes the **block** (today every matrix is built per hour with `num_rows_per_t = 2(G+L+N)`, [gep_problem_operational.py:1344](gep_problem_operational.py#L1344)), and the ramp limit **depends on the investment `u`**, so it enters the Benders cut.

---

## The cut formulation

### Subproblem
For block `b` at investment `u`, with `y_t = (p_t, f_t, md_t)`:

```
Q_b(u) = min   Σ_{t∈b} w · ( cᵀ p_t + VOLL · 1ᵀ md_t )
         s.t.  E y_t = D_t                                    [ λ_t , free ]      (balance, per hour)
               p_{g,t} ≤ A_{g,t} Pmax_g u_g                   [ μ^cap_{g,t} ≥ 0 ] (capacity, per hour)
               other box rows (p ≥ 0, f bounds, md bounds)    [ μ^box_{t}  ≥ 0 ]  (rhs free of u)
               p_{g,t} − p_{g,t−1} ≤ R_g u_g                  [ ν⁺_{g,t} ≥ 0 ]    (t not first in b)
               p_{g,t−1} − p_{g,t} ≤ R_g u_g                  [ ν⁻_{g,t} ≥ 0 ]
```

### Cut
`Q_b` is convex in `u`, and by LP duality it is the maximum over dual-feasible points of a function **affine in `u`**:

```
Q_b(u) = max_{λ,μ,ν ≥ 0 feasible}  Σ_{t∈b} λ_tᵀ D_t
                                   − Σ_{t∈b} Σ_g μ^cap_{g,t} A_{g,t} Pmax_g u_g
                                   − Σ_{t∈b} μ^box_tᵀ h_t
                                   − Σ_{g} Σ_{t∈b, t>t₀} ( ν⁺_{g,t} + ν⁻_{g,t} ) R_g u_g
```

Any single dual-feasible point therefore gives a valid lower bound for **every** `u`, which is the optimality cut

```
θ_b  ≥  Σ_g slope_{b,g} · u_g  +  const_b
```

with

```
slope_{b,g} = − [ Σ_{t∈b} μ^cap_{g,t} · A_{g,t} Pmax_g   +   Σ_{t∈b, t>t₀} ( ν⁺_{g,t} + ν⁻_{g,t} ) · R_g ]
const_b     =   Σ_{t∈b} λ_tᵀ D_t  −  Σ_{t∈b} μ^box_tᵀ h_t
```

**The only change from today is the second term of the slope.** Its shape is identical to the first: a non-negative multiplier times a coefficient of `u`. The capacity rows carry `A_{g,t}Pmax_g`, the ramp rows carry `R_g`. Setting `ν ≡ 0` recovers exactly the current cut, which is the formal statement of the backward-compatibility requirement below.

Three properties worth keeping in mind:
- **Validity needs only dual feasibility, not optimality.** That is what lets an inexact subproblem answer — the learned or constructed dual — still produce a legal cut, and it stays true with ramping as long as `ν ≥ 0` and the ramp rows are included in the dual-feasibility check.
- **Ramp rows belong in the slope, never in the constant**, because their rhs contains `u`. The same rule already distinguishes the capacity rows (slope) from the line and unmet-demand bounds (constant).
- **No cross-block terms.** Ramping does not couple blocks, so `Q(u) = Σ_b Q_b(u)` and the decomposition is unchanged; aggregating a group of blocks still just sums their slopes and constants.

### Mapping to the code
`dual_vals = concat([−μ, −λ])`, and [gep_benders.py:212](gep_benders.py#L212) computes the slope as `duals[:, G:2G] · (−A_coeffs)`, reading the coefficient straight from `ineq_cm`. Since the ramp rows sit in `ineq_cm` with coefficient `−R_g` in column `g`, the implementation is the same expression over a larger row set:

- `compute_investment_duals`: gather the block's **capacity rows and ramp rows**, multiply each dual by the negated `ineq_cm` coefficient, sum.
- `compute_per_timestep_cuts`: `constraint_nrs` (the rhs terms) stays as it is — line bounds and unmet-demand bounds — since ramp rows are `u`-dependent and belong to the slope.
- `find_benders_cut_batch_for_group` and `find_benders_cut_batch`: same change, and their per-timestep strides become per-block.

---

## Backwards compatibility — the current problem keeps working

Non-negotiable for every change below:

- `ramping = "false"` (or `B = 1`) must produce **byte-identical matrices** and the same run results as today. Ramp rows are appended only when ramping is on, so row indices for the existing rows do not move.
- The cut code takes the ramp row set as **empty** in that case, so the expression collapses to today's formula rather than branching.
- `X` keeps its current layout `[D, cap_ub]` when ramping is off; the `ramp_ub` block is appended only when it is on. Existing datasets, checkpoints and `split_X` callers are then untouched.
- Regression check before anything else lands: the saved 3-node dataset rebuilds to identical `ineq_cm`, `eq_cm`, `X`, and an existing Benders run reproduces its recorded UB/LB trajectory.

---

## Implementation steps

**1. Config.** `ramp_block_hours` (B) in `ED_args` / `Benders_args`; `ramping`, `ramping_value` are already parsed by `gep_config_parser.py`. Assert `sample_duration % B == 0` at startup.

**2. `gep_problem_operational.py`** — the block becomes the unit.
- Variables per block `B·(G+L+N)`; rows per block `B·2(G+L+N)` box rows **then** `2G(B−1)` ramp rows; equalities `B·N`.
- `X` per block `[D (B·N), cap_ub (B·G), ramp_ub (G)]`. The ramp limit is constant across the block while `cap_ub` varies by hour through availability, so it cannot be derived from the existing columns — it is a new input block, and the first thing the network will need.
- Update `split_X`, `capacity_ub_indices`, `missed_demand_ub_indices`, new `ramp_ub_indices`, `split_dec_vars_from_Y`, `obj_fn`, `dual_obj_fn`, `ineq_resid`/`ineq_dist`, `eq_resid`, `split_ineq_constraints`.

**3. `gep_problem.py`** — ramp rows in the GEP matrices with the `u` coefficient `−R_g`.

**4. `gep_benders.py`** — block-aware slicing in `find_subproblem_cm_rhs_obj_from_mats`; the cut change above; cut grouping over blocks; `num_alpha` and `_ensure_master_model` follow the block count. Subproblem count drops from T to T/B, each B× larger.

**5. `create_gep_dataset.py`** — sample **contiguous B-hour windows**; one labelled instance per block (Gurobi solves the block LP, storing `y`, `obj`, `mu`, `lamb` in the block layout). Instance count falls by ~B for the same hours; decide whether to raise the sample count.

---

## Verification
1. **Regression (do this first):** `ramping = "false"` rebuilds identical matrices and reproduces an existing run's numbers.
2. **Against the Pyomo model:** for one instance and investment, the matrix block LP objective equals `gep_exact_solver.py` with ramping on, to 1e-9.
3. **Ramping actually binds:** on a labelled block, some ramp rows are tight and the optimum is anticipatory — a unit runs before it is needed. Otherwise B or `ramping_value` is too loose to test anything.
4. **The cut is valid:** `--cut-selection single --benders-setup Exact` on one small sample must reach the `--solve-direct` objective. A missing ramp term in the slope shows up as LB above UB, or as non-convergence — not as an exception.
5. **Duals:** weak duality per block (`dual_obj_fn ≤ obj_fn`) with the ramp multipliers included.

## Notes for the ML stage (later)
- The time-expanded graph is the natural formulation: `(g,t)` linked to `(g,t−1)` with limit `R_g·u_g`, alongside the line edges within each hour.
- The merit-order fill does **not** survive as a sort, since the block optimum is anticipatory. Whether flow-first keeps an exact closed-form fill depends on whether the per-node block problem is a min-cost flow on that time-expanded graph — worth settling before designing the network.
