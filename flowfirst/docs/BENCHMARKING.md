# Making the Gurobi baseline in the Benders benchmark realistic

Everything below concerns `gep_benders.py`, class `BendersSolver`.

## What happens now

`BendersSolver._solve_sub_lp` creates a new Gurobi model on every
economic-dispatch subproblem solve: a new `Model`, new variables, and new
constraints built from dense matrices. The timer wraps only `optimize()`, so
the reported time excludes construction. Each solve still starts cold: no
basis from the previous iteration, and a full presolve.

## What to change

Build the subproblem model once and reuse it, the way the master already
works. `BendersSolver._ensure_master_model` builds the master once and stores
it as `self.master_model`, and `_add_new_cuts_to_master` appends cuts to that
same model.

Between solves only the right-hand side changes: `b_ineq` carries the master's
capacity decisions and `b_eq` the hourly demand, while `A_ineq` and `A_eq` are
fixed for a given grid. The per-solve update is two right-hand-side
assignments on a model that already exists.

Because the model object survives, Gurobi keeps the previous optimal basis and
re-solves from it with dual simplex. The basis is that previous optimum
expressed as a vertex, which is the form simplex restarts from, and Gurobi
holds it internally. Reusing the model is the whole mechanism; `Start` and
`PStart` are for the case where the model is gone.

## Why this is the right baseline

Consecutive master iterations change only the capacities, so each subproblem
differs from the previous one in its right-hand side alone, which in practice
is where the solve would start from.

A lower capacity can make the previous solution infeasible. The previous basis
stays usable: changing the right-hand side leaves it dual feasible, which is
the condition dual simplex starts from, and dual simplex restores primal
feasibility in a few iterations.

## What it changes

Measured on the full instance (20 countries, 44 lines, 107 generators, so 171
variables per subproblem LP) on an M2 Pro: 0.32 ms per solve for a fresh model
against 0.16 ms for a persistent one. The current setup reports about twice
the per-solve time of a realistic implementation.

## Two practical notes

- `_solve_sub_lp` is the thread-safe path and takes its environment from
  `_get_worker_env`. Gurobi models are not thread-safe, so cache one
  persistent model per worker, next to that environment.
- Measure whether presolve helps or hurts on repeated solves this small.

## What to report

Solve time from the persistent model, the one-off build cost separately, the
thread count (`_solve_sub_lp` already pins `Threads=1`), and the machine. Log
simplex iterations per subproblem as well: at 171 variables the wall clock is
close to Gurobi's API floor, so the iteration count is the more informative
number.
