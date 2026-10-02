"""
Solver-independent model layer for the GEP experiments.

Models are built in Pyomo and solved through Pyomo's persistent interfaces
(pyomo.contrib.solver), so the same model runs on Gurobi or HiGHS:

    "gurobi" -> gurobi_persistent
    "highs"  -> highs (persistent)

Persistent matters for Benders: the master keeps its model between iterations
and only receives the new cuts, and the hourly ED LP is built once and then
re-solved with only its right-hand side changed.

Fair comparison between methods rests on three things kept identical here for
every model, whichever method builds it:
  * one solver for the whole run (direct, master, subproblems, gap-gate re-solves);
  * one parameter profile (threads, seed, gap, time limit, LP tolerance), mapped
    onto each solver's own option names by solver_options();
  * one timing definition: SolveStats.solve_time is the time inside the solver's
    own optimize()/run() call -- the span the gurobipy code timed around
    m.optimize() -- while the Pyomo bookkeeping around it (model translation,
    pushing updates, loading the solution) is reported separately as sync_time.
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

import pyomo.environ as pyo   # registers the solver plugins with the factory
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus, TerminationCondition
from pyomo.core.expr.numeric_expr import LinearExpression

SOLVERS = ("gurobi", "highs")
_PYOMO_NAME = {"gurobi": "gurobi_persistent", "highs": "highs"}

#! Pyomo re-scans the whole model for structural changes before every solve by
#! default. The hourly LP only ever changes its right-hand side, so every check
#! except the parameter push is switched off for it; that keeps a re-solve at
#! about a millisecond. (The two interfaces name their switches differently.)
PARAMS_ONLY_UPDATES = "params_only"


def check_solver_name(name):
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver={name!r}. Choose from {SOLVERS}.")
    return name


def _adopt_gurobi_env(env):
    """Make Pyomo's Gurobi interfaces use `env` instead of opening their own.

    Pyomo shares one class-level gurobipy.Env across its Gurobi solvers and
    would start it with no parameters, which bypasses the WLS keys make_env()
    reads. It also closes that Env once its last solver is garbage collected,
    so one extra client is registered to keep ours alive for the process.
    """
    from pyomo.contrib.solver.solvers.gurobi.gurobi_direct_base import GurobiDirectBase
    if GurobiDirectBase._gurobipy_env is env:
        return
    if GurobiDirectBase._gurobipy_env is None:
        GurobiDirectBase._gurobipy_env = env
        GurobiDirectBase._num_gurobipy_env_clients += 1


def make_solver(name, gurobi_env=None, auto_updates=None):
    """A fresh persistent Pyomo solver for `name` ("gurobi" or "highs")."""
    check_solver_name(name)
    if name == "gurobi" and gurobi_env is not None:
        _adopt_gurobi_env(gurobi_env)
    opt = SolverFactory(_PYOMO_NAME[name])
    if not opt.available():
        hint = "pip install highspy" if name == "highs" else "check the gurobipy install and licence"
        raise RuntimeError(f"Solver {name!r} is not available ({opt.available()}); {hint}.")
    if auto_updates == PARAMS_ONLY_UPDATES:
        for key in list(opt.config.auto_updates.keys()):
            setattr(opt.config.auto_updates, key, key == "update_parameters")
    return opt


def solver_options(name, *, threads=1, seed=0, time_limit=None, mip_gap=None,
                   lp_optimality_tol=None, log_file=None, tee=False):
    """One parameter profile, in the native option names of `name`.

    Every option the old gurobipy code set explicitly has a counterpart here;
    anything not listed (feasibility tolerances, algorithm choice, ...) is left
    at the solver's own default, as it was before.

    Logging is always set explicitly -- also to "off" -- because a persistent
    model keeps its parameters between solves: a log file set for one master
    solve would otherwise keep collecting every later one.
    """
    check_solver_name(name)
    log = bool(log_file) or bool(tee)
    if name == "gurobi":
        opts = {
            "Threads": threads,
            "Seed": seed,
            "OutputFlag": int(log),
            "LogToConsole": int(bool(tee)),
            "LogFile": log_file or "",
        }
        if time_limit is not None:
            opts["TimeLimit"] = float(time_limit)
        if mip_gap is not None:
            opts["MIPGap"] = float(mip_gap)
        if lp_optimality_tol is not None:
            opts["OptimalityTol"] = float(lp_optimality_tol)
    else:
        opts = {
            "threads": threads,
            "random_seed": seed,
            "output_flag": log,
            "log_to_console": bool(tee),
            "log_file": log_file or "",
        }
        if time_limit is not None:
            opts["time_limit"] = float(time_limit)
        if mip_gap is not None:
            opts["mip_rel_gap"] = float(mip_gap)
        if lp_optimality_tol is not None:
            #! HiGHS' counterpart of Gurobi's OptimalityTol (reduced-cost tolerance).
            opts["dual_feasibility_tolerance"] = float(lp_optimality_tol)
    return opts


@dataclass
class SolveStats:
    status: str             # Pyomo termination condition, e.g. "convergenceCriteriaSatisfied"
    has_solution: bool
    objective: float        # incumbent objective, nan without a solution
    bound: float            # best dual bound (MIP) or the LP optimum, nan if unknown
    gap: float              # |objective - bound| / |objective|, same formula for both solvers
    nodes: int              # branch-and-bound nodes, -1 for an LP
    iterations: int         # simplex iterations
    solve_time: float       # inside the solver's optimize()/run()
    sync_time: float        # Pyomo bookkeeping in the same call (translate/update/load)
    results: object

    @property
    def optimal(self):
        return self.status == TerminationCondition.convergenceCriteriaSatisfied.name

    @property
    def hit_time_limit(self):
        return self.status == TerminationCondition.maxTimeLimit.name


def _timer_total(timer, name):
    """Total time of the (possibly nested) timer called `name`, 0 if it never ran."""
    total = 0.0
    stack = [timer]
    while stack:
        t = stack.pop()
        for key, sub in t.timers.items():
            if key == name:
                total += sub.total_time
            stack.append(sub)
    return total


def _extra_int(res, *keys):
    """First of `keys` the solver reported in extra_info (Gurobi and HiGHS name them differently)."""
    for key in keys:
        if key in res.extra_info:
            try:
                return int(res.extra_info[key])
            except (TypeError, ValueError):
                pass
    return -1


def solve(opt, model, options, tee=False):
    """Solve `model`, load the solution if there is one, and report on it.

    Never raises on a non-optimal status: the caller decides, because a master
    stopped by its time limit still carries a usable incumbent and bound.
    """
    timer = HierarchicalTimer()
    t0 = time.perf_counter()
    res = opt.solve(model, solver_options=options, timer=timer, tee=tee,
                    load_solutions=False, raise_exception_on_nonoptimal_result=False)
    has_solution = res.solution_status in (SolutionStatus.optimal, SolutionStatus.feasible)
    if has_solution:
        res.solution_loader.load_vars()
    wall = time.perf_counter() - t0
    solve_time = _timer_total(timer, "optimize")

    obj = res.incumbent_objective
    bound = res.objective_bound
    obj = float(obj) if obj is not None else np.nan
    bound = float(bound) if bound is not None and np.isfinite(bound) else np.nan
    gap = abs(obj - bound) / max(abs(obj), 1e-10) if np.isfinite(obj) and np.isfinite(bound) else np.nan

    return SolveStats(
        status=res.termination_condition.name,
        has_solution=has_solution,
        objective=obj, bound=bound, gap=gap,
        nodes=_extra_int(res, "NodeCount", "mip_node_count"),
        iterations=_extra_int(res, "IterCount", "simplex_iteration_count"),
        solve_time=solve_time,
        sync_time=max(0.0, wall - solve_time),
        results=res,
    )


def linear_expr(coefs, xs):
    """sum_k coefs[k] * xs[k] as a single Pyomo LinearExpression (no operator overloading)."""
    return LinearExpression(constant=0.0, linear_coefs=list(coefs), linear_vars=list(xs))


def add_matrix_constraints(model, name, x, A, rhs, sense):
    """Add the rows `A @ x <= rhs` (sense "<=") or `A @ x == rhs` (sense "==") as model.<name>.

    `A` may be dense or scipy-sparse; `rhs` is an array or an indexed mutable
    Param (for models whose right-hand side changes between solves). Row r of A
    becomes model.<name>[r], so duals come back in the original row order.
    """
    A = sp.csr_matrix(A)
    xs = [x[j] for j in range(A.shape[1])]
    indptr, indices, data = A.indptr, A.indices.tolist(), A.data.tolist()
    rhs_at = (lambda r: rhs[r]) if isinstance(rhs, pyo.Param) else (lambda r, _b=np.asarray(rhs, float).tolist(): _b[r])

    def rule(m, r):
        s, e = indptr[r], indptr[r + 1]
        if s == e:
            #! An all-zero row is either trivially true or makes the model infeasible;
            #! no solver needs to see it. None of the GEP matrices has one.
            return pyo.Constraint.Skip
        body = linear_expr(data[s:e], [xs[j] for j in indices[s:e]])
        return (None, body, rhs_at(r)) if sense == "<=" else (body, rhs_at(r))

    if sense not in ("<=", "=="):
        raise ValueError(f"sense must be '<=' or '==', got {sense!r}")
    model.add_component(name, pyo.Constraint(range(A.shape[0]), rule=rule))
    return getattr(model, name)


def build_matrix_model(obj, A_ineq, b_ineq, A_eq=None, b_eq=None, *,
                       integer=None, lb=None, ub=None):
    """min obj @ x  s.t.  A_ineq x <= b_ineq,  A_eq x == b_eq.

    `integer` is a boolean mask (default all continuous); `lb`/`ub` are scalars
    or arrays, None meaning unbounded -- the gurobipy code used lb=-inf
    explicitly, and Pyomo's default Var is free, so that carries over as-is.
    """
    obj = np.asarray(obj, dtype=float).reshape(-1)
    n = obj.size
    integer = np.zeros(n, dtype=bool) if integer is None else np.asarray(integer, dtype=bool)
    lbs = np.full(n, np.nan) if lb is None else np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    ubs = np.full(n, np.nan) if ub is None else np.broadcast_to(np.asarray(ub, dtype=float), (n,))

    def bounds(m, j):
        lo, hi = lbs[j], ubs[j]
        return (None if np.isnan(lo) or np.isneginf(lo) else float(lo),
                None if np.isnan(hi) or np.isposinf(hi) else float(hi))

    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(n), bounds=bounds,
                  domain=lambda m, j: pyo.Integers if integer[j] else pyo.Reals)
    nz = np.flatnonzero(obj)
    m.obj = pyo.Objective(expr=linear_expr(obj[nz], [m.x[j] for j in nz]), sense=pyo.minimize)
    add_matrix_constraints(m, "ineq", m.x, A_ineq, b_ineq, "<=")
    if A_eq is not None:
        add_matrix_constraints(m, "eq", m.x, A_eq, b_eq, "==")
    return m


def constraint_list(model, *names):
    """The model's constraint rows in index order across `names` -- the dual layout."""
    rows = []
    for name in names:
        comp = getattr(model, name, None)
        if comp is not None:
            rows.extend(comp[r] for r in sorted(comp.keys()))
    return rows


def model_values(model):
    return np.array([model.x[j].value for j in sorted(model.x.keys())], dtype=float)


class _GurobiRHSPath:
    """Hot path for MatrixLP on Gurobi: set the rows' RHS and call optimize() directly."""
    def __init__(self, opt, rows, xs, n_ineq):
        import gurobipy
        self.m = opt._solver_model
        self.cons = [opt._pyomo_con_to_solver_con_map[c] for c in rows]
        self.vars = [opt._pyomo_var_to_solver_var_map[v] for v in xs]
        if any(isinstance(c, tuple) for c in self.cons):
            raise TypeError("range constraints have two Gurobi rows")
        self.optimal = gurobipy.GRB.OPTIMAL

    def solve(self, rhs):
        self.m.setAttr("RHS", self.cons, rhs.tolist())
        t0 = time.perf_counter()
        self.m.optimize()
        dt = time.perf_counter() - t0
        if self.m.Status != self.optimal:
            raise RuntimeError(f"Subproblem status={self.m.Status} (gurobi) -- duals unavailable.")
        return (self.m.ObjVal, np.array(self.m.getAttr("X", self.vars)),
                np.array(self.m.getAttr("Pi", self.cons)), dt)


class _HighsRHSPath:
    """Hot path for MatrixLP on HiGHS: change the row bounds and call run() directly."""
    def __init__(self, opt, rows, xs, n_ineq):
        import highspy
        self.h = opt._solver_model
        self.rows = np.array([opt._pyomo_con_to_solver_con_map[c] for c in rows], dtype=np.int32)
        self.cols = np.array([opt._pyomo_var_to_solver_var_map[id(v)] for v in xs], dtype=np.int64)
        self.n_ineq = n_ineq
        self.lower = np.full(len(rows), -highspy.kHighsInf)
        self.optimal = highspy.HighsModelStatus.kOptimal

    def solve(self, rhs):
        lower = self.lower.copy()
        lower[self.n_ineq:] = rhs[self.n_ineq:]          # equalities: lower == upper == rhs
        self.h.changeRowsBounds(len(self.rows), self.rows, lower, rhs)
        t0 = time.perf_counter()
        self.h.run()
        dt = time.perf_counter() - t0
        status = self.h.getModelStatus()
        if status != self.optimal:
            raise RuntimeError(f"Subproblem status={self.h.modelStatusToString(status)} (highs) "
                               "-- duals unavailable.")
        sol = self.h.getSolution()
        return (self.h.getInfo().objective_function_value,
                np.asarray(sol.col_value)[self.cols], np.asarray(sol.row_dual)[self.rows], dt)


_RHS_PATHS = {"gurobi": _GurobiRHSPath, "highs": _HighsRHSPath}


class MatrixLP:
    """An LP  min c@x  s.t.  A_ineq x <= b_ineq,  A_eq x == b_eq  (x free) kept in a
    persistent solver, re-solved for new right-hand sides.

    Built for the hourly ED subproblem, whose matrices are identical every hour:
    only b_ineq (capacity A*Pmax*u, demand bounds) and b_eq (demand) change, so
    a re-solve pushes just the new bounds and the solver warm-starts from the
    previous hour's basis.

    The model is built in Pyomo and its first solve goes through Pyomo, which
    translates it and sets the parameters. Later solves then write the new
    right-hand side straight into that solver model and call its optimize()/run():
    Pyomo's generic solve() spends ~0.5 ms per call on output capture and config
    objects, more than the LP itself, and this LP is solved T times per Benders
    iteration. The fast path uses Pyomo's own row/column maps; if a Pyomo
    version ever lays them out differently, it is skipped and every solve goes
    through Pyomo as the first one did. Results are identical either way.

    Duals are returned in Gurobi's Pi convention, d(objective)/d(rhs) -- HiGHS'
    row_dual uses the same one, and Pyomo passes both through unchanged: all
    inequality rows first, then the equality rows, the layout the cut code
    indexes into. solve_time is the optimize()/run() call alone on both paths.
    """
    def __init__(self, solver_name, obj, A_ineq, A_eq, options, gurobi_env=None):
        A_ineq = sp.csr_matrix(A_ineq)
        A_eq = sp.csr_matrix(A_eq)
        R, Re = A_ineq.shape[0], A_eq.shape[0]
        empty = np.flatnonzero(np.diff(A_ineq.indptr) == 0).tolist() + \
                [R + r for r in np.flatnonzero(np.diff(A_eq.indptr) == 0)]
        if empty:
            raise ValueError(f"MatrixLP needs every row to have a nonzero; rows {empty} are empty "
                             "and would drop out of the dual vector.")

        obj = np.asarray(obj, dtype=float).reshape(-1)
        m = pyo.ConcreteModel()
        m.x = pyo.Var(range(obj.size))
        m.b_ineq = pyo.Param(range(R), mutable=True, initialize=0.0)
        m.b_eq = pyo.Param(range(Re), mutable=True, initialize=0.0)
        nz = np.flatnonzero(obj)
        m.obj = pyo.Objective(expr=linear_expr(obj[nz], [m.x[j] for j in nz]), sense=pyo.minimize)
        add_matrix_constraints(m, "ineq", m.x, A_ineq, m.b_ineq, "<=")
        add_matrix_constraints(m, "eq", m.x, A_eq, m.b_eq, "==")

        self.model = m
        self.options = options
        self.solver_name = solver_name
        self.opt = make_solver(solver_name, gurobi_env, auto_updates=PARAMS_ONLY_UPDATES)
        self._rows = constraint_list(m, "ineq", "eq")
        self._xs = [m.x[j] for j in range(obj.size)]
        self._b_ineq = [m.b_ineq[r] for r in range(R)]
        self._b_eq = [m.b_eq[r] for r in range(Re)]
        self._n_ineq = R
        self._fast = None
        self._fast_tried = False

    def _solve_pyomo(self, b_ineq, b_eq):
        for p, v in zip(self._b_ineq, b_ineq.tolist()):
            p.set_value(v)
        for p, v in zip(self._b_eq, b_eq.tolist()):
            p.set_value(v)
        stats = solve(self.opt, self.model, self.options)
        if not stats.optimal:
            raise RuntimeError(f"Subproblem status={stats.status} ({self.solver_name}) "
                               "-- duals unavailable.")
        duals = stats.results.solution_loader.get_duals(self._rows)
        dual_vec = np.array([duals[c] for c in self._rows], dtype=float)
        x = np.array([v.value for v in self._xs], dtype=float)
        return stats.objective, x, dual_vec, stats.solve_time

    def solve(self, b_ineq, b_eq):
        """Returns (objective, x, duals, solve_time)."""
        b_ineq = np.asarray(b_ineq, dtype=float).reshape(-1)
        b_eq = np.asarray(b_eq, dtype=float).reshape(-1)
        if self._fast is not None:
            return self._fast.solve(np.concatenate([b_ineq, b_eq]))

        out = self._solve_pyomo(b_ineq, b_eq)
        if not self._fast_tried:
            self._fast_tried = True
            try:
                self._fast = _RHS_PATHS[self.solver_name](self.opt, self._rows, self._xs, self._n_ineq)
            except (AttributeError, KeyError, TypeError, ImportError) as exc:
                print(f"[solver_backend] RHS fast path unavailable ({exc!r}); "
                      f"solving every LP through Pyomo", flush=True)
        return out


def write_model(model, path):
    """Write `model` to an LP/MPS file (by extension) that gurobi_cl and the highs CLI both read.

    Labels are symbolic (u[3], alpha[0], cuts[12]) so the file can be read
    against the iteration log.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fmt = "mps" if path.endswith(".mps") else "lp"
    model.write(path, format=fmt, io_options={"symbolic_solver_labels": True})
    return path
