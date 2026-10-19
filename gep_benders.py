import copy
import numpy as np
import torch
import json
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from gep_problem import GEPProblemSet, ramp_block_rows
from gep_problem_operational import GEPOperationalProblemSet
from create_gep_dataset import create_gep_ed_dataset
from data_wrangling import ramping_rates
from gep_config_parser import *
from networks import DualClassificationNetEndToEnd, DualNet, DualNetEndToEnd, PrimalNetEndToEnd
from devices import KNOWN_DEVICES, resolve_device_name
from paths import add_path_args, ensure_dir, resolve_roots, under_repo, under_root
import pyomo.environ as pyo
import solver_backend as sb

import os

#! Gurobi is optional: a HiGHS run never touches it. Only the licence helpers
#! below (and Cut_selection_experiment.py, which builds gurobipy models itself)
#! need it.
try:
    import gurobipy as gp
except ImportError:
    gp = None

CONFIG_FILE_NAME        = "configs/config.toml"

torch.set_num_threads(1)   # match the solvers' Threads=1 and the paper's single-thread protocol
#! Licence limits of the size-limited environment bundled with the pip
#! gurobipy wheel. Anything past these needs a real licence.
#! Restarts for every KMeans grouping. This was 300, which at k=100 over a
#! 2190-hour sample cost 36-56 s in the first Benders iteration -- more than that
#! iteration's master solve and LP solves combined -- for a grouping the run then
#! keeps for good. 10 is sklearn's classic default and still well above the
#! 'auto' behaviour (a single init for k-means++).
KMEANS_N_INIT = 10

RESTRICTED_LICENCE_VARS = 2000
RESTRICTED_LICENCE_CONSTRS = 2000


def make_env(quiet=True):
    if gp is None:
        raise RuntimeError("gurobipy is not installed; use --solver highs or pip install gurobipy.")
    env = gp.Env(empty=True)
    if quiet:
        env.setParam("OutputFlag", 0)

    # lic = os.environ.get("GRB_LICENSE_FILE")
    lic = os.environ.get("GRB_LICENSE_FILE") or os.path.expanduser("~/gurobi.lic")
    print(f"[make_env] GRB_LICENSE_FILE={lic}, exists={os.path.exists(lic) if lic else False}", flush=True)
    if lic and os.path.exists(lic):
        with open(lic) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip().upper(), val.strip()
                if key == "WLSACCESSID":
                    env.setParam("WLSACCESSID", val); print("[make_env] set WLSACCESSID", flush=True)
                elif key == "WLSSECRET":
                    env.setParam("WLSSECRET", val); print("[make_env] set WLSSECRET", flush=True)
                elif key == "LICENSEID":
                    env.setParam("LICENSEID", int(val)); print(f"[make_env] set LICENSEID={val}", flush=True)
    env.start()
    return env

_shared_env = None


def get_shared_env():
    """One Env reused across solvers on the main thread.

    Callers that build many BendersSolvers in a loop -- gen_GEP/solve_for_train.py
    makes one per GEP instance -- otherwise start one Env per instance and never
    dispose of it. That is merely wasteful with a node-locked licence, but a WLS
    licence authenticates over the network at start() and each live Env holds a
    session, so 80 instances can exhaust the concurrent-session limit partway
    through a run.
    """
    global _shared_env
    if _shared_env is None:
        _shared_env = make_env()
    return _shared_env


def build_capacity_demand_features(s):
    """
    Timestep features z_t = [D_{1,t}..D_{N,t}, A_{1,t}Pmax_1..A_{G,t}Pmax_G],
    read from the cached SampleStructure (no matrix rebuild).
    """
    demand_features = s.b_eq_t                 # (T, N)
    capacity_potential_features = s.apmax      # (T, G)
    X = np.concatenate([demand_features, capacity_potential_features], axis=1)
    return X, demand_features, capacity_potential_features


def make_single_group(T):
    return [list(range(T))]


def make_full_multicut_groups(T):
    return [[t] for t in range(T)]


def make_kmeans_capacity_demand_groups(s, K, random_state=0):
    """
    KMeans grouping based on [demand, A*Pmax].
    No ED solve and no reference investment required.
    """
    X, _, _ = build_capacity_demand_features(s)
    X = s.per_unit(X)

    if K <= 1:
        labels = np.zeros(X.shape[0], dtype=int)
        return make_single_group(X.shape[0]), labels, X

    K_eff = min(K, X.shape[0])

    X_scaled = StandardScaler().fit_transform(X)

    labels = KMeans(
        n_clusters=K_eff,
        random_state=random_state,
        n_init=KMEANS_N_INIT
    ).fit_predict(X_scaled)

    groups = [np.where(labels == k)[0].tolist() for k in range(K_eff)]

    return groups, labels, X



def compute_investment_duals(data, dual_vals, s):
    """Cut slope per timestep: lambda_{g,t} = mu^ub_{g,t} * A_{g,t} Pmax_g. Shape (T, G)."""
    G = data.num_g
    return np.asarray(dual_vals)[:, G:2 * G] * s.apmax


def compute_shadow_prices(data, dual_vals):
    """
    Per-timestep nodal shadow prices: the duals of the node-balance equalities.
    Returns array of shape (T, N).
    """
    duals = np.asarray(dual_vals)
    eq_dual_start = 2 * (data.num_g + data.num_l + data.num_n)
    return duals[:, eq_dual_start:eq_dual_start + data.num_n]


def compute_per_timestep_cuts(data, dual_vals, b_ineqs, b_eqs, s):
    """
    Disaggregated Benders cut per timestep t, in the form
        slope_t @ u - theta_t <= rhs_t
    Summing (slope_t, rhs_t) over a group gives exactly find_benders_cut_batch_for_group.
    Returns slopes (T, G) and rhs (T,).
    """
    G, L, N = data.num_g, data.num_l, data.num_n
    duals = np.asarray(dual_vals)

    slopes = compute_investment_duals(data, duals, s)

    constraint_nrs = np.concatenate([
        2*G + np.arange(L),
        2*G + L + np.arange(L),
        2*G + 2*L + N + np.arange(N),
    ])
    rhs = -np.sum(duals[:, constraint_nrs] * b_ineqs[:, constraint_nrs], axis=1)

    eq_dual_start = 2 * (G + L + N)
    rhs -= np.sum(duals[:, eq_dual_start:eq_dual_start + N] * b_eqs, axis=1)

    return slopes, rhs


def compute_per_unit_cuts(data, dual_vals, ramp_duals, b_ineqs, b_eqs, s):
    """
    Disaggregated Benders cut per subproblem unit k, slope_k @ u - theta_k <= rhs_k.
    Without ramping a unit is one hour and this is compute_per_timestep_cuts. With
    ramping a unit is a block of s.B hours: its hours' terms are summed, and the ramp
    rows add sum_rows nu * R_g to the slope -- their rhs R_g u_g depends on u, so they
    belong in the slope and never in the constant. Returns slopes (K, G) and rhs (K,).
    """
    slopes, rhs = compute_per_timestep_cuts(data, dual_vals, b_ineqs, b_eqs, s)
    if s.B == 1:
        return slopes, rhs
    slopes = slopes.reshape(s.K, s.B, -1).sum(axis=1) + np.asarray(ramp_duals) @ s.ramp_u_coef
    return slopes, rhs.reshape(s.K, s.B).sum(axis=1)


def compute_installed_capacity_features(demand_features, capacity_potential_features, investments, scale=None):
    """
    Primal, investment-dependent features: [D_{n,t}, A_{g,t} Pmax_g u_g].
    Same space as the static features, with each generator column scaled by how much of it is built.
    Generators that are not built drop out; heavily built ones dominate, which is what decides
    whether hour t is capacity constrained at the current investment.

    Columns are in MW on both sides, so no scaling is applied by default. Any `scale` passed here
    must be investment-independent: standardising the columns would divide generator g by u_g
    and cancel the weighting.
    """
    X = np.concatenate([demand_features,
                        capacity_potential_features * np.asarray(investments, dtype=float)[None, :]], axis=1)
    if scale is not None:
        X = X / scale[None, :]
    return X


def make_kmeans_dual_groups(features, K, random_state=0):
    """
    Adaptive grouping: cluster timesteps on dual-based features, shape (T, d)
    (nodal shadow prices or cut slopes). No per-column standardisation: columns share
    units, so raw Euclidean distance is meaningful.
    Returns a list of timestep-index groups.
    """
    X = np.asarray(features)
    T = X.shape[0]
    K_eff = min(K, T)
    if K_eff <= 1:
        return [list(range(T))]
    labels = KMeans(n_clusters=K_eff, n_init=KMEANS_N_INIT, random_state=random_state).fit_predict(X)
    return [np.where(labels == k)[0].tolist() for k in range(K_eff) if (labels == k).any()]


def make_cut_groups(s, cut_selection="single", cut_selection_k=1, random_state=0):
    """Groups of subproblem units (hours, or ramping blocks of s.B hours)."""
    T = s.K
    cut_selection = cut_selection.lower()

    if cut_selection == "single":
        groups = make_single_group(T)
        info = {
            "labels": np.zeros(T, dtype=int),
            "group_sizes": [len(g) for g in groups],
        }

    elif cut_selection == "full":
        groups = make_full_multicut_groups(T)
        info = {
            "labels": np.arange(T),
            "group_sizes": [len(g) for g in groups],
        }

    elif cut_selection in ("kmeans", "kmeans_dynamic"):
        groups, labels, X = make_kmeans_capacity_demand_groups(
            s, K=cut_selection_k, random_state=random_state,
        )
        info = {
            "labels": labels,
            "features": X,
            "group_sizes": [len(g) for g in groups],
        }

    else:
        raise ValueError(
            f"Unknown cut_selection={cut_selection}. "
            "Choose from 'single', 'kmeans', 'full'."
        )

    groups = [g for g in groups if len(g) > 0]
    info["group_sizes"] = [len(g) for g in groups]

    return groups, info

import scipy.sparse as sp

#! Horizons up to this length keep the dense path, so results recorded before the
#! sparse builders landed stay byte-reproducible; longer ones must go sparse.
DENSE_HORIZON_LIMIT = 256

#! One numerical profile for every solve of a run -- direct, master, hourly LPs and
#! gap-gate re-solves: primal feasibility (Gurobi FeasibilityTol, HiGHS
#! primal_feasibility_tolerance) and dual feasibility (OptimalityTol /
#! dual_feasibility_tolerance), both at Gurobi's minimum. The hourly duals become the
#! cuts, and a grouped cut sums them over its whole cluster, so noise at the 1e-6
#! default adds up into near-parallel cuts the master cannot tell apart.
NUMERIC_TOL = 1e-9

#! Benders stops at this relative UB-LB gap; the direct solve gets it as its MIP gap,
#! so both certify the same optimality.
BENDERS_REL_TOL = 1e-4

#! Master MIP gap, the same in the inexact and exact phases. The lower bound is the
#! master's bound, so it is valid at any gap; the gap only has to sit below the
#! Benders certificate for the incumbent to let UB-LB close. A tighter gap only makes
#! every master slower.
MASTER_MIP_GAP = BENDERS_REL_TOL / 10

#! Default LP phase of a gep_benders.py run: the master's u stays continuous until the relaxed
#! problem is solved to this gap, then turns integer with every cut kept. Cheap LP masters build
#! the first cuts, so the integer masters start from a tight relaxation. Off with --no-lp-phase
#! (or "lp_phase_tol": null in Benders_args). BendersSolver itself defaults to off, so library
#! callers such as gen_GEP/solve_for_train.py keep integer trajectories.
LP_PHASE_TOL = 1e-3

#! The lower-bound stall trigger for exact refinement only counts once the inexact UB-LB gap
#! is below this: early iterations can stall the bound for many iterations while the master is
#! still far from converged (one generator at a time, before the cuts say anything).
REFINE_STALL_MAX_GAP = 0.01

#! Default exact-refinement switch of a gep_benders.py run: the inexact phase ends once its UB-LB
#! gap is below this. Surrogate cuts stall at the surrogate's own error level (0.4-0.8 % at 12
#! nodes), far above LP_PHASE_TOL, so without it the switch waits for a repeated investment and
#! spends integer masters that cannot move the lower bound. Off with --refine-gap-tol 0 (or
#! "refine_gap_tol": null in Benders_args); BendersSolver itself defaults to off.
REFINE_GAP_TOL = 1e-2

#! Cut groupings that are built once and never revisited. Because the groups are
#! fixed, these carry one recourse variable per group rather than per hour.
STATIC_GROUPED_CUTS = ("full", "kmeans")


def _to_numpy(a):
    """Dense numpy view of a torch tensor, scipy sparse matrix or array (small slices only)."""
    if sp.issparse(a):
        return a.toarray()
    if torch.is_tensor(a):
        return a.detach().cpu().numpy()
    return np.asarray(a)


class SampleStructure:
    """
    Everything Benders reads from one sample's constraint matrices.

    Built straight from the problem primitives -- the full-horizon matrices are
    never materialised. get_sample_matrices() would kron the per-hour block into
    a dense (G + T*R) x (G + T*C) array, which is 1.6 TB at T=8760 for the 6-node
    case; peak memory here is O(T*G) instead of O(T^2).

    That substitution is exact because the per-hour operational block is the same
    for every hour: the only time-varying entries are pGenAva * pUnitCap, and they
    sit in the *investment* columns, which are kept separately as `apmax`.

    With ramping (data.set_ramping(B), B > 1) the subproblem is a block of B hours,
    since ramp rows couple the hours inside a block. Everything per hour above stays
    per hour; the subproblem *unit* becomes the block: K = T/B units, each with the
    B hourly blocks on the diagonal plus 2G(B-1) ramp rows (A_ineq_unit, A_eq_unit).
    B = 1 makes the unit the hour and every unit array the hourly one.
    """
    def __init__(self, data, sample):
        G, L, N = data.num_g, data.num_l, data.num_n
        time_range = data.time_ranges[sample]
        T  = len(time_range)
        R  = 2 * (G + L + N)      # inequality rows per hour
        Re = N                    # equality rows per hour
        C  = data.n_var_per_t     # variables per hour
        self.G, self.T, self.R, self.Re, self.C = G, T, R, Re, C

        # (1) investment-only rows for the master: the -I_G block that
        #     build_ineq_cm_rhs_sample prepends via block_diag(ui_g, ...), with
        #     the leading G entries of the inequality RHS (3.1k, all zero).
        self.A_base = -np.eye(G)

        # (2) A_{g,t} * Pmax_g -- the only time-varying coefficients, and they
        #     live in the investment columns rather than the operational block.
        unit_cap = np.array([float(data.pUnitCap[g]) for g in data.G])
        self.apmax = np.empty((T, G))
        for i, t in enumerate(time_range):
            ava = np.array([float(data.pGenAva.get((*g, t), 1.0)) for g in data.G])
            self.apmax[i] = ava * unit_cap

        # (3) per-hour operational blocks, identical for every hour. Stored once
        #     and broadcast, so s.A_ineq_t[t] keeps working at zero memory cost.
        A_ineq_block = np.zeros((R, C))
        A_ineq_block[0:G,                 0:G]           = -np.eye(G)   # 3.1h production lb
        A_ineq_block[G:2*G,               0:G]           =  np.eye(G)   # 3.1b production ub
        A_ineq_block[2*G:2*G+L,           G:G+L]         = -np.eye(L)   # 3.1d lineflow lb
        A_ineq_block[2*G+L:2*G+2*L,       G:G+L]         =  np.eye(L)   # 3.1e lineflow ub
        A_ineq_block[2*G+2*L:2*G+2*L+N,   G+L:G+L+N]     = -np.eye(N)   # 3.1i missed demand lb
        A_ineq_block[2*G+2*L+N:R,         G+L:G+L+N]     =  np.eye(N)   # 3.1j missed demand ub

        A_eq_block = np.concatenate([
            _to_numpy(data.node_to_gen_mask),
            _to_numpy(data.lineflow_mask),
            np.eye(N),
        ], axis=1)

        self.A_ineq_block = A_ineq_block
        self.A_eq_block   = A_eq_block
        self.A_ineq_t = np.broadcast_to(A_ineq_block, (T, R, C))
        self.A_eq_t   = np.broadcast_to(A_eq_block,   (T, Re, C))

        # (4) right-hand sides, one row per hour -- these builders are already O(T)
        ineq_rhs = _to_numpy(data.build_ineq_rhs_sample(time_range))
        self.b_base   = ineq_rhs[:G].copy()
        self.b_ineq_t = ineq_rhs[G:G + T * R].reshape(T, R).copy()
        self.b_eq_t   = _to_numpy(data.build_eq_rhs_sample(time_range)).reshape(T, Re).copy()

        # (5) subproblem units: hours, or ramping blocks of B hours
        B = getattr(data, "ramp_block_hours_", 1)
        if T % B:
            raise ValueError(f"Sample of {T} hours does not split into ramping blocks of {B}.")
        self.B, self.K = B, T // B
        self.n_ramp = 2 * G * (B - 1)
        if B == 1:
            self.A_ineq_unit, self.A_eq_unit = A_ineq_block, A_eq_block
        else:
            r, c, v, gen = ramp_block_rows(B, G, C)
            A_ramp = np.zeros((self.n_ramp, B * C))
            A_ramp[r, c] = v
            self.A_ineq_unit = np.vstack([np.kron(np.eye(B), A_ineq_block), A_ramp])
            self.A_eq_unit = np.kron(np.eye(B), A_eq_block)
            #! Ramp row i bounds generator ramp_gen[i] by R_g u_g: ramp_u_coef maps the
            #! ramp duals of a block onto its cut slope, and the ramp rhs at u.
            R_g = data.ramp_limits()
            self.ramp_gen = gen
            self.ramp_u_coef = np.zeros((self.n_ramp, G))
            self.ramp_u_coef[np.arange(self.n_ramp), gen] = R_g[gen]

    def per_unit(self, X):
        """Per-hour rows (T, d) -> per-unit rows (K, B*d), the hours of a block side by side."""
        X = np.asarray(X)
        return X if self.B == 1 else X.reshape(self.K, -1)

    def ramp_rhs_at(self, investments):
        """Ramp-row rhs R_g u_g of one block at a given investment, shape (n_ramp,)."""
        return self.ramp_u_coef @ np.asarray(investments, dtype=float)

    def b_ineq_at(self, investments):
        """Inequality RHS for all hours at a given investment, shape (T, R)."""
        b = self.b_ineq_t.copy()
        b[:, self.G:2 * self.G] = self.apmax * np.asarray(investments, dtype=float)[None, :]
        return b

class BendersSolver():
    def __init__(self, gep_data, operational_data, sample, primal_net=None, dual_net=None, exact=True, 
                 exact_refinement=True, max_investment=100000, init_investment = "Zero", 
                 cut_selection="single",cut_selection_k=1,parallel_subproblems = False, n_workers = None,
                 dynamic_cluster_features="price", env=None,
                 gap_gate_threshold=None, gap_gate_action="resolve",
                 solver="gurobi", solver_log_dir=None, solver_tee=False, dump_master=False,
                 master_mip_gap=MASTER_MIP_GAP, lp_phase_tol=None, lp_phase_max_iters=50,
                 refine_change_tol=None, refine_stall_iters=None, refine_stall_tol=1e-4,
                 refine_gap_tol=None):

        self.gep_data = gep_data
        self.operational_data = operational_data
        self.primal_net = primal_net
        self.dual_net = dual_net
        self.exact = exact
        self.exact_refinement = exact_refinement
        self.sample = sample
        self.total_time_subproblem_exact = 0
        self.total_time_subproblem_pdl = 0
        self.total_time_master = 0
        self.pWeight = self.gep_data.pWeight 
        self.max_investment = max_investment
        self.best_upper_bound = np.inf
        self.best_lower_bound = -np.inf

        self.cut_selection = cut_selection
        self.cut_selection_k = cut_selection_k
        self.cut_groups = None
        self.cut_group_info = None
        # Adaptive (dynamic) grouping, Law & Mallapragada (2026):
        #   "kmeans_dynamic"        -> adapt-G-I: one theta per timestep, old grouped cuts kept as-is
        #   "kmeans_dynamic_shared" -> adapt-G-S: one theta per group, all historical cuts
        #                              re-aggregated under the new grouping every iteration
        # dynamic_cluster_features:
        #   "price"    -> nodal shadow prices, (T, N)
        #   "slope"    -> cut slopes / investment duals lambda_s, (T, G)
        #   "capacity" -> primal [D, A*Pmax*u] at the current investment, (T, N+G)
        if dynamic_cluster_features not in ("price", "slope", "capacity"):
            raise ValueError(f"Unknown dynamic_cluster_features={dynamic_cluster_features}. "
                             "Choose 'price', 'slope' or 'capacity'.")
        self.dynamic_cluster_features = dynamic_cluster_features

        #! Gap gate: on an inexact iteration, a subproblem whose relative duality gap exceeds
        #! the threshold is not trusted to produce a cut. "resolve" solves it exactly and cuts
        #! from those duals; "drop" leaves it out of the cut. None disables the gate entirely,
        #! which is the behaviour of every run made before this existed.
        if gap_gate_action not in ("resolve", "drop"):
            raise ValueError(f"Unknown gap_gate_action={gap_gate_action}. Choose 'resolve' or 'drop'.")
        self.gap_gate_threshold = None if gap_gate_threshold is None else float(gap_gate_threshold)
        self.gap_gate_action = gap_gate_action
        self._last_gate_stats = None
        self.gate_flagged_hist = []
        self.gate_resolved_hist = []
        self.gate_dropped_hist = []
        self.gate_time_hist = []
        self.gap_rel_max_hist = []
        # exact iterations: hourly cuts at u_k vs the LP optima (NaN on PDL iterations)
        self._last_cut_tight = None
        self.cut_tight_rel_hist = []
        self.cut_tight_rel_max_hour_hist = []
        self._cd_features = None      # cached (demand, capacity potential) blocks for "capacity" features
        self._cut_hist_slopes = []    # adapt-G-S: per-iteration (T, G) disaggregated cut slopes
        self._cut_hist_rhs = []       # adapt-G-S: per-iteration (T,) disaggregated cut rhs
        #! Pyomo's solver interfaces swap sys.stdout (HiGHS even the file descriptor)
        #! around every solve, so they cannot run from several threads at once. The
        #! hourly LPs are therefore always solved in sequence -- which is also the
        #! single-thread protocol the comparison with the direct solve assumes.
        if parallel_subproblems:
            print("[solver] parallel_subproblems is ignored: Pyomo solves are not thread-safe, "
                  "subproblems run sequentially", flush=True)
        self.parallel_subproblems = False
        self.n_workers = n_workers

        #! One solver for every model of this run -- direct, master, hourly LPs and
        #! gap-gate re-solves -- so a comparison never mixes two solvers.
        self.solver = sb.check_solver_name(solver)
        #! Debugging aids for a master that stalls (see _master_log_path): native solver
        #! log per solve, optional live echo, and optional LP dump of each master.
        self.solver_log_dir = solver_log_dir
        self.solver_tee = bool(solver_tee)
        self.dump_master = bool(dump_master)
        if self.dump_master and not self.solver_log_dir:
            raise ValueError("dump_master needs solver_log_dir: that is where the model files go.")
        if self.solver_log_dir:
            os.makedirs(self.solver_log_dir, exist_ok=True)

        self.master_model = None
        self.master_u = None          # Pyomo Var of investments
        self.master_alpha = None      # Pyomo Var of recourse (alpha/theta)
        self.master_num_alpha = None
        self._master_opt = None
        self._master_time_limit = 3600.0  # safety net; solve_with_benders tightens it to the remaining wall-clock
        self._master_mip_gap = float(master_mip_gap)
        if not 0.0 <= self._master_mip_gap < BENDERS_REL_TOL:
            raise ValueError(f"master_mip_gap={master_mip_gap} must be below the Benders stopping "
                             f"tolerance {BENDERS_REL_TOL:g}, or UB-LB cannot close.")
        #! LP phase (two-phase Benders): the master's u is continuous until the relaxed problem
        #! is solved to lp_phase_tol, so the first cuts come from cheap LP masters; then u turns
        #! integer and every cut is kept. A cut is valid for any u and the LP master's bound is a
        #! valid lower bound, so only the stopping tests (and the incumbent upper bound, whose u
        #! is fractional) wait for the integer phase. None: integer from the start, as before.
        self.lp_phase_tol = None if lp_phase_tol is None else float(lp_phase_tol)
        self.lp_phase_max_iters = int(lp_phase_max_iters)
        self._lp_phase = self.lp_phase_tol is not None
        self._lp_iters = 0
        self.lp_phase_hist = []
        #! Practical convergence of the inexact phase. With surrogate cuts the master keeps moving a
        #! few units back and forth instead of repeating an investment exactly, so the exact-repeat
        #! test alone may never end the phase. Either trigger below acts like a repeat (exact
        #! refinement, or a stop without it); both are off when None.
        #!   refine_change_tol: the investment moved by less than this share of built capacity (MW);
        #!   refine_stall_iters / refine_stall_tol: the lower bound rose by less than the tol over that
        #!   many iterations, counted only once the inexact gap is below REFINE_STALL_MAX_GAP.
        self.refine_change_tol = None if refine_change_tol is None else float(refine_change_tol)
        self.refine_stall_iters = None if not refine_stall_iters else int(refine_stall_iters)
        self.refine_stall_tol = float(refine_stall_tol)
        #! refine_gap_tol: the inexact UB-LB gap is below this, checked in the LP phase as well.
        #! When set, the LP phase also never hands over to the integer master on surrogate cuts:
        #! if it would end while still inexact, the run switches to exact cuts first.
        self.refine_gap_tol = float(refine_gap_tol) if refine_gap_tol else None
        self.refine_reason = None
        self._num_cuts_in_master = 0  # how many cuts already added
        self._cut_stats = []          # per cut in the master: (min |slope|, max |slope|, |rhs|)
        self._sub_lp = None           # persistent hourly ED LP (solver_backend.MatrixLP)
        self._sub_lp_src = None

        # --- master diagnostics, one entry per iteration (iteration 0 has no master) ---
        self.master_status_hist = []
        self.master_obj_hist = []
        self.master_bound_hist = []
        self.master_gap_hist = []
        self.master_nodes_hist = []
        self.master_iters_hist = []
        self.master_sync_hist = []    # Pyomo bookkeeping around the master solve
        self.master_ncuts_hist = []
        self.cut_coef_min_hist = []
        self.cut_coef_max_hist = []
        self.cut_rhs_max_hist = []

        # --- surrogate duality-gap tracking (inexact iters only) ---
        self.gap_abs_mean_hist   = []
        self.gap_abs_median_hist = []
        self.gap_rel_mean_hist   = []
        self.gap_rel_median_hist = []
        self.gap_abs_total_hist  = []   # == UB-LB bracket at this investment
        self.gap_neg_count_hist  = []   # feasibility audit: should be 0
        self.gap_t_hist          = []   # per-timestep arrays, one per iter (for distributions)
        self._last_gap_stats     = None

        self.wall_iter_hist = []
        self.wall_master_hist = []
        self.wall_sub_hist = []

        self.X_all = []
        self.objs_all = []
        self.dual_solutions_all = []
        self.primal_solutions_all = []

        self.primal_opt_gap_all = []
        self.dual_opt_gap_all = []

        self.exact_iterations = 0
        self.inexact_iterations = 0

        self.ub_hist = []
        self.lb_hist = []
        self.master_time_hist = []
        self.sub_time_hist = []
        self.exact_flag_hist = []
        self.iter_hist = []

        self.inv_hist = []  # list[list[float]] length = #iters
        self.investment_init_method = init_investment # Zero by Default, also option: "HalfMax"
        #! Shared by default, so building many solvers in a loop does not open
        #! one Gurobi environment (and, under WLS, one licence session) each.
        #! Opened lazily: a HiGHS run never needs one.
        self._env = env
        self._struct = None
        self._struct_src = None

    @property
    def env(self):
        if self._env is None:
            self._env = get_shared_env()
        return self._env

    def _gurobi_env(self):
        """The Env Pyomo's Gurobi interface should use; None under HiGHS."""
        return self.env if self.solver == "gurobi" else None

    def _log_file(self, name):
        return os.path.join(self.solver_log_dir, name) if self.solver_log_dir else None

    def _options(self, **kw):
        """This run's parameter profile for one solve (see solver_backend.solver_options)."""
        return sb.solver_options(self.solver, threads=1, seed=0, lp_optimality_tol=NUMERIC_TOL,
                                 feasibility_tol=NUMERIC_TOL, **kw)

    def _structure(self, data, sample):
        src = self._struct_src
        if src is None or src[0] is not data or src[1] != sample:
            self._struct = SampleStructure(data, sample)
            self._struct_src = (data, sample)
        return self._struct

    def check_licence(self, data, label=""):
        """Fail now if the licence cannot take a model this size.

        Without this the first refusal arrives from m.optimize() after the
        instance has been unpickled and the model built -- minutes into a
        cluster job at 20 nodes, with an error naming neither the model size
        nor the licence limit.
        """
        if self.solver != "gurobi":
            return   # HiGHS has no licence to run out of
        n_vars = int(data.ydim)
        n_constrs = int(data.nineq + data.neq)

        probe = gp.Model("licence_probe", env=self.env)
        probe.setParam("OutputFlag", 0)
        probe.addMVar(shape=n_vars, lb=-gp.GRB.INFINITY)
        try:
            probe.update()
        except gp.GurobiError as exc:
            lic = os.environ.get("GRB_LICENSE_FILE") or "<unset>"
            raise SystemExit(
                f"\nGurobi licence cannot handle this model.\n"
                f"  model:   {n_vars:,} variables, {n_constrs:,} constraints"
                f"{f'   ({label})' if label else ''}\n"
                f"  licence: refused at this size -- almost certainly the "
                f"size-limited licence bundled with the pip gurobipy wheel "
                f"({RESTRICTED_LICENCE_VARS:,} vars / {RESTRICTED_LICENCE_CONSTRS:,} constraints)\n"
                f"  GRB_LICENSE_FILE: {lic}\n"
                f"  gurobi error: {exc}\n\n"
                f"A named-user academic licence is node-locked and will not work on a\n"
                f"compute node. Load a site licence module, or point GRB_LICENSE_FILE at\n"
                f"a floating (TOKENSERVER) or WLS licence.\n"
            ) from exc
        finally:
            probe.dispose()

    @property
    def X(self):
        return torch.cat(self.X_all, dim=0)
    @property
    def objs(self):
        return np.concatenate(self.objs_all, axis=0)
    
    @property
    def dual_solutions(self):
        return np.concatenate(self.dual_solutions_all, axis=0)
    
    @property
    def primal_solutions(self):
        return np.concatenate(self.primal_solutions_all, axis=0)
    
    def save_data(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pickle.dump(self.X, open(os.path.join(folder_path, "X.pkl"), "wb"))
        pickle.dump(self.objs, open(os.path.join(folder_path, "objs.pkl"), "wb"))
        pickle.dump(self.dual_solutions, open(os.path.join(folder_path, "dual_solutions.pkl"), "wb"))
        pickle.dump(self.primal_solutions, open(os.path.join(folder_path, "primal_solutions.pkl"), "wb"))

    def _hourly_lp(self, struct):
        """The persistent hourly ED LP for this sample, built once.

        Every hour shares the same matrices (see SampleStructure), so one model
        serves them all: each hour only pushes its right-hand side, and the solver
        warm-starts from the previous hour's basis.
        """
        if self._sub_lp is None or self._sub_lp_src is not struct:
            obj = self.operational_data.obj_coeff.detach().cpu().numpy() * self.pWeight
            #! With ramping the LP is one block: B hours of the same objective.
            obj = np.tile(obj, struct.B) if struct.B > 1 else obj
            self._sub_lp = sb.MatrixLP(
                self.solver, obj, struct.A_ineq_unit, struct.A_eq_unit,
                self._options(),
                gurobi_env=self._gurobi_env(),
            )
            self._sub_lp_src = struct
        return self._sub_lp

    def solve_matrix_problem(self, data, i, inv_decision=None):
        """Direct (monolithic) solve of sample i: the reference the Benders runs are compared to.

        Same solver and parameter profile as Benders: Threads=1, Seed=0, NUMERIC_TOL,
        MIP gap BENDERS_REL_TOL (= Benders' stopping tolerance), one hour of solver time. Timing is
        split so the comparison can use like for like (self.direct_info):
          solve_time -- inside the solver's optimize()/run(), the quantity the
                        Benders total_time sums;
          build_time -- building the model in Pyomo and translating it to the solver;
          wall_time  -- the whole call.
        """
        t_wall = time.perf_counter()

        #! Sparse beyond a few hundred hours: the dense matrices are quadratic in
        #! the horizon (1.6 TB at T=8760, 6 nodes) while the CSR is ~10 MB.
        T = len(data.time_ranges[i])
        use_sparse = T > DENSE_HORIZON_LIMIT
        ineq_cm, ineq_rhs, eq_cm, eq_rhs = data.get_sample_matrices(i, sparse=use_sparse)

        #! Investments integer, operation continuous; every variable free (lb=-inf),
        #! as in the gurobipy version -- the sign constraints are rows of ineq_cm.
        integer = np.zeros(data.ydim, dtype=bool)
        integer[:data.num_g] = True
        lb = np.full(data.ydim, -np.inf)
        ub = np.full(data.ydim, np.inf)
        if inv_decision is not None:
            # For plotting: the value function at a fixed investment.
            lb[:data.num_g] = ub[:data.num_g] = np.asarray(inv_decision, dtype=float).reshape(-1)

        m = sb.build_matrix_model(
            _to_numpy(data.obj_coeff), ineq_cm if use_sparse else np.array(ineq_cm), np.array(ineq_rhs),
            eq_cm if use_sparse else np.array(eq_cm), np.array(eq_rhs),
            integer=integer, lb=lb, ub=ub,
        )
        t_pyomo = time.perf_counter() - t_wall

        log_file = self._log_file(f"direct_sample{i}.log")
        opt = sb.make_solver(self.solver, self._gurobi_env())
        st = sb.solve(opt, m, self._options(time_limit=3600, mip_gap=BENDERS_REL_TOL, log_file=log_file),
                      tee=self.solver_tee)

        if not st.has_solution:
            raise RuntimeError(f"Direct solve found no feasible solution (status {st.status}, {self.solver}).")
        self.direct_info = {
            "solver": self.solver,
            "status": st.status,
            "mip_gap": st.gap,
            "obj_bound": st.bound,
            "hit_time_limit": st.hit_time_limit,
            "solve_time": st.solve_time,
            "build_time": t_pyomo + st.sync_time,
            "wall_time": time.perf_counter() - t_wall,
            "nodes": st.nodes,
            "target_mip_gap": BENDERS_REL_TOL,
            "numeric_tol": NUMERIC_TOL,
        }
        print(f"Obj: {st.objective:g}  gap: {st.gap:.2e}  status: {st.status}  "
              f"[{self.solver}] solve {st.solve_time:.2f}s, build {self.direct_info['build_time']:.2f}s"
              + (f"  log: {log_file}" if log_file else ""), flush=True)

        return sb.model_values(m), st.objective

    def solve_matrix_problem_simple(
        self,
        obj,
        A_ineq,
        b_ineq,
        A_eq,
        b_eq,
        master,
        investment=None,
        num_alpha=1,
    ):
        """
        Solve either the master problem or an ED subproblem, as a one-off model.

        If master=True:
            variables are [u_1, ..., u_G, alpha_1, ..., alpha_K]
            where K = num_alpha.

        If master=False:
            variables are the ED subproblem variables.

        Benders itself uses the persistent master and _hourly_lp; this stays for
        scripts that solve single problems from matrices.
        """
        obj = np.asarray(obj, dtype=float).reshape(-1)
        ydim = obj.size

        if not master:
            lp = sb.MatrixLP(self.solver, obj, np.asarray(A_ineq, dtype=float), np.asarray(A_eq, dtype=float),
                             self._options(), gurobi_env=self._gurobi_env())
            return lp.solve(b_ineq, b_eq)

        num_u = ydim - num_alpha
        integer = np.arange(ydim) < num_u
        lb = np.zeros(ydim)
        ub = np.full(ydim, np.inf)
        if investment is not None:
            lb[:num_u] = ub[:num_u] = np.asarray(investment, dtype=float)
        m = sb.build_matrix_model(obj, np.asarray(A_ineq, dtype=float), np.asarray(b_ineq, dtype=float),
                                  integer=integer, lb=lb, ub=ub)
        st = sb.solve(sb.make_solver(self.solver, self._gurobi_env()), m,
                      self._options(mip_gap=self._master_mip_gap))
        if not st.has_solution:
            raise RuntimeError(f"Master found no feasible solution (status {st.status}).")
        return st.objective, sb.model_values(m), [], st.solve_time

    def solve_matrix_problem_PDL(self, X):
        '''
        Solver Matrix problem (ED) using Primal and Dual Learning
        Returns:
        - Primal Obj& Dual Obj,
        - Primal var (production, flow, unmet demand)
        - Dual var (mu,lambda)
        '''
        start_time = time.time()
        primal_sol = self.primal_net(X)
        mu, lamb = self.dual_net(X)
        inference_time = time.time() - start_time

        mu *= self.pWeight
        lamb *= self.pWeight
        
        #! Total_obj_val is the primal objective value, since it is used as the upper bound in Benders decomposition.
        
        primal_obj_val = np.sum(self.operational_data.obj_fn(X, primal_sol).detach().numpy())

        #! Economic dispatch objective does not include the pWeight, so we need to multiply by it.
        primal_obj_val *= self.pWeight

        dual_obj_val = np.sum(self.operational_data.dual_obj_fn(X, mu, lamb).detach().numpy())
        #! Negate duals, for some reason these are flipped in Gurobi.
        dual_sol = torch.concat([-mu, -lamb], dim=1).squeeze()

        # per-timestep gap; note mu,lamb already scaled by pWeight above
        primal_t = self.operational_data.obj_fn(X, primal_sol).detach().cpu().numpy().reshape(-1) * self.pWeight
        dual_t   = self.operational_data.dual_obj_fn(X, mu, lamb).detach().cpu().numpy().reshape(-1)
        self._last_gap_stats = self._compute_gap_stats(primal_t, dual_t)

        return primal_obj_val, dual_obj_val, primal_sol.detach().numpy(), dual_sol.detach().numpy(), inference_time

    @staticmethod
    def _compute_gap_stats(primal_t, dual_t, eps=1e-9):
        primal_t = np.asarray(primal_t, float).reshape(-1)
        dual_t   = np.asarray(dual_t,   float).reshape(-1)
        gap_t = primal_t - dual_t
        rel_t = gap_t / np.clip(np.abs(primal_t), eps, None)
        return {
            "gap_abs_mean":   float(np.mean(gap_t)),
            "gap_abs_median": float(np.median(gap_t)),
            "gap_rel_mean":   float(np.mean(rel_t)),
            "gap_rel_median": float(np.median(rel_t)),
            "gap_abs_total":  float(np.sum(gap_t)),
            "n_negative_gap": int((gap_t < -1e-6).sum()),
            "gap_t": gap_t,
            #! Per-subproblem series the gap gate needs: the relative gap it thresholds on,
            #! and the two objectives it has to correct when a subproblem is re-solved.
            "rel_t": rel_t,
            "primal_t": primal_t,
            "dual_t": dual_t,
        }
    
    def _ensure_cut_groups(self, struct):
        """Build the static cut grouping once. kmeans_dynamic(_shared) reclusters
        every iteration and therefore does not come through here."""
        if self.cut_groups is None:
            self.cut_groups, self.cut_group_info = make_cut_groups(
                struct, cut_selection=self.cut_selection,
                cut_selection_k=self.cut_selection_k,
            )
            print(f"Cut selection: {self.cut_selection}, groups={len(self.cut_groups)}", flush=True)
            print("Group sizes:", self.cut_group_info["group_sizes"], flush=True)
        return self.cut_groups

    def _ensure_master_model(self, data, sample):
        """Build the master model once. Cheap to call repeatedly."""
        if self.master_model is not None:
            return

        s = self._structure(data, sample)

        # Determine number of recourse variables.
        # "single" uses one aggregate recourse var. The static grouped modes (full,
        # kmeans) cluster once and never regroup, so one theta per *group* is
        # exactly equivalent to one per timestep: the groups partition the hours, no
        # cut mixes two groups, and only the per-group sum is ever constrained. Same
        # bound, same trajectory, far fewer columns and no degenerate plateau.
        # kmeans_dynamic reclusters every iteration, so its cuts must keep keying off
        # per-timestep thetas to stay valid across regroupings.
        if self.cut_selection == "single":
            num_alpha = 1
        elif self.cut_selection == "kmeans_dynamic_shared":
            # adapt-G-S: one theta per group (unused thetas stay at their lb of 0)
            num_alpha = max(1, min(self.cut_selection_k, s.K))
        elif self.cut_selection in STATIC_GROUPED_CUTS:
            num_alpha = len(self._ensure_cut_groups(s))
        else:
            # one theta per subproblem unit: an hour, or a ramping block
            num_alpha = s.K

        self.master_num_alpha = num_alpha
        m = pyo.ConcreteModel("Benders Master Persistent")

        # Variables
        m.u = pyo.Var(range(data.num_g), domain=pyo.NonNegativeReals if self._lp_phase else pyo.NonNegativeIntegers,
                      bounds=(0.0, 100000.0))
        #! If we are solving the master problem, we know the investments are positive, so the lowerbound of each recourse value can be set to 0.
        m.alpha = pyo.Var(range(num_alpha), domain=pyo.NonNegativeReals)

        # Objective: c_u^T u + sum_k alpha_k
        obj_u = data.obj_coeff[:data.num_g].detach().cpu().numpy()
        self._u_list = [m.u[g] for g in range(data.num_g)]
        self._alpha_list = [m.alpha[k] for k in range(num_alpha)]
        m.obj = pyo.Objective(
            expr=sb.linear_expr(list(obj_u) + [1.0] * num_alpha, self._u_list + self._alpha_list),
            sense=pyo.minimize,
        )

        sb.add_matrix_constraints(m, "inv_base", m.u, s.A_base, s.b_base, "<=")
        m.cuts = pyo.ConstraintList()

        self.master_model = m
        self.master_u = m.u
        self.master_alpha = m.alpha
        self._master_opt = sb.make_solver(self.solver, self._gurobi_env())
        self._num_cuts_in_master = 0
        self._cut_constrs = []
        self._cut_stats = []


    def _add_new_cuts_to_master(self, all_cuts):
        """Only push cuts that aren't already in the model.
        adapt-G-S rebuilds every cut under the new grouping, so its cuts are replaced wholesale."""
        m = self.master_model
        if self.cut_selection == "kmeans_dynamic_shared":
            if self._cut_constrs:
                #! A fresh ConstraintList; the persistent solver sees the old rows go.
                m.del_component(m.cuts)
                m.cuts = pyo.ConstraintList()
            self._cut_constrs = []
            self._cut_stats = []
            new_cuts = all_cuts
        else:
            new_cuts = all_cuts[self._num_cuts_in_master:]
        if not new_cuts:
            return

        num_g = len(self._u_list)
        xs = self._u_list + self._alpha_list

        for cut_lhs, cut_rhs in new_cuts:
            row = np.asarray(cut_lhs, dtype=float).reshape(-1)
            nz = np.flatnonzero(row)
            self._cut_constrs.append(m.cuts.add(
                (None, sb.linear_expr(row[nz], [xs[j] for j in nz]), float(cut_rhs))
            ))
            #! Coefficient range of each cut, for the master diagnostics: a slope
            #! range of 1e9+ (or rhs that large) is a classic reason for a MIP to stall.
            slope = np.abs(row[:num_g])
            slope = slope[slope > 0]
            self._cut_stats.append((slope.min() if slope.size else np.nan,
                                    slope.max() if slope.size else np.nan,
                                    abs(float(cut_rhs))))

        self._num_cuts_in_master = len(all_cuts)

    def _end_lp_phase(self, reason):
        """Turn the master's u integer for the rest of the run; the persistent solver picks
        the domain change up on the next solve. All cuts stay."""
        for u in self._u_list:
            u.domain = pyo.NonNegativeIntegers
        self._lp_phase = False
        print(f"[benders] LP phase done after {self._lp_iters} LP masters ({reason}); "
              f"master is integer from now on, with all {self._num_cuts_in_master} cuts", flush=True)

    def _switch_to_exact(self, reason):
        """End the inexact phase: from the next iteration on, subproblems are solved exactly."""
        self.refine_reason = reason
        self.exact = True
        print(f"[benders] inexact phase converged in practice: {reason}; switching to exact "
              f"refinement" + (" (LP phase continues on exact cuts)" if self._lp_phase else ""),
              flush=True)

    def _master_log_path(self, sample, it, ext):
        """<solver_log_dir>/master_sample<s>_iter<i>.<ext>, or None when logging is off.

        One file per master solve, so a stalled solve is simply the last file:
        the solver writes it live, so it shows the search state while stuck.
        """
        return self._log_file(f"master_sample{sample}_iter{it:04d}.{ext}")

    @staticmethod
    def get_crossover_metrics(iter_df):
        cross_rows = iter_df[
            (iter_df["exact_mode"] == True) &
            (iter_df["exact_mode"].shift(1) == False)
        ]

        if len(cross_rows) == 0:
            return {
                "has_crossover": False,
                "cross_iter": None,
                "ub_cross": None,
                "lb_cross": None,
                "gap_cross_pct": None,
                "lb_cross_ratio_pct": None,
            }

        cross_idx = cross_rows.index[0]

        ub_cross = float(iter_df.loc[cross_idx, "UB"])
        lb_cross = float(iter_df.loc[cross_idx, "LB"])
        lb_final = float(iter_df["LB"].iloc[-1])
        gap_cross_pct = float(iter_df.loc[cross_idx, "gap_rel"]) * 100.0

        lb_cross_ratio_pct = (
            100.0 * lb_cross / lb_final
            if abs(lb_final) > 1e-12 else None
        )

        return {
            "has_crossover": True,
            "cross_iter": int(iter_df.loc[cross_idx, "iter"]),
            "ub_cross": ub_cross,
            "lb_cross": lb_cross,
            "gap_cross_pct": gap_cross_pct,
            "lb_cross_ratio_pct": lb_cross_ratio_pct,
        }

    def solve_master_problem(self, data, compact, sample, investments,
                         obj_val, benders_cuts, investment=None):
        self._ensure_master_model(data, sample)
        self._add_new_cuts_to_master(benders_cuts)

        m = self.master_model
        u = self.master_u
        it = len(self.iter_hist)   # the Benders iteration this master solve belongs to

        # Final-evaluation call: fix u = investment via temporary bounds.
        if investment is not None:
            inv_arr = np.asarray(investment, dtype=float)
            old_bounds = [(u[g].lb, u[g].ub) for g in range(len(inv_arr))]
            for g, v in enumerate(inv_arr):
                u[g].setlb(v)
                u[g].setub(v)

        stats = np.array(self._cut_stats, dtype=float).reshape(-1, 3)
        coef_min = float(np.nanmin(stats[:, 0])) if len(stats) and np.isfinite(stats[:, 0]).any() else np.nan
        coef_max = float(np.nanmax(stats[:, 1])) if len(stats) and np.isfinite(stats[:, 1]).any() else np.nan
        rhs_max = float(stats[:, 2].max()) if len(stats) else np.nan

        log_file = self._master_log_path(sample, it, "log")
        if self.dump_master:
            sb.write_model(m, self._master_log_path(sample, it, "lp"))
        #! Printed before the solve, so a master that never returns still says what it was given.
        print(f"[master] iter {it}: {len(self._cut_constrs)} cuts, |slope| in [{coef_min:.3g}, {coef_max:.3g}], "
              f"|rhs| <= {rhs_max:.3g}, gap {self._master_mip_gap:g}, limit {self._master_time_limit:.0f}s"
              + (f", log: {log_file}" if log_file else ""), flush=True)

        st = sb.solve(self._master_opt, m,
                      self._options(time_limit=self._master_time_limit, mip_gap=self._master_mip_gap,
                                    log_file=log_file),
                      tee=self.solver_tee)
        inference_time = st.solve_time

        if not st.has_solution:
            raise RuntimeError(f"Master found no feasible solution (status {st.status}, {self.solver}).")
        if st.hit_time_limit:
            print(f"[master] time limit hit, MIP gap {st.gap:.2e}", flush=True)
        print(f"[master] iter {it}: {st.status} in {st.solve_time:.2f}s (+{st.sync_time:.2f}s pyomo), "
              f"obj {st.objective:.6g}, bound {st.bound:.6g}, gap {st.gap:.2e}, nodes {st.nodes}", flush=True)
        #! Without a finite bound (should not happen for a MIP) fall back to the objective, as before.
        self._last_master_bound = st.bound if np.isfinite(st.bound) else None

        self._last_master_stats = {
            "master_status": st.status,
            "master_obj": st.objective,
            "master_bound": st.bound,
            "master_gap": st.gap,
            "master_nodes": st.nodes,
            "master_iters": st.iterations,
            "master_sync": st.sync_time,
            "master_ncuts": len(self._cut_constrs),
            "cut_coef_min": coef_min,
            "cut_coef_max": coef_max,
            "cut_rhs_max": rhs_max,
        }

        new_investments = np.array([u[g].value for g in range(len(self._u_list))], dtype=float)
        alpha_vals = np.array([a.value for a in self._alpha_list], dtype=float)

        obj_u = data.obj_coeff[:data.num_g].detach().cpu().numpy()
        investment_cost = float(obj_u @ new_investments)
        alpha_total = float(alpha_vals.sum())

        if investment is not None:
            for g, (lo, hi) in enumerate(old_bounds):
                u[g].setlb(lo)
                u[g].setub(hi)

        return [investment_cost, alpha_total], new_investments, inference_time

    def _apply_gap_gate(self, s, b_ineqs_np, b_eqs_np, dual_vals, primal_total, dual_total):
        """Act on subproblems whose predicted duality gap is too large to trust.

        Returns (dual_vals, keep_mask, primal_total, dual_total). The gate reads the
        per-hour relative gap computed alongside the prediction, so it costs nothing
        when no hour is flagged.

        Dropping an hour from the cut is legitimate because the recourse variable bounds
        a SUM of subproblem values and every one of them is non-negative (generation cost
        plus VOLL times unmet demand), so a cut over a subset still lower-bounds it -- it
        is merely weaker. Re-solving instead replaces that hour's duals with exact ones,
        which both tightens the cut and corrects the bounds this hour contributes.
        """
        keep_mask = np.ones(s.T, dtype=bool)
        if self.gap_gate_threshold is None or self._last_gap_stats is None:
            self._last_gate_stats = None
            return dual_vals, keep_mask, primal_total, dual_total

        t0 = time.time()
        rel_t = np.asarray(self._last_gap_stats["rel_t"], dtype=float)
        flagged = np.flatnonzero(rel_t > self.gap_gate_threshold)
        action = self.gap_gate_action

        #! An empty cut would leave the master unchanged and the loop would stall without
        #! erroring, so the fallback is to pay for exact solves rather than stop improving.
        if action == "drop" and len(flagged) == s.T and s.T > 0:
            print(f"[gap gate] all {s.T} hours exceed {self.gap_gate_threshold:.3%}; "
                  f"re-solving them instead of dropping every cut", flush=True)
            action = "resolve"

        n_resolved = 0
        if len(flagged) and action == "resolve":
            lp = self._hourly_lp(s)
            primal_t = np.asarray(self._last_gap_stats["primal_t"], dtype=float)
            dual_t = np.asarray(self._last_gap_stats["dual_t"], dtype=float)
            dual_vals = np.array(dual_vals, dtype=float, copy=True)
            for t in flagged:
                ov, _, dv, _ = lp.solve(b_ineqs_np[t], b_eqs_np[t])
                dual_vals[t] = dv
                #! Exact hour: its primal and dual contributions both become the LP optimum.
                primal_total += ov - primal_t[t]
                dual_total += ov - dual_t[t]
            n_resolved = len(flagged)
        elif len(flagged):
            keep_mask[flagged] = False

        elapsed = time.time() - t0
        #! Charged to the exact-solver budget, not to the surrogate's, so the timing split stays honest.
        self.total_time_subproblem_exact += elapsed if n_resolved else 0.0
        self._last_gate_stats = {
            "n_flagged": int(len(flagged)),
            "n_resolved": int(n_resolved),
            "n_dropped": int(len(flagged) - n_resolved),
            "gate_time": float(elapsed),
            "gap_rel_max": float(rel_t.max()) if rel_t.size else 0.0,
        }
        if len(flagged):
            print(f"[gap gate] {len(flagged)}/{s.T} hours above {self.gap_gate_threshold:.3%} "
                  f"(max {rel_t.max():.3%}) -> {'re-solved' if n_resolved else 'dropped from the cut'} "
                  f"in {elapsed:.2f}s", flush=True)
        return dual_vals, keep_mask, primal_total, dual_total

    def find_benders_cut_batch_for_group(
        self, data, compact, sample,
        dual_vals, b_ineqs, b_eqs,
        timestep_indices, alpha_index, num_alpha,
        struct, shared_alpha=False,
    ):
        if compact:
            raise NotImplementedError("Grouped cuts currently support compact=False only.")
        s = struct
        G = data.num_g
        num_rows_per_t_ineq = 2 * (G + data.num_l + data.num_n)
        timestep_indices = np.asarray(timestep_indices, dtype=int)

        benders_cut_lhs = np.zeros((1, G + num_alpha))
        benders_cut_rhs = 0.0
        if shared_alpha:
            # one theta per group: the cut bounds that group's recourse directly
            benders_cut_lhs[0, G + alpha_index] = -1.0
        else:
            # -1 on theta_t for every hour in this group
            benders_cut_lhs[0, G + timestep_indices] = -1.0

        # LHS: sum over the group of mu^ub_{g,t} * A_{g,t} Pmax_g
        dual_slice = dual_vals[timestep_indices][:, G:2 * G]            # (Tg, G)
        benders_cut_lhs[0, :G] = (dual_slice * s.apmax[timestep_indices]).sum(axis=0)

        # RHS: inequality rows with a nonzero constant rhs
        constraint_nrs = np.concatenate([
            2 * G + np.arange(data.num_l),                                # flow lower bounds
            2 * G + data.num_l + np.arange(data.num_l),                   # flow upper bounds
            2 * G + 2 * data.num_l + data.num_n + np.arange(data.num_n),  # missed-demand upper bounds
        ])
        ineq_duals = dual_vals[np.ix_(timestep_indices, constraint_nrs)]
        ineq_rhs = b_ineqs[np.ix_(timestep_indices, constraint_nrs)]
        benders_cut_rhs += -np.sum(ineq_duals * ineq_rhs)

        # RHS: node-balance equalities
        eq_duals = dual_vals[timestep_indices, num_rows_per_t_ineq:num_rows_per_t_ineq + data.num_n]
        benders_cut_rhs += -np.sum(eq_duals * b_eqs[timestep_indices])

        return benders_cut_lhs, benders_cut_rhs
    
    def _capacity_features(self, data, sample, investments):
        """[D, A*Pmax*u] in MW at the current investment."""
        s = self._structure(data, sample)
        return compute_installed_capacity_features(s.b_eq_t, s.apmax, investments)

    def find_benders_cuts_grouped_batch(
        self, data, compact, sample,
        dual_vals, b_ineqs, b_eqs,
        struct, investments=None, keep_mask=None, ramp_duals=None,
    ):
        s = struct
        if s.B > 1:
            return self._block_cuts(data, sample, dual_vals, ramp_duals, b_ineqs, b_eqs, s, investments)

        #! Hours the gap gate dropped contribute nothing to a cut. Zeroing their duals removes
        #! them from every sum below (slope and rhs alike); the grouped modes additionally leave
        #! their theta out, which makes the cut tighter and is still valid because each dropped
        #! subproblem value is non-negative. Clustering still sees the unmasked duals, so the
        #! groups -- and therefore the theta bookkeeping -- do not shift under the gate.
        dual_cut = dual_vals
        if keep_mask is not None and not keep_mask.all():
            dual_cut = np.array(dual_vals, dtype=float, copy=True)
            dual_cut[~keep_mask] = 0.0

        if self.cut_selection == "single":
            return [self.find_benders_cut_batch(data, compact, sample,
                                                dual_cut, b_ineqs, b_eqs, s)]

        # kmeans_dynamic(_shared): recluster every iteration; others: cluster once
        if self.cut_selection in ("kmeans_dynamic", "kmeans_dynamic_shared"):
            if self.dynamic_cluster_features == "price":
                features = compute_shadow_prices(data, dual_vals)
            elif self.dynamic_cluster_features == "capacity":
                features = self._capacity_features(data, sample, investments)
            else:
                features = compute_investment_duals(data, dual_vals, s)
            self.cut_groups = make_kmeans_dual_groups(features, self.cut_selection_k)

        if self.cut_selection == "kmeans_dynamic_shared":
            slopes, rhs = compute_per_timestep_cuts(data, dual_cut, b_ineqs, b_eqs, s)
            self._cut_hist_slopes.append(slopes)
            self._cut_hist_rhs.append(rhs)
            hist_slopes = np.stack(self._cut_hist_slopes)    # (I, T, G)
            hist_rhs = np.stack(self._cut_hist_rhs)          # (I, T)

            G = data.num_g
            num_alpha = max(1, min(self.cut_selection_k, s.T))
            cuts = []
            for k, group in enumerate(self.cut_groups):
                group = np.asarray(group, dtype=int)
                group_slopes = hist_slopes[:, group, :].sum(axis=1)
                group_rhs = hist_rhs[:, group].sum(axis=1)
                for it in range(hist_slopes.shape[0]):
                    cut_lhs = np.zeros((1, G + num_alpha))
                    cut_lhs[0, :G] = group_slopes[it]
                    cut_lhs[0, G + k] = -1.0
                    cuts.append((cut_lhs, float(group_rhs[it])))
            return cuts

        if self.cut_selection != "kmeans_dynamic":
            self._ensure_cut_groups(s)

        # Static groupings get one theta per group (see _ensure_master_model);
        # kmeans_dynamic regroups every iteration and stays on per-hour thetas.
        shared_alpha = self.cut_selection in STATIC_GROUPED_CUTS
        num_alpha = len(self.cut_groups) if shared_alpha else s.T

        #! Keep each group's original index: it is what the cut keys its theta off
        #! under shared recourse, so dropping an emptied group must not renumber the
        #! ones after it.
        groups = list(enumerate(self.cut_groups))
        if keep_mask is not None and not keep_mask.all():
            kept = set(np.flatnonzero(keep_mask).tolist())
            groups = [(k, [t for t in group if t in kept]) for k, group in groups]
            groups = [(k, group) for k, group in groups if group]
        return [
            self.find_benders_cut_batch_for_group(
                data=data, compact=compact, sample=sample,
                dual_vals=dual_cut, b_ineqs=b_ineqs, b_eqs=b_eqs,
                timestep_indices=group, alpha_index=k, num_alpha=num_alpha,
                struct=s, shared_alpha=shared_alpha,
            )
            for k, group in groups
        ]
    
    def _block_cuts(self, data, sample, dual_vals, ramp_duals, b_ineqs, b_eqs, s, investments):
        """Cuts when the subproblem unit is a ramping block (s.B > 1).

        Same modes and theta bookkeeping as the hourly path, over blocks instead of
        hours: groups are groups of blocks, per-unit thetas are per block, and a
        group's cut sums its blocks' disaggregated cuts (compute_per_unit_cuts), which
        is what the hourly path computes when every block is one hour.
        """
        G = data.num_g
        slopes_u, rhs_u = compute_per_unit_cuts(data, dual_vals, ramp_duals, b_ineqs, b_eqs, s)

        def cut(units, num_alpha, alpha_cols):
            lhs = np.zeros((1, G + num_alpha))
            lhs[0, :G] = slopes_u[units].sum(axis=0)
            lhs[0, G + np.asarray(alpha_cols, dtype=int)] = -1.0
            return lhs, float(rhs_u[units].sum())

        if self.cut_selection == "single":
            return [cut(np.arange(s.K), 1, [0])]

        if self.cut_selection in ("kmeans_dynamic", "kmeans_dynamic_shared"):
            if self.dynamic_cluster_features == "price":
                features = compute_shadow_prices(data, dual_vals)
            elif self.dynamic_cluster_features == "capacity":
                features = self._capacity_features(data, sample, investments)
            else:
                features = compute_investment_duals(data, dual_vals, s)
            self.cut_groups = make_kmeans_dual_groups(s.per_unit(features), self.cut_selection_k)

        if self.cut_selection == "kmeans_dynamic_shared":
            self._cut_hist_slopes.append(slopes_u)
            self._cut_hist_rhs.append(rhs_u)
            hist_slopes = np.stack(self._cut_hist_slopes)    # (I, K, G)
            hist_rhs = np.stack(self._cut_hist_rhs)          # (I, K)
            num_alpha = max(1, min(self.cut_selection_k, s.K))
            cuts = []
            for k, group in enumerate(self.cut_groups):
                group = np.asarray(group, dtype=int)
                for it in range(hist_slopes.shape[0]):
                    lhs = np.zeros((1, G + num_alpha))
                    lhs[0, :G] = hist_slopes[it, group].sum(axis=0)
                    lhs[0, G + k] = -1.0
                    cuts.append((lhs, float(hist_rhs[it, group].sum())))
            return cuts

        if self.cut_selection != "kmeans_dynamic":
            self._ensure_cut_groups(s)
        shared_alpha = self.cut_selection in STATIC_GROUPED_CUTS
        num_alpha = len(self.cut_groups) if shared_alpha else s.K
        return [cut(np.asarray(group, dtype=int), num_alpha, [k] if shared_alpha else group)
                for k, group in enumerate(self.cut_groups)]

    def solve_subproblems(self, data, compact, sample, investments, exact=True):
        '''
        Solve all hourly subproblems (exact LPs or PDL) at the given investment
        and return the Benders cuts. Uses the cached SampleStructure, so the
        full-horizon matrices are never held here.
        '''
        if compact:
            raise NotImplementedError("solve_subproblems supports compact=False only.")

        t_fn_start = time.time()

        # --- Phase 0: per-sample structure (built once, then cached) ---
        t0 = time.time()
        s = self._structure(data, sample)
        num_timesteps = s.T
        t_getmat = time.time() - t0          # large on the first call, ~0 afterwards

        # --- Phase 1: right-hand sides for every hour at this investment ---
        t0 = time.time()
        b_ineqs_np = s.b_ineq_at(investments)   # (T, R), capacity rows set to A*Pmax*u
        b_eqs_np   = s.b_eq_t                   # (T, N), demand
        t_build = time.time() - t0

        t_solve_wall = 0.0
        t_pdl = 0.0
        self._last_cut_tight = None
        ramp_duals = None

        if not exact and s.B > 1:
            raise NotImplementedError("Ramping blocks need exact subproblems: the PDL nets "
                                      "predict single hours without ramp rows.")

        if exact:
            lp = self._hourly_lp(s)

            obj_vals    = [None] * s.K
            primal_vals = [None] * num_timesteps
            dual_vals   = [None] * num_timesteps
            inf_times   = [None] * s.K

            # --- Phase 2: LP solves (sequential: see __init__ on parallel_subproblems) ---
            t0 = time.time()
            if s.B == 1:
                for t in range(num_timesteps):
                    obj_vals[t], primal_vals[t], dual_vals[t], inf_times[t] = lp.solve(b_ineqs_np[t], b_eqs_np[t])
            else:
                #! One LP per block. Its duals are split back into the per-hour layout
                #! [ineq R | eq N] the cut, clustering and diagnostics code reads, plus
                #! the block's ramp duals, which only the cut slope needs.
                B, R, N = s.B, s.R, s.Re
                ramp_rhs = s.ramp_rhs_at(investments)
                ramp_duals = np.empty((s.K, s.n_ramp))
                for k in range(s.K):
                    hrs = slice(k * B, (k + 1) * B)
                    obj_vals[k], x, duals, inf_times[k] = lp.solve(
                        np.concatenate([b_ineqs_np[hrs].ravel(), ramp_rhs]), b_eqs_np[hrs].ravel())
                    hour_ineq = duals[:B * R].reshape(B, R)
                    ramp_duals[k] = duals[B * R:B * R + s.n_ramp]
                    hour_eq = duals[B * R + s.n_ramp:].reshape(B, N)
                    dual_vals[hrs] = list(np.concatenate([hour_ineq, hour_eq], axis=1))
                    primal_vals[hrs] = list(np.asarray(x).reshape(B, -1))
            t_solve_wall = time.time() - t0

            primal_obj_val_total = float(np.sum(obj_vals))
            dual_obj_val_total   = primal_obj_val_total
            inference_time_total = float(np.sum(inf_times))

            #! Cut tightness: each unit's cut (hour, or ramping block) evaluated at this
            #! investment is that unit's dual objective, which strong duality makes equal to its LP optimum. Anything
            #! well above NUMERIC_TOL means the duals, not the master, carry the error.
            slopes_t, rhs_t = compute_per_unit_cuts(data, np.array(dual_vals), ramp_duals, b_ineqs_np, b_eqs_np, s)
            cut_at_u = slopes_t @ np.asarray(investments, dtype=float) - rhs_t
            err_t = cut_at_u - np.asarray(obj_vals, dtype=float)
            self._last_cut_tight = {
                "cut_tight_rel": float(err_t.sum()) / max(abs(primal_obj_val_total), 1e-10),
                "cut_tight_rel_max_hour": float(np.max(np.abs(err_t) / np.maximum(np.abs(obj_vals), 1.0))),
            }

        else:
            # --- PDL branch ---
            t0 = time.time()
            X = torch.tensor(np.concatenate(
                [b_eqs_np, b_ineqs_np[:, self.operational_data.capacity_ub_indices]],
                axis=1,
            ))
            primal_obj_val_total, dual_obj_val_total, primal_vals, dual_vals, inference_time_total = \
                self.solve_matrix_problem_PDL(X)
            t_pdl = time.time() - t0

        # --- Phase 2b: gap gate (inexact iterations only; a no-op unless a threshold is set) ---
        keep_mask = None
        if not exact:
            dual_vals, keep_mask, primal_obj_val_total, dual_obj_val_total = self._apply_gap_gate(
                s, b_ineqs_np, b_eqs_np, dual_vals, primal_obj_val_total, dual_obj_val_total
            )
        else:
            self._last_gate_stats = None

        # --- Phase 3: cut building ---
        t0 = time.time()
        benders_cuts = self.find_benders_cuts_grouped_batch(
            data=data, compact=compact, sample=sample,
            dual_vals=np.array(dual_vals), b_ineqs=b_ineqs_np, b_eqs=b_eqs_np,
            struct=s,
            investments=np.asarray(investments, dtype=float),
            keep_mask=keep_mask,
            ramp_duals=ramp_duals,
        )
        t_cuts = time.time() - t0

        t_fn_total = time.time() - t_fn_start

        if exact:
            print(f"[solve_subproblems EXACT T={num_timesteps} {self.solver}] total={t_fn_total:.2f}s | "
                  f"struct={t_getmat:.3f}s build={t_build:.3f}s "
                  f"solve_wall={t_solve_wall:.2f}s (sum_solve={inference_time_total:.2f}s, "
                  f"pyomo={t_solve_wall - inference_time_total:.2f}s) cuts={t_cuts:.2f}s", flush=True)
        else:
            print(f"[solve_subproblems PDL T={num_timesteps}] total={t_fn_total:.2f}s | "
                  f"struct={t_getmat:.3f}s build={t_build:.3f}s pdl={t_pdl:.2f}s "
                  f"cuts={t_cuts:.2f}s", flush=True)

        print(f"Inference time: {inference_time_total}", flush=True)
        return primal_obj_val_total, dual_obj_val_total, benders_cuts, inference_time_total

    def find_benders_cut_batch(self, data, compact, sample, dual_vals, b_ineqs, b_eqs, struct):
        """Single aggregated cut over all hours (non-compact)."""
        s = struct
        G = data.num_g
        num_rows_per_t_ineq = 2 * (G + data.num_l + data.num_n)

        benders_cut_lhs = np.zeros((1, G + 1))
        benders_cut_lhs[0, -1] = -1.0                                     # alpha
        benders_cut_lhs[0, :G] = (dual_vals[:, G:2 * G] * s.apmax).sum(axis=0)

        constraint_nrs = np.concatenate([
            2 * G + np.arange(data.num_l),
            2 * G + data.num_l + np.arange(data.num_l),
            2 * G + 2 * data.num_l + data.num_n + np.arange(data.num_n),
        ])
        benders_cut_rhs = -np.sum(dual_vals[:, constraint_nrs] * b_ineqs[:, constraint_nrs])
        benders_cut_rhs -= np.sum(dual_vals[:, num_rows_per_t_ineq:num_rows_per_t_ineq + data.num_n] * b_eqs)

        return benders_cut_lhs, benders_cut_rhs


    def _update_cut_list(self, benders_cut_all, benders_cuts):
        # adapt-G-S returns the full rebuilt cut set; every other mode returns only new cuts.
        if self.cut_selection == "kmeans_dynamic_shared":
            benders_cut_all[:] = benders_cuts
        else:
            benders_cut_all.extend(benders_cuts)

    def _refine_trigger(self, investments_all, lb_now, unit_cap):
        """Why the inexact phase counts as converged in practice, or None (see refine_change_tol)."""
        if self.refine_change_tol is not None and len(investments_all) >= 2:
            cur = investments_all[-1].to(torch.float64).numpy()
            prev = investments_all[-2].to(torch.float64).numpy()
            change = float(np.abs(cur - prev) @ unit_cap) / max(float(prev @ unit_cap), 1.0)
            if change < self.refine_change_tol:
                return f"investment moved {change:.2%} of built capacity < {self.refine_change_tol:.2%}"
        k = self.refine_stall_iters
        if k and len(self.lb_hist) >= k and self.ub_hist:
            gap = (min(self.ub_hist[-k:]) - lb_now) / max(1.0, abs(min(self.ub_hist[-k:])))
            rise = (lb_now - self.lb_hist[-k]) / max(1.0, abs(lb_now))
            if gap < REFINE_STALL_MAX_GAP and rise < self.refine_stall_tol:
                return f"lower bound rose {rise:.1e} < {self.refine_stall_tol:g} over {k} iterations at gap {gap:.2%}"
        return None

    def _investment_repeated(self, investments_iter_k, investments_all):
        # adapt-G-S rebuilds the master under a new grouping every iteration, so it can cycle
        # through several investments; treat a return to ANY earlier investment as stalled.
        # All other modes keep the original check against the previous iteration only.
        inv = investments_iter_k.to(torch.float64)
        previous = investments_all if self.cut_selection == "kmeans_dynamic_shared" else investments_all[-1:]
        return any(torch.allclose(inv, p.to(torch.float64), atol=1e-6) for p in previous)

    def _lower_bound(self, obj_val_master):
        lb = getattr(self, "_last_master_bound", None)
        lower_bound = lb if lb is not None else obj_val_master[0] + obj_val_master[1]
        self.best_lower_bound = max(self.best_lower_bound, lower_bound)
        return self.best_lower_bound

    MASTER_STAT_KEYS = ("master_status", "master_obj", "master_bound", "master_gap", "master_nodes",
                        "master_iters", "master_sync", "master_ncuts",
                        "cut_coef_min", "cut_coef_max", "cut_rhs_max")

    def _record_master_stats(self, stats):
        """One row of master diagnostics per iteration; iteration 0 has no master solve."""
        empty = {k: np.nan for k in self.MASTER_STAT_KEYS}
        empty.update(master_status="", master_nodes=-1, master_iters=-1, master_ncuts=0)
        stats = stats or empty
        self.master_status_hist.append(stats["master_status"])
        self.master_obj_hist.append(stats["master_obj"])
        self.master_bound_hist.append(stats["master_bound"])
        self.master_gap_hist.append(stats["master_gap"])
        self.master_nodes_hist.append(stats["master_nodes"])
        self.master_iters_hist.append(stats["master_iters"])
        self.master_sync_hist.append(stats["master_sync"])
        self.master_ncuts_hist.append(stats["master_ncuts"])
        self.cut_coef_min_hist.append(stats["cut_coef_min"])
        self.cut_coef_max_hist.append(stats["cut_coef_max"])
        self.cut_rhs_max_hist.append(stats["cut_rhs_max"])

    def master_diagnostics(self):
        """Per-iteration master diagnostics as columns for the iteration log."""
        return {
            "master_status": self.master_status_hist,
            "master_obj": self.master_obj_hist,
            "master_bound": self.master_bound_hist,
            "master_gap": self.master_gap_hist,
            "master_nodes": self.master_nodes_hist,
            "master_simplex_iters": self.master_iters_hist,
            "t_master_pyomo": self.master_sync_hist,
            "master_ncuts": self.master_ncuts_hist,
            "cut_coef_min": self.cut_coef_min_hist,
            "cut_coef_max": self.cut_coef_max_hist,
            "cut_rhs_max": self.cut_rhs_max_hist,
        }

    def solve_with_benders(self, data, compact, sample):

        # Create lists for algorithm
        investments_all = [] # list of tensors of size (num_g), one for every iteration
        obj_val_subproblems_all = [] # list of floats, one for every iteration
        benders_cut_all = [] # list of benders cuts ([lhs],rhs), one for every iteration

        # Parameters for Benders algorithm
        rel_tol = BENDERS_REL_TOL         # stopping gap, matches Proxy Benders paper
        wall_limit = 3600.0               # one hour per sample
        t_start = time.time()
        self.hit_time_limit = False
        unit_cap = np.array([float(data.pUnitCap[g]) for g in data.G])   # MW per unit, for refine_change_tol
        self._last_master_stats = None
        upper_bound, lower_bound = np.inf, -np.inf

        # Start Benders algorithm
        optimal = False
        i = 0
        while not optimal and i < 1000:
            elapsed = time.time() - t_start
            if elapsed > wall_limit:
                print(f"[benders] wall-clock limit reached after {elapsed:.0f}s", flush=True)
                self.hit_time_limit = True
                break
            self._master_time_limit = max(1.0, wall_limit - elapsed)

            self._last_gap_stats = None
            t_iter_start = time.time()
            in_lp = self._lp_phase         # this iteration's master is (or would be) the LP relaxation
            lp_repeated = False
            print("-"*50)
            print("Iteration", i, "Exact:", self.exact)

            # Find the investment decisions
            if i == 0:

                # Generate initial investment solution
                if self.investment_init_method == "Zero":
                    investments_iter_k = [0. for _ in range(data.num_g)] #TODO find better initial solution?
                elif self.investment_init_method == "HalfMax":
                    investments_iter_k = (
                        self.operational_data.pUnitInvestment.max(dim=0).values / 2
                    ).to(torch.float64)
                else:
                    raise ValueError(f"Invalid investment initialization method: {self.investment_init_method}")
                # investments_iter_k = self.operational_data.opt_targets['y_investment'][0] #! Test with optimal solution
                # Calculate objective of master problem of this solution
                obj_val_master = 0
                for g_idx, g in enumerate(data.G):
                    obj_val_master += data.pInvCost[g] * data.pUnitCap[g] * investments_iter_k[g_idx]
                obj_val_master = [obj_val_master, 0] # alpha is zero in the first iteration
            else:
                # Solve master problem to find investments
                print("Solving the master problem in iteration", i)
                obj_val_master, investments_iter_k, inference_time_master = self.solve_master_problem(data,compact,sample,torch.stack(investments_all),torch.tensor(obj_val_subproblems_all),benders_cut_all)
                self.total_time_master += inference_time_master
                self._lp_iters += int(in_lp)

            # Add investment values of current iteration to list
            print("The investment decisions are", investments_iter_k)
            investments_iter_k = torch.tensor(investments_iter_k)

            self.inv_hist.append(investments_iter_k.detach().cpu().numpy().tolist())

            #! In the LP phase a repeated u only means the relaxation has converged: end the phase
            #! (below) instead of triggering exact refinement or a stop.
            if in_lp and i > 0 and self._investment_repeated(investments_iter_k, investments_all):
                lp_repeated = True
                investments_all.append(investments_iter_k)
            elif self.exact == False and i > 0 and self._investment_repeated(investments_iter_k, investments_all):
                print("!! Investments are the same as last iteration")
                self.refine_reason = "investment repeated"
                if self.exact_refinement:
                    self.exact = True
                else:
                    print("Stopping Benders decomposition because exact refinement is not used.")
                    print("Upper bound:", self.best_upper_bound) #! Return the best upper bound found so far if exact refinement is not used
                    lower_bound = self._lower_bound(obj_val_master)
                    print("Lower bound:", lower_bound)
                    print(f"Duality gap: {(self.best_upper_bound - lower_bound)/np.abs(self.best_upper_bound)}")
                    break
            else:
                # if self.exact == False:
                #     # print("!! !! Different investments than last iteration")
                investments_all.append(investments_iter_k)
                if self.exact == False and i > 0 and not in_lp:
                    _lb = getattr(self, "_last_master_bound", None)
                    lb_now = max(self.best_lower_bound, _lb if _lb is not None else sum(obj_val_master))
                    reason = self._refine_trigger(investments_all, lb_now, unit_cap)
                    if reason:
                        self.refine_reason = reason
                        print(f"[benders] inexact phase converged in practice: {reason}; "
                              + ("switching to exact refinement" if self.exact_refinement
                                 else "stopping (exact refinement is off)"), flush=True)
                        if self.exact_refinement:
                            self.exact = True
                        else:
                            lower_bound = self._lower_bound(obj_val_master)
                            break
            
            if self.exact:
                self.exact_iterations += 1
            else:
                self.inexact_iterations += 1
            # Solve subproblems to find new cuts
            #primal_obj_val_total, dual_obj_val_total, benders_cut, inference_time_subproblems_total = self.solve_subproblems(data,compact,sample,investments_iter_k, exact=self.exact)
            primal_obj_val_total, dual_obj_val_total, benders_cuts, inference_time_subproblems_total = self.solve_subproblems(
                data,
                compact,
                sample,
                investments_iter_k,
                exact=self.exact,
            )
            print(f"SOlved Subproblem: inf time: {inference_time_subproblems_total}")

            # if not self.exact:
                # PDL solve
                # exact_primal_obj_val_total, exact_dual_obj_val_total, _, _ = self.solve_subproblems(data,compact,sample,investments_iter_k, exact=True)

                # primal_opt_gap = (primal_obj_val_total - exact_primal_obj_val_total) / exact_primal_obj_val_total
                # dual_opt_gap = (dual_obj_val_total - exact_dual_obj_val_total) / exact_dual_obj_val_total

                # print(f"Primal opt gap: {primal_opt_gap}, Dual opt gap: {dual_opt_gap}")
                # print(f"Primal obj val: {primal_obj_val_total}, Dual obj val: {dual_obj_val_total}")

                # self.primal_opt_gap_all.append(primal_opt_gap)
                # self.dual_opt_gap_all.append(dual_opt_gap)
            
            if self.exact:
                self.total_time_subproblem_exact += inference_time_subproblems_total
            else:
                self.total_time_subproblem_pdl += inference_time_subproblems_total

            # Add total objective value of all subproblems of current iteration together to list
            obj_val_subproblems_all.append(primal_obj_val_total)

            # Check for optimality
            lower_bound = self._lower_bound(obj_val_master)
            upper_bound = obj_val_master[0] + primal_obj_val_total
            print(f"UB={upper_bound:.4f}, LB={lower_bound:.4f}, ")
            # --- LOG UB/LB PER ITERATION ---
            self.ub_hist.append(float(upper_bound))
            self.lb_hist.append(float(lower_bound))
            self.iter_hist.append(int(i))
            self.exact_flag_hist.append(bool(self.exact))
            self.sub_time_hist.append(float(inference_time_subproblems_total))

            gs = self._last_gap_stats
            if gs is not None and not self.exact:
                self.gap_abs_mean_hist.append(gs["gap_abs_mean"])
                self.gap_abs_median_hist.append(gs["gap_abs_median"])
                self.gap_rel_mean_hist.append(gs["gap_rel_mean"])
                self.gap_rel_median_hist.append(gs["gap_rel_median"])
                self.gap_abs_total_hist.append(gs["gap_abs_total"])
                self.gap_neg_count_hist.append(gs["n_negative_gap"])
                self.gap_t_hist.append(gs["gap_t"])
            else:  # exact iter: gap between predictions is undefined
                for h in (self.gap_abs_mean_hist, self.gap_abs_median_hist,
                          self.gap_rel_mean_hist, self.gap_rel_median_hist,
                          self.gap_abs_total_hist):
                    h.append(np.nan)
                self.gap_neg_count_hist.append(0)
                self.gap_t_hist.append(None)

            #! One row per iteration, so the gate's effect can be attributed afterwards.
            gate = self._last_gate_stats
            self.gate_flagged_hist.append(gate["n_flagged"] if gate else 0)
            self.gate_resolved_hist.append(gate["n_resolved"] if gate else 0)
            self.gate_dropped_hist.append(gate["n_dropped"] if gate else 0)
            self.gate_time_hist.append(gate["gate_time"] if gate else 0.0)
            self.gap_rel_max_hist.append(gate["gap_rel_max"] if gate else np.nan)
            tight = self._last_cut_tight
            self.cut_tight_rel_hist.append(tight["cut_tight_rel"] if tight else np.nan)
            self.cut_tight_rel_max_hour_hist.append(tight["cut_tight_rel_max_hour"] if tight else np.nan)
            self.lp_phase_hist.append(bool(in_lp))

            # master time only exists when i>0
            if i == 0:
                self.master_time_hist.append(0.0)
            else:
                self.master_time_hist.append(float(inference_time_master))
            self._record_master_stats(self._last_master_stats if i > 0 else None)

            # Check for optimality -- not in the LP phase: its u is fractional, so neither its
            # upper bound nor a repeated u says anything about the integer problem yet.
            if self.exact and not in_lp:
                # print("Found upper bound:",upper_bound)
                if upper_bound < self.best_upper_bound:
                    self.best_upper_bound = upper_bound

                same_investment = (
                    len(investments_all) >= 2
                    and torch.allclose(
                        investments_all[-1].to(torch.float64),
                        investments_all[-2].to(torch.float64),
                        atol=1e-6,
                    )
                )


                if (upper_bound - lower_bound) / max(1.0, abs(upper_bound)) < rel_tol:
                    optimal = True
                    print('Done! Optimal solution found')
                    print('Total number of iterations needed:', i)
                    print('Optimal objective value:', upper_bound)
                elif same_investment:
                    optimal = True
                    print('Done! Investment repeated in exact mode.')
                    print('Stopping to avoid cycling.')
                    print(f'Upper bound: {upper_bound}')
                    print(f'Lower bound: {lower_bound}')
                    print(f'Gap: {upper_bound - lower_bound}')
                else:
                    # Add Benders cut of current iteration to list
                    # benders_cut_all.append(benders_cut)
                    self._update_cut_list(benders_cut_all, benders_cuts)
                    print(f"Subproblems solved. Added {len(benders_cuts)} Benders cuts.")
                    # print('Subproblems solved. Benders_cut:',benders_cut)
            else:
                    # Add Benders cut of current iteration to list
                    self._update_cut_list(benders_cut_all, benders_cuts)
                    print(f"Subproblems solved. Added {len(benders_cuts)} Benders cuts.")
            #! Gap switch to exact refinement, in the LP phase too. sub_exact: whether this
            #! iteration's cuts (and so its gap) came from exact subproblems.
            gap_now = (upper_bound - lower_bound) / max(1.0, abs(upper_bound))
            sub_exact = self.exact
            gap_switch = self.refine_gap_tol is not None and self.exact_refinement
            if gap_switch and not sub_exact and i > 0 and gap_now < self.refine_gap_tol:
                self._switch_to_exact(f"gap {gap_now:.1e} < {self.refine_gap_tol:g}")
            if in_lp and i > 0:
                lp_gap = gap_now
                lp_done = (lp_gap < self.lp_phase_tol or lp_repeated
                           or self._lp_iters >= self.lp_phase_max_iters)
                if gap_switch and not sub_exact and lp_done:
                    #! Never go integer on surrogate cuts: stay in the LP phase on exact cuts.
                    if not self.exact:
                        self._switch_to_exact("LP phase ended inexact")
                elif lp_gap < self.lp_phase_tol:
                    self._end_lp_phase(f"relaxation gap {lp_gap:.1e} < {self.lp_phase_tol:g}")
                elif lp_repeated:
                    self._end_lp_phase("investment repeated")
                elif self._lp_iters >= self.lp_phase_max_iters:
                    self._end_lp_phase(f"cap of {self.lp_phase_max_iters} reached, relaxation gap still {lp_gap:.1e}")
            self.wall_master_hist.append(
                            float(inference_time_master) if i > 0 else 0.0
                        )
            self.wall_sub_hist.append(float(inference_time_subproblems_total))
            self.wall_iter_hist.append(time.time() - t_iter_start)
            i += 1

        #! Everything solve_with_benders spent, Pyomo bookkeeping, clustering and cut
        #! building included -- next to total_time, which only sums solver calls.
        self.wall_time = time.time() - t_start
        return upper_bound, lower_bound, benders_cut_all, investments_all, obj_val_subproblems_all, i
    

    def plot_benders_cuts(self, min_investment, max_investment, steps, index, benders_cuts_all, investments_all, obj_val_subproblems_all, upper_bound):
        """
        Only works with a single investment variable!

        To visualize the value function, we solve the entire problem for a range of investment values specified by min_investment and max_investment, with number of steps steps.
        We then plot the value function against the investment values.

        Then, we plot the benders cuts. The cut itself corresponds to a subgradient of the subproblems (operational costs). For the cuts to intersect the value function,
        we need to account for the gradient investment costs. Hence, the formula of the cut is: (c + mu)x - b, where x is the investment variable, c is the investment costs,
        mu is the dual variable and b is the rhs of the cut.
        """

         # Create a grid of investment values
        investment_decisions = torch.linspace(min_investment, max_investment, steps)
        # Add the investments from the Benders iterations
        investments_tensor = torch.tensor([inv.item() if isinstance(inv, torch.Tensor) else float(inv) for inv in investments_all])
        # Combine and deduplicate
        all_investments = torch.cat([investment_decisions, investments_tensor])
        investment_decisions_unique, _ = torch.sort(torch.unique(all_investments))

        # Calculate the value function for each investment value
        value_function = []
        for investment in investment_decisions_unique:
            primal_val, obj_val = self.solve_matrix_problem(self.gep_data, index, investment)
            value_function.append(obj_val)

        value_function = torch.tensor(value_function)
        min_idx = torch.argmin(value_function)
        min_investment_val = investment_decisions_unique[min_idx]
        min_value = value_function[min_idx]

        primal_val, known_optimal_obj = self.solve_matrix_problem(self.gep_data, index, None)
        known_optimal_inv = primal_val[0]
        
        print(f"Known optimal investment: {known_optimal_inv}, Known optimal value: {known_optimal_obj}")

        if benders_cuts_all is not None and investments_all is not None:
            for i, (cut_lhs, cut_rhs) in enumerate(benders_cuts_all):
                # Ensure cut_lhs is a 1D array or tensor
                if isinstance(cut_lhs, torch.Tensor):
                    cut_lhs = cut_lhs.squeeze().numpy()
                if isinstance(cut_rhs, torch.Tensor):
                    cut_rhs = cut_rhs.item()

                # print(f"Benders cut {i}: {cut_lhs}, {cut_rhs}")

                a = cut_lhs[0]
                b = cut_lhs[1]
                # 
                theta_vals = [self.gep_data.obj_coeff[:self.gep_data.num_g].item()*x + a*x - cut_rhs for x in investment_decisions_unique]
                plt.plot(investment_decisions_unique.numpy(), theta_vals, label=f"Benders Cut {i}")
                print(f"Investments all: {investments_all[i]}")
                plt.scatter([investments_all[i]], [0], color='green')

        # Plot the value function
        plt.plot(investment_decisions_unique.numpy(), value_function.numpy(), label="Value Function")
        plt.scatter([known_optimal_inv], [known_optimal_obj], color='blue', label='Known Optimum')
        plt.xlabel('Investment')
        plt.ylabel('Value')
        plt.ylim(-1e6, 1e7)
        plt.title(f"Value Function. predicted inv.: {investments_all[-1].item():.2f} ({known_optimal_inv:.2f}), predicted obj.: {upper_bound:.2f} ({known_optimal_obj:.2f})")
        plt.legend()
        plt.show()
            

if __name__ == "__main__":
    import argparse
    '''
    ARGS_FILE_NAME option:
    - "config.json": Default config for experiments. (3 Node)
    - "config-4node.json": Config for 4-node experiments.
    - "config-5node.json": Config for 5-node experiments.
    - "config-6node.json": Config for 6-node experiments.

    TO use the solver directly, solve with python gep_benders.py --solve-direct
    by default solve direct is False
    '''
    ## Step 1: parse the input data
    #! No abbreviated options: argparse would otherwise read e.g. --sample as --sample-duration
    #! (or refuse it as ambiguous), and a typo could silently set a different option.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "-s", "--solve-direct",
        action="store_true",
        help="Solve the full GEP directly with Gurobi, without Benders decomposition.",
        default=False
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="configs/config.json",
        help="Path to run config JSON file, e.g. configs/config.json. "
             "Relative paths resolve against the repository, not the working directory."
    )

    #! --home-path / --data-root / --output-root
    add_path_args(parser)

    #! The trained nets are inputs here, produced by an earlier main.py run.
    #! They stay config-driven; these override them for cluster runs.
    parser.add_argument(
        "--primal-net-dir", "--primal_net_dir",
        dest="primal_net_dir",
        default=None,
        help="Override Benders_args.primal_net_directory.",
    )
    parser.add_argument(
        "--dual-net-dir", "--dual_net_dir",
        dest="dual_net_dir",
        default=None,
        help="Override Benders_args.dual_net_directory. Ignored for a flow-first primal "
             "directory, whose dual is constructed rather than loaded.",
    )

    #! Only read when --primal-net-dir points at a flow-first run (a directory with model.pt).
    parser.add_argument(
        "--flowfirst-weights", "--flowfirst_weights",
        dest="flowfirst_weights",
        default=None,
        help="Which checkpoint of a flow-first run to use: model.pt (default), "
             "model_best.pt, or model_ema.pt for a run trained with --ema.",
    )
    parser.add_argument(
        "--flowfirst-polish-sweeps", "--flowfirst_polish_sweeps",
        dest="flowfirst_polish_sweeps",
        type=int,
        default=None,
        help="Exact line-search sweeps over the predicted flows before each subproblem "
             "answer (default 1). 0 uses the network's raw flows.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        choices=list(KNOWN_DEVICES),
        default=None,
        help="Device for net inference. Defaults to 'auto' (CUDA when present, "
             "else CPU), independently of whatever device the nets were trained on.",
    )

    parser.add_argument(
        "--sample-duration", "--sample_duration",
        dest="sample_duration",
        type=int,
        default=None,
        help="Override Benders_args.sample_duration from the config. "
             "Left unset, the config value is used.",
    )

    parser.add_argument(
        "--cut-selection", "--cut_selection",
        dest="cut_selection",
        choices=["single", "full", "kmeans", "kmeans_dynamic", "kmeans_dynamic_shared"],
        default=None,
        help="Override Benders_args.cut_selection. Left unset, uses the config.",
    )
    parser.add_argument(
        "--cut-selection-k", "--cut_selection_k",
        dest="cut_selection_k",
        type=int,
        default=None,
        help="Number of clusters/groups for kmeans/kmeans_dynamic(_shared). "
             "Ignored for 'single' and 'full'.",
    )
    #! Per-subproblem certificate gate, off unless a threshold is given.
    parser.add_argument(
        "--gap-gate-threshold", "--gap_gate_threshold",
        dest="gap_gate_threshold",
        type=float,
        default=None,
        help="On inexact iterations, treat an hour whose relative duality gap exceeds this "
             "as untrusted, e.g. 0.01 for 1 percent. Unset disables the gate.",
    )
    parser.add_argument(
        "--gap-gate-action", "--gap_gate_action",
        dest="gap_gate_action",
        choices=["resolve", "drop"],
        default=None,
        help="What to do with a flagged hour: 'resolve' solves it exactly and cuts from those "
             "duals, 'drop' leaves it out of the cut (and out of its cluster). Default resolve.",
    )
    parser.add_argument(
        "--ramp-block-hours", "--ramp_block_hours",
        dest="ramp_block_hours",
        type=int,
        default=None,
        help="Override Benders_args.ramp_block_hours: ramp limits between consecutive hours "
             "inside blocks of this many hours (never across blocks). 1 or unset: no ramping. "
             "Must divide sample_duration; needs --benders-setup Exact.",
    )
    parser.add_argument(
        "--master-mip-gap", "--master_mip_gap",
        dest="master_mip_gap",
        type=float,
        default=None,
        help=f"Override Benders_args.master_mip_gap: the master's relative MIP gap in both "
             f"phases. Must be below the Benders stopping tolerance {BENDERS_REL_TOL:g}. "
             f"Default {MASTER_MIP_GAP:g}.",
    )
    parser.add_argument(
        "--lp-phase-tol", "--lp_phase_tol",
        dest="lp_phase_tol",
        type=float,
        default=None,
        help=f"LP phase: the master's investments stay continuous (LP master) until the relaxed "
             f"problem's UB-LB gap is below this, then turn integer with every cut kept. "
             f"Default {LP_PHASE_TOL:g}, or Benders_args.lp_phase_tol.",
    )
    parser.add_argument(
        "--no-lp-phase", "--no_lp_phase",
        dest="no_lp_phase",
        action="store_true",
        help="Integer master from the first iteration (no LP phase).",
    )
    parser.add_argument(
        "--lp-phase-max-iters", "--lp_phase_max_iters",
        dest="lp_phase_max_iters",
        type=int,
        default=None,
        help="Safety cap on LP-phase iterations (default 50); the phase also ends when the LP "
             "master repeats an investment.",
    )
    parser.add_argument(
        "--refine-gap-tol", "--refine_gap_tol",
        dest="refine_gap_tol",
        type=float,
        default=None,
        help=f"Inexact phase: switch to exact refinement once the UB-LB gap is below this, in the "
             f"LP phase too; the LP phase then also never ends on surrogate cuts. "
             f"Default {REFINE_GAP_TOL:g}, or Benders_args.refine_gap_tol; 0: off.",
    )
    parser.add_argument(
        "--refine-change-tol", "--refine_change_tol",
        dest="refine_change_tol",
        type=float,
        default=None,
        help="Inexact phase: switch to exact refinement once the investment moves by less than "
             "this share of built capacity (MW) between iterations, e.g. 0.01. Unset: only an "
             "exactly repeated investment ends the inexact phase.",
    )
    parser.add_argument(
        "--refine-stall-iters", "--refine_stall_iters",
        dest="refine_stall_iters",
        type=int,
        default=None,
        help=f"Inexact phase: switch to exact refinement once the lower bound has risen by less "
             f"than --refine-stall-tol over this many iterations, counted only once the inexact "
             f"gap is below {REFINE_STALL_MAX_GAP * 100:g}%%. Unset: off.",   # %% : argparse formats help
    )
    parser.add_argument(
        "--refine-stall-tol", "--refine_stall_tol",
        dest="refine_stall_tol",
        type=float,
        default=None,
        help="Relative lower-bound rise that counts as a stall (default 1e-4).",
    )
    parser.add_argument(
        "--benders-setup", "--benders_setup",
        dest="benders_setup",
        choices=["Exact", "Inexact", "Inexact_Refine", "All"],
        default=None,
        help="Override Benders_args.benders_setup. Left unset, uses the config.",
    )
    parser.add_argument(
        "--samples",
        dest="samples",
        type=int,
        nargs="+",
        default=None,
        help="Solve only these samples (0-based), e.g. --samples 2, to rerun or debug one "
             "instance. Default: every sample of the year.",
    )
    parser.add_argument(
        "--specific-name", "--specific_name",
        dest="specific_name",
        default=None,
        help="Override Benders_args.specific_name, the label in the output paths "
            "(iter_logs_<benders_setup>_<specific_name> and the summary CSV name).",
    )
    parser.add_argument(
        "--ground-truth", action="store_true", default=False,
        help="Also solve the full monolithic GEP per sample (dense matrices; small T only).",
    )
    #! Solver choice applies to every model of the run (direct, master, subproblems),
    #! so a direct-vs-Benders comparison always uses one solver.
    parser.add_argument(
        "--solver",
        choices=list(sb.SOLVERS),
        default=None,
        help="Optimisation solver, through Pyomo. Overrides Benders_args.solver; default gurobi.",
    )
    parser.add_argument(
        "--solver-log-dir", "--solver_log_dir",
        dest="solver_log_dir",
        default=None,
        help="Write the solver's own log for every master solve (and the direct solve) here, one "
             "file per solve, written live -- a stalled master is the last file. Relative paths "
             "resolve under the output root.",
    )
    parser.add_argument(
        "--solver-tee", "--solver_tee",
        dest="solver_tee",
        action="store_true",
        default=False,
        help="Also stream the master/direct solver log to the console.",
    )
    parser.add_argument(
        "--dump-master", "--dump_master",
        dest="dump_master",
        action="store_true",
        default=False,
        help="Write each master problem to an .lp file next to its log (needs --solver-log-dir), "
             "to replay it offline with gurobi_cl or highs.",
    )

    args_cli = parser.parse_args()



    print("Parsing the config file")
    #! Repo-anchored, so any spelling of the path works and the file is found
    #! whatever the working directory is.
    RUN_CONFIG_FILE = under_repo(args_cli.config)

    data = parse_config(under_repo(CONFIG_FILE_NAME))
    experiment = data["experiment"]
    outputs_config = data["outputs_config"]


    with open(RUN_CONFIG_FILE, "r") as file:
        args = json.load(file)

    if args_cli.sample_duration is not None:
        args["Benders_args"]["sample_duration"] = args_cli.sample_duration
        print(f"[override] sample_duration = {args_cli.sample_duration}")
    if args_cli.benders_setup is not None:
        args["Benders_args"]["benders_setup"] = args_cli.benders_setup
        print(f"[override] benders_setup = {args_cli.benders_setup}")
    if args_cli.specific_name is not None:
        args["Benders_args"]["specific_name"] = args_cli.specific_name
        print(f"[override] specific_name = {args_cli.specific_name}")

    if args_cli.gap_gate_threshold is not None:
        args["Benders_args"]["gap_gate_threshold"] = args_cli.gap_gate_threshold
        print(f"[override] gap_gate_threshold = {args_cli.gap_gate_threshold}")
    if args_cli.ramp_block_hours is not None:
        args["Benders_args"]["ramp_block_hours"] = args_cli.ramp_block_hours
        print(f"[override] ramp_block_hours = {args_cli.ramp_block_hours}")
    #! Ramping runs keep their own result files: the block length goes into the name.
    RAMP_B = int(args["Benders_args"].get("ramp_block_hours") or 1)
    if RAMP_B > 1 and not args_cli.solve_direct and args["Benders_args"]["benders_setup"] != "Exact":
        raise SystemExit("--ramp-block-hours needs --benders-setup Exact: the PDL nets predict "
                         "single hours and have no ramp rows.")
    for _key in ("refine_gap_tol", "refine_change_tol", "refine_stall_iters", "refine_stall_tol"):
        if getattr(args_cli, _key) is not None:
            args["Benders_args"][_key] = getattr(args_cli, _key)
            print(f"[override] {_key} = {getattr(args_cli, _key)}")
    if args_cli.no_lp_phase:
        args["Benders_args"]["lp_phase_tol"] = None
        print("[override] lp_phase_tol = None (no LP phase)")
    elif args_cli.lp_phase_tol is not None:
        args["Benders_args"]["lp_phase_tol"] = args_cli.lp_phase_tol
        print(f"[override] lp_phase_tol = {args_cli.lp_phase_tol}")
    #! On by default; a config can switch it off with "lp_phase_tol": null.
    args["Benders_args"].setdefault("lp_phase_tol", LP_PHASE_TOL)
    args["Benders_args"].setdefault("refine_gap_tol", REFINE_GAP_TOL)
    if args_cli.lp_phase_max_iters is not None:
        args["Benders_args"]["lp_phase_max_iters"] = args_cli.lp_phase_max_iters
        print(f"[override] lp_phase_max_iters = {args_cli.lp_phase_max_iters}")
    if args_cli.master_mip_gap is not None:
        args["Benders_args"]["master_mip_gap"] = args_cli.master_mip_gap
        print(f"[override] master_mip_gap = {args_cli.master_mip_gap}")
    if args_cli.gap_gate_action is not None:
        args["Benders_args"]["gap_gate_action"] = args_cli.gap_gate_action
        print(f"[override] gap_gate_action = {args_cli.gap_gate_action}")

    if args_cli.solver is not None:
        args["Benders_args"]["solver"] = args_cli.solver
        print(f"[override] solver = {args_cli.solver}")
    SOLVER = sb.check_solver_name(args["Benders_args"].get("solver", "gurobi"))
    if args_cli.dump_master and not args_cli.solver_log_dir:
        raise SystemExit("--dump-master needs --solver-log-dir: that is where the .lp files go.")

    if args_cli.cut_selection is not None:
        args["Benders_args"]["cut_selection"] = args_cli.cut_selection
        print(f"[override] cut_selection = {args_cli.cut_selection}")
    if args_cli.cut_selection_k is not None:
        args["Benders_args"]["cut_selection_k"] = args_cli.cut_selection_k
        print(f"[override] cut_selection_k = {args_cli.cut_selection_k}")

    # Coupling check: k-based strategies need a sensible k.
    _cs = args["Benders_args"]["cut_selection"]
    _k  = args["Benders_args"].get("cut_selection_k", 1)
    _needs_k = _cs in ("kmeans", "kmeans_dynamic", "kmeans_dynamic_shared")
    if _needs_k and (_k is None or _k < 1):
        raise SystemExit(
            f"cut_selection='{_cs}' needs --cut-selection-k >= 1 "
            f"(got {_k}). Pass it, or set it in the config."
        )
    if not _needs_k and args_cli.cut_selection_k is not None:
        print(f"[note] cut_selection='{_cs}' ignores k; --cut-selection-k={_k} has no effect.")

    #! Derived from the config itself rather than from its filename, so an
    #! absolute or ./-prefixed --config no longer fails.
    NumNode = len(args["Benders_args"]["N"])
    #! Ramping runs get a results folder of their own, <N>Node_Block<B>, laid out exactly
    #! like <N>Node (same file names), so analysis scripts only need the other folder.
    BLOCK_SUFFIX = f"_Block{RAMP_B}" if RAMP_B > 1 else ""

    roots = resolve_roots(args, args_cli)
    data_root = roots["data_root"]
    output_root = roots["output_root"]

    SOLVER_LOG_DIR = under_root(args_cli.solver_log_dir, output_root) if args_cli.solver_log_dir else None

    print(f"Run config:   {RUN_CONFIG_FILE}  ({NumNode} nodes)")
    print(f"Dataset root: {data_root}")
    print(f"Output root:  {output_root}")
    print(f"Solver:       {SOLVER}" + (f"  (logs: {SOLVER_LOG_DIR})" if SOLVER_LOG_DIR else ""))

    #! Inference device for the loaded nets, independent of the device they
    #! were trained on. CLI beats the config; "auto" is the fallback.
    BENDERS_DEVICE = resolve_device_name(args_cli.device or args.get("device") or "auto")
    print(f"Device:       {BENDERS_DEVICE}")

    #! Surfaced at startup rather than only when the summary CSV is written at
    #! the very end of a run. configs/config-6node.json trips this.
    _exp_dir = args["Benders_args"].get("exp_save_directory") or ""
    if _exp_dir and os.path.basename(_exp_dir.rstrip("/")) != f"{NumNode}Node":
        print(
            f"[paths] WARNING: exp_save_directory '{_exp_dir}' does not match the "
            f"{NumNode}-node config. Summary CSVs go there anyway, alongside another "
            f"node count's results; iteration logs go to outputs/Benders/{NumNode}Node/."
        )

    print(args)


    for i, experiment_instance in enumerate(experiment["experiments"]):
        # Setup output dataframe
        df_res = pd.DataFrame(columns=["setup_time", "presolve_time", "barrier_time", "crossover_time", "restore_time", "objective_value"])

        for j in range(experiment["repeats"]):
            # Run one experiment for j repeats
            run_name = f"train:{args['train']}_rho:{args['rho']}_rhomax:{args['rho_max']}_alpha:{args['alpha']}"

            benders_args = args["Benders_args"]
            ED_args = args["ED_args"]

            # For nodes, just use first letters: ['BEL', 'GER', 'NED'] → 'B-G-N'
            nodes_str = "-".join([n[0] for n in benders_args['N']])
            
            # For generators, count per node: [['BEL', 'WindOn'], ['BEL', 'Gas'],...] = 'B3-G2-N2'
            gen_counts = {}
            for g in benders_args['G']:
                node = g[0]
                gen_counts[node] = gen_counts.get(node, 0) + 1
            gens_str = "-".join([f"{node[0]}{count}" for node, count in gen_counts.items()])
            
            # For lines, just count: [['BEL', 'GER'], ['BEL', 'NED'], ['GER', 'NED']] → 'L3'
            lines_str = f"L{len(benders_args['L'])}"

            # Create a shortened filename
            ed_data_save_path = under_root(
                (f"data/ED_data/ED_N{nodes_str}_G{gens_str}_{lines_str}"
                 f"_c{int(benders_args['benders_compact'])}"
                 f"_s{int(benders_args['scale_problem'])}"
                 f"_p{int(benders_args['perturb_operating_costs'])}"
                 f"_smp{benders_args['2n_synthetic_samples']}.pkl"),
                data_root,
            )

            gep_data_save_path = under_root(
                f"data/GEP_data/sample_duration:{benders_args['sample_duration']}_N:{nodes_str}_G:{gens_str}_L:{lines_str}.pkl",
                data_root,
            )
            # Prep problem data:
            # Prep problem data:
            if args_cli.solve_direct:
                if not os.path.exists(gep_data_save_path):
                    directory = os.path.dirname(gep_data_save_path)
                    os.makedirs(directory, exist_ok=True)
                    create_gep_ed_dataset(
                        args=args,
                        problem_args=benders_args,
                        inputs=experiment_instance,
                        problem_type="GEP",
                        save_path=gep_data_save_path
                    )
            else:
                if not os.path.exists(ed_data_save_path):
                    directory = os.path.dirname(ed_data_save_path)
                    os.makedirs(directory, exist_ok=True)
                    create_gep_ed_dataset(
                        args=args,
                        problem_args=benders_args,
                        inputs=experiment_instance,
                        problem_type="ED",
                        save_path=ed_data_save_path
                    )
                if not os.path.exists(gep_data_save_path):
                    directory = os.path.dirname(gep_data_save_path)
                    os.makedirs(directory, exist_ok=True)
                    create_gep_ed_dataset(
                        args=args,
                        problem_args=benders_args,
                        inputs=experiment_instance,
                        problem_type="GEP",
                        save_path=gep_data_save_path
                    )

            # Load data:
            if args_cli.solve_direct:
                operational_data = None
                with open(gep_data_save_path, 'rb') as file:
                    gep_data = pickle.load(file)
            else:
                with open(ed_data_save_path, 'rb') as file:
                    operational_data = pickle.load(file)
                with open(gep_data_save_path, 'rb') as file:
                    gep_data = pickle.load(file)

            #! Ramping is a property of the run, not of the cached dataset, so the same
            #! pickle serves both. config.toml's `ramping` only drives gep_exact_solver.py
            #! (ramping over the whole horizon); the matrix pipeline reads the block length.
            gep_data.set_ramping(RAMP_B)
            #! Ramp rates are baked into the cached pickle when the dataset is built, so read
            #! them again from the generation data (ramping_rate column, or ramping_value in
            #! inputs/scalars.toml without it): an edit to either would otherwise be ignored.
            rates = ramping_rates(experiment_instance["generation_data"], experiment_instance["ramping_value"])
            cached = gep_data.pRamping
            if not isinstance(cached, dict) or any(cached.get(g) != rates.get(g) for g in gep_data.G):
                print("[ramping] ramp rates taken from the generation data, not the cached dataset")
            gep_data.pRamping = rates
            if RAMP_B > 1:
                gep_data.ramp_limits()   # fails here, naming the generators, if a rate is missing
                print(f"[ramping] blocks of {RAMP_B} h, R_g = rate_g * UnitCap_g: " +
                      ", ".join(f"{n} {t} {rates[(n, t)]:g}" for n, t in gep_data.G))
            elif str(experiment_instance.get("ramping", "false")).lower() == "true":
                print("[ramping] WARNING: config.toml has ramping = true, but ramp_block_hours is "
                      "unset, so this run has no ramp rows. Set --ramp-block-hours to ramp.")

            # !Load primal and dual net
            if not args_cli.solve_direct:
                if args_cli.primal_net_dir or "primal_net_directory" in args["Benders_args"]:
                    primal_net_directory = under_root(
                        args_cli.primal_net_dir or args["Benders_args"]["primal_net_directory"],
                        output_root,
                    )
                else:
                    raise ValueError("Please provide a directory for the primal net in the config file under Benders_args with key 'primal_net_directory'")
                    primal_net_directory = "outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalCompletionClassification/repeat:0"
                
                #! A flow-first run holds one network and no dual checkpoint: its prices are
                #! constructed from the predicted flows, so the pair of interfaces this solver
                #! calls comes from one directory and dual_net_directory is not read.
                flowfirst_weights = (args_cli.flowfirst_weights
                                     or benders_args.get("flowfirst_weights", "model.pt"))
                #! By the presence of the checkpoint, not by importing flowfirst: its trainer sets the
                #! global default dtype to float64 at import time, which the PDL path must not inherit.
                use_flowfirst = os.path.exists(os.path.join(primal_net_directory, flowfirst_weights))

                if use_flowfirst:
                    dual_net_directory = primal_net_directory
                elif args_cli.dual_net_dir or "dual_net_directory" in args["Benders_args"]:
                    dual_net_directory = under_root(
                        args_cli.dual_net_dir or args["Benders_args"]["dual_net_directory"],
                        output_root,
                    )
                else:
                    raise ValueError("Please provide a directory for the dual net in the config file under Benders_args with key 'dual_net_directory'")
                    dual_net_directory = "outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalCompletionClassification/repeat:0"
                print(f"Primal Net Directory: {primal_net_directory}")
                print(f"Dual Net Directory: {dual_net_directory}")

                if use_flowfirst:
                    from flowfirst.dual import load_flowfirst_pair

                    polish_sweeps = int(args_cli.flowfirst_polish_sweeps
                                        if args_cli.flowfirst_polish_sweeps is not None
                                        else benders_args.get("flowfirst_polish_sweeps", 1))
                    print(f"[flowfirst] {flowfirst_weights} with {polish_sweeps} polish sweep(s); "
                          f"the dual is constructed from the flows, no dual checkpoint is read")
                    primal_net, dual_net = load_flowfirst_pair(
                        primal_net_directory, operational_data,
                        weights=flowfirst_weights, polish_sweeps=polish_sweeps,
                        device=BENDERS_DEVICE,
                    )
                else:
                    primal_model_args = json.load(open(os.path.join(primal_net_directory, "args.json")))
                    dual_model_args = json.load(open(os.path.join(dual_net_directory, "args.json")))

                    #! args.json records the device the nets were TRAINED on. Without
                    #! this override a GPU-trained checkpoint would demand a GPU here,
                    #! where we only run inference. Use this run's own device instead.
                    primal_model_args["device"] = BENDERS_DEVICE
                    dual_model_args["device"] = BENDERS_DEVICE

                    #! The run's own args.json decides the architecture. Forcing main.py's
                    #! best_args here rebuilt every primal net at 2 x 28 * xdim, which does not
                    #! load a run trained with any other depth or width. Only runs from before
                    #! main.py recorded its post-override args need the fallback: they store
                    #! hidden_size_factor false for a net that was trained at 28.
                    if not primal_model_args.get("hidden_size_factor"):
                        primal_model_args["hidden_size_factor"] = 28
                        primal_model_args["n_layers"] = 2
                        print("[pdl] args.json records no usable hidden_size_factor; assuming main.py's "
                              "best_args (28 x xdim, 2 layers)")
                    print(f"[pdl] PrimalNetEndToEnd n_layers={primal_model_args['n_layers']} "
                          f"hidden_size_factor={primal_model_args['hidden_size_factor']} from {primal_net_directory}")
                    primal_net = PrimalNetEndToEnd(primal_model_args, operational_data)
                    if args["dual_classification"]:
                        dual_net = DualClassificationNetEndToEnd(dual_model_args, operational_data)
                    else:
                        dual_net = DualNetEndToEnd(dual_model_args, operational_data)
                    primal_net.load_state_dict(torch.load(os.path.join(primal_net_directory, "primal_weights.pth"), weights_only=True, map_location = BENDERS_DEVICE), strict = False)
                    dual_net.load_state_dict(torch.load(os.path.join(dual_net_directory, "dual_weights.pth"), weights_only=True, map_location = BENDERS_DEVICE), strict = False)
                    primal_net.eval()
                    dual_net.eval()

            # Solve single sample with matrix formulation
            start_exact = True
            exact_refinement = False

            # benders_setups = [(True, False), # Exact Benders
            #                     (False, False), # Inexact Benders
            #                     (False, True)] # Inexact Benders with exact refinement

            benders_setups = [(False, True)] # Inexact Benders with exact refinement

            if benders_args["benders_setup"] == "Exact":
                benders_setups = [(True, False)]
            elif benders_args["benders_setup"] == "Inexact":
                benders_setups = [(False, False)]
            elif benders_args["benders_setup"] == "Inexact_Refine":
                benders_setups = [(False, True)]
            elif benders_args["benders_setup"] == "All":
                benders_setups = [(True, False), # Exact Benders
                                    (False, False), # Inexact Benders
                                    (False, True)] # Inexact Benders with exact refinement
            else:
                raise ValueError("Invalid Benders setup specified in config file. Please choose from 'exact', 'inexact', 'inexact_refine' or 'all'.")
            

            # experiment_data = {"opt_obj": [], "upper_bound": [], "lower_bound": [], "total_iterations": [], "exact_iterations": [], "inexact_iterations": [], "total_time": [], "total_time_master": [], "total_time_subproblem_exact": [], "total_time_subproblem_pdl": []}
            samples = 8760 // benders_args["sample_duration"] # SAMPLE
            sample_ids = list(range(samples)) if args_cli.samples is None else args_cli.samples
            if any(not 0 <= s_ < samples for s_ in sample_ids):
                raise SystemExit(f"--samples {args_cli.samples}: this run has samples 0..{samples - 1}.")
            # sample = 0
            for (start_exact, exact_refinement) in benders_setups:
                all_results = []
                #! Logs of one setup in a folder of their own, so "All" does not overwrite them.
                setup_tag = "direct" if args_cli.solve_direct else f"start_exact{start_exact}_ref{exact_refinement}"
                solver_kwargs = dict(
                    solver=SOLVER,
                    solver_log_dir=os.path.join(SOLVER_LOG_DIR, setup_tag) if SOLVER_LOG_DIR else None,
                    solver_tee=args_cli.solver_tee,
                    dump_master=args_cli.dump_master,
                    master_mip_gap=args["Benders_args"].get("master_mip_gap", MASTER_MIP_GAP),
                    lp_phase_tol=args["Benders_args"].get("lp_phase_tol"),
                    lp_phase_max_iters=args["Benders_args"].get("lp_phase_max_iters", 50),
                    refine_change_tol=args["Benders_args"].get("refine_change_tol"),
                    refine_stall_iters=args["Benders_args"].get("refine_stall_iters"),
                    refine_stall_tol=args["Benders_args"].get("refine_stall_tol", 1e-4),
                    refine_gap_tol=args["Benders_args"].get("refine_gap_tol"),
                )
                #! The tolerance profile goes into every result row, so a table can say what it compares.
                tol_profile = {
                    "ramp_block_hours": RAMP_B,
                    "benders_rel_tol": BENDERS_REL_TOL,
                    "master_mip_gap": solver_kwargs["master_mip_gap"],
                    "lp_phase_tol": solver_kwargs["lp_phase_tol"],
                    "refine_gap_tol": solver_kwargs["refine_gap_tol"] or None,
                    "refine_change_tol": solver_kwargs["refine_change_tol"],
                    "refine_stall_iters": solver_kwargs["refine_stall_iters"],
                    "numeric_tol": NUMERIC_TOL,
                }
                for repeat in range(1):
                    for sample in sample_ids:
                        if args_cli.solve_direct:
                            # Solve Directly with Solver
                            primal_net = None
                            dual_net = None
                            print(f"Solving sample {sample} directly with {SOLVER} without Benders decomposition.")
                            solver = BendersSolver(gep_data=gep_data, operational_data=operational_data, 
                                                   primal_net=primal_net, dual_net=dual_net, sample=sample, 
                                                   exact=start_exact, exact_refinement=exact_refinement,
                                                    max_investment=benders_args["max_investment"],
                                                   init_investment=benders_args["init_investment"],
                                                    cut_selection=benders_args["cut_selection"],
                                                    cut_selection_k=benders_args["cut_selection_k"],
                                                    parallel_subproblems=benders_args["parallel_subproblems"],
                                                    n_workers=benders_args["n_workers"],
                                                    dynamic_cluster_features=benders_args.get("dynamic_cluster_features", "price"),
                                                    gap_gate_threshold=benders_args.get("gap_gate_threshold"),
                                                    gap_gate_action=benders_args.get("gap_gate_action", "resolve"),
                                                    **solver_kwargs)
                            y, obj = solver.solve_matrix_problem(gep_data, sample, inv_decision=None)
                            info = solver.direct_info

                            print(f"Direct exact GEP optimum: {obj}")
                            print(f"Direct investment decision: {y[:gep_data.num_g]}")
                            print(f"Direct solve time: {info['solve_time']:.2f}s "
                                  f"(build {info['build_time']:.2f}s, wall {info['wall_time']:.2f}s)")

                            result = {
                                "repeat": repeat,
                                "sample": sample,
                                "method": "direct_exact",
                                "opt_obj": obj,
                                "upper_bound": obj,
                                "lower_bound": obj,
                                "total_iterations": 1,
                                "exact_iterations": 1,
                                "inexact_iterations": 0,
                                #! Solver time only, like the Benders total_time (a sum of solver
                                #! calls); building the model is reported apart in build_time.
                                "total_time": info["solve_time"],
                                "total_time_master": 0.0,
                                "total_time_subproblem_exact": 0.0,
                                "total_time_subproblem_pdl": 0.0,
                                "mip_gap": info["mip_gap"],
                                **tol_profile,
                                "obj_bound": info["obj_bound"],
                                "hit_time_limit": info["hit_time_limit"],
                                "investments": y[:gep_data.num_g].tolist(),
                                "solver": SOLVER,
                                "status": info["status"],
                                "build_time": info["build_time"],
                                "wall_time": info["wall_time"],
                                "nodes": info["nodes"],
                            }
                            all_results.append(result)
                            continue

                        else:
                            solver = BendersSolver(gep_data=gep_data, operational_data=operational_data, primal_net=primal_net, dual_net=dual_net, sample=sample, 
                                                   exact=start_exact, exact_refinement=exact_refinement, max_investment=benders_args["max_investment"],
                                                   init_investment=benders_args["init_investment"],
                                                    cut_selection=benders_args["cut_selection"],
                                                    cut_selection_k=benders_args["cut_selection_k"],
                                                    parallel_subproblems=benders_args["parallel_subproblems"],
                                                    n_workers=benders_args["n_workers"],
                                                    dynamic_cluster_features=benders_args.get("dynamic_cluster_features", "price"),
                                                    gap_gate_threshold=benders_args.get("gap_gate_threshold"),
                                                    gap_gate_action=benders_args.get("gap_gate_action", "resolve"),
                                                    **solver_kwargs)
        
                            # Solve for the ground truth if falg is set to True
                            if args_cli.ground_truth:
                                y, obj = solver.solve_matrix_problem(gep_data, sample)

                            # Solve single sample with Benders decomposition
                            # sample = 1 # solution = Obj: 2374.99
                            # compact = False
                            # Solving with Benders decomposition
                            upper_bound, lower_bound, benders_cuts_all, investments_all, obj_val_subproblems_all, iterations = solver.solve_with_benders(gep_data, benders_args['benders_compact'], sample)

                            iter_df = pd.DataFrame({
                                "sample": sample,
                                "iter": solver.iter_hist,
                                "UB": solver.ub_hist,
                                "LB": solver.lb_hist,
                                "gap_abs": np.array(solver.ub_hist) - np.array(solver.lb_hist),
                                "gap_rel": (np.array(solver.ub_hist) - np.array(solver.lb_hist)) / np.maximum(1.0, np.abs(np.array(solver.ub_hist))),
                                "exact_mode": solver.exact_flag_hist,
                                "t_master": solver.master_time_hist,
                                "t_sub": solver.sub_time_hist,
                                "t_iter_wall":    solver.wall_iter_hist,     # wall-clock per iter
                                "gap_abs_mean":   solver.gap_abs_mean_hist,
                                "gap_abs_median": solver.gap_abs_median_hist,
                                "gap_rel_mean":   solver.gap_rel_mean_hist,
                                "gap_rel_median": solver.gap_rel_median_hist,
                                "gap_abs_total":  solver.gap_abs_total_hist,
                                "gap_neg_count":  solver.gap_neg_count_hist,
                                "gate_flagged":   solver.gate_flagged_hist,
                                "gate_resolved":  solver.gate_resolved_hist,
                                "gate_dropped":   solver.gate_dropped_hist,
                                "gate_time":      solver.gate_time_hist,
                                "gap_rel_max":    solver.gap_rel_max_hist,
                                "cut_tight_rel":  solver.cut_tight_rel_hist,
                                "cut_tight_rel_max_hour": solver.cut_tight_rel_max_hour_hist,
                                "lp_phase":       solver.lp_phase_hist,
                                #! Master state per iteration -- status, bound, gap, nodes and the
                                #! cut coefficient range -- for tracing a master that stalls.
                                **solver.master_diagnostics(),
                            })
                            crossover_metrics = BendersSolver.get_crossover_metrics(iter_df)
                            iter_df["investment"] = [json.dumps(v) for v in solver.inv_hist]
                            specific_name = args["Benders_args"].get("specific_name", "")
                            if SOLVER != "gurobi":
                                specific_name = f"{specific_name}_{SOLVER}"
                            benders_setup_str = args["Benders_args"].get("benders_setup", "")
                            # if samples == 1:
                            #     out_dir = f"outputs/Benders/{NumNode}Node/Full_Time/iter_logs_{benders_setup_str}_{specific_name}"
                            # else:
                            out_dir = under_root(
                                f"outputs/Benders/{NumNode}Node{BLOCK_SUFFIX}/Sample_{benders_args['sample_duration']}/iter_logs_{benders_setup_str}_{specific_name}",
                                output_root,
                            )

                            ensure_dir(out_dir)
                            iter_df.to_csv(
                                os.path.join(out_dir, f"iterlog_sample{sample}_start_exact{start_exact}_ref{exact_refinement}.csv"),
                                index=False
                            )

                            # ! If you want to save data for the first sample for plotting, uncomment the following line.
                            # if start_exact and sample == 0:
                            #     solver.save_data(f"experiment-output/ch7/3nodes/benders_data")
                            
                            # print(f"Known optimum: {obj}")
                            # print(y[:gep_data.num_g])
                            print(f"Iterations: {iterations}")
                            print(f"Total time master: {solver.total_time_master}, Total time subproblem_exact: {solver.total_time_subproblem_exact}, Total time subproblem_pdl: {solver.total_time_subproblem_pdl}")
                            print(f"Total time: {solver.total_time_master + solver.total_time_subproblem_exact + solver.total_time_subproblem_pdl}")

                            # Store results in a dict for this run
                            result = {
                                "repeat": repeat,
                                "sample": sample,
                                # "opt_obj": obj,
                                "upper_bound": upper_bound,
                                "lower_bound": lower_bound,
                                "total_iterations": iterations,
                                "exact_iterations": solver.exact_iterations,
                                "inexact_iterations": solver.inexact_iterations,
                                "total_time": solver.total_time_master + solver.total_time_subproblem_exact + solver.total_time_subproblem_pdl,
                                "total_time_master": solver.total_time_master,
                                "total_time_subproblem_exact": solver.total_time_subproblem_exact,
                                "total_time_subproblem_pdl": solver.total_time_subproblem_pdl,
                                "hit_time_limit": solver.hit_time_limit,
                                "investments": investments_all[-1].tolist() if len(investments_all) > 0 else None,
                                "has_crossover": crossover_metrics["has_crossover"],
                                "lb_cross": crossover_metrics["lb_cross"],
                                "gap_cross_pct": crossover_metrics["gap_cross_pct"],
                                "lb_cross_ratio_pct": crossover_metrics["lb_cross_ratio_pct"],
                                "solver": SOLVER,
                                **tol_profile,
                                #! Whole solve_with_benders, Pyomo bookkeeping and cut building included.
                                "wall_time": solver.wall_time,
                                #! What ended the inexact phase: "investment repeated", or the practical trigger.
                                "refine_reason": solver.refine_reason,
                            }
                            all_results.append(result)
                            # break
                        
                #! Set to True if saving data.
                if True:
                    experiment_data_df = pd.DataFrame(all_results)

                    exp_save_directory = under_root(
                        (args["Benders_args"].get("exp_save_directory") or f"outputs/Benders/{NumNode}Node").rstrip("/")
                        + BLOCK_SUFFIX,
                        output_root,
                    )

                    #! The Sample_<duration> level has to be created here. It used
                    #! to exist only as a side effect of the iter_logs makedirs
                    #! above, which shared the same literal prefix by coincidence.
                    sample_dir = ensure_dir(
                        os.path.join(exp_save_directory, f"Sample_{str(benders_args['sample_duration'])}")
                    )

                    if args_cli.solve_direct:
                        specific_name = "direct_exact"
                        #! Gurobi keeps its old file name, so existing analysis scripts still find it.
                        data_save_path = os.path.join(sample_dir, "Gurobi_Solution.csv" if SOLVER == "gurobi"
                                                      else f"{SOLVER}_Solution.csv")

                    else:
                        specific_name = args["Benders_args"].get("specific_name", "")
                        #! Runs on another solver must not overwrite the Gurobi results.
                        if SOLVER != "gurobi":
                            specific_name = f"{specific_name}_{SOLVER}"

                        if samples == 1:
                            data_save_path = os.path.join(sample_dir, f"experiment_data_full_time_sample_duration:{benders_args['sample_duration']}_start_exact:{start_exact}_exact_refinement:{exact_refinement}_{specific_name}.csv")
                        else:
                            data_save_path = os.path.join(sample_dir, f"experiment_data_sample_duration:{benders_args['sample_duration']}_start_exact:{start_exact}_exact_refinement:{exact_refinement}_{specific_name}.csv")

                    experiment_data_df.to_csv(data_save_path, index=False)


