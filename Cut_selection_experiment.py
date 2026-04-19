"""
One-sample Benders cut-selection experiment for GEP.

This file imports BendersSolver from gep_benders.py and compares different
ways of aggregating timestep Benders cuts for ONE fixed sample:

    1. single      : all timesteps aggregated into one cut (current baseline)
    2. kmeans      : KMeans clusters based on [demand, A*Pmax*u_ref]
    3. stress      : quantile bins based on total_demand / total_available_capacity
    4. full        : one individual cut per timestep (multi-cut benchmark)

Important design choice:
    The groups are STATIC within one run. They are created once before Benders
    starts using a reference investment vector u_ref. This keeps the meaning of
    each alpha_k fixed across iterations.

Expected usage from project root:

    python gep_benders_cut_selection_experiment.py

or import in notebook:

    from gep_benders_cut_selection_experiment import run_cut_selection_experiment

    df = run_cut_selection_experiment(
        gep_data=gep_data,
        operational_data=operational_data,
        sample=0,
        compact=False,
        u_ref=investment,
        strategies=(
            ("single", None),
            ("kmeans", 6),
            ("stress", 6),
            ("full", None),
        ),
    )
"""

import os
import time
import pickle
import argparse
from typing import List, Tuple, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
import torch
import gurobipy as gp
from gurobipy import GRB

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from gep_benders import BendersSolver

import json

from networks import (
    PrimalNetEndToEnd,
    DualNetEndToEnd,
    DualClassificationNetEndToEnd,
)

GroupList = List[List[int]]
Cut = Tuple[np.ndarray, float]


# -----------------------------------------------------------------------------
# Fast clustering/grouping utilities: no ED solve needed
# -----------------------------------------------------------------------------

def _to_numpy_investment(investment) -> np.ndarray:
    if isinstance(investment, torch.Tensor):
        return investment.detach().cpu().numpy().astype(float)
    return np.asarray(investment, dtype=float)


def build_capacity_demand_features(data, sample: int, investment) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build cheap timestep features without solving ED.

    Feature vector per timestep:
        z_t = [D_{1,t}, ..., D_{N,t}, A_{1,t}Pmax_1 u_1, ..., A_{G,t}Pmax_G u_G]

    Returns:
        X                  shape (T, num_n + num_g)
        demand_features    shape (T, num_n)
        capacity_features  shape (T, num_g)
    """
    ineq_cm_sample, _, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample)

    T = len(data.time_ranges[sample])
    num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)
    num_rows_per_t_eq = data.num_n
    investment_np = _to_numpy_investment(investment)

    demand_features = np.zeros((T, data.num_n), dtype=float)
    capacity_features = np.zeros((T, data.num_g), dtype=float)

    for t in range(T):
        # Demand RHS from node-balance equality rows.
        row_eq = t * num_rows_per_t_eq
        demand_features[t, :] = (
            eq_rhs_sample[row_eq:row_eq + data.num_n]
            .detach()
            .cpu()
            .numpy()
        )

        # Available invested capacity from rows:
        #     p_g - A_{g,t} Pmax_g u_g <= 0
        row_ineq = data.num_g + t * num_rows_per_t_ineq
        for g in range(data.num_g):
            p_ub_row = row_ineq + data.num_g + g
            apmax = float((-ineq_cm_sample[p_ub_row, g]).detach().cpu().numpy())
            capacity_features[t, g] = apmax * investment_np[g]

    X = np.concatenate([demand_features, capacity_features], axis=1)
    return X, demand_features, capacity_features


def make_single_group(T: int) -> GroupList:
    return [list(range(T))]


def make_full_multicut_groups(T: int) -> GroupList:
    return [[t] for t in range(T)]


def make_kmeans_capacity_demand_groups(
    data,
    sample: int,
    investment,
    K: int,
    random_state: int = 0,
) -> Tuple[GroupList, np.ndarray, np.ndarray]:
    """
    KMeans grouping based on [demand, A*Pmax*u_ref].
    This is cheap: no ED subproblem solve is required.
    """
    X, _, _ = build_capacity_demand_features(data, sample, investment)

    if K <= 1:
        labels = np.zeros(X.shape[0], dtype=int)
        return make_single_group(X.shape[0]), labels, X

    K_eff = min(K, X.shape[0])
    X_scaled = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=K_eff, random_state=random_state, n_init="auto").fit_predict(X_scaled)
    groups = [np.where(labels == k)[0].tolist() for k in range(K_eff)]
    return groups, labels, X


def make_stress_bin_groups(
    data,
    sample: int,
    investment,
    K: int,
) -> Tuple[GroupList, np.ndarray, np.ndarray]:
    """
    Very cheap interpretable grouping by stress quantiles:

        stress_t = total_demand_t / (total_available_invested_capacity_t + eps)

    Returns:
        groups, labels, stress
    """
    _, demand_features, capacity_features = build_capacity_demand_features(data, sample, investment)
    total_demand = demand_features.sum(axis=1)
    total_capacity = capacity_features.sum(axis=1)
    stress = total_demand / (total_capacity + 1e-9)

    T = len(stress)
    if K <= 1:
        labels = np.zeros(T, dtype=int)
        return make_single_group(T), labels, stress

    K_eff = min(K, T)
    order = np.argsort(stress)
    split = np.array_split(order, K_eff)

    groups = [list(x) for x in split]
    labels = np.empty(T, dtype=int)
    for k, group in enumerate(groups):
        labels[group] = k

    return groups, labels, stress


def make_groups(
    strategy: str,
    data,
    sample: int,
    investment,
    K: Optional[int] = None,
    random_state: int = 0,
) -> Tuple[GroupList, Dict[str, Any]]:
    """
    Create static timestep groups for one run.
    """
    T = len(data.time_ranges[sample])
    strategy = strategy.lower()

    if strategy == "single":
        groups = make_single_group(T)
        labels = np.zeros(T, dtype=int)
        extra = {"labels": labels}
    elif strategy == "full":
        groups = make_full_multicut_groups(T)
        labels = np.arange(T)
        extra = {"labels": labels}
    elif strategy == "kmeans":
        if K is None:
            raise ValueError("K must be provided for strategy='kmeans'.")
        groups, labels, X = make_kmeans_capacity_demand_groups(
            data=data,
            sample=sample,
            investment=investment,
            K=K,
            random_state=random_state,
        )
        extra = {"labels": labels, "features": X}
    elif strategy == "stress":
        if K is None:
            raise ValueError("K must be provided for strategy='stress'.")
        groups, labels, stress = make_stress_bin_groups(
            data=data,
            sample=sample,
            investment=investment,
            K=K,
        )
        extra = {"labels": labels, "stress": stress}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Remove accidental empty groups, just in case.
    groups = [g for g in groups if len(g) > 0]
    extra["group_sizes"] = [len(g) for g in groups]
    return groups, extra


# -----------------------------------------------------------------------------
# Grouped-cut Benders solver
# -----------------------------------------------------------------------------

class BendersSolverCutSelection(BendersSolver):
    """
    Exact Benders solver variant that supports grouped timestep cuts.

    Master variables:
        [u_1, ..., u_G, alpha_1, ..., alpha_K]

    Each group k has its own alpha_k and receives one aggregated group cut per
    Benders iteration.
    """

    def solve_matrix_problem_simple_grouped_master(
        self,
        obj,
        A_ineq,
        b_ineq,
        A_eq,
        b_eq,
        num_alpha: int,
        investment=None,
    ):
        m = gp.Model("Grouped master problem", env=self.env)
        m.setParam("MIPGap", 1e-8)
        m.setParam("OptimalityTol", 1e-9)

        ydim = obj.size
        num_u = ydim - num_alpha

        vtypes = np.array([GRB.INTEGER for _ in range(num_u)] + [GRB.CONTINUOUS for _ in range(num_alpha)])
        x = m.addMVar(shape=ydim, lb=0, vtype=vtypes, name="x")

        if investment is not None:
            m.addConstr(x[:num_u] == np.array(investment, dtype=float))

        m.setObjective(np.array(obj, dtype=float) @ x, GRB.MINIMIZE)

        m.addConstr(np.array(A_ineq, dtype=float) @ x <= np.array(b_ineq, dtype=float), name="ineq")

        # Kept for compatibility, but grouped master normally has no real equalities.
        if A_eq is not None and len(A_eq) > 0:
            m.addConstr(np.array(A_eq, dtype=float) @ x == np.array(b_eq, dtype=float), name="eq")

        start_time = time.time()
        m.optimize()
        inference_time = time.time() - start_time

        if m.status != GRB.OPTIMAL:
            raise RuntimeError(f"Grouped master not solved to optimality. Gurobi status={m.status}")

        return m.ObjVal, x.X, inference_time

    def find_master_problem_cm_rhs_obj_grouped(
        self,
        data,
        compact: bool,
        sample: int,
        benders_cuts: Sequence[Cut],
        num_alpha: int,
        alpha_lb: float = -1e6,
        investment_ub: float = 100000.0,
    ):
        investment_obj = data.obj_coeff[:data.num_g].detach().cpu().numpy()
        obj = np.concatenate([investment_obj, np.ones(num_alpha)], axis=0)

        ineq_cm_sample, ineq_rhs_sample, _, _ = data.get_sample_matrices(sample)

        # Original investment lower-bound rows, e.g. -u_g <= 0.
        A_ineq = ineq_cm_sample[:data.num_g, :data.num_g].detach().cpu().numpy()
        A_ineq = np.concatenate([A_ineq, np.zeros((data.num_g, num_alpha))], axis=1)
        b_ineq = ineq_rhs_sample[:data.num_g].detach().cpu().numpy()

        # Lower bound alpha_k >= alpha_lb, written as -alpha_k <= -alpha_lb.
        # Original code used alpha >= -1e6 through -alpha <= 1e6.
        for k in range(num_alpha):
            row = np.zeros((1, data.num_g + num_alpha))
            row[0, data.num_g + k] = -1.0
            A_ineq = np.concatenate([A_ineq, row], axis=0)
            b_ineq = np.concatenate([b_ineq, np.array([-alpha_lb])], axis=0)

        # Add all Benders cuts from all previous iterations.
        for cut_lhs, cut_rhs in benders_cuts:
            A_ineq = np.concatenate([A_ineq, cut_lhs], axis=0)
            b_ineq = np.concatenate([b_ineq, np.array([cut_rhs])], axis=0)

        # Investment upper bounds.
        for g in range(data.num_g):
            row = np.zeros((1, data.num_g + num_alpha))
            row[0, g] = 1.0
            A_ineq = np.concatenate([A_ineq, row], axis=0)
            b_ineq = np.concatenate([b_ineq, np.array([investment_ub])], axis=0)

        # Dummy equality, matching your existing helper convention.
        A_eq = np.zeros((1, data.num_g + num_alpha))
        b_eq = np.zeros((1,))

        return obj, A_ineq, b_ineq, A_eq, b_eq

    def solve_master_problem_grouped(
        self,
        data,
        compact: bool,
        sample: int,
        benders_cuts: Sequence[Cut],
        num_alpha: int,
        investment=None,
    ):
        obj, A_ineq, b_ineq, A_eq, b_eq = self.find_master_problem_cm_rhs_obj_grouped(
            data=data,
            compact=compact,
            sample=sample,
            benders_cuts=benders_cuts,
            num_alpha=num_alpha,
        )

        obj_val, primal_val, inference_time = self.solve_matrix_problem_simple_grouped_master(
            obj=obj,
            A_ineq=A_ineq,
            b_ineq=b_ineq,
            A_eq=A_eq,
            b_eq=b_eq,
            num_alpha=num_alpha,
            investment=investment,
        )

        investments = primal_val[:data.num_g]
        alpha_vals = primal_val[data.num_g:data.num_g + num_alpha]

        investment_cost = float(obj[:data.num_g] @ investments)
        alpha_total = float(np.sum(alpha_vals))

        obj_val_master = [investment_cost, alpha_total]
        return obj_val_master, investments, alpha_vals, inference_time

    def find_benders_cut_for_timesteps(
        self,
        data,
        compact: bool,
        sample: int,
        dual_vals: np.ndarray,
        b_ineqs: np.ndarray,
        b_eqs: np.ndarray,
        timestep_indices: Sequence[int],
        alpha_index: int,
        num_alpha: int,
    ) -> Cut:
        """
        Build one grouped Benders cut for selected timesteps.

        Uses the same sign convention as your find_benders_cut_batch:
            cut_lhs @ [u, alpha] <= cut_rhs
        with coefficient -1 on this group's alpha.
        """
        if compact:
            raise NotImplementedError("This grouped experiment currently supports compact=False only.")

        ineq_cm_sample, _, _, _ = data.get_sample_matrices(sample)
        num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)

        timestep_indices = np.array(list(timestep_indices), dtype=int)

        cut_lhs = np.zeros((1, data.num_g + num_alpha), dtype=float)
        cut_rhs = 0.0
        cut_lhs[0, data.num_g + alpha_index] = -1.0

        # Investment coefficients from production upper-bound duals.
        for g in range(data.num_g):
            dual_idx = data.num_g + g
            coeff_sum = 0.0
            for t in timestep_indices:
                p_ub_row = data.num_g + t * num_rows_per_t_ineq + data.num_g + g
                apmax = float((-ineq_cm_sample[p_ub_row, g]).detach().cpu().numpy())
                coeff_sum += float(dual_vals[t, dual_idx]) * apmax
            cut_lhs[0, g] = coeff_sum

        # RHS terms from nonzero inequality RHS constraints.
        constraint_nrs = []
        constraint_nrs += [2 * data.num_g + l for l in range(data.num_l)]
        constraint_nrs += [2 * data.num_g + data.num_l + l for l in range(data.num_l)]
        constraint_nrs += [2 * data.num_g + 2 * data.num_l + data.num_n + n for n in range(data.num_n)]
        constraint_nrs = np.array(constraint_nrs, dtype=int)

        cut_rhs += -float(np.sum(
            dual_vals[timestep_indices[:, None], constraint_nrs]
            * b_ineqs[timestep_indices[:, None], constraint_nrs]
        ))

        # RHS terms from equality constraints.
        eq_dual_start = num_rows_per_t_ineq
        eq_duals = dual_vals[timestep_indices, eq_dual_start:eq_dual_start + data.num_n]
        cut_rhs += -float(np.sum(eq_duals * b_eqs[timestep_indices]))

        return cut_lhs, cut_rhs

    def solve_subproblems_grouped_exact(
        self,
        data,
        compact: bool,
        sample: int,
        investments,
        groups: GroupList,
    ):
        """
        Solve all timestep ED subproblems exactly, then aggregate cuts according
        to the provided static groups.
        """
        if compact:
            raise NotImplementedError("This grouped experiment currently supports compact=False only.")

        T = len(data.time_ranges[sample])
        ineq_cm_sample, ineq_rhs_sample, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample)

        obj_vals = []
        primal_vals = []
        dual_vals = []
        inference_times = []
        b_ineqs = []
        b_eqs = []

        for t in range(T):
            obj, A_ineq, b_ineq, A_eq, b_eq = self.find_subproblem_cm_rhs_obj_from_mats(
                data=data,
                compact=compact,
                investments=investments,
                time_step=t,
                ineq_cm_sample=ineq_cm_sample,
                ineq_rhs_sample=ineq_rhs_sample,
                eq_cm_sample=eq_cm_sample,
                eq_rhs_sample=eq_rhs_sample,
            )

            obj_val, primal_val, dual_val, inference_time = self.solve_matrix_problem_simple(
                obj=obj,
                A_ineq=A_ineq,
                b_ineq=b_ineq,
                A_eq=A_eq,
                b_eq=b_eq,
                master=False,
            )

            obj_vals.append(obj_val)
            primal_vals.append(primal_val)
            dual_vals.append(dual_val)
            inference_times.append(inference_time)
            b_ineqs.append(b_ineq)
            b_eqs.append(b_eq)

        obj_vals = np.array(obj_vals, dtype=float)
        dual_vals = np.array(dual_vals, dtype=float)
        b_ineqs = np.array(b_ineqs, dtype=float)
        b_eqs = np.array(b_eqs, dtype=float)

        group_cuts: List[Cut] = []
        for k, timestep_indices in enumerate(groups):
            cut = self.find_benders_cut_for_timesteps(
                data=data,
                compact=compact,
                sample=sample,
                dual_vals=dual_vals,
                b_ineqs=b_ineqs,
                b_eqs=b_eqs,
                timestep_indices=timestep_indices,
                alpha_index=k,
                num_alpha=len(groups),
            )
            group_cuts.append(cut)

        primal_obj_val_total = float(np.sum(obj_vals))
        dual_obj_val_total = primal_obj_val_total
        inference_time_total = float(np.sum(inference_times))

        return primal_obj_val_total, dual_obj_val_total, group_cuts, inference_time_total

    def solve_with_benders_grouped(
        self,
        data,
        compact: bool,
        sample: int,
        groups: GroupList,
        max_iterations: int = 1000,
        epsilon: float = 1e-6,
    ):
        """
        Exact Benders with fixed grouped cuts.
        """
        if not self.exact:
            raise ValueError("This cut-selection experiment is intended for exact Benders only.")
        if compact:
            raise NotImplementedError("This grouped experiment currently supports compact=False only.")

        benders_cut_all: List[Cut] = []
        investments_all = []
        obj_val_subproblems_all = []

        num_alpha = len(groups)
        optimal = False
        i = 0
        upper_bound = np.inf
        lower_bound = -np.inf

        while not optimal and i < max_iterations:
            print("-" * 50)
            print(f"Iteration {i}; groups={num_alpha}")

            if i == 0:
                if self.investment_init_method == "Zero":
                    investments_iter_k = np.zeros(data.num_g, dtype=float)
                elif self.investment_init_method == "HalfMax":
                    investments_iter_k = (
                        self.operational_data.pUnitInvestment.max(dim=0).values / 2
                    ).detach().cpu().numpy().astype(float)
                else:
                    raise ValueError(f"Invalid investment initialization method: {self.investment_init_method}")

                investment_cost = 0.0
                for g_idx, g in enumerate(data.G):
                    investment_cost += data.pInvCost[g] * data.pUnitCap[g] * investments_iter_k[g_idx]
                obj_val_master = [float(investment_cost), 0.0]
                alpha_vals = np.zeros(num_alpha, dtype=float)
                inference_time_master = 0.0
            else:
                obj_val_master, investments_iter_k, alpha_vals, inference_time_master = self.solve_master_problem_grouped(
                    data=data,
                    compact=compact,
                    sample=sample,
                    benders_cuts=benders_cut_all,
                    num_alpha=num_alpha,
                )
                self.total_time_master += inference_time_master

            print("Investment decisions:", investments_iter_k)
            investments_iter_k_tensor = torch.tensor(investments_iter_k, dtype=torch.float64)
            investments_all.append(investments_iter_k_tensor)
            self.inv_hist.append(investments_iter_k_tensor.detach().cpu().numpy().tolist())

            self.exact_iterations += 1

            primal_obj_val_total, dual_obj_val_total, group_cuts, inference_time_sub = self.solve_subproblems_grouped_exact(
                data=data,
                compact=compact,
                sample=sample,
                investments=investments_iter_k_tensor,
                groups=groups,
            )
            self.total_time_subproblem_exact += inference_time_sub

            obj_val_subproblems_all.append(primal_obj_val_total)

            lower_bound = float(obj_val_master[0] + obj_val_master[1])
            upper_bound = float(obj_val_master[0] + primal_obj_val_total)
            if upper_bound < self.best_upper_bound:
                self.best_upper_bound = upper_bound

            print(f"UB={upper_bound:.6f}, LB={lower_bound:.6f}, gap={upper_bound - lower_bound:.6f}")

            self.ub_hist.append(upper_bound)
            self.lb_hist.append(lower_bound)
            self.iter_hist.append(int(i))
            self.exact_flag_hist.append(True)
            self.sub_time_hist.append(float(inference_time_sub))
            self.master_time_hist.append(float(inference_time_master))

            abs_gap = upper_bound - lower_bound
            rel_gap = abs_gap / max(1.0, abs(upper_bound))
            abs_tol = 1e-3
            rel_tol = 1e-8
            if abs_gap <= abs_tol or rel_gap <= rel_tol:
                optimal = True
                print("Done! Optimal solution found.")
            else:
                # Add all group cuts from this iteration.
                benders_cut_all.extend(group_cuts)
                print(f"Added {len(group_cuts)} cuts. Total cuts: {len(benders_cut_all)}")

            i += 1

        return upper_bound, lower_bound, benders_cut_all, investments_all, obj_val_subproblems_all, i


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_one_strategy(
    gep_data,
    operational_data,
    sample: int,
    compact: bool,
    u_ref,
    strategy: str,
    K: Optional[int],
    init_investment: str = "Zero",
    max_iterations: int = 1000,
    random_state: int = 0,
) -> Dict[str, Any]:
    groups, group_info = make_groups(
        strategy=strategy,
        data=gep_data,
        sample=sample,
        investment=u_ref,
        K=K,
        random_state=random_state,
    )

    print("=" * 80)
    print(f"Strategy={strategy}, K={K}, num_groups={len(groups)}")
    print("Group sizes:", group_info["group_sizes"])

    solver = BendersSolverCutSelection(
        gep_data=gep_data,
        operational_data=operational_data,
        primal_net=None,
        dual_net=None,
        sample=sample,
        exact=True,
        exact_refinement=False,
        init_investment=init_investment,
    )

    start_time = time.time()
    upper_bound, lower_bound, cuts, investments_all, sub_objs, iterations = solver.solve_with_benders_grouped(
        data=gep_data,
        compact=compact,
        sample=sample,
        groups=groups,
        max_iterations=max_iterations,
    )
    total_wall_time = time.time() - start_time

    result = {
        "sample": sample,
        "strategy": strategy,
        "K_requested": K,
        "num_groups": len(groups),
        "group_sizes": group_info["group_sizes"],
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "gap_abs": upper_bound - lower_bound,
        "gap_rel": (upper_bound - lower_bound) / max(1.0, abs(upper_bound)),
        "iterations": iterations,
        "num_cuts_total": len(cuts),
        "total_wall_time": total_wall_time,
        "total_time_master": solver.total_time_master,
        "total_time_subproblem_exact": solver.total_time_subproblem_exact,
        "final_investment": investments_all[-1].detach().cpu().numpy().tolist() if investments_all else None,
    }

    return result


def run_cut_selection_experiment(
    gep_data,
    operational_data,
    sample: int = 0,
    compact: bool = False,
    u_ref=None,
    strategies: Sequence[Tuple[str, Optional[int]]] = (("single", None), ("kmeans", 6), ("stress", 6), ("full", None)),
    init_investment: str = "Zero",
    max_iterations: int = 1000,
    random_state: int = 0,
    save_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    if u_ref is None:
        u_ref = torch.full((gep_data.num_g,), 300.0, dtype=torch.float64)

    results = []
    for strategy, K in strategies:
        res = run_one_strategy(
            gep_data=gep_data,
            operational_data=operational_data,
            sample=sample,
            compact=compact,
            u_ref=u_ref,
            strategy=strategy,
            K=K,
            init_investment=init_investment,
            max_iterations=max_iterations,
            random_state=random_state,
        )
        results.append(res)

    df = pd.DataFrame(results)

    if save_csv_path is not None:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"Saved results to {save_csv_path}")

    return df


# -----------------------------------------------------------------------------
# CLI entry point for your current paths
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gep-data-path", type=str, default="data/GEP_data/sample_duration:120_N:B-G-F_G:B2-G2-F2_L:L3.pkl")
    parser.add_argument("--ed-data-path", type=str, default="data/ED_data/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15.pkl")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--investment", type=float, default=1.0, help="Reference investment for static clustering.")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--skip-full", action="store_true", help="Skip full individual timestep cuts if master gets too slow.")
    parser.add_argument("--save-csv", type=str, default="outputs/Benders/cut_selection_one_sample/results_sample0.csv")
    args = parser.parse_args()

    with open(args.gep_data_path, "rb") as f:
        gep_data = pickle.load(f)
    with open(args.ed_data_path, "rb") as f:
        operational_data = pickle.load(f)

    u_ref = torch.full((gep_data.num_g,), float(args.investment), dtype=torch.float64)



    k_values = [4, 6, 8, 10]

    strategies = [("single", None)]

    for k in k_values:
        strategies.append(("kmeans", k))
        strategies.append(("stress", k))

    if not args.skip_full:
        strategies.append(("full", None))

    df = run_cut_selection_experiment(
        gep_data=gep_data,
        operational_data=operational_data,
        sample=args.sample,
        compact=False,
        u_ref=u_ref,
        strategies=strategies,
        max_iterations=args.max_iterations,
        save_csv_path=args.save_csv,
    )

    print("\nFinal summary:")
    print(df[[
        "strategy", "K_requested", "num_groups", "iterations", "num_cuts_total",
        "upper_bound", "lower_bound", "gap_abs", "total_wall_time",
        "total_time_master", "total_time_subproblem_exact"
    ]])


if __name__ == "__main__":
    main()
