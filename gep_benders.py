import copy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch
import json
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


from gep_problem import GEPProblemSet
from gep_problem_operational import GEPOperationalProblemSet
from create_gep_dataset import create_gep_ed_dataset
from gep_config_parser import *
from networks import DualClassificationNetEndToEnd, DualNet, DualNetEndToEnd, PrimalNetEndToEnd

CONFIG_FILE_NAME        = "config.toml"

def build_capacity_demand_features(data, sample):
    """
    Build cheap timestep features without solving ED and without a reference investment.

    Feature vector per timestep:
        z_t = [D_{1,t}, ..., D_{N,t}, A_{1,t}Pmax_1, ..., A_{G,t}Pmax_G]

    This avoids using an arbitrary u_ref.
    """
    ineq_cm_sample, _, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample)

    T = len(data.time_ranges[sample])
    num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)
    num_rows_per_t_eq = data.num_n

    demand_features = np.zeros((T, data.num_n), dtype=float)
    capacity_potential_features = np.zeros((T, data.num_g), dtype=float)

    for t in range(T):
        # Demand RHS from node-balance equality rows.
        row_eq = t * num_rows_per_t_eq
        demand_features[t, :] = (
            eq_rhs_sample[row_eq:row_eq + data.num_n]
            .detach()
            .cpu()
            .numpy()
        )

        # Capacity potential A_{g,t} Pmax_g from rows:
        #     p_g - A_{g,t} Pmax_g u_g <= 0
        row_ineq = data.num_g + t * num_rows_per_t_ineq

        for g in range(data.num_g):
            p_ub_row = row_ineq + data.num_g + g
            apmax = float((-ineq_cm_sample[p_ub_row, g]).detach().cpu().numpy())
            capacity_potential_features[t, g] = apmax

    X = np.concatenate([demand_features, capacity_potential_features], axis=1)

    return X, demand_features, capacity_potential_features


def make_single_group(T):
    return [list(range(T))]


def make_full_multicut_groups(T):
    return [[t] for t in range(T)]


def make_kmeans_capacity_demand_groups(data, sample, K, random_state=0):
    """
    KMeans grouping based on [demand, A*Pmax].
    No ED solve and no reference investment required.
    """
    X, _, _ = build_capacity_demand_features(data, sample)

    if K <= 1:
        labels = np.zeros(X.shape[0], dtype=int)
        return make_single_group(X.shape[0]), labels, X

    K_eff = min(K, X.shape[0])

    X_scaled = StandardScaler().fit_transform(X)

    labels = KMeans(
        n_clusters=K_eff,
        random_state=random_state,
        n_init="auto"
    ).fit_predict(X_scaled)

    groups = [np.where(labels == k)[0].tolist() for k in range(K_eff)]

    return groups, labels, X


def make_stress_bin_groups(data, sample, K):
    """
    Stress grouping without reference investment.

    stress_t = total_demand_t / (total_capacity_potential_t + eps)

    where total_capacity_potential_t = sum_g A_{g,t} Pmax_g.
    """
    _, demand_features, capacity_potential_features = build_capacity_demand_features(
        data=data,
        sample=sample,
    )

    total_demand = demand_features.sum(axis=1)
    total_capacity_potential = capacity_potential_features.sum(axis=1)

    stress = total_demand / (total_capacity_potential + 1e-9)

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


def make_cut_groups(data, sample, cut_selection="single", cut_selection_k=1, random_state=0):
    """
    Create timestep groups for cut aggregation.

    cut_selection options:
        single  : all timesteps in one group (Aggregate all cuts into one)
        kmeans  : KMeans on [D, A*Pmax], no u_ref
        stress  : stress bins using total_demand / total_A_Pmax
        full    : one group per timestep (Add all cuts no aggregation)
    """
    T = len(data.time_ranges[sample])
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

    elif cut_selection == "kmeans":
        groups, labels, X = make_kmeans_capacity_demand_groups(
            data=data,
            sample=sample,
            K=cut_selection_k,
            random_state=random_state,
        )
        info = {
            "labels": labels,
            "features": X,
            "group_sizes": [len(g) for g in groups],
        }

    elif cut_selection == "stress":
        groups, labels, stress = make_stress_bin_groups(
            data=data,
            sample=sample,
            K=cut_selection_k,
        )
        info = {
            "labels": labels,
            "stress": stress,
            "group_sizes": [len(g) for g in groups],
        }

    else:
        raise ValueError(
            f"Unknown cut_selection={cut_selection}. "
            "Choose from 'single', 'kmeans', 'stress', 'full'."
        )

    groups = [g for g in groups if len(g) > 0]
    info["group_sizes"] = [len(g) for g in groups]

    return groups, info

class BendersSolver():
    def __init__(self, gep_data, operational_data, sample, primal_net=None, dual_net=None, exact=True, 
                 exact_refinement=True, max_investment=100000, init_investment = "Zero", cut_selection="single",cut_selection_k=1,):
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

        self.cut_selection = cut_selection
        self.cut_selection_k = cut_selection_k
        self.cut_groups = None
        self.cut_group_info = None

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
        self.env = gp.Env(empty=True)
        self.env.setParam("OutputFlag", 0)
        self.env.start()


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

    def solve_matrix_problem(self, data, i, inv_decision=None):

        # env = gp.Env(empty=True)
        # env.setParam("OutputFlag",0)
        # env.start()

        # Create a new model
        m = gp.Model("Matrix problem", env=self.env)
        m.setParam('MIPGap', 1e-16)
        # m.setParam('FeasibilityTol', 1e-9)   # For constraint feasibility
        # m.setParam('IntFeasTol', 1e-9)   
        # m.setParam('OptimalityTol', 1e-9)    # For duality/optimality (LP)     

        # Create variables
        # x = m.addMVar(shape=data.ydim, vtype=GRB.CONTINUOUS, name="x")
        #! Important! We need the lb=-GRB.INFINITY, because otherwise the lower bound is automatically set to 0 by Gurobi.
        vtypes = np.array([GRB.INTEGER for _ in range(data.num_g)])
        # vtypes = np.array([GRB.CONTINUOUS for _ in range(data.num_g)])
        vtypes = np.concatenate((vtypes, np.array([GRB.CONTINUOUS for _ in range(data.ydim-data.num_g)])))
        x = m.addMVar(shape=data.ydim, lb=-GRB.INFINITY, vtype=vtypes, name="x")


        # Set objective
        obj = np.array(data.obj_coeff)
        m.setObjective(obj @ x, GRB.MINIMIZE)

        # Add ineq constraints

        # A = np.array(data.ineq_cm[i])
        # b = np.array(data.ineq_rhs[i])
        # m.addConstr(A @ x <= b, name="ineq")

        # # Add eq constraints
        # A = np.array(data.eq_cm[i])
        # b = np.array(data.eq_rhs[i])
        # m.addConstr(A @ x == b, name="eq")

        ineq_cm, ineq_rhs, eq_cm, eq_rhs = data.get_sample_matrices(i) # TODO: Added only for optimise dataset
        A = np.array(ineq_cm)
        b = np.array(ineq_rhs)
        m.addConstr(A @ x <= b, name="ineq")

        A = np.array(eq_cm)
        b = np.array(eq_rhs)
        m.addConstr(A @ x == b, name="eq")
        # For plotting
        if inv_decision is not None:
            m.addConstr(x[:data.num_g] == inv_decision)

        #! Enforce max investment
        # m.addConstr(x[:data.num_g] <= self.max_investment)

        # Optimize model
        m.optimize()

        # print(x.X)
        print(f"Obj: {m.ObjVal:g}")

        return x.X, m.ObjVal

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
        Solve either the master problem or an ED subproblem.

        If master=True:
            variables are [u_1, ..., u_G, alpha_1, ..., alpha_K]
            where K = num_alpha.

        If master=False:
            variables are the ED subproblem variables.
        """

        m = gp.Model("Matrix problem", env=self.env)
        m.setParam("OptimalityTol", 1e-9)

        ydim = obj.size

        if master:
            m.setParam("MIPGap", 1e-8)

            num_u = ydim - num_alpha

            vtypes = np.array(
                [GRB.INTEGER for _ in range(num_u)]
                + [GRB.CONTINUOUS for _ in range(num_alpha)]
            )

            x = m.addMVar(shape=ydim, lb=0, vtype=vtypes, name="x")

            if investment is not None:
                m.addConstr(x[:num_u] == np.array(investment, dtype=float))

        else:
            vtypes = np.array([GRB.CONTINUOUS for _ in range(ydim)])
            x = m.addMVar(shape=ydim, lb=-GRB.INFINITY, vtype=vtypes, name="x")

        obj = np.array(obj, dtype=float)
        m.setObjective(obj @ x, GRB.MINIMIZE)

        A = np.array(A_ineq, dtype=float)
        b = np.array(b_ineq, dtype=float)
        m.addConstr(A @ x <= b, name="ineq")

        if not master:
            A = np.array(A_eq, dtype=float)
            b = np.array(b_eq, dtype=float)
            m.addConstr(A @ x == b, name="eq")

        start_time = time.time()
        m.optimize()
        inference_time = time.time() - start_time

        if master:
            dual_val = []
        else:
            if m.status == GRB.OPTIMAL:
                dual_val = m.getAttr("Pi", m.getConstrs())
            else:
                print(f"Warning: Gurobi status = {m.status}. Cannot retrieve duals.")
                if m.status == 4:
                    m.computeIIS()
                    m.write("model_infeasible.ilp")
                    print("Wrote infeasible model to model_infeasible.ilp")
                raise RuntimeError("Subproblem not solved to optimality — duals unavailable.")

        return m.ObjVal, x.X, dual_val, inference_time

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

        return primal_obj_val, dual_obj_val, primal_sol.detach().numpy(), dual_sol.detach().numpy(), inference_time

    def solve_master_problem(self, data,compact,sample,investments,obj_val,benders_cuts, investment=None):
        # Solves the master problem in Benders decomposition
        # Returns the optimal objective function value in two parts: investment costs and value of alpha
        # And returns the optimal investment solution

        obj, A_ineq, b_ineq, A_eq, b_eq = self.find_master_problem_cm_rhs_obj(data,compact,sample,investments,obj_val,benders_cuts)

        obj_val, primal_val, dual_val, inference_time = self.solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, True, investment)

        obj_val_master = [obj_val-primal_val[-1], primal_val[-1]] # primal_val[-1] is the value of alpha

        # The new investments are the primal variables of the master problem, except for alpha
        new_investments = primal_val[:-1]

        return obj_val_master, new_investments, inference_time
    

    def solve_master_problem(
        self,
        data,
        compact,
        sample,
        investments,
        obj_val,
        benders_cuts,
        investment=None,
    ):
        """
        Solve the Benders master problem.

        Works for:
            single cut: one alpha
            grouped cuts: multiple alpha_k
        """

        obj, A_ineq, b_ineq, A_eq, b_eq = self.find_master_problem_cm_rhs_obj(
            data,
            compact,
            sample,
            investments,
            obj_val,
            benders_cuts,
        )

        if self.cut_selection == "single":
            num_alpha = 1
        else:
            if self.cut_groups is None:
                self.cut_groups, self.cut_group_info = make_cut_groups(
                    data=data,
                    sample=sample,
                    cut_selection=self.cut_selection,
                    cut_selection_k=self.cut_selection_k,
                )
            num_alpha = len(self.cut_groups)

        obj_val, primal_val, dual_val, inference_time = self.solve_matrix_problem_simple(
            obj,
            A_ineq,
            b_ineq,
            A_eq,
            b_eq,
            True,
            investment,
            num_alpha=num_alpha,
        )

        new_investments = primal_val[:data.num_g]
        alpha_vals = primal_val[data.num_g:data.num_g + num_alpha]

        investment_cost = float(obj[:data.num_g] @ new_investments)
        alpha_total = float(np.sum(alpha_vals))

        obj_val_master = [investment_cost, alpha_total]

        return obj_val_master, new_investments, inference_time

    def find_master_problem_cm_rhs_obj(
        self,
        data,
        compact,
        sample,
        investments,
        obj_val,
        benders_cuts,
    ):
        """
        Flexible master builder.

        If cut_selection == "single":
            master variables are [u_1, ..., u_G, alpha]

        If cut_selection in {"kmeans", "stress", "full"}:
            master variables are [u_1, ..., u_G, alpha_1, ..., alpha_K]

        Objective:
            min investment_cost + sum_k alpha_k
        """

        if self.cut_selection == "single":
            num_alpha = 1
        else:
            if self.cut_groups is None:
                self.cut_groups, self.cut_group_info = make_cut_groups(
                    data=data,
                    sample=sample,
                    cut_selection=self.cut_selection,
                    cut_selection_k=self.cut_selection_k,
                )
                print(f"Cut selection: {self.cut_selection}, groups={len(self.cut_groups)}")
                print("Group sizes:", self.cut_group_info["group_sizes"])

            num_alpha = len(self.cut_groups)

        # Objective: investment cost + sum alpha_k
        obj = data.obj_coeff[:data.num_g].detach().numpy()
        obj = np.concatenate((obj, np.ones(num_alpha)), axis=0)

        ineq_cm_sample, ineq_rhs_sample, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample)

        # Original investment lower-bound rows, usually -u_g <= 0
        A_ineq = ineq_cm_sample[:data.num_g, :data.num_g].detach().numpy()
        A_ineq = np.concatenate(
            (A_ineq, np.zeros((data.num_g, num_alpha))),
            axis=1,
        )
        b_ineq = ineq_rhs_sample[:data.num_g].detach().numpy()

        # Dummy equality, same convention as your old code
        A_eq = np.zeros((1, data.num_g + num_alpha))
        b_eq = np.zeros((1))

        # Lower bound for each alpha:
        # alpha_k >= -1e6  <=>  -alpha_k <= 1e6
        for k in range(num_alpha):
            lb_constraint = np.zeros((1, data.num_g + num_alpha))
            lb_constraint[0, data.num_g + k] = -1.0
            A_ineq = np.concatenate((A_ineq, lb_constraint), axis=0)
            b_ineq = np.concatenate((b_ineq, np.array([1e6])), axis=0)

        # Add Benders cuts
        for cut_lhs, cut_rhs in benders_cuts:
            A_ineq = np.concatenate((A_ineq, cut_lhs), axis=0)
            b_ineq = np.concatenate((b_ineq, np.array([cut_rhs])), axis=0)

        # Investment upper bounds
        rhs = 100000.0
        for g in range(data.num_g):
            ub_constraint = np.zeros((1, data.num_g + num_alpha))
            ub_constraint[0, g] = 1.0
            A_ineq = np.concatenate((A_ineq, ub_constraint), axis=0)
            b_ineq = np.concatenate((b_ineq, np.array([rhs])), axis=0)

        return obj, A_ineq, b_ineq, A_eq, b_eq

    def find_subproblem_cm_rhs_obj_from_mats(
        self,
        data,
        compact,
        investments,
        time_step,
        ineq_cm_sample,
        ineq_rhs_sample,
        eq_cm_sample,
        eq_rhs_sample,
    ):
        '''
        More efficient way to find the constraint matrix
        '''
        num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)
        num_rows_per_t_eq = data.num_n
        num_columns_per_t = data.n_var_per_t
        columns_ui = range(data.num_g)

        column_index = data.num_g + time_step * num_columns_per_t
        obj = self.operational_data.obj_coeff.detach().numpy() * self.pWeight

        if compact:
            obj = np.concatenate((np.zeros(data.num_g), obj), axis=0)

        row_index_ineq = data.num_g + time_step * num_rows_per_t_ineq
        A_ineq = ineq_cm_sample[
            row_index_ineq:row_index_ineq + num_rows_per_t_ineq,
            column_index:column_index + num_columns_per_t
        ].detach().numpy()

        if compact:
            A_ineq = np.concatenate((
                ineq_cm_sample[
                    row_index_ineq:row_index_ineq + num_rows_per_t_ineq,
                    columns_ui
                ].detach().numpy(),
                A_ineq
            ), axis=1)

        b_ineq = ineq_rhs_sample[
            row_index_ineq:row_index_ineq + num_rows_per_t_ineq
        ].clone().detach().numpy()

        if not compact:
            for g in range(data.num_g):
                upper_bound_p = investments[g] * -ineq_cm_sample[row_index_ineq + data.num_g + g, g]
                b_ineq[data.num_g + g] = upper_bound_p

        row_index_eq = time_step * num_rows_per_t_eq
        A_eq = eq_cm_sample[
            row_index_eq:row_index_eq + num_rows_per_t_eq,
            column_index:column_index + num_columns_per_t
        ]

        if compact:
            A_eq = np.concatenate((
                eq_cm_sample[row_index_eq:row_index_eq + num_rows_per_t_eq, columns_ui],
                A_eq
            ), axis=1)

        b_eq = eq_rhs_sample[row_index_eq:row_index_eq + num_rows_per_t_eq]

        if compact:
            ui_g = np.eye(data.num_g)
            ui_g = np.concatenate((ui_g, np.zeros((data.num_g, num_columns_per_t))), axis=1)
            A_eq = np.concatenate((A_eq, ui_g), 0)
            b_eq = np.concatenate((b_eq, investments), 0)

        return obj, A_ineq, b_ineq, A_eq, b_eq
    
    def find_benders_cut_batch_for_group(
        self,
        data,
        compact,
        sample,
        dual_vals,
        b_ineqs,
        b_eqs,
        timestep_indices,
        alpha_index,
        num_alpha,
    ):
        """
        Build one grouped Benders cut for a subset of timesteps.

        Master variables:
            [u_1, ..., u_G, alpha_1, ..., alpha_K]

        Cut convention:
            cut_lhs @ [u, alpha] <= cut_rhs

        with coefficient -1 on alpha_{alpha_index}.
        """

        if compact:
            raise NotImplementedError("Grouped cuts currently support compact=False only.")

        ineq_cm_sample, _, _, _ = data.get_sample_matrices(sample)

        num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)
        timestep_indices = np.array(timestep_indices, dtype=int)

        benders_cut_lhs = np.zeros((1, data.num_g + num_alpha))
        benders_cut_rhs = 0.0

        # This group's alpha
        benders_cut_lhs[0, data.num_g + alpha_index] = -1.0

        # LHS: investment coefficients
        for g in range(data.num_g):
            dual_idx = data.num_g + g

            ineq_row_indices = np.array([
                data.num_g + t * num_rows_per_t_ineq + data.num_g + g
                for t in timestep_indices
            ])

            ui_coeffs = (
                dual_vals[timestep_indices, dual_idx]
                * -ineq_cm_sample[ineq_row_indices, g].detach().numpy()
            )

            benders_cut_lhs[0, g] = np.sum(ui_coeffs)

        # RHS: inequality RHS terms with nonzero RHS
        constraint_nrs = []

        # line flow lower bounds
        constraint_nrs += [2 * data.num_g + l for l in range(data.num_l)]

        # line flow upper bounds
        constraint_nrs += [2 * data.num_g + data.num_l + l for l in range(data.num_l)]

        # missed demand upper bounds
        constraint_nrs += [
            2 * data.num_g + 2 * data.num_l + data.num_n + n
            for n in range(data.num_n)
        ]

        constraint_nrs = np.array(constraint_nrs)

        ineq_duals = dual_vals[np.ix_(timestep_indices, constraint_nrs)]
        ineq_rhs = b_ineqs[np.ix_(timestep_indices, constraint_nrs)]

        benders_cut_rhs += -np.sum(ineq_duals * ineq_rhs)

        # RHS: equality RHS terms
        eq_dual_start = num_rows_per_t_ineq
        eq_duals = dual_vals[
            timestep_indices,
            eq_dual_start:eq_dual_start + data.num_n,
        ]

        benders_cut_rhs += -np.sum(eq_duals * b_eqs[timestep_indices])

        return benders_cut_lhs, benders_cut_rhs
    

    def find_benders_cuts_grouped_batch(
        self,
        data,
        compact,
        sample,
        dual_vals,
        b_ineqs,
        b_eqs,
    ):
        """
        Return a list of Benders cuts.

        If cut_selection == "single":
            returns [one aggregated cut]

        If cut_selection in {"kmeans", "stress", "full"}:
            returns one cut per timestep group
        """

        if self.cut_selection == "single":
            cut = self.find_benders_cut_batch(
                data,
                compact,
                sample,
                dual_vals,
                b_ineqs,
                b_eqs,
            )
            return [cut]

        if self.cut_groups is None:
            self.cut_groups, self.cut_group_info = make_cut_groups(
                data=data,
                sample=sample,
                cut_selection=self.cut_selection,
                cut_selection_k=self.cut_selection_k,
            )

            print(f"Cut selection: {self.cut_selection}, groups={len(self.cut_groups)}")
            print("Group sizes:", self.cut_group_info["group_sizes"])

        num_alpha = len(self.cut_groups)

        cuts = []

        for k, group in enumerate(self.cut_groups):
            cut = self.find_benders_cut_batch_for_group(
                data=data,
                compact=compact,
                sample=sample,
                dual_vals=dual_vals,
                b_ineqs=b_ineqs,
                b_eqs=b_eqs,
                timestep_indices=group,
                alpha_index=k,
                num_alpha=num_alpha,
            )
            cuts.append(cut)

        return cuts
    
    def solve_subproblems(self, data, compact, sample, investments, exact=True):
        time_range = data.time_ranges[sample]
        num_timesteps = len(time_range)

        # Load sample matrices ONCE
        ineq_cm_sample, ineq_rhs_sample, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample)

        if exact:
            obj_vals = []
            primal_vals = []
            dual_vals = []
            inference_times = []

            # Still need b_ineqs / b_eqs for the batch cut at the end
            b_ineqs = []
            b_eqs = []

            for time_step in range(num_timesteps):
                obj, A_ineq, b_ineq, A_eq, b_eq = self.find_subproblem_cm_rhs_obj_from_mats(
                    data=data,
                    compact=compact,
                    investments=investments,
                    time_step=time_step,
                    ineq_cm_sample=ineq_cm_sample,
                    ineq_rhs_sample=ineq_rhs_sample,
                    eq_cm_sample=eq_cm_sample,
                    eq_rhs_sample=eq_rhs_sample,
                )

                obj_val, primal_val, dual_val, inference_time = self.solve_matrix_problem_simple(
                    obj, A_ineq, b_ineq, A_eq, b_eq, False
                )

                obj_vals.append(obj_val)
                primal_vals.append(primal_val)
                dual_vals.append(dual_val)
                inference_times.append(inference_time)
                b_ineqs.append(b_ineq)
                b_eqs.append(b_eq)

            primal_obj_val_total = np.sum(obj_vals)
            dual_obj_val_total = primal_obj_val_total
            inference_time_total = np.sum(inference_times)

            # self.objs_all.append(np.array(obj_vals))
            # self.dual_solutions_all.append(np.array(dual_vals))
            # self.primal_solutions_all.append(np.array(primal_vals))

            b_ineqs_np = np.stack(b_ineqs)
            b_eqs_np = np.stack(b_eqs)

        else:
            # For PDL, only build what is needed for X and the cut
            b_ineqs = []
            b_eqs = []

            num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)
            num_rows_per_t_eq = data.num_n
            num_columns_per_t = data.n_var_per_t

            for time_step in range(num_timesteps):
                row_index_ineq = data.num_g + time_step * num_rows_per_t_ineq
                row_index_eq = time_step * num_rows_per_t_eq

                b_ineq = ineq_rhs_sample[
                    row_index_ineq:row_index_ineq + num_rows_per_t_ineq
                ].clone().detach().numpy()

                for g in range(data.num_g):
                    upper_bound_p = investments[g] * -ineq_cm_sample[row_index_ineq + data.num_g + g, g]
                    b_ineq[data.num_g + g] = upper_bound_p

                b_eq = eq_rhs_sample[
                    row_index_eq:row_index_eq + num_rows_per_t_eq
                ]

                b_ineqs.append(b_ineq)
                b_eqs.append(b_eq)

            b_ineqs_np = np.stack(b_ineqs)
            b_eqs_np = np.stack(b_eqs)

            X = torch.tensor(
                np.concatenate(
                    [b_eqs_np, b_ineqs_np[:, self.operational_data.capacity_ub_indices]],
                    axis=1
                )
            )
            # self.X_all.append(X) # Save memory so we do not append X

            primal_obj_val_total, dual_obj_val_total, primal_vals, dual_vals, inference_time_total = self.solve_matrix_problem_PDL(X)

        benders_cuts = self.find_benders_cuts_grouped_batch(
            data=data,
            compact=compact,
            sample=sample,
            dual_vals=np.array(dual_vals),
            b_ineqs=b_ineqs_np,
            b_eqs=b_eqs_np,
        )

        print(f"Inference time: {inference_time_total}")

        return primal_obj_val_total, dual_obj_val_total, benders_cuts, inference_time_total

    # def solve_subproblems(self, data, compact, sample, investments, exact=True):
    #     # Solves the subproblems in Benders decomposition
    #     # Returns the optimal objective function value of the subproblems for all time periods added together
    #     # And returns the dual values of the ui_g = investment constraints

    #     # Calculate information about subproblem sizes
    #     time_range = data.time_ranges[sample]
    #     num_timesteps = len(time_range)

    #     objs = []
    #     A_ineqs = []
    #     b_ineqs = []
    #     A_eqs = []
    #     b_eqs = []

    #     for time_step in range(num_timesteps):
    #         # Find constraint matrices, right hand side vectors and objective vector of subproblem
    #         obj, A_ineq, b_ineq, A_eq, b_eq = self.find_subproblem_cm_rhs_obj(data,compact,sample,investments,time_step)

    #         objs.append(obj)
    #         A_ineqs.append(A_ineq)
    #         b_ineqs.append(b_ineq)
    #         A_eqs.append(A_eq)
    #         b_eqs.append(b_eq)

    #     obj_vals = []
    #     primal_vals = []
    #     dual_vals = []
    #     inference_times = []
    #     # Save X to investigate the distribution of the data samples generated by Benders decomposition
    #     b_eqs_np = np.stack(b_eqs)
    #     b_ineqs_np = np.stack(b_ineqs)
    #     X = torch.tensor(np.concatenate([b_eqs_np, b_ineqs_np[:, self.operational_data.capacity_ub_indices]], axis=1))
    #     self.X_all.append(X)
    #     if exact:
    #         for obj, A_ineq, b_ineq, A_eq, b_eq in zip(objs, A_ineqs, b_ineqs, A_eqs, b_eqs):
    #             obj_val, primal_val, dual_val, inference_time = self.solve_matrix_problem_simple(obj, A_ineq, b_ineq, A_eq, b_eq, False)
    #             obj_vals.append(obj_val)
    #             primal_vals.append(primal_val)
    #             dual_vals.append(dual_val)
    #             inference_times.append(inference_time)
    #         # Add objective value to the total
    #         primal_obj_val_total = np.sum(obj_vals)
    #         inference_time_total = np.sum(inference_times)
    #         dual_obj_val_total = primal_obj_val_total
    #         self.objs_all.append(np.array(obj_vals))
    #         self.dual_solutions_all.append(np.array(dual_vals))
    #         self.primal_solutions_all.append(np.array(primal_vals))
    #     else:
    #         primal_obj_val_total, dual_obj_val_total, primal_vals, dual_vals, inference_time_total = self.solve_matrix_problem_PDL(X)

    #     benders_cut_lhs, benders_cut_rhs = self.find_benders_cut_batch(data, compact, sample, np.array(dual_vals), np.array(b_ineqs), np.array(b_eqs))
            
    #     # Obtain the final Benders cut of all subproblems together
    #     benders_cut = benders_cut_lhs, benders_cut_rhs
    #     print(f"Benders cut: {benders_cut}")
    #     print(f"Inference time: {inference_time_total}")
    #     return primal_obj_val_total, dual_obj_val_total, benders_cut, inference_time_total

    def find_subproblem_cm_rhs_obj(self, data,compact,sample,investments,time_step):
        ineq_cm_sample, ineq_rhs_sample, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample) # TODO: Added only for optimise dataset
        # Calculate information about subproblem sizes
        num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n 
        num_rows_per_t_eq = data.num_n # energy balance equality for each node
        num_columns_per_t = data.n_var_per_t
        columns_ui = range(data.num_g)

        # Find objective of subproblem
        column_index = data.num_g + time_step*num_columns_per_t # first g columns are ui_g variables
        # obj = data.obj_coeff[column_index:column_index + num_columns_per_t].detach().numpy() # sample index is not needed, obj is same for all samples
        obj = self.operational_data.obj_coeff.detach().numpy() * self.pWeight

        if compact:
            obj = np.concatenate((np.zeros(data.num_g), obj), axis=0) # add 0 for ui_g variables

        # Find constraint ineq submatrices
        row_index = data.num_g + time_step*num_rows_per_t_ineq # first g rows are 3.1k constraints
        # A_ineq = data.ineq_cm[sample,row_index:row_index + num_rows_per_t_ineq,column_index:column_index + num_columns_per_t] #take submatrix of time_step
        # if compact:
        #     A_ineq = np.concatenate((data.ineq_cm[sample,row_index:row_index + num_rows_per_t_ineq,columns_ui],A_ineq), axis=1) # add ui_g columns

        A_ineq = ineq_cm_sample[row_index:row_index + num_rows_per_t_ineq,
                        column_index:column_index + num_columns_per_t].detach().numpy()
        if compact:
            A_ineq = np.concatenate((
                ineq_cm_sample[row_index:row_index + num_rows_per_t_ineq, columns_ui].detach().numpy(),
                A_ineq
            ), axis=1)

        b_ineq = ineq_rhs_sample[row_index:row_index + num_rows_per_t_ineq].clone().detach().numpy() # TODO: added for optimise dataset

        # Find constraint ineq rhs
        #! Beware for ineq rhs, we need to clone it, otherwise it will be a view and we will modify the original data.ineq_rhs

        # b_ineq = data.ineq_rhs[sample,row_index:row_index + num_rows_per_t_ineq].clone()    # ORIGNAL CODE
        if not compact:
            # Replace investment variables with constants in right hand side, NOT needed in compact form
            for g in range(data.num_g):
                #upper_bound_p = investments[g]*-data.ineq_cm[sample,row_index+ data.num_g + g,g] #first g constraints are 3.1c, we want to take 3.1b coeff of ui_g
                upper_bound_p = investments[g] * -ineq_cm_sample[row_index + data.num_g + g, g] # TODO: added for optimise dataset
                b_ineq[data.num_g+g] = upper_bound_p  # second set of g constraints are 3.1b, we want to replace rhs 0 of 3.1b with upper_bound_p
                
        # Find constraint eq submatrices
        row_index = time_step*num_rows_per_t_eq
        # A_eq = data.eq_cm[sample,row_index:row_index + num_rows_per_t_eq,column_index:column_index + num_columns_per_t] #take submatrix of time_step
        # if compact:
        #     A_eq = np.concatenate((data.eq_cm[sample,row_index:row_index + num_rows_per_t_eq,columns_ui],A_eq), axis=1) # add ui_g columns
        # b_eq = data.eq_rhs[sample,row_index:row_index + num_rows_per_t_eq]

        A_eq = eq_cm_sample[row_index:row_index + num_rows_per_t_eq,
                            column_index:column_index + num_columns_per_t]
        if compact:
            A_eq = np.concatenate((
                eq_cm_sample[row_index:row_index + num_rows_per_t_eq, columns_ui],
                A_eq
            ), axis=1)
        b_eq = eq_rhs_sample[row_index:row_index + num_rows_per_t_eq]  # TODO: added for optimise dataset

        # Fix investments: add constraint ui_g = investments, ONLY in compact form
        if compact:
            ui_g = np.eye(data.num_g)
            ui_g = np.concatenate((ui_g,np.zeros((data.num_g,num_columns_per_t))), axis=1)
            A_eq = np.concatenate((A_eq,ui_g),0)
            b_eq = np.concatenate((b_eq,investments),0) 

        return obj, A_ineq, b_ineq, A_eq, b_eq

    def find_benders_cut(self, data, compact, sample, investments, old_benders_cut, time_step, b_ineq, b_eq, obj_val, dual_val):
        ineq_cm_sample, ineq_rhs_sample, eq_cm_sample, eq_rhs_sample = data.get_sample_matrices(sample) # TODO: Added only for optimise dataset
        benders_cut_lhs = old_benders_cut[0]
        benders_cut_rhs = old_benders_cut[1]

        # Find the coefficients of ui_g (lhs of Benders cut)
        for g in range(data.num_g):
            if compact:
                # Add dual variables of ui_g = investments constraint (last g equalities)
                coeff_ui = dual_val[-(data.num_g-g)]
            else:
                # Add dual term for upperbound on p constraint
                num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n 
                # coefficient of ui_g is - sum(pi_g,t * GA_g,t for t) * UCAP_g
                # we take this from 3.1b constraint in the original problem
                row_index = data.num_g + time_step*num_rows_per_t_ineq + data.num_g # we want the 3.1b constraints
                # First g constraints are lower bound, we want the upper bound, therefore offset of num_g.
                # coeff_ui = dual_val[data.num_g + g] * -data.ineq_cm[sample,row_index+g,g]
                coeff_ui = dual_val[data.num_g + g] * -ineq_cm_sample[row_index + g, g] # TODO: added for optimise dataset

            benders_cut_lhs[0,g] = benders_cut_lhs[0,g] + coeff_ui

        # Compute right hand side of Benders cut
        if compact:
            # Add objective of subproblem
            benders_cut_rhs += -obj_val
            # Add dual term for ui_g = investments constraint
            for g in range(data.num_g):
                # rhs is - dual * -investment
                benders_cut_rhs += - dual_val[-(data.num_g-g)]*-investments[g]
        else:
            # Create array of constraint nr's of inequalties of which we want to include the dual term (3.1d,3.1e,3.1j)
            # because we only need to consider the constraints of which the rhs is not 0
            constraint_nrs = []
            constraint_nrs.extend([2*data.num_g+l for l in range(data.num_l)]) # 3.1d: Lineflow lower bound
            constraint_nrs.extend([2*data.num_g+data.num_l+l for l in range(data.num_l)]) # 3.1e: Lineflow upper bound
            constraint_nrs.extend([2*data.num_g+2*data.num_l+data.num_n+n for n in range(data.num_n)]) # 3.1j: Missed demand upper bound

            # Add dual term for inequalities 
            for constraint_nr in constraint_nrs:
                benders_cut_rhs += -dual_val[constraint_nr] * b_ineq[constraint_nr]
                
            # Add dual term for equalities
            num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for p_g, f_l and md_n
            for constraint_nr in range(data.num_n):
                benders_cut_rhs += -dual_val[num_rows_per_t_ineq+constraint_nr] * b_eq[constraint_nr]
        
        new_benders_cut = benders_cut_lhs, benders_cut_rhs

        return new_benders_cut

    def find_benders_cut_batch(self, data, compact, sample, dual_vals, b_ineqs, b_eqs):
        """
        Vectorized Benders cut aggregation (non-compact case) over time steps.
        Returns the full cut (lhs, rhs) as a tuple.
        """
        ineq_cm_sample, _, _, _ = data.get_sample_matrices(sample) # TODO: Added only for optimise dataset
        T = dual_vals.shape[0]
        num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n) # lower and upper bounds for production, lineflow and missed demand

        # Initialize cut terms
        benders_cut_lhs = np.zeros((1, data.num_g+1)) # coefficients for ui_g and for alpha
        benders_cut_rhs = 0

        benders_cut_lhs[0,-1] = -1 # coeff for alpha: -1

        # LHS: Sum dual contributions to ui_g variables
        for g in range(data.num_g):
            # Index of upper bound on p_g at time t
            dual_indices = data.num_g + g  # First num_g are lower bounds, we want upper bound
            ineq_row_indices = np.array([
                data.num_g + t * num_rows_per_t_ineq + data.num_g + g for t in range(T)
            ])
            #ui_coeffs = dual_vals[:, dual_indices] * -data.ineq_cm[sample, ineq_row_indices, g].detach().numpy()  # Multiply by the dual variables of production upper bound
            ui_coeffs = dual_vals[:, dual_indices] * -ineq_cm_sample[ineq_row_indices, g].detach().numpy() # TODO: added for optimise dataset
            benders_cut_lhs[0, g] = np.sum(ui_coeffs) # Sum over all subproblems

        # RHS: Add dual contributions from inequality RHS
        constraint_nrs = []
        # 3.1d: Line flow lower bounds
        constraint_nrs += [2 * data.num_g + l for l in range(data.num_l)]
        # 3.1e: Line flow upper bounds
        constraint_nrs += [2 * data.num_g + data.num_l + l for l in range(data.num_l)]
        # 3.1j: Missed demand upper bounds
        constraint_nrs += [2 * data.num_g + 2 * data.num_l + n for n in range(data.num_n)]

        constraint_nrs = np.array(constraint_nrs)  # shape: (C,)
        ineq_duals = dual_vals[:, constraint_nrs]     # shape: (T, C)
        ineq_rhs = b_ineqs[:, constraint_nrs]         # shape: (T, C)
        benders_cut_rhs += -np.sum(ineq_duals * ineq_rhs)

        # RHS: Add dual contributions from equality RHS
        eq_dual_start = num_rows_per_t_ineq
        eq_duals = dual_vals[:, eq_dual_start:eq_dual_start + data.num_n]  # shape: (T, N)
        benders_cut_rhs += -np.sum(eq_duals * b_eqs)

        return benders_cut_lhs, benders_cut_rhs



    def solve_with_benders(self, data, compact, sample):

        # Create lists for algorithm
        investments_all = [] # list of tensors of size (num_g), one for every iteration
        obj_val_subproblems_all = [] # list of floats, one for every iteration
        benders_cut_all = [] # list of benders cuts ([lhs],rhs), one for every iteration

        # Parameters for Benders algorithm
        epsilon = 1e-6

        # Start Benders algorithm
        optimal = False
        i = 0
        while not optimal and i < 1000:
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

            # Add investment values of current iteration to list
            print("The investment decisions are", investments_iter_k)
            investments_iter_k = torch.tensor(investments_iter_k)

            self.inv_hist.append(investments_iter_k.detach().cpu().numpy().tolist())

            if self.exact == False and i > 0 and torch.allclose(investments_iter_k.to(torch.float64), investments_all[-1].to(torch.float64), atol=1e-6):
                print("!! Investments are the same as last iteration")
                if self.exact_refinement:
                    self.exact = True
                else:
                    print("Stopping Benders decomposition because exact refinement is not used.")
                    print("Upper bound:", self.best_upper_bound) #! Return the best upper bound found so far if exact refinement is not used
                    lower_bound = obj_val_master[0] + obj_val_master[1]
                    print("Lower bound:", lower_bound)
                    print(f"Duality gap: {(self.best_upper_bound - lower_bound)/np.abs(self.best_upper_bound)}")
                    break
            else:
                # if self.exact == False:
                #     # print("!! !! Different investments than last iteration")
                investments_all.append(investments_iter_k)
            
            if self.exact:
                self.exact_iterations += 1
            else:
                self.inexact_iterations += 1
            print(f"Before Subproblem SOlve")
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
            lower_bound = obj_val_master[0] + obj_val_master[1]
            upper_bound = obj_val_master[0] + primal_obj_val_total
            print(f"UB={upper_bound:.4f}, LB={lower_bound:.4f}, ")
            # --- LOG UB/LB PER ITERATION ---
            self.ub_hist.append(float(upper_bound))
            self.lb_hist.append(float(lower_bound))
            self.iter_hist.append(int(i))
            self.exact_flag_hist.append(bool(self.exact))
            self.sub_time_hist.append(float(inference_time_subproblems_total))
            # master time only exists when i>0 in your code
            if i == 0:
                self.master_time_hist.append(0.0)
            else:
                self.master_time_hist.append(float(inference_time_master))

            # Check for optimality
            if self.exact:
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


                if upper_bound - lower_bound < epsilon:
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
                    benders_cut_all.extend(benders_cuts)
                    print(f"Subproblems solved. Added {len(benders_cuts)} Benders cuts.")
                    # print('Subproblems solved. Benders_cut:',benders_cut)
            else:
                    # Add Benders cut of current iteration to list
                    benders_cut_all.extend(benders_cuts)
                    print(f"Subproblems solved. Added {len(benders_cuts)} Benders cuts.")

            i += 1
            
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--solve-direct",
        action="store_true",
        help="Solve the full GEP directly with Gurobi, without Benders decomposition.",
        default=False
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.json",
        help="Path to run config JSON file, e.g. config.json, config-5node.json, config-6node.json"
    )
    args_cli = parser.parse_args()



    print("Parsing the config file")
    RUN_CONFIG_FILE = args_cli.config
    NumNode = None
    if RUN_CONFIG_FILE == "config.json":
        NumNode = 3
    elif RUN_CONFIG_FILE == "config-4node.json":
        NumNode = 4
    elif RUN_CONFIG_FILE == "config-5node.json":
        NumNode = 5
    elif RUN_CONFIG_FILE == "config-6node.json":
        NumNode = 6
    else:
        raise ValueError("Invalid config file name.")

    data = parse_config(CONFIG_FILE_NAME)
    experiment = data["experiment"]
    outputs_config = data["outputs_config"]


    with open(RUN_CONFIG_FILE, "r") as file:
        args = json.load(file)
    
    print(args)





    # Train the model:
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
            ed_data_save_path = (f"data/ED_data/ED_N{nodes_str}_G{gens_str}_{lines_str}"
                            f"_c{int(benders_args['benders_compact'])}"
                            f"_s{int(benders_args['scale_problem'])}"
                            f"_p{int(benders_args['perturb_operating_costs'])}"
                            f"_smp{benders_args['2n_synthetic_samples']}.pkl")

            gep_data_save_path = f"data/GEP_data/sample_duration:{benders_args['sample_duration']}_N:{nodes_str}_G:{gens_str}_L:{lines_str}.pkl"
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

            # !Load primal and dual net
            # primal_net_directory = "experiment-output/ch7/3nodes/primal_model"
            # dual_net_directory = "experiment-output/ch7/3nodes/dual_model"
            # primal_net_directory = "outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalCompletionClassification/repeat:0"
            # dual_net_directory = "outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalCompletionClassification/repeat:0"
            if not args_cli.solve_direct:
                if "primal_net_directory" in args["Benders_args"]:
                    primal_net_directory = args["Benders_args"]["primal_net_directory"]
                else:
                    raise ValueError("Please provide a directory for the primal net in the config file under Benders_args with key 'primal_net_directory'")
                    primal_net_directory = "outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalCompletionClassification/repeat:0"
                
                if "dual_net_directory" in args["Benders_args"]:
                    dual_net_directory = args["Benders_args"]["dual_net_directory"]
                else:
                    raise ValueError("Please provide a directory for the dual net in the config file under Benders_args with key 'dual_net_directory'")
                    dual_net_directory = "outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalCompletionClassification/repeat:0"
                print(f"Primal Net Directory: {primal_net_directory}")
                print(f"Dual Net Directory: {dual_net_directory}")
                primal_model_args = json.load(open(os.path.join(primal_net_directory, "args.json")))
                dual_model_args = json.load(open(os.path.join(dual_net_directory, "args.json")))
                
                best_args = {'primal_lr': 0.0006785456069117277, 'hidden_size_factor': 28, 'n_layers': 2, 'decay': 0.9989743016070536, 'batch_size': 2048}  #! Temporary, for primal net
                primal_model_args["primal_lr"] = best_args["primal_lr"]
                primal_model_args["hidden_size_factor"] = best_args["hidden_size_factor"]
                primal_model_args["n_layers"] = best_args["n_layers"]
                primal_model_args["decay"] = best_args["decay"]
                primal_model_args["batch_size"] = best_args["batch_size"]
                primal_net = PrimalNetEndToEnd(primal_model_args, operational_data)
                if args["dual_classification"]:
                    dual_net = DualClassificationNetEndToEnd(dual_model_args, operational_data)
                else:
                    dual_net = DualNetEndToEnd(dual_model_args, operational_data)
                primal_net.load_state_dict(torch.load(os.path.join(primal_net_directory, "primal_weights.pth"), weights_only=True), strict = False)
                dual_net.load_state_dict(torch.load(os.path.join(dual_net_directory, "dual_weights.pth"), weights_only=True), strict = False)
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
            # sample = 0
            for (start_exact, exact_refinement) in benders_setups:
                all_results = []
                for repeat in range(1):
                    for sample in range(samples):
                        if args_cli.solve_direct:
                            # Solve Directly with Solver
                            primal_net = None
                            dual_net = None
                            print(f"Solving sample {sample} directly with Gurobi without Benders decomposition.")
                            solver = BendersSolver(gep_data=gep_data, operational_data=operational_data, 
                                                   primal_net=primal_net, dual_net=dual_net, sample=sample, 
                                                   exact=start_exact, exact_refinement=exact_refinement,
                                                    max_investment=benders_args["max_investment"],
                                                   init_investment=benders_args["init_investment"],
                                                    cut_selection=benders_args["cut_selection"],
                                                    cut_selection_k=benders_args["cut_selection_k"],)
                            start_time_direct = time.time()
                            y, obj = solver.solve_matrix_problem(gep_data, sample, inv_decision=None)
                            total_time_direct = time.time() - start_time_direct

                            print(f"Direct exact GEP optimum: {obj}")
                            print(f"Direct investment decision: {y[:gep_data.num_g]}")
                            print(f"Direct total time: {total_time_direct}")

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
                                "total_time": total_time_direct,
                                "total_time_master": 0.0,
                                "total_time_subproblem_exact": 0.0,
                                "total_time_subproblem_pdl": 0.0,
                                "investments": y[:gep_data.num_g].tolist()
                            }
                            all_results.append(result)
                            continue

                        else:
                            solver = BendersSolver(gep_data=gep_data, operational_data=operational_data, primal_net=primal_net, dual_net=dual_net, sample=sample, 
                                                   exact=start_exact, exact_refinement=exact_refinement, max_investment=benders_args["max_investment"],
                                                   init_investment=benders_args["init_investment"],
                                                    cut_selection=benders_args["cut_selection"],
                                                    cut_selection_k=benders_args["cut_selection_k"],)
        
                            # Solve for the ground truth
                            y, obj = solver.solve_matrix_problem(gep_data, sample) # solution = Obj: 2374.99

                            # Solve single sample with Benders decomposition
                            # sample = 1 # solution = Obj: 2374.99
                            # compact = False
                            # Solving a subproblem
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
                            })
                            iter_df["investment"] = [json.dumps(v) for v in solver.inv_hist]
                            specific_name = args["Benders_args"].get("specific_name", "")
                            benders_setup_str = args["Benders_args"].get("benders_setup", "")
                            # if samples == 1:
                            #     out_dir = f"outputs/Benders/{NumNode}Node/Full_Time/iter_logs_{benders_setup_str}_{specific_name}"
                            # else:
                            out_dir = f"outputs/Benders/{NumNode}Node/Sample_{benders_args['sample_duration']}/iter_logs_{benders_setup_str}_{specific_name}"

                            os.makedirs(out_dir, exist_ok=True)
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

                            obj_val_master, investments_iter_k, inference_time_master = solver.solve_master_problem(gep_data,False,sample,investments_all,None,benders_cuts_all, investment=y[:gep_data.num_g])

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
                                # Optionally, add investments or other details:
                                # "investments": y[:gep_data.num_g].tolist()
                                "investments": investments_all[-1].tolist() if len(investments_all) > 0 else None,
                            }
                            all_results.append(result)
                            # break
                        
                #! Set to True if saving data.
                if True:
                    experiment_data_df = pd.DataFrame(all_results)
                    if not os.path.exists(args["Benders_args"]["exp_save_directory"]):
                        os.makedirs(args["Benders_args"]["exp_save_directory"])
                    
                    if args_cli.solve_direct:
                        specific_name = "direct_exact"
                        out_dir = f"outputs/Benders/{NumNode}Node/Sample_{benders_args['sample_duration']}"
                        os.makedirs(out_dir, exist_ok=True)
                        data_save_path = os.path.join(args["Benders_args"]["exp_save_directory"], f"Sample_{str(benders_args['sample_duration'])}", f"Gurobi_Solution.csv")
                
                    else:
                        specific_name = args["Benders_args"].get("specific_name", "")
                        
                        if samples == 1:
                            data_save_path = os.path.join(args["Benders_args"]["exp_save_directory"], f"Sample_{str(benders_args['sample_duration'])}",f"experiment_data_full_time_sample_duration:{benders_args['sample_duration']}_start_exact:{start_exact}_exact_refinement:{exact_refinement}_{specific_name}.csv")
                        else:
                            data_save_path = os.path.join(args["Benders_args"]["exp_save_directory"], f"Sample_{str(benders_args['sample_duration'])}", f"experiment_data_sample_duration:{benders_args['sample_duration']}_start_exact:{start_exact}_exact_refinement:{exact_refinement}_{specific_name}.csv")
                        
                    experiment_data_df.to_csv(data_save_path, index=False)


            # ! Plotting optimality gap per iteration
            if not start_exact:
                # Plot optimality gap per iteration
                # Only works for last sample
                tab10 = plt.get_cmap("tab10")
                primal_color = tab10(0)  # blue
                dual_color = tab10(1)    # orange

                plt.figure(figsize=(8, 5))
                plt.rcParams.update({
                    "axes.titlesize": 20,
                    "axes.labelsize": 18,
                    "xtick.labelsize": 16,
                    "ytick.labelsize": 16,
                    "legend.fontsize": 16,
                    "font.size": 16
                })

                # plt.plot(np.array(solver.primal_opt_gap_all)*100, label="Primal opt gap", color=primal_color, linewidth=2, marker='o')
                # plt.plot(np.array(solver.dual_opt_gap_all)*100, label="Dual opt gap", color=dual_color, linewidth=2, marker='o')

                # plt.xlabel("Benders Iteration")
                # plt.ylabel("Optimality Gap (%)")
                # plt.title("Primal and Dual Optimality Gap per Benders Iteration")
                # # Add line at 0
                # plt.axhline(0, color='black', linewidth=1)
                # plt.legend(loc='best', frameon=True)
                # plt.grid(True, linestyle='--', alpha=0.6)
                # plt.tight_layout()
                # plt.savefig("experiment-output/ch7/3nodes/benders_test_data_exact.pdf", dpi=300, bbox_inches='tight')
                # plt.show()


'''
PseudoCode
'''

