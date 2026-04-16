import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from gep_benders import BendersSolver
import torch

def load_investment_path(csv_path):
    df = pd.read_csv(csv_path)
    invs = []
    for s in df["investment"]:
        inv = np.array(ast.literal_eval(s), dtype=float)
        inv[np.abs(inv) < 1e-8] = 0.0
        invs.append(inv)
    return df, invs


def collect_exact_benders_lambdas_for_investments(
    solver,
    data,
    compact,
    sample,
    investments_list,
):
    all_lambdas = []
    all_iters = []

    for k, inv in enumerate(investments_list):
        time_range = data.time_ranges[sample]
        T = len(time_range)

        lambdas_this_inv = []

        for t in range(T):
            obj, A_ineq, b_ineq, A_eq, b_eq = solver.find_subproblem_cm_rhs_obj(
                data, compact, sample, torch.tensor(inv, dtype=torch.float64), t
            )

            obj_val, primal_val, dual_val, solve_time = solver.solve_matrix_problem_simple(
                obj, A_ineq, b_ineq, A_eq, b_eq, master=False
            )

            dual_val = np.asarray(dual_val, dtype=float)

            num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)

            # lambda are the equality duals, one per node
            lamb = dual_val[num_rows_per_t_ineq:num_rows_per_t_ineq + data.num_n]

            lambdas_this_inv.append(lamb)

        lambdas_this_inv = np.stack(lambdas_this_inv, axis=0)  # [T, num_nodes]
        all_lambdas.append(lambdas_this_inv)
        all_iters.extend([k] * T)

    all_lambdas = np.concatenate(all_lambdas, axis=0)  # [num_iters*T, num_nodes]
    return all_lambdas, np.array(all_iters)


def summarize_lambda_distribution(lambdas, node_names=None, round_decimals=6):
    if node_names is None:
        node_names = [f"node_{i}" for i in range(lambdas.shape[1])]

    rows = []
    for j, node in enumerate(node_names):
        vals = np.round(lambdas[:, j], round_decimals)
        counts = Counter(vals)

        for value, count in sorted(counts.items()):
            rows.append({
                "node": node,
                "lambda_value": value,
                "count": count,
                "share": count / len(vals),
            })

    return pd.DataFrame(rows)


def plot_lambda_distribution(lambdas, node_names=None, bins=30):
    if node_names is None:
        node_names = [f"node_{i}" for i in range(lambdas.shape[1])]

    for j, node in enumerate(node_names):
        plt.figure(figsize=(7, 4))
        plt.hist(lambdas[:, j], bins=bins)
        plt.xlabel(r"$\lambda$")
        plt.ylabel("Frequency")
        plt.title(f"Exact Benders encountered lambda distribution - {node}")
        plt.tight_layout()
        plt.show()



RENEWABLE_TECHS = {"SunPV", "WindOn", "WindOff"}


def get_generator_q_values(data, availability_percentile=90):
    """
    q_g = 1 for dispatchable generators.
    q_g = renewable availability percentile for renewables.
    """
    q = {}

    for g in data.G:
        node, tech = g

        if tech in RENEWABLE_TECHS:
            vals = []
            for t in data.T:
                vals.append(float(data.pGenAva[(node, tech, t)]))
            q[g] = max(np.percentile(vals, availability_percentile), 1e-6)
        else:
            q[g] = 1.0

    return q


def compute_node_capacity_bound(data):
    """
    C_n = max_t demand_n,t + total export capacity from node n.
    """
    C = {}

    for n in data.N:
        max_demand = max(float(data.pDemand[(n, t)]) for t in data.T)

        export_cap = 0.0
        for l in data.L:
            i, j = l
            if i == n:
                export_cap += float(data.pExpCap[l])
            elif j == n:
                export_cap += float(data.pImpCap[l])

        C[n] = max_demand + export_cap

    return C

def collect_lambdas_for_sampled_investments(solver, data, sample, investment_samples, compact=False):
    all_lambdas = []

    T = len(data.time_ranges[sample])
    num_rows_per_t_ineq = 2 * (data.num_g + data.num_l + data.num_n)

    for inv in investment_samples:
        lambdas_this_inv = []

        for t in range(T):
            obj, A_ineq, b_ineq, A_eq, b_eq = solver.find_subproblem_cm_rhs_obj(
                data,
                compact,
                sample,
                torch.tensor(inv, dtype=torch.float64),
                t,
            )

            obj_val, primal_val, dual_val, solve_time = solver.solve_matrix_problem_simple(
                obj, A_ineq, b_ineq, A_eq, b_eq, master=False
            )

            dual_val = np.asarray(dual_val, dtype=float)

            # Gurobi equality duals
            lamb = dual_val[num_rows_per_t_ineq:num_rows_per_t_ineq + data.num_n]

            # Convert to model convention: remove pWeight scaling and flip sign
            lamb = -lamb / float(data.pWeight)

            lambdas_this_inv.append(lamb)

        all_lambdas.append(np.stack(lambdas_this_inv, axis=0))

    return np.concatenate(all_lambdas, axis=0)


def summarize_lambda_classes(lambdas, node_names):
    rows = []

    for j, node in enumerate(node_names):
        vals = np.round(lambdas[:, j], 6)
        unique, counts = np.unique(vals, return_counts=True)

        for v, c in zip(unique, counts):
            rows.append({
                "node": node,
                "lambda_value": v,
                "count": int(c),
                "share": c / len(vals),
            })

    return pd.DataFrame(rows)


def sample_node_bound_investment(
    data,
    C,
    q,
    r_choices=None,
    r_probs=None,
    alpha_choices=(0.2, 1.0, 5.0),
    max_investment=100000,
):
    """
    Sample investment u_g using node-level effective capacity:

        sum_g q_g * pUnitCap_g * u_g ~= r_n * C_n

    Then split node budget across generators using Dirichlet.
    """
    if r_choices is None:
        r_choices = np.array([0.2, 0.5, 0.8, 0.95, 1.0, 1.05, 1.2])

    if r_probs is None:
        r_probs = np.array([0.08, 0.12, 0.18, 0.20, 0.20, 0.14, 0.08])

    investments = {}

    for n in data.N:
        gens_at_node = [g for g in data.G if g[0] == n]

        if len(gens_at_node) == 0:
            continue

        r_n = np.random.choice(r_choices, p=r_probs)
        E_n = r_n * C[n]

        alpha = np.random.choice(alpha_choices)
        shares = np.random.dirichlet(alpha * np.ones(len(gens_at_node)))

        for share, g in zip(shares, gens_at_node):
            p_unit = float(data.pUnitCap[g])
            q_g = max(float(q[g]), 1e-8)

            # effective capacity share -> investment units
            u_g = share * E_n / (q_g * p_unit)
            u_g = min(max(u_g, 0.0), max_investment)

            investments[g] = u_g

    # return in data.G order
    return np.array([investments[g] for g in data.G], dtype=float)

def generate_node_bound_investment_samples(
    data,
    num_samples=5000,
    availability_percentile=90,
    max_investment=100000,
):
    C = compute_node_capacity_bound(data)
    q = get_generator_q_values(data, availability_percentile)

    samples = []

    for _ in range(num_samples):
        inv = sample_node_bound_investment(
            data=data,
            C=C,
            q=q,
            max_investment=max_investment,
        )
        samples.append(inv)

    return np.stack(samples), C, q


if __name__ == "__main__":
    import pickle
    #csv_path = "outputs/Benders/3Node/Sample_120/iter_logs_exact_5/iterlog_sample1_start_exactTrue_refFalse.csv"

    ed_data_save_path = "data/ED_data/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15.pkl"
    gep_data_save_path = "data/GEP_data/sample_duration:120_N:B-G-F_G:B2-G2-F2_L:L3.pkl"

    with open(ed_data_save_path, "rb") as f:
        operational_data = pickle.load(f)

    with open(gep_data_save_path, "rb") as f:
        gep_data = pickle.load(f)

    node_names = list(gep_data.N)

    # df_iter, investments_list = load_investment_path(csv_path)
    # print(f"Create Solver with {len(investments_list)} investments to analyze...")
    investment_samples, C, q = generate_node_bound_investment_samples(
        data=operational_data,
        num_samples=50,
        availability_percentile=90,
        max_investment=100000,
    )
    solver = BendersSolver(
        gep_data=gep_data,
        operational_data=operational_data,
        primal_net=None,
        dual_net=None,
        sample=0,
        exact=True,
        exact_refinement=False,
    )

    lambdas = collect_lambdas_for_sampled_investments(
        solver=solver,
        data=gep_data,
        sample=0,
        investment_samples=investment_samples,
        compact=False,
    )

    lambda_dist_df = summarize_lambda_classes(lambdas, list(gep_data.N))
    print(lambda_dist_df)

    # solver = BendersSolver(
    #     gep_data=gep_data,
    #     operational_data=operational_data,
    #     primal_net=None,
    #     dual_net=None,
    #     sample=1,
    #     exact=True,
    #     exact_refinement=False,
    # )

    # lambdas, iter_ids = collect_exact_benders_lambdas_for_investments(
    #     solver=solver,
    #     data=gep_data,
    #     compact=False,
    #     sample=1,
    #     investments_list=investments_list,
    # )

    # node_names = list(gep_data.N)

    # pWeight = float(gep_data.pWeight)

    # # Convert Gurobi exact equality duals to model lambda convention
    # lambdas_model = -lambdas / pWeight

    # lambda_dist_df = summarize_lambda_distribution(
    #     lambdas_model,
    #     node_names=list(gep_data.N),
    #     round_decimals=6,
    # )

    # print(lambda_dist_df)
    # print(f"Dual Net all classes are ")
    # plot_lambda_distribution(lambdas_model, node_names=list(gep_data.N))