import os
import pickle
import numpy as np
import pandas as pd
import math

# =========================
# Paths
# =========================
DATASETS = {
    # "D_uniform": "data/ED_data/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15.pkl",
    "D_CABCap": "data/ED_data/Constraint/3Loc/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_CapSobol_GenConst_ui_constraint1000_smp15_renewMaxInvTrue_NodeConst.pkl",
    "D_CAB": "data/ED_data/Constraint/3Loc/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_ui_constraint1000_smp15_GenConst_lbFalse_renewMaxInvTrue.pkl",
    # "D_CAB-R50": "data/ED_data/Constraint/3Loc/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_ui_constraint10000_smp15_GenConst_lbFalse_renewPerc50.pkl",
    # "D_CAB-R90": "data/ED_data/Constraint/3Loc/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_ui_constraint10000_smp15_GenConst_lbFalse_renewPerc90.pkl",
}

import matplotlib.pyplot as plt

def plot_generator_effective_capacity_grid(
    datasets,
    bins=80,
    density=True,
    alpha=0.75,
    figsize=(15, 8),
    save_dir=None,
):
    """
    For each dataset, make a 2x3 subplot figure:
    one subplot per generator, showing the distribution of
    effective capacity E = U * Pmax * A.
    Renewable generator titles are colored red.
    """
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file for {name}: {path}")
            continue

        data = load_dataset(path)
        D, A, U, E = reconstruct_arrays(data)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for g_idx, g in enumerate(data.G):
            ax = axes[g_idx]
            label = f"{g[0]}-{g[1]}"
            is_ren = is_renewable(g)

            vals = E[:, g_idx]
            vmax = float(np.max(vals))
            if vmax <= 0:
                vmax = 1.0
            bin_edges = np.linspace(0.0, vmax, bins + 1)

            ax.hist(
                vals,
                bins=bin_edges,
                density=density,
                alpha=alpha,
            )

            ax.set_title(label, color=("red" if is_ren else "black"))
            ax.set_xlabel("Effective capacity")
            ax.set_ylabel("Density" if density else "Count")
            ax.grid(True, linestyle="--", alpha=0.4)

        fig.suptitle(f"{name}: per-generator effective-capacity distributions", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{name}_generator_effective_capacity_grid.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {out_path}")

        plt.show()


# =========================
# Helpers
# =========================
def is_renewable(gen):
    _, tech = gen
    return str(tech).lower() in {"sunpv", "windon", "windoff"}

def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_time_index_for_sample(data, i):
    # Matches build_X():
    # t = i % len(self.T) + 1
    return (i % len(data.T)) + 1

def reconstruct_arrays(data):
    """
    Reconstruct per-sample:
      - demand matrix D[sample, node]
      - availability matrix A[sample, gen]   (historical availability if applicable)
      - investment matrix U[sample, gen]     (None for direct-capacity datasets)
      - effective capacity matrix E[sample, gen]

    Supports:
      1. previous investment-based datasets
      2. direct-capacity Sobol datasets
    """
    X = to_numpy(data.X).astype(float)
    n_samples = X.shape[0]

    D = X[:, :data.num_n].copy()
    E = X[:, data.num_n:data.num_n + data.num_g].copy()

    A = np.zeros((n_samples, data.num_g), dtype=float)

    # historical availability matrix, useful for plotting/reference
    for i in range(n_samples):
        t = get_time_index_for_sample(data, i)
        for g_idx, g in enumerate(data.G):
            A[i, g_idx] = float(data.pGenAva.get((*g, t), 1.0))

    # Old dataset mode: has pUnitInvestment and E should be reconstructed from it
    if hasattr(data, "pUnitInvestment") and data.pUnitInvestment is not None:
        U = to_numpy(data.pUnitInvestment).astype(float)

        # Recompute E from old formula to stay explicit/consistent
        E_old = np.zeros((n_samples, data.num_g), dtype=float)
        for i in range(n_samples):
            t = get_time_index_for_sample(data, i)
            for g_idx, g in enumerate(data.G):
                a = float(data.pGenAva.get((*g, t), 1.0))
                E_old[i, g_idx] = U[i, g_idx] * float(data.pUnitCap[g]) * a

        E = E_old
    else:
        # New direct-capacity mode
        U = None

    return D, A, U, E
def collapse_metric(u, eff, n_bins=50):
    """
    Quantify how much different investments collapse into similar effective capacities.
    For each eff-capacity bin, compute std(u). Then average across non-empty bins.
    Higher => more collapse.
    """
    eff = np.asarray(eff, dtype=float)
    u = np.asarray(u, dtype=float)

    if np.allclose(eff.max(), eff.min()):
        return 0.0

    bins = np.linspace(eff.min(), eff.max(), n_bins + 1)
    idx = np.digitize(eff, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    stds = []
    counts = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 2:
            stds.append(np.std(u[mask]))
            counts.append(mask.sum())

    if len(stds) == 0:
        return 0.0

    return float(np.average(stds, weights=counts))

def nearest_neighbor_distance(X, max_rows=5000, seed=0):
    """
    Cheap diversity proxy in actual ED-input space.
    Uses median nearest-neighbor distance after optional subsampling.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)

    if X.shape[0] > max_rows:
        idx = rng.choice(X.shape[0], size=max_rows, replace=False)
        X = X[idx]

    # Compute squared distances without sklearn
    G = X @ X.T
    sq = np.sum(X**2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2 * G
    dist2 = np.maximum(dist2, 0.0)

    np.fill_diagonal(dist2, np.inf)
    nn = np.sqrt(np.min(dist2, axis=1))
    return float(np.median(nn))

def summarize_dataset(name, data):
    D, A, U, E = reconstruct_arrays(data)

    renewable_indices = [i for i, g in enumerate(data.G) if is_renewable(g)]
    dispatchable_indices = [i for i, g in enumerate(data.G) if not is_renewable(g)]

    total_demand = D.sum(axis=1)
    total_ren_eff = E[:, renewable_indices].sum(axis=1) if renewable_indices else np.zeros(len(D))
    total_disp_cap = E[:, dispatchable_indices].sum(axis=1) if dispatchable_indices else np.zeros(len(D))
    net_load = total_demand - total_ren_eff
    ren_share = total_ren_eff / np.maximum(total_demand, 1e-9)

    X_actual = np.concatenate([D, E], axis=1)

    summary = {
        "dataset": name,
        "n_samples": int(E.shape[0]),
        "n_timesteps_base": int(len(data.T)),
        "num_nodes": int(data.num_n),
        "num_gens": int(data.num_g),
        "mean_total_demand": float(np.mean(total_demand)),
        "std_total_demand": float(np.std(total_demand)),
        "mean_total_ren_effcap": float(np.mean(total_ren_eff)),
        "std_total_ren_effcap": float(np.std(total_ren_eff)),
        "mean_total_disp_cap": float(np.mean(total_disp_cap)),
        "std_total_disp_cap": float(np.std(total_disp_cap)),
        "mean_net_load": float(np.mean(net_load)),
        "std_net_load": float(np.std(net_load)),
        "mean_ren_share": float(np.mean(ren_share)),
        "std_ren_share": float(np.std(ren_share)),
        "nn_dist_actual_input": nearest_neighbor_distance(X_actual),
        "input_rank_95pct_var": effective_rank_95(X_actual),
        "has_investment_samples": U is not None,
    }

    for g_idx in renewable_indices:
        g = data.G[g_idx]
        gname = f"{g[0]}-{g[1]}"

        a = A[:, g_idx]
        e = E[:, g_idx]

        summary[f"{gname}_a_mean"] = float(np.mean(a))
        summary[f"{gname}_a_std"] = float(np.std(a))
        summary[f"{gname}_eff_mean"] = float(np.mean(e))
        summary[f"{gname}_eff_std"] = float(np.std(e))

        if U is not None:
            u = U[:, g_idx]
            summary[f"{gname}_u_mean"] = float(np.mean(u))
            summary[f"{gname}_u_std"] = float(np.std(u))
            summary[f"{gname}_collapse_u_given_eff"] = collapse_metric(u, e, n_bins=50)
        else:
            summary[f"{gname}_u_mean"] = np.nan
            summary[f"{gname}_u_std"] = np.nan
            summary[f"{gname}_collapse_u_given_eff"] = np.nan

    return summary

def effective_rank_95(X):
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2
    if var.sum() <= 0:
        return 0
    cum = np.cumsum(var) / var.sum()
    return int(np.searchsorted(cum, 0.95) + 1)

def print_nice_summary(df):
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
    print("\n=== Compact summary ===")
    cols = [
        "dataset",
        "n_samples",
        "mean_total_demand",
        "std_total_demand",
        "mean_total_ren_effcap",
        "std_total_ren_effcap",
        "mean_net_load",
        "std_net_load",
        "mean_ren_share",
        "std_ren_share",
        "nn_dist_actual_input",
        "input_rank_95pct_var",
    ]
    print(df[cols].to_string(index=False))

    print("\n=== Renewable collapse metrics ===")
    collapse_cols = [c for c in df.columns if "collapse_u_given_eff" in c]
    show_cols = ["dataset"] + collapse_cols
    print(df[show_cols].to_string(index=False))

# =========================
# Run
# =========================
rows = []
for name, path in DATASETS.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing file for {name}: {path}")
        continue

    print(f"Loading {name} from {path}")
    data = load_dataset(path)
    rows.append(summarize_dataset(name, data))

summary_df = pd.DataFrame(rows)

if len(summary_df) == 0:
    raise FileNotFoundError("No dataset files found.")

print_nice_summary(summary_df)

# Optional: save
summary_df.to_csv("ed_dataset_diversity_summary.csv", index=False)
print("\nSaved full summary to ed_dataset_diversity_summary.csv")

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def get_node_gen_indices(data):
    """
    Returns a dict:
        node -> list of generator indices connected to that node
    """
    node_to_gidx = {n: [] for n in data.N}
    for g_idx, g in enumerate(data.G):
        node_to_gidx[g[0]].append(g_idx)
    return node_to_gidx


def compute_total_demand_and_node_stress(data):
    """
    Returns:
        total_demand: [n_samples]
        stress_per_node: dict[node] -> [n_samples]

    stress_n = demand_n / (sum effective capacity of generators at node n + eps)
    """
    D, A, U, E = reconstruct_arrays(data)

    total_demand = D.sum(axis=1)

    node_to_gidx = get_node_gen_indices(data)
    eps = 1e-9

    stress_per_node = {}
    for node_idx, node in enumerate(data.N):
        local_gen_idx = node_to_gidx[node]
        local_eff_cap = E[:, local_gen_idx].sum(axis=1) if len(local_gen_idx) > 0 else np.zeros(E.shape[0])
        stress_per_node[node] = D[:, node_idx] / (local_eff_cap + eps)

    return total_demand, stress_per_node

def plot_total_demand_histograms(
    datasets,
    bins=60,
    density=True,
    alpha=0.55,
    figsize=(8, 5),
    save_path=None,
):
    """
    Overlay histogram of total demand for all datasets.
    """
    plt.figure(figsize=figsize)

    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file for {name}: {path}")
            continue

        data = load_dataset(path)
        total_demand, _ = compute_total_demand_and_node_stress(data)

        plt.hist(
            total_demand,
            bins=bins,
            density=density,
            alpha=alpha,
            label=name,
        )

    plt.xlabel("Total demand")
    plt.ylabel("Density" if density else "Count")
    plt.title("Distribution of total demand")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_node_demand_density_grid(
    datasets,
    bins=80,
    density=True,
    alpha=0.75,
    figsize_per_subplot=(5, 4),
    save_dir=None,
):
    """
    For each dataset, create a subplot grid with one histogram per node,
    showing the distribution of nodal demand.

    Parameters
    ----------
    datasets : dict
        {"dataset_name": "path/to/file.pkl", ...}
    bins : int
        Number of histogram bins.
    density : bool
        If True, plot density; otherwise plot counts.
    alpha : float
        Histogram transparency.
    figsize_per_subplot : tuple
        Approximate size per subplot (width, height).
    save_dir : str or None
        Optional folder to save figures.
    """
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing file for {name}: {path}")
            continue

        data = load_dataset(path)
        D, A, U, E = reconstruct_arrays(data)

        num_nodes = data.num_n

        # Make layout automatic
        ncols = min(3, num_nodes)
        nrows = math.ceil(num_nodes / ncols)

        figsize = (figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Make axes always iterable
        if num_nodes == 1:
            axes = np.array([axes])
        axes = np.array(axes).reshape(-1)

        for n_idx, node in enumerate(data.N):
            ax = axes[n_idx]

            vals = D[:, n_idx]
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))

            if np.isclose(vmin, vmax):
                vmin = max(0.0, vmin - 1.0)
                vmax = vmax + 1.0

            bin_edges = np.linspace(vmin, vmax, bins + 1)

            ax.hist(
                vals,
                bins=bin_edges,
                density=density,
                alpha=alpha,
            )

            ax.set_title(str(node))
            ax.set_xlabel("Demand")
            ax.set_ylabel("Density" if density else "Count")
            ax.grid(True, linestyle="--", alpha=0.4)

        # Hide unused axes
        for j in range(num_nodes, len(axes)):
            axes[j].axis("off")

        fig.suptitle(f"{name}: per-node demand distributions", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{name}_node_demand_density_grid.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"Saved plot to {out_path}")

        plt.show()

def plot_node_log_stress_histograms(
    datasets,
    bins=60,
    density=True,
    alpha=0.55,
    figsize=(15, 4),
    save_path=None,
):
    """
    Overlay histogram of log(1 + stress) across datasets.
    """
    first_data = None
    for _, path in datasets.items():
        if os.path.exists(path):
            first_data = load_dataset(path)
            break

    if first_data is None:
        raise FileNotFoundError("No dataset files found.")

    num_nodes = first_data.num_n
    fig, axes = plt.subplots(1, num_nodes, figsize=figsize, sharey=False)

    if num_nodes == 1:
        axes = [axes]

    for node_idx, node in enumerate(first_data.N):
        ax = axes[node_idx]

        for name, path in datasets.items():
            if not os.path.exists(path):
                continue

            data = load_dataset(path)
            _, stress_per_node = compute_total_demand_and_node_stress(data)
            vals = np.asarray(stress_per_node[node], dtype=float)

            vals_log = np.log1p(vals)

            ax.hist(
                vals_log,
                bins=bins,
                density=density,
                alpha=alpha,
                label=name,
            )

        ax.set_title(f"{node}")
        ax.set_xlabel(r"$\log(1+\mathrm{stress})$")
        if node_idx == 0:
            ax.set_ylabel("Density" if density else "Count")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(datasets), frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(r"Per-location $\log(1+\mathrm{stress})$ distributions", fontsize=14, y=1.08)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.show()

# =========================
# Plot 2x3 grid per dataset
# =========================
# plot_generator_effective_capacity_grid(
#     DATASETS,
#     bins=80,
#     density=True,
#     alpha=0.75,
#     figsize=(15, 8),
#     save_dir="figures/generator_effcap_grid_plots",
# )

# plot_total_demand_histograms(
#     DATASETS,
#     bins=60,
#     density=True,
#     alpha=0.5,
#     figsize=(8, 5),
#     save_path="figures/dataset_demand_and_stress/total_demand_hist.png",
# )

plot_node_demand_density_grid(
    DATASETS,
    bins=80,
    density=True,
    alpha=0.75,
    save_dir="figures/node_demand_density_plots",
)

plot_node_log_stress_histograms(
    DATASETS,
    bins=60,
    density=True,
    alpha=0.5,
    figsize=(15, 4),
    save_path="figures/dataset_demand_and_stress/node_stress_overlayed.png",
)