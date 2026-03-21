"""
Diagnostic evaluation for understanding PDL performance on constrained dataset.
Slices performance by: investment level, demand, active constraints, dual structure.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
# add ../netork to path

from networks import PrimalNetEndToEnd, DualNet, DualNetEndToEnd, DualClassificationNetEndToEnd


def diagnose_model(
    data,
    exp_path,
    repeat=0,
    train_frac=0.8,
    valid_frac=0.1,
    hidden_size_factor=28,
    split="test",
):
    """
    Full diagnostic evaluation of a trained model.
    Returns a rich per-sample DataFrame for analysis.
    
    Args:
        data: dataset object
        exp_path: path to experiment folder
        split: "test", "valid", or "all"
    """
    # Load args and model
    with open(os.path.join(exp_path, "args.json"), "r") as f:
        args = json.load(f)
    args["hidden_size_factor"] = hidden_size_factor
    directory = os.path.join(exp_path, f"repeat:{repeat}")

    primal_net = PrimalNetEndToEnd(args, data=data)
    primal_net.load_state_dict(
        torch.load(os.path.join(directory, "primal_weights.pth"), weights_only=True),
        strict=False,
    )
    primal_net.eval()

    if args.get("dual_classification", False):
        dual_net = DualClassificationNetEndToEnd(args, data=data)
    elif args.get("dual_completion", False):
        dual_net = DualNetEndToEnd(args, data=data)
    else:
        dual_net = DualNet(args, data=data)
    dual_net.load_state_dict(
        torch.load(os.path.join(directory, "dual_weights.pth"), weights_only=True),
        strict=False,
    )
    dual_net.eval()

    # Select indices
    n_total = data.X.shape[0]
    train_size = int(train_frac * n_total)
    valid_size = int(valid_frac * n_total)
    if split == "test":
        indices = torch.arange(n_total)[train_size + valid_size:]
    elif split == "valid":
        indices = torch.arange(n_total)[train_size:train_size + valid_size]
    else:
        indices = torch.arange(n_total)

    X = data.X[indices]
    Y_target = data.opt_targets["y_operational"][indices]
    mu_target = data.opt_targets["mu_operational"][indices]
    lamb_target = data.opt_targets["lamb_operational"][indices]
    opt_obj_values = data.opt_targets["obj"][indices]

    with torch.no_grad():
        Y_pred = primal_net(X)
        mu_pred, lamb_pred = dual_net(X)

    # ---- Primal metrics ----
    pred_obj = data.obj_fn(X, Y_pred).detach()
    target_obj = data.obj_fn(X, Y_target).detach()
    primal_opt_gap = ((pred_obj - target_obj) / target_obj.abs().clamp(min=1e-8) * 100).cpu().numpy()

    ineq_dist = data.ineq_dist(X, Y_pred)
    eq_resid = data.eq_resid(X, Y_pred)
    ineq_max = ineq_dist.max(dim=1)[0].cpu().numpy()
    ineq_mean = ineq_dist.mean(dim=1).cpu().numpy()
    eq_max = eq_resid.abs().max(dim=1)[0].cpu().numpy()

    # ---- Dual metrics ----
    dual_obj_pred = data.dual_obj_fn(X, mu_pred, lamb_pred).detach()
    dual_obj_gt = data.dual_obj_fn(X, mu_target, lamb_target).detach()
    dual_opt_gap = ((dual_obj_gt - dual_obj_pred) / dual_obj_gt.abs().clamp(min=1e-8) * 100).cpu().numpy()

    # ---- Per-sample structural features ----
    # Investment per generator
    inv = data.pUnitInvestment[indices].cpu().numpy()  # [N, G]
    total_inv = inv.sum(axis=1)

    # Demand per node
    demand = X[:, :data.num_n].cpu().numpy()  # [N, num_n]
    total_demand = demand.sum(axis=1)

    # Capacity upper bounds (from X)
    capacity_ub = X[:, data.num_n:].cpu().numpy()  # [N, num_g]
    total_capacity = capacity_ub.sum(axis=1)

    # Capacity-to-demand ratio (how tight is the problem?)
    cap_demand_ratio = total_capacity / (total_demand + 1e-8)

    # Count active constraints at optimum (ineq residual ≈ 0 means binding)
    ineq_resid_target = data.ineq_resid(X, Y_target).detach()
    ineq_rhs_target = data.split_X(X)[1]
    # A constraint is "active" if |Ax - b| < tol (close to the boundary)
    tol = 1e-4
    active_ineq = (ineq_rhs_target - Y_target @ data.ineq_cm.T).abs() < tol
    n_active_ineq = active_ineq.sum(dim=1).cpu().numpy()

    # Count non-zero duals at optimum
    n_nonzero_mu = (mu_target.abs() > tol).sum(dim=1).cpu().numpy()
    n_nonzero_lamb = (lamb_target.abs() > tol).sum(dim=1).cpu().numpy()

    # Dual prediction error per constraint type
    mu_err = (mu_pred - mu_target).abs()
    lamb_err = (lamb_pred - lamb_target).abs()

    # Split mu error by constraint type: [lb_p, ub_p, lb_f, ub_f, lb_md, ub_md]
    sizes = [data.num_g, data.num_g, data.num_l, data.num_l, data.num_n, data.num_n]
    mu_err_split = torch.split(mu_err, sizes, dim=1)
    constraint_names = ["mu_lb_prod", "mu_ub_prod", "mu_lb_flow", "mu_ub_flow", "mu_lb_md", "mu_ub_md"]

    # ---- Build DataFrame ----
    df = pd.DataFrame({
        "sample_idx": indices.cpu().numpy(),
        "pred_obj": pred_obj.cpu().numpy(),
        "target_obj": target_obj.cpu().numpy(),
        "primal_opt_gap": primal_opt_gap,
        "dual_obj_pred": dual_obj_pred.cpu().numpy(),
        "dual_obj_gt": dual_obj_gt.cpu().numpy(),
        "dual_opt_gap": dual_opt_gap,
        "ineq_max": ineq_max,
        "ineq_mean": ineq_mean,
        "eq_max": eq_max,
        "total_investment": total_inv,
        "total_demand": total_demand,
        "total_capacity": total_capacity,
        "cap_demand_ratio": cap_demand_ratio,
        "n_active_ineq": n_active_ineq,
        "n_nonzero_mu": n_nonzero_mu,
        "n_nonzero_lamb": n_nonzero_lamb,
        "mu_mae": mu_err.mean(dim=1).cpu().numpy(),
        "lamb_mae": lamb_err.mean(dim=1).cpu().numpy(),
    })

    # Add per-generator investment columns
    for g_idx, g in enumerate(data.G):
        node, tech = g
        df[f"inv_{node}_{tech}"] = inv[:, g_idx]
        df[f"cap_{node}_{tech}"] = capacity_ub[:, g_idx]

    # Add per-constraint-type dual error
    for name, err in zip(constraint_names, mu_err_split):
        df[f"err_{name}"] = err.mean(dim=1).cpu().numpy()

    # Add per-node demand
    for n_idx, n in enumerate(data.N):
        df[f"demand_{n}"] = demand[:, n_idx]

    df["experiment"] = os.path.basename(exp_path)

    return df


def plot_diagnostics(df, title="Diagnostic Analysis"):
    """
    Create diagnostic plots from the diagnose_model output.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16)

    # 1. Dual gap vs capacity-to-demand ratio
    ax = axes[0, 0]
    ax.scatter(df["cap_demand_ratio"], df["dual_opt_gap"], alpha=0.3, s=8)
    ax.set_xlabel("Capacity / Demand ratio")
    ax.set_ylabel("Dual opt gap (%)")
    ax.set_title("Dual gap vs tightness")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Dual gap vs number of active constraints
    ax = axes[0, 1]
    ax.scatter(df["n_active_ineq"], df["dual_opt_gap"], alpha=0.3, s=8)
    ax.set_xlabel("# Active inequality constraints")
    ax.set_ylabel("Dual opt gap (%)")
    ax.set_title("Dual gap vs constraint complexity")
    ax.grid(True, alpha=0.3)

    # 3. Dual gap vs number of non-zero duals
    ax = axes[0, 2]
    ax.scatter(df["n_nonzero_mu"], df["dual_opt_gap"], alpha=0.3, s=8)
    ax.set_xlabel("# Non-zero mu (at optimum)")
    ax.set_ylabel("Dual opt gap (%)")
    ax.set_title("Dual gap vs dual sparsity")
    ax.grid(True, alpha=0.3)

    # 4. Dual error by constraint type (boxplot)
    ax = axes[1, 0]
    err_cols = [c for c in df.columns if c.startswith("err_mu_")]
    err_data = df[err_cols]
    err_data.columns = [c.replace("err_mu_", "") for c in err_cols]
    err_data.boxplot(ax=ax, rot=45)
    ax.set_ylabel("Mean absolute dual error")
    ax.set_title("Dual error by constraint type")

    # 5. Primal gap vs total demand
    ax = axes[1, 1]
    ax.scatter(df["total_demand"], df["primal_opt_gap"], alpha=0.3, s=8)
    ax.set_xlabel("Total demand")
    ax.set_ylabel("Primal opt gap (%)")
    ax.set_title("Primal gap vs demand")
    ax.grid(True, alpha=0.3)

    # 6. Distribution of dual opt gap
    ax = axes[1, 2]
    ax.hist(df["dual_opt_gap"], bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Dual opt gap (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"Dual gap distribution (mean={df['dual_opt_gap'].mean():.2f}%)")
    ax.axvline(df["dual_opt_gap"].median(), color="red", linestyle="--", label=f"median={df['dual_opt_gap'].median():.2f}%")
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def print_diagnostic_summary(df):
    """Print a text summary of where the model struggles."""
    print("=" * 60)
    print(f"DIAGNOSTIC SUMMARY: {df['experiment'].iloc[0]}")
    print(f"N samples: {len(df)}")
    print("=" * 60)
    print(f"\n--- Overall Performance ---")

    print(
        f"  Primal gap:  "
        f"mean={df['primal_opt_gap'].mean():.3f}%  "
        f"median={df['primal_opt_gap'].median():.3f}%  "
        f"p90={df['primal_opt_gap'].quantile(0.9):.3f}%  "
        f"p99={df['primal_opt_gap'].quantile(0.99):.3f}%  "
        f"max={df['primal_opt_gap'].max():.3f}%"
    )

    print(
        f"  Dual gap:    "
        f"mean={df['dual_opt_gap'].mean():.3f}%  "
        f"median={df['dual_opt_gap'].median():.3f}%  "
        f"p90={df['dual_opt_gap'].quantile(0.9):.3f}%  "
        f"p99={df['dual_opt_gap'].quantile(0.99):.3f}%  "
        f"max={df['dual_opt_gap'].max():.3f}%"
    )

    # Slice by capacity-demand ratio
    print(f"\n--- Dual gap by capacity tightness ---")
    df["tightness_bin"] = pd.cut(df["cap_demand_ratio"], bins=[0, 1.0, 1.5, 2.0, 3.0, float("inf")],
                                  labels=["<1 (infeasible)", "1-1.5 (tight)", "1.5-2 (moderate)", "2-3 (loose)", ">3 (excess)"])
    for bin_name, group in df.groupby("tightness_bin", observed=True):
        print(f"  {bin_name:20s}  n={len(group):5d}  dual_gap_mean={group['dual_opt_gap'].mean():8.3f}%  primal_gap_mean={group['primal_opt_gap'].mean():8.3f}%")

    # Slice by number of active constraints
    print(f"\n--- Dual gap by # active constraints ---")
    df["active_bin"] = pd.cut(df["n_active_ineq"], bins=[0, 5, 10, 15, 20, float("inf")],
                               labels=["0-5", "6-10", "11-15", "16-20", ">20"])
    for bin_name, group in df.groupby("active_bin", observed=True):
        print(f"  {bin_name:10s}  n={len(group):5d}  dual_gap_mean={group['dual_opt_gap'].mean():8.3f}%")

    # Dual error by constraint type
    print(f"\n--- Dual error by constraint type ---")
    err_cols = [c for c in df.columns if c.startswith("err_mu_")]
    for col in err_cols:
        name = col.replace("err_mu_", "")
        print(f"  {name:15s}  mean_err={df[col].mean():.6f}  max_err={df[col].max():.6f}")
    print(f"  {'lamb (eq)':15s}  mean_err={df['lamb_mae'].mean():.6f}  max_err={df['lamb_mae'].max():.6f}")

    # Worst samples
    print(f"\n--- Top 10 worst dual gap samples ---")
    worst = df.nlargest(10, "dual_opt_gap")[["sample_idx", "dual_opt_gap", "primal_opt_gap",
                                              "cap_demand_ratio", "n_active_ineq", "n_nonzero_mu",
                                              "total_demand", "total_capacity"]]
    print(worst.to_string(index=False))

    # Correlation
    print(f"\n--- Correlation with dual_opt_gap ---")
    numeric_cols = ["cap_demand_ratio", "n_active_ineq", "n_nonzero_mu", "total_demand",
                    "total_capacity", "total_investment", "mu_mae", "lamb_mae"]
    for col in numeric_cols:
        corr = df["dual_opt_gap"].corr(df[col])
        print(f"  {col:25s}  r={corr:+.4f}")

if __name__ == "__main__":
    base_dir = "outputs/PDL/ED/3Nodes-FraBelGer"
    baseline_path = f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BaselineConst2"
    s3l_path = f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-ConstBaseline_Reg0.99"


    data = pickle.load(open("data/ED_data/Constraint/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_ui_constraint1000_smp15_GenConst_lbFalse.pkl", "rb"))
    exp_path = "outputs/PDL/ED/3Nodes-FraBelGer/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BaselineConst2"

    # Get rich per-sample diagnostics
    df = diagnose_model(data, exp_path)

    # Text summary — this is the key output
    print_diagnostic_summary(df)

    # Visual plots
    fig = plot_diagnostics(df, title="ConstBaseline2 Diagnostics")
    plt.show()

    # Compare two models on the same data
    df_baseline = diagnose_model(data, baseline_path)
    df_s3l = diagnose_model(data, s3l_path)
    print_diagnostic_summary(df_baseline)
    print_diagnostic_summary(df_s3l)


    

    #####

    # Baseline model on unconstrained dataset
    base_data = pickle.load(open("data/ED_data/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15.pkl", "rb"))
    baseline_unconst_path = f"{base_dir}/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BendersBaseline"  # adjust to your actual path

    df_baseline_unconst = diagnose_model(base_data, baseline_unconst_path)
    print_diagnostic_summary(df_baseline_unconst)
    fig = plot_diagnostics(df_baseline_unconst, title="Baseline (Unconstrained Dataset) Diagnostics")
    plt.show()