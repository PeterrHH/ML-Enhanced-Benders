import os
import json
import numpy as np
import pandas as pd
import torch
import pickle
from networks import PrimalNetEndToEnd, DualNet, DualNetEndToEnd, DualClassificationNetEndToEnd


def get_investment_cap(const_data):
    cap = const_data.pUnitInvestment.max(dim=0)[0]
    return cap


def get_feasible_mask(base_data, cap):
    """
    Returns a boolean mask over all samples in base_data where EVERY
    generator's investment is <= the corresponding cap from const_data.
    
    Args:
        base_data: baseline dataset with .pUnitInvestment [N, num_g]
        cap: tensor [num_g] from get_investment_cap(const_data)
    
    Returns:
        mask: BoolTensor [N] — True if sample satisfies all caps
    """
    # base_data.pUnitInvestment: [N, num_g]
    # cap: [num_g]  →  broadcast to [1, num_g]
    within_cap = base_data.pUnitInvestment <= cap.unsqueeze(0)  # [N, num_g]
    mask = within_cap.all(dim=1)  # [N] — True only if ALL generators are within cap

    print(f"Total samples: {len(mask)}")
    print(f"Feasible samples (all gens within cap): {mask.sum().item()} "
          f"({mask.float().mean().item()*100:.1f}%)")
    
    # Show per-generator stats
    for g in range(cap.shape[0]):
        g_ok = within_cap[:, g].sum().item()
        g_max = base_data.pUnitInvestment[:, g].max().item()
        print(f"  Gen {g}: {g_ok}/{len(mask)} within cap "
              f"(cap={cap[g].item():.2f}, data_max={g_max:.2f})")
    
    return mask


def evaluate_filtered(
    base_data,
    const_data,
    exp_path,
    repeat=0,
    train_frac=0.8,
    valid_frac=0.1,
    hidden_size_factor=28,
):
    """
    Evaluate a trained model on the baseline dataset, but only on instances
    where all generators' investments are within the constraint dataset's cap.
    
    Args:
        base_data:  baseline dataset (loaded from pickle)
        const_data: constraint dataset (loaded from pickle)
        exp_path:   path to the experiment folder (contains args.json, repeat:0/, etc.)
        repeat:     repeat index
        train_frac: train split fraction (to derive test indices)
        valid_frac: valid split fraction
        hidden_size_factor: override for hidden_size_factor if needed
    
    Returns:
        df: DataFrame with per-sample evaluation metrics for feasible test samples
    """
    # 1. Get the investment cap from the constraint dataset
    cap = get_investment_cap(const_data)
    
    # 2. Get feasibility mask over ALL base_data samples
    feasible_mask = get_feasible_mask(base_data, cap)
    
    # 3. Compute test indices
    n_total = base_data.X.shape[0]
    train_size = int(train_frac * n_total)
    valid_size = int(valid_frac * n_total)
    all_indices = torch.arange(n_total)
    test_indices = all_indices[train_size + valid_size:]
    
    # 4. Intersect: test indices that are also feasible
    test_feasible_mask = feasible_mask[test_indices]  # [n_test]
    filtered_test_indices = test_indices[test_feasible_mask]
    
    print(f"\nTest set: {len(test_indices)} samples")
    print(f"Feasible test samples: {len(filtered_test_indices)} "
          f"({len(filtered_test_indices)/len(test_indices)*100:.1f}%)")
    
    if len(filtered_test_indices) == 0:
        print("WARNING: No feasible test samples found!")
        return pd.DataFrame()
    
    # 5. Load model
    args_path = os.path.join(exp_path, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)
    args["hidden_size_factor"] = hidden_size_factor
    
    directory = os.path.join(exp_path, f"repeat:{repeat}")
    
    primal_net = PrimalNetEndToEnd(args, data=base_data)
    primal_net.load_state_dict(
        torch.load(os.path.join(directory, "primal_weights.pth"), weights_only=True),
        strict=False,
    )
    primal_net.eval()
    
    # 6. Evaluate on filtered test set
    X = base_data.X[filtered_test_indices]
    Y_target = base_data.opt_targets["y_operational"][filtered_test_indices]
    
    with torch.no_grad():
        Y_pred = primal_net(X)
    
    ineq_dist = base_data.ineq_dist(X, Y_pred)
    eq_resid = base_data.eq_resid(X, Y_pred)
    
    pred_obj = base_data.obj_fn(X, Y_pred).detach().cpu().numpy()
    opt_obj = base_data.obj_fn(X, Y_target).detach().cpu().numpy()
    opt_gap = (pred_obj - opt_obj) / np.abs(opt_obj) * 100.0
    
    ineq_max = torch.max(ineq_dist, dim=1)[0].detach().cpu().numpy()
    ineq_mean = torch.mean(ineq_dist, dim=1).detach().cpu().numpy()
    eq_max = torch.max(torch.abs(eq_resid), dim=1)[0].detach().cpu().numpy()
    eq_mean = torch.mean(torch.abs(eq_resid), dim=1).detach().cpu().numpy()
    
    total_investment = base_data.pUnitInvestment[filtered_test_indices].sum(dim=1).detach().cpu().numpy()
    total_demand = X[:, :base_data.num_n].sum(dim=1).detach().cpu().numpy()
    
    df = pd.DataFrame({
        "sample_idx": filtered_test_indices.cpu().numpy(),
        "predicted_obj": pred_obj,
        "optimal_obj": opt_obj,
        "opt_gap": opt_gap,
        "ineq_max": ineq_max,
        "ineq_mean": ineq_mean,
        "eq_max": eq_max,
        "eq_mean": eq_mean,
        "total_investment": total_investment,
        "total_demand": total_demand,
    })
    df["experiment"] = os.path.basename(exp_path)
    
    return df


def evaluate_filtered_dual(
    base_data,
    const_data,
    exp_path,
    repeat=0,
    train_frac=0.8,
    valid_frac=0.1,
    hidden_size_factor=28,
):
    """
    Same as evaluate_filtered but also evaluates the dual network.
    Returns a DataFrame with both primal and dual metrics.
    """
    cap = get_investment_cap(const_data)
    feasible_mask = get_feasible_mask(base_data, cap)
    
    n_total = base_data.X.shape[0]
    train_size = int(train_frac * n_total)
    valid_size = int(valid_frac * n_total)
    test_indices = torch.arange(n_total)[train_size + valid_size:]
    
    test_feasible_mask = feasible_mask[test_indices]
    filtered_test_indices = test_indices[test_feasible_mask]
    
    print(f"\nTest set: {len(test_indices)} | Feasible: {len(filtered_test_indices)} "
          f"({len(filtered_test_indices)/len(test_indices)*100:.1f}%)")
    
    if len(filtered_test_indices) == 0:
        return pd.DataFrame()
    
    # Load args and models
    args_path = os.path.join(exp_path, "args.json")
    with open(args_path, "r") as f:
        args = json.load(f)
    args["hidden_size_factor"] = hidden_size_factor
    
    directory = os.path.join(exp_path, f"repeat:{repeat}")
    
    primal_net = PrimalNetEndToEnd(args, data=base_data)
    primal_net.load_state_dict(
        torch.load(os.path.join(directory, "primal_weights.pth"), weights_only=True),
        strict=False,
    )
    primal_net.eval()
    
    # Load dual net (handle different architectures)
    if args.get("dual_classification", False):
        dual_net = DualClassificationNetEndToEnd(args, data=base_data)
    elif args.get("dual_completion", False):
        dual_net = DualNetEndToEnd(args, data=base_data)
    else:
        dual_net = DualNet(args, data=base_data)
    
    dual_net.load_state_dict(
        torch.load(os.path.join(directory, "dual_weights.pth"), weights_only=True),
        strict=False,
    )
    dual_net.eval()
    
    # Evaluate
    X = base_data.X[filtered_test_indices]
    Y_target = base_data.opt_targets["y_operational"][filtered_test_indices]
    
    with torch.no_grad():
        Y_pred = primal_net(X)
        mu, lamb = dual_net(X)
    
    # Primal metrics
    pred_obj = base_data.obj_fn(X, Y_pred).detach().cpu().numpy()
    opt_obj = base_data.obj_fn(X, Y_target).detach().cpu().numpy()
    primal_opt_gap = (pred_obj - opt_obj) / np.abs(opt_obj) * 100.0
    
    ineq_dist = base_data.ineq_dist(X, Y_pred)
    eq_resid = base_data.eq_resid(X, Y_pred)
    ineq_max = torch.max(ineq_dist, dim=1)[0].detach().cpu().numpy()
    ineq_mean = torch.mean(ineq_dist, dim=1).detach().cpu().numpy()
    eq_max = torch.max(torch.abs(eq_resid), dim=1)[0].detach().cpu().numpy()
    eq_mean = torch.mean(torch.abs(eq_resid), dim=1).detach().cpu().numpy()
    
    # Dual metrics
    dual_obj = base_data.dual_obj_fn(X, mu, lamb).detach().cpu().numpy()
    
    # Ground truth duals (if available)
    if "mu_operational" in base_data.opt_targets and "lamb_operational" in base_data.opt_targets:
        mu_gt = base_data.opt_targets["mu_operational"][filtered_test_indices]
        lamb_gt = base_data.opt_targets["lamb_operational"][filtered_test_indices]
        dual_obj_gt = base_data.dual_obj_fn(X, mu_gt, lamb_gt).detach().cpu().numpy()
        dual_opt_gap = (dual_obj_gt - dual_obj) / np.abs(dual_obj_gt) * 100.0
    else:
        dual_obj_gt = opt_obj  # strong duality: dual* = primal*
        dual_opt_gap = (opt_obj - dual_obj) / np.abs(opt_obj) * 100.0
    
    total_investment = base_data.pUnitInvestment[filtered_test_indices].sum(dim=1).detach().cpu().numpy()
    total_demand = X[:, :base_data.num_n].sum(dim=1).detach().cpu().numpy()
    
    df = pd.DataFrame({
        "sample_idx": filtered_test_indices.cpu().numpy(),
        "predicted_obj": pred_obj,
        "optimal_obj": opt_obj,
        "primal_opt_gap": primal_opt_gap,
        "dual_obj": dual_obj,
        "dual_obj_gt": dual_obj_gt,
        "dual_opt_gap": dual_opt_gap,
        "ineq_max": ineq_max,
        "ineq_mean": ineq_mean,
        "eq_max": eq_max,
        "eq_mean": eq_mean,
        "total_investment": total_investment,
        "total_demand": total_demand,
    })
    df["experiment"] = os.path.basename(exp_path)
    
    return df


def compare_models_filtered(
    base_data,
    const_data,
    experiments: dict,
    repeat=0,
    train_frac=0.8,
    valid_frac=0.1,
    hidden_size_factor=28,
    include_dual=True,
):
    """
    Evaluate multiple models on the baseline dataset, filtered by constraint
    dataset investment caps. Returns a summary DataFrame.
    
    Args:
        base_data:   baseline dataset
        const_data:  constraint dataset (for computing investment caps)
        experiments: dict mapping name -> experiment path, e.g.
                     {"Baseline": "outputs/PDL/ED/.../Baseline",
                      "S3L_0.99": "outputs/PDL/ED/.../ConstBaseline_Reg0.99"}
        include_dual: if True, also evaluate dual network
    
    Returns:
        summary_df: DataFrame with one row per experiment
        per_sample_dfs: dict of name -> per-sample DataFrame
    
    Usage:
        experiments = {
            "BaselineConstDataset": "Constraint/learn_primal:...-BaselineConstDataset",
            "ConstBaseline2":       "Constraint/learn_primal:...-BaselineConstraintRenewables2",
            "Const_Reg_0.99":       "Constraint/learn_primal:...-ConstBaseline_Reg0.99",
        }
        summary, details = compare_models_filtered(base_data, const_data, experiments)
        print(summary.to_string())
    """
    # Compute cap and mask once
    cap = get_investment_cap(const_data)
    feasible_mask = get_feasible_mask(base_data, cap)
    
    n_total = base_data.X.shape[0]
    train_size = int(train_frac * n_total)
    valid_size = int(valid_frac * n_total)
    test_indices = torch.arange(n_total)[train_size + valid_size:]
    filtered_test_indices = test_indices[feasible_mask[test_indices]]
    
    # print(f"\n{'='*60}")
    # print(f"Evaluating {len(experiments)} models on {len(filtered_test_indices)} "
    #       f"feasible test samples (out of {len(test_indices)} total)")
    # print(f"{'='*60}\n")
    
    per_sample_dfs = {}
    summary_rows = []
    
    for name, exp_path in experiments.items():
        print(f"\n--- {name} ---")
        
        if include_dual:
            df = evaluate_filtered_dual(
                base_data, const_data, exp_path,
                repeat=repeat,
                train_frac=train_frac,
                valid_frac=valid_frac,
                hidden_size_factor=hidden_size_factor,
            )
        else:
            df = evaluate_filtered(
                base_data, const_data, exp_path,
                repeat=repeat,
                train_frac=train_frac,
                valid_frac=valid_frac,
                hidden_size_factor=hidden_size_factor,
            )
        
        if df.empty:
            print(f"  SKIPPED (no feasible samples)")
            continue
        
        per_sample_dfs[name] = df
        
        row = {
            "experiment": name,
            "n_samples": len(df),
            "primal_opt_gap_mean": df.get("primal_opt_gap", df.get("opt_gap")).mean(),
            "primal_opt_gap_max": df.get("primal_opt_gap", df.get("opt_gap")).max(),
            "ineq_max_mean": df["ineq_max"].mean(),
            "ineq_mean_mean": df["ineq_mean"].mean(),
            "eq_max_mean": df["eq_max"].mean(),
            "eq_mean_mean": df["eq_mean"].mean(),
        }
        
        if include_dual and "dual_opt_gap" in df.columns:
            row["dual_opt_gap_mean"] = df["dual_opt_gap"].mean()
            row["dual_opt_gap_max"] = df["dual_opt_gap"].max()
        
        summary_rows.append(row)
        
        print(f"  Primal gap: {row['primal_opt_gap_mean']:.4f}% (mean), "
              f"{row['primal_opt_gap_max']:.4f}% (max)")
        if "dual_opt_gap_mean" in row:
            print(f"  Dual gap:   {row['dual_opt_gap_mean']:.4f}% (mean), "
                  f"{row['dual_opt_gap_max']:.4f}% (max)")
        print(f"  Ineq max:   {row['ineq_max_mean']:.6f} (mean)")
    
    summary_df = pd.DataFrame(summary_rows)
    
    return summary_df, per_sample_dfs


if __name__ == "__main__":
    import pickle

    # 1. Load both datasets
    const_data = pickle.load(open(
        "data/ED_data/Constraint/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_ui_constraint1000_smp15_GenConst_lbFalse_renewMaxInvTrue.pkl", 'rb'))
    # base_data = pickle.load(open(
    #     "data/ED_data/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15.pkl", 'rb'))
    base_data = pickle.load(open(
        "data/ED_data/Constraint/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_ui_constraint1000_smp15_GenConst_lbFalse.pkl", 'rb'))

    # 2. Define experiments — use your actual paths from the screenshot
    base_dir = "outputs/PDL/ED/3Nodes-FraBelGer"
    experiments = {
        "BaselineConstDataset":   f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BaselineConst2",
        "ConstBaseline2":         f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BaselineConstraintRenewables2",
        "BlendRepairs":           f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BlendRepairs",
        "BlendRepairsNoPenalty":  f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BlendRepairsNoPenalty",
        "DualityGap":             f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-ConstDualityGap",
        "ConstRenewables2":       f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-BaselineConstraintRenewables2",
        "Const_Reg_0.99":         f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-ConstBaseline_Reg0.99",
        "Const_Reg_0.9":          f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-ConstBaseline_Reg0.9",
        "Const_Ref_0.7":          f"{base_dir}/Constraint/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-ConstBaseline_Ref0.7",
    }

    # 3. Run — one call does everything
    summary, details = compare_models_filtered(
        base_data, 
        const_data, 
        experiments,
        include_dual=True,
    )

    # 4. Print results
    print(summary.to_string(index=False))