"""Build or load the ED dataset used by the flow-first experiments.

Thin wrapper around the existing ``create_gep_ed_dataset`` so that every
script (fill tests, training) sees the same instances, labelled once by
Gurobi. Run from the repo root to build the dataset and print a summary:

    python -m flowfirst.dataset [path/to/config.json]

The config may carry ``ED_args.var_cost_override``, a list of
``[country, technology, cost]`` entries that replace the variable cost from
``inputs/iGEP_data_generation.csv`` before the instances are built. It exists
to make every dispatchable cost in the system distinct.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import gurobipy as gp
import numpy as np
import torch

from create_gep_dataset import create_gep_ed_dataset
from gep_config_parser import parse_config

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent
TOML_PATH = str(REPO_ROOT / "configs" / "config.toml")
DEFAULT_CONFIG = str(PKG_DIR / "config-3node.json")


def dataset_path(args):
    ed = args["ED_args"]
    return str(PKG_DIR / "datasets" / f"{ed['specific_name']}_smp{ed['2n_synthetic_samples']}.pkl")


def apply_cost_overrides(inputs, overrides):
    """Return inputs with the variable cost replaced for the listed generators."""
    if not overrides:
        return inputs
    gen = inputs["generation_data"].copy()
    for country, tech, cost in overrides:
        mask = (gen["Country"] == country) & (gen["Technology"] == tech)
        assert mask.any(), f"no generator {country}-{tech} in the generation data"
        gen.loc[mask, "VarCost_kEUR_per_MWh"] = float(cost)
    return {**inputs, "generation_data": gen}


def load_or_create(args_path=DEFAULT_CONFIG, seed=0):
    """Return (data, args). Builds and pickles the dataset on first use."""
    with open(args_path) as f:
        args = json.load(f)
    path = dataset_path(args)
    if not os.path.exists(path):
        np.random.seed(seed)
        torch.manual_seed(seed)
        gp.setParam("OutputFlag", 0)  # silence the per-instance Gurobi log
        inputs = parse_config(TOML_PATH)["experiment"]["experiments"][0]
        inputs = apply_cost_overrides(inputs, args["ED_args"].get("var_cost_override", []))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        create_gep_ed_dataset(args=args, problem_args=args["ED_args"], inputs=inputs,
                              problem_type="ED", save_path=path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data, args


def summarize(data):
    X = data.X
    N, G = data.num_n, data.num_g
    D = X[:, :N]
    cap = X[:, N:N + G]
    y = data.opt_targets["y_operational"]
    p, f, md = data.split_dec_vars_from_Y(y)
    lamb = -data.opt_targets["lamb_operational"]  # stored sign-flipped; this is the price

    print(f"instances: {X.shape[0]}, xdim: {X.shape[1]}, label time: {data.opt_label_runtime:.1f}s")
    print("generators and variable cost [kEUR/MWh]:")
    for g, c in zip(data.G, data.cost_vec.tolist(), strict=True):
        print(f"  {g[0]}-{g[1]:<8} {c:g}")
    print(f"VOLL: {data.pVOLL}")

    node_cap = cap @ data.node_to_gen_mask.T
    ratio = node_cap / D
    print("local capacity / demand per node, quantiles 5/25/50/75/95 %:")
    for i, n in enumerate(data.N):
        q = torch.quantile(ratio[:, i], torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], dtype=ratio.dtype))
        print(f"  {n}: " + "  ".join(f"{v:.2f}" for v in q.tolist()) + f"   share<1: {(ratio[:, i] < 1).float().mean():.2f}")

    f_lb = torch.tensor([-data.pImpCap[l] for l in data.L], dtype=f.dtype)
    f_ub = torch.tensor([data.pExpCap[l] for l in data.L], dtype=f.dtype)
    sat = (f <= f_lb + 1e-6) | (f >= f_ub - 1e-6)
    print("optimal solution statistics:")
    print(f"  instances with unmet demand > 1e-6 MW: {(md.sum(1) > 1e-6).float().mean():.3f}")
    print(f"  instances with at least one saturated line: {sat.any(1).float().mean():.3f}")
    print("  per-line saturation rate: " + ", ".join(f"{l[0]}-{l[1]} {s:.2f}" for l, s in zip(data.L, sat.float().mean(0).tolist(), strict=True)))
    print("  mean |f*| / line limit: " + ", ".join(f"{v:.2f}" for v in (f.abs() / torch.maximum(f_ub, -f_lb)).mean(0).tolist()))
    alphabet = sorted(set(data.cost_vec.tolist()) | {data.pVOLL})
    hist = {a: 0 for a in alphabet}
    off = 0
    for v in lamb.flatten().tolist():
        close = [a for a in alphabet if abs(v - a) < 1e-7]
        if close:
            hist[close[0]] += 1
        else:
            off += 1
    total = lamb.numel()
    print("nodal price alphabet occupancy (share of node-instances):")
    for a in alphabet:
        print(f"  {a:g}: {hist[a] / total:.3f}")
    print(f"  not in alphabet (kink / degenerate): {off / total:.3f}")


if __name__ == "__main__":
    args_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    data, _ = load_or_create(args_path)
    summarize(data)
