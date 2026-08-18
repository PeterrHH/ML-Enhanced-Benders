import copy, pickle, os
import numpy as np
import sys
import pandas as pd
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from create_gep_dataset import create_gep_ed_dataset
from gep_config_parser import parse_config
import json

# ======================================================================
# 1. Pool: pivot demand & availability to Time-indexed wide tables
# ======================================================================
def precompute_pool(inputs, countries, pool_times=None):
    """pool_times: Time labels allowed for sampling (the training-block hours).
    None -> all hours, which WILL leak your eval hours. Pass the complement."""
    dem = inputs["demand_data"]
    ava = inputs["generation_availability_data"]
    dem = dem[dem["Country"].isin(countries)]
    ava = ava[ava["Country"].isin(countries)]

    dem_wide = dem.pivot(index="Time", columns="Country", values="Demand_MW")
    ava_wide = ava.pivot(index="Time", columns=["Country", "Technology"],
                         values="Availability_pu")

    if pool_times is None:
        print("WARNING: pool_times=None -> all hours used. Eval hours will leak.")
        pool_times = dem_wide.index.to_numpy()
    pool_times = np.asarray(sorted(set(pool_times)))

    return {
        "pool_times": pool_times,
        "dem_wide":   dem_wide.loc[pool_times],   # [T_pool, N]
        "ava_wide":   ava_wide.loc[pool_times],   # [T_pool, (country,tech)]  renewables
        "countries":  list(dem_wide.columns),
    }


# ======================================================================
# 2. Build ONE engineered GEP instance
# ======================================================================
def build_one_gep_instance(
    inputs, base_problem_args, args, pool, save_path,
    horizon,
    demand_scale=1.0,          # global load-level multiplier (primary spread lever)
    node_multipliers=None,     # {country: factor} spatial reallocation
    preserve_total=True,       # renormalize spatial shift to hold system total
    cinv_perturb=0.0,          # +/- fraction on InvCost per generator in G (master-only)
    seed=0,
):
    rng = np.random.default_rng(seed)
    countries = pool["countries"]

    # --- draw the instance's hours: uniform, with replacement, across the year ---
    # (scarcity hook: replace uniform p with a stress-weighted p here if needed)
    idx   = rng.choice(len(pool["pool_times"]), size=horizon, replace=True)
    drawn = pool["pool_times"][idx]
    new_times = np.arange(1, horizon + 1)

    # --- demand: node reallocation -> preserve total -> global scale ---
    D = pool["dem_wide"].loc[drawn].to_numpy().copy()        # [horizon, N]
    if node_multipliers:
        mult = np.array([node_multipliers.get(c, 1.0) for c in countries])
        orig_tot = D.sum(axis=1, keepdims=True)
        D = D * mult
        if preserve_total:
            D *= orig_tot / (D.sum(axis=1, keepdims=True) + 1e-9)
    D *= demand_scale

    demand_new = pd.DataFrame([
        {"Country": c, "Time": int(t), "Demand_MW": float(D[k, j])}
        for k, t in enumerate(new_times)
        for j, c in enumerate(countries)
    ])

    # --- availability: real rows re-indexed to new times (NEVER perturbed) ---
    A = pool["ava_wide"].loc[drawn]                          # [horizon, (c,tech)]
    ava_rows = []
    for k, t in enumerate(new_times):
        for (c, tech), val in A.iloc[k].items():
            if pd.notna(val):
                ava_rows.append({"Country": c, "Technology": tech,
                                 "Time": int(t), "Availability_pu": float(val)})
    ava_new = pd.DataFrame(ava_rows)

    # --- C^inv perturbation (only generators in G; invisible to the ED) ---
    gen = inputs["generation_data"].copy()
    cinv_factors = {}
    if cinv_perturb > 0.0:
        for (c, tech) in [tuple(g) for g in base_problem_args["G"]]:
            f = float(rng.uniform(1 - cinv_perturb, 1 + cinv_perturb))
            cinv_factors[f"{c}_{tech}"] = f
            m = (gen["Country"] == c) & (gen["Technology"] == tech)
            gen.loc[m, "InvCost_kEUR_MW_year"] *= f

    # --- assemble modified inputs; everything else copied by reference ---
    inst_inputs = dict(inputs)
    inst_inputs["demand_data"] = demand_new
    inst_inputs["generation_availability_data"] = ava_new
    inst_inputs["generation_data"] = gen
    inst_inputs["times"] = list(map(int, new_times))

    problem_args = copy.deepcopy(base_problem_args)
    problem_args["sample_duration"] = horizon    # drives the annualization weight

    data = create_gep_ed_dataset(
        args=args, problem_args=problem_args, inputs=inst_inputs,
        problem_type="GEP", save_path=save_path,
    )

    meta = {
        "save_path": save_path, "horizon": horizon, "seed": seed,
        "demand_scale": demand_scale, "node_multipliers": node_multipliers,
        "preserve_total": preserve_total, "cinv_perturb": cinv_perturb,
        "cinv_factors": cinv_factors, "drawn_hours": drawn.tolist(),
    }
    return data, meta


# ======================================================================
# 3. Driver
# ======================================================================
COUNTRIES = ["BEL", "GER", "FRA"]
HORIZON   = 219                       # <-- set from your solve-time budget
OUT_DIR   = "data/GEP_for_training"
POOL_TIMES = None                     # <-- replace with training-block hours (else leaks)
os.makedirs(OUT_DIR, exist_ok=True)

# Each dict = one "case". demand_scale swept wide (primary lever);
# a few spatial cases; C^inv on everywhere for mix diversity.
REGIMES = [
    dict(demand_scale=0.5, node_multipliers=None,                                   cinv_perturb=0.5),
    dict(demand_scale=0.7,  node_multipliers=None,                                   cinv_perturb=0.3),
    dict(demand_scale=0.85, node_multipliers=None,                                   cinv_perturb=0.3),
    dict(demand_scale=1.15, node_multipliers=None,                                   cinv_perturb=0.3),
    dict(demand_scale=1.3,  node_multipliers=None,                                   cinv_perturb=0.3),
    dict(demand_scale=1.5, node_multipliers=None,                                   cinv_perturb=0.5),
    dict(demand_scale=1.1,  node_multipliers={"FRA": 1.4, "GER": 0.8, "BEL": 0.8},   cinv_perturb=0.3),
    dict(demand_scale=0.9,  node_multipliers={"GER": 1.4, "FRA": 0.8, "BEL": 0.8},   cinv_perturb=0.3),
    dict(demand_scale=1.15, node_multipliers={"BEL": 1.5, "GER": 0.8, "FRA": 0.9},   cinv_perturb=0.3),
    dict(demand_scale=0.7, node_multipliers={"BEL": 1.2, "GER": 0.8, "FRA": 0.8},   cinv_perturb=0.5),
    dict(demand_scale=1.2,  node_multipliers={"GER": 1.4, "FRA": 0.8, "BEL": 0.8},   cinv_perturb=0.5),
    dict(demand_scale=0.9,  node_multipliers={"FRA": 1.2, "GER": 0.8, "BEL": 0.8},   cinv_perturb=0.5), 
    # add cases up to your solve budget
]
with open("configs/config.json", "r") as file:
    args = json.load(file)
input_data = parse_config("configs/config.toml") # Reads the input data using config.toml's experiment.inputs.data path.

gep_ed_data = input_data["experiment"]["experiments"][0]
pool = precompute_pool(gep_ed_data, COUNTRIES, pool_times=POOL_TIMES)

manifest = []
for i, reg in enumerate(REGIMES):
    path = os.path.join(OUT_DIR, f"gep_instance_{i:03d}.pkl")
    _, meta = build_one_gep_instance(
        inputs=gep_ed_data, base_problem_args=args["Benders_args"], args=args,
        pool=pool, save_path=path, horizon=HORIZON, seed=1000 + i, **reg,
    )
    meta["instance_id"] = i
    manifest.append(meta)

with open(os.path.join(OUT_DIR, "manifest.pkl"), "wb") as f:
    pickle.dump(manifest, f)
print(f"Built {len(manifest)} GEP instances -> {OUT_DIR}")