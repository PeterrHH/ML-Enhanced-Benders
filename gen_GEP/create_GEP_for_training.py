import argparse, copy, glob, pickle, os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from create_gep_dataset import create_gep_ed_dataset
from gep_config_parser import parse_config
from paths import (
    add_path_args,
    ensure_dir,
    resolve_roots,
    topology_tag,
    under_repo,
    under_root,
)

# ======================================================================
# 5. Driver config
# ======================================================================
HORIZON     = 219
N_INSTANCES = 80
MASTER_SEED = 42
#! Directory name is built from the config's topology further down, so a
#! 3-node and a 20-node run cannot overwrite each other and you can tell which
#! is which from the name alone. This is only the prefix.
OUT_PREFIX  = "data/GEP_perturb_mix"
POOL_TIMES  = None            # <sque-- Could be used to restrict the pool of hours for sampling

USE_CLASS_MIXTURE = True       # True: renewable-heavy/high-demand/middle mixture
                               # False: original pure-random perturbation
RENEWABLE_FRAC    = 0.4        # (class mixture only) share of renewable-heavy instances

# ======================================================================
# 1. Pool
# ======================================================================
def precompute_pool(inputs, countries, pool_times=None):
    dem = inputs["demand_data"]; ava = inputs["generation_availability_data"]
    dem = dem[dem["Country"].isin(countries)]
    ava = ava[ava["Country"].isin(countries)]
    dem_wide = dem.pivot(index="Time", columns="Country", values="Demand_MW")
    ava_wide = ava.pivot(index="Time", columns=["Country", "Technology"],
                         values="Availability_pu")
    if pool_times is None:
        print("WARNING: pool_times=None -> all hours used. Eval hours will leak.")
        pool_times = dem_wide.index.to_numpy()
    pool_times = np.asarray(sorted(set(pool_times)))
    return {"pool_times": pool_times,
            "dem_wide": dem_wide.loc[pool_times],
            "ava_wide": ava_wide.loc[pool_times],
            "countries": list(dem_wide.columns)}

# ======================================================================
# 2. Availability perturbation — interior values only
# ======================================================================
def perturb_availability(A, rng, sigma=0.0, hi=0.999, lo=1e-6):
    if sigma <= 0.0:
        return A
    A = A.copy()
    mask = (A > lo) & (A < hi)
    noise = rng.normal(0.0, sigma, size=A.shape)
    A[mask] = np.clip(A[mask] + noise[mask], 0.0, 1.0)
    return A

# ======================================================================
# 3. Build ONE instance
# ======================================================================
def build_one_gep_instance(
    inputs, base_problem_args, args, pool, save_path, horizon,
    demand_scale=1.0,
    cinv_perturb=0.0,
    renew_cinv_scale=1.0,          # <1 = cheaper renewables (drives optima to build more solar/wind)
    demand_lognormal_sigma=0.0,
    avail_sigma=0.0,
    seed=0,
):
    rng = np.random.default_rng(seed)
    countries = pool["countries"]
    idx   = rng.choice(len(pool["pool_times"]), size=horizon, replace=True)
    drawn = pool["pool_times"][idx]
    new_times = np.arange(1, horizon + 1)

    # --- demand: global scale (+ optional per-hour lognormal jitter) ---
    D = pool["dem_wide"].loc[drawn].to_numpy().copy()
    D *= demand_scale
    if demand_lognormal_sigma > 0.0:
        f = rng.lognormal(mean=-0.5*demand_lognormal_sigma**2,
                          sigma=demand_lognormal_sigma, size=D.shape)
        D = D * f
    demand_new = pd.DataFrame([
        {"Country": c, "Time": int(t), "Demand_MW": float(D[k, j])}
        for k, t in enumerate(new_times) for j, c in enumerate(countries)
    ])

    # --- availability: real drawn rows, interior-only perturbation ---
    ava_cols = list(pool["ava_wide"].columns)
    A = pool["ava_wide"].loc[drawn].to_numpy().copy()
    A = perturb_availability(A, rng, sigma=avail_sigma)
    ava_rows = []
    for k, t in enumerate(new_times):
        for j, (c, tech) in enumerate(ava_cols):
            v = A[k, j]
            if not np.isnan(v):
                ava_rows.append({"Country": c, "Technology": tech,
                                 "Time": int(t), "Availability_pu": float(v)})
    ava_new = pd.DataFrame(ava_rows)

    # --- C^inv perturbation: per-generator jitter + renewable-class cost skew ---
    RENEW = {"WindOff", "WindOn", "SunPV"}
    gen = inputs["generation_data"].copy()
    cinv_factors = {}
    for (c, tech) in [tuple(g) for g in base_problem_args["G"]]:
        fac = float(rng.uniform(1 - cinv_perturb, 1 + cinv_perturb)) if cinv_perturb > 0 else 1.0
        if tech in RENEW:
            fac *= renew_cinv_scale
        cinv_factors[f"{c}_{tech}"] = fac
        m = (gen["Country"] == c) & (gen["Technology"] == tech)
        gen.loc[m, "InvCost_kEUR_MW_year"] *= fac

    inst_inputs = dict(inputs)
    inst_inputs["demand_data"] = demand_new
    inst_inputs["generation_availability_data"] = ava_new
    inst_inputs["generation_data"] = gen
    inst_inputs["times"] = list(map(int, new_times))

    problem_args = copy.deepcopy(base_problem_args)
    problem_args["sample_duration"] = horizon

    data = create_gep_ed_dataset(
        args=args, problem_args=problem_args, inputs=inst_inputs,
        problem_type="GEP", save_path=save_path,
    )
    meta = {"save_path": save_path, "horizon": horizon, "seed": seed,
            "demand_scale": demand_scale, "cinv_perturb": cinv_perturb,
            "renew_cinv_scale": renew_cinv_scale,
            "demand_lognormal_sigma": demand_lognormal_sigma,
            "avail_sigma": avail_sigma, "cinv_factors": cinv_factors,
            "drawn_hours": drawn.tolist()}
    return data, meta

# ======================================================================
# 4. Samplers
# ======================================================================
def sample_perturbation_random(rng,
                               demand_scale_range=(0.6, 1.4),
                               cinv_perturb=0.3,
                               demand_lognormal_sigma=0.1,
                               avail_sigma=0.02):
    """Original: pure-random perturbation, symmetric C^inv, no renewable skew."""
    return dict(_class="random",
                demand_scale           = float(rng.uniform(*demand_scale_range)),
                cinv_perturb           = cinv_perturb,
                renew_cinv_scale       = 1.0,
                demand_lognormal_sigma = demand_lognormal_sigma,
                avail_sigma            = avail_sigma)

def sample_perturbation_mixture(rng, renewable_frac=0.4):
    """Class mixture: deliberate share pushing optima into the high-renewable region."""
    r = rng.uniform()
    if r < renewable_frac:
        return dict(_class="renewable_heavy",
                    demand_scale=float(rng.uniform(0.5, 0.9)),
                    cinv_perturb=0.2,
                    renew_cinv_scale=float(rng.uniform(0.35, 0.7)),
                    demand_lognormal_sigma=0.1, avail_sigma=0.02)
    elif r < renewable_frac + 0.25:
        return dict(_class="high_demand",
                    demand_scale=float(rng.uniform(1.1, 1.5)),
                    cinv_perturb=0.3, renew_cinv_scale=1.0,
                    demand_lognormal_sigma=0.1, avail_sigma=0.02)
    else:
        return dict(_class="middle",
                    demand_scale=float(rng.uniform(0.7, 1.3)),
                    cinv_perturb=0.3,
                    renew_cinv_scale=float(rng.uniform(0.7, 1.2)),
                    demand_lognormal_sigma=0.1, avail_sigma=0.02)

# ======================================================================
# Run
# ======================================================================
parser = argparse.ArgumentParser(
    description="Build perturbed GEP instances for PDL training data harvesting."
)
add_path_args(parser)                       # --home-path / --data-root / --output-root
parser.add_argument(
    "-c", "--config",
    default="configs/config.json",
    help="Run config JSON. Relative paths resolve against the repository. "
         "The topology (N/G/L) is read from its Benders_args.",
)
parser.add_argument(
    "--toml-config", "--toml_config",
    dest="toml_config",
    default="configs/config.toml",
    help="Input-data TOML config.",
)
parser.add_argument(
    "-n", "--n-instances", "--n_instances",
    dest="n_instances",
    type=int,
    default=N_INSTANCES,
    help=f"How many GEP instances to build (default {N_INSTANCES}). Larger "
         f"topologies cost far more to solve in stage 2, so fewer instances "
         f"are usually the right trade.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=HORIZON,
    help=f"Hours per instance (default {HORIZON}). Appears in the output "
         f"directory name, so different horizons never mix.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=MASTER_SEED,
    help=f"Master seed for the perturbation sampler (default {MASTER_SEED}).",
)
cli_args = parser.parse_args()

N_INSTANCES = cli_args.n_instances
HORIZON     = cli_args.horizon
MASTER_SEED = cli_args.seed

with open(under_repo(cli_args.config), "r") as file:
    args = json.load(file)

roots = resolve_roots(args, cli_args)
data_root = roots["data_root"]

#! e.g. data/GEP_perturb_mix_N20_G107_L44_H219 -- the topology and horizon are
#! in the name, so runs of different sizes coexist instead of overwriting one
#! another. solve_for_train.py rebuilds this exact name from the same config.
OUT_DIR = ensure_dir(under_root(
    f"{OUT_PREFIX}_{topology_tag(args)}_H{HORIZON}", data_root
))

#! Instance count is NOT in the directory name, so re-running with fewer
#! instances leaves the higher-numbered files from a previous run behind and
#! stage 2 would solve those too. Say so rather than let it pass silently.
_existing = len(glob.glob(os.path.join(OUT_DIR, "gep_instance_*.pkl")))
if _existing > N_INSTANCES:
    print(
        f"[warn] {OUT_DIR} already holds {_existing} instances but only "
        f"{N_INSTANCES} will be written. The remaining {_existing - N_INSTANCES} "
        f"are from an earlier run and stage 2 would solve them too. "
        f"Delete the directory first if that is not what you want."
    )

#! Derived from the config, not hardcoded: with a hardcoded 3-country list a
#! 20-node config would silently pool only BEL/GER/FRA and build instances
#! whose demand and availability do not match their own topology.
COUNTRIES = list(dict.fromkeys(args["Benders_args"]["N"]))

print(f"Run config: {under_repo(cli_args.config)}")
print(f"Countries ({len(COUNTRIES)}): {COUNTRIES}")
print(f"Generators: {len(args['Benders_args']['G'])}  Lines: {len(args['Benders_args']['L'])}")
print(f"Instances:  {N_INSTANCES}  Horizon: {HORIZON}h  Seed: {MASTER_SEED}")
print(f"Output dir: {OUT_DIR}")

input_data = parse_config(under_repo(cli_args.toml_config))
gep_ed_data = input_data["experiment"]["experiments"][0]
pool = precompute_pool(gep_ed_data, COUNTRIES, pool_times=POOL_TIMES)

master_rng = np.random.default_rng(MASTER_SEED)
print(f"Sampling mode: {'CLASS MIXTURE' if USE_CLASS_MIXTURE else 'PURE RANDOM'}")

manifest = []
for i in range(N_INSTANCES):
    if USE_CLASS_MIXTURE:
        reg = sample_perturbation_mixture(master_rng, renewable_frac=RENEWABLE_FRAC)
    else:
        reg = sample_perturbation_random(master_rng)
    inst_class = reg.pop("_class")
    path = os.path.join(OUT_DIR, f"gep_instance_{i:03d}.pkl")
    _, meta = build_one_gep_instance(
        inputs=gep_ed_data, base_problem_args=args["Benders_args"], args=args,
        pool=pool, save_path=path, horizon=HORIZON, seed=1000 + i, **reg,
    )
    meta["instance_id"] = i
    meta["class"] = inst_class
    manifest.append(meta)

with open(os.path.join(OUT_DIR, "manifest.pkl"), "wb") as f:
    pickle.dump(manifest, f)

from collections import Counter
print(f"Built {len(manifest)} GEP instances -> {OUT_DIR}")
print("class counts:", dict(Counter(m["class"] for m in manifest)))