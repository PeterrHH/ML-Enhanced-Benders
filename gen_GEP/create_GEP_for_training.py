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
def perturb_availability(A, rng, sigma=0.0, hi=0.999, lo=1e-6, block=1):
    """Interior-only availability noise. With block > 1 one draw is shared by the
    `block` consecutive hours of each window, so the noise adds no hour-to-hour jumps
    for the ramp limits to react to."""
    if sigma <= 0.0:
        return A
    A = A.copy()
    mask = (A > lo) & (A < hi)
    if block == 1:
        noise = rng.normal(0.0, sigma, size=A.shape)
    else:
        noise = np.repeat(rng.normal(0.0, sigma, size=(A.shape[0] // block, A.shape[1])), block, axis=0)
    A[mask] = np.clip(A[mask] + noise[mask], 0.0, 1.0)
    return A


def draw_hours(pool, horizon, rng, block=1):
    """Hours of the pool to build an instance from, as positions into pool_times.

    block == 1: `horizon` independent hours, drawn with replacement (the original
    sampler, same random stream).
    block > 1: horizon // block windows of `block` consecutive hours. A window may
    start only where its whole span lies in the pool, so it never runs past the end
    of the year or into hours held out by POOL_TIMES. The instance renumbers its hours
    1..horizon and ramping blocks start at its first hour, so window k is exactly
    ramping block k, and no ramp limit links two windows drawn from different times.
    """
    times = pool["pool_times"]
    if block == 1:
        return rng.choice(len(times), size=horizon, replace=True)
    starts = np.flatnonzero(times[block - 1:] - times[:len(times) - block + 1] == block - 1)
    if len(starts) == 0:
        raise ValueError(f"The pool has no run of {block} consecutive hours to draw a window from.")
    picked = rng.choice(starts, size=horizon // block, replace=True)
    return (picked[:, None] + np.arange(block)[None, :]).ravel()

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
    ramp_block_hours=1,
):
    rng = np.random.default_rng(seed)
    countries = pool["countries"]
    B = int(ramp_block_hours)
    idx   = draw_hours(pool, horizon, rng, block=B)
    drawn = pool["pool_times"][idx]
    new_times = np.arange(1, horizon + 1)

    # --- demand: global scale (+ optional lognormal jitter: per hour, or per window when B > 1) ---
    D = pool["dem_wide"].loc[drawn].to_numpy().copy()
    D *= demand_scale
    if demand_lognormal_sigma > 0.0:
        shape = D.shape if B == 1 else (D.shape[0] // B, D.shape[1])
        f = rng.lognormal(mean=-0.5*demand_lognormal_sigma**2,
                          sigma=demand_lognormal_sigma, size=shape)
        D = D * (f if B == 1 else np.repeat(f, B, axis=0))
    demand_new = pd.DataFrame([
        {"Country": c, "Time": int(t), "Demand_MW": float(D[k, j])}
        for k, t in enumerate(new_times) for j, c in enumerate(countries)
    ])

    # --- availability: real drawn rows, interior-only perturbation ---
    ava_cols = list(pool["ava_wide"].columns)
    A = pool["ava_wide"].loc[drawn].to_numpy().copy()
    A = perturb_availability(A, rng, sigma=avail_sigma, block=B)
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
            "ramp_block_hours": B,
            "window_starts": drawn[::B].tolist() if B > 1 else None,
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
parser.add_argument(
    "--ramp-block-hours", "--ramp_block_hours",
    dest="ramp_block_hours",
    type=int,
    default=1,
    help="Build instances for ramping inside blocks of this many hours: hours are drawn "
         "as whole windows of consecutive hours, one per ramping block, and demand and "
         "availability noise is drawn once per window. 1 (default) draws single hours, "
         "as before. The horizon must be a multiple of it, e.g. 216 for 4, 6 or 8.",
)
cli_args = parser.parse_args()

N_INSTANCES = cli_args.n_instances
HORIZON     = cli_args.horizon
MASTER_SEED = cli_args.seed
RAMP_B      = cli_args.ramp_block_hours
if RAMP_B < 1:
    raise SystemExit(f"--ramp-block-hours must be at least 1, got {RAMP_B}.")
if HORIZON % RAMP_B:
    _near = max(RAMP_B, (HORIZON // RAMP_B) * RAMP_B)
    raise SystemExit(f"--horizon {HORIZON} is not a multiple of --ramp-block-hours {RAMP_B}; "
                     f"use e.g. --horizon {_near} (216 works for blocks of 4, 6 and 8).")

with open(under_repo(cli_args.config), "r") as file:
    args = json.load(file)

roots = resolve_roots(args, cli_args)
data_root = roots["data_root"]

#! e.g. data/GEP_perturb_mix_N20_G107_L44_H219 -- the topology and horizon are
#! in the name, so runs of different sizes coexist instead of overwriting one
#! another. solve_for_train.py rebuilds this exact name from the same config.
#! Ramping instances get _B<hours> as well: their hours come in windows, so they must
#! never mix with an hourly set of the same horizon.
OUT_DIR = ensure_dir(under_root(
    f"{OUT_PREFIX}_{topology_tag(args)}_H{HORIZON}" + (f"_B{RAMP_B}" if RAMP_B > 1 else ""), data_root
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
print(f"Instances:  {N_INSTANCES}  Horizon: {HORIZON}h  Seed: {MASTER_SEED}"
      + (f"  Ramping blocks: {RAMP_B}h ({HORIZON // RAMP_B} windows per instance)" if RAMP_B > 1 else ""))
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
        pool=pool, save_path=path, horizon=HORIZON, seed=1000 + i, ramp_block_hours=RAMP_B, **reg,
    )
    meta["instance_id"] = i
    meta["class"] = inst_class
    manifest.append(meta)

with open(os.path.join(OUT_DIR, "manifest.pkl"), "wb") as f:
    pickle.dump(manifest, f)

from collections import Counter
print(f"Built {len(manifest)} GEP instances -> {OUT_DIR}")
print("class counts:", dict(Counter(m["class"] for m in manifest)))