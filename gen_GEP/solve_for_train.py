import argparse, os, glob, pickle, copy, re, time
import numpy as np
import torch
import json
from sklearn.cluster import KMeans
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from gep_benders import BendersSolver
from gep_problem_operational import GEPOperationalProblemSet
from paths import (
    add_path_args,
    ensure_dir,
    harvest_path,
    resolve_roots,
    topology_tag,
    under_repo,
    under_root,
)

#! Must match create_GEP_for_training.py. Both build the directory name from
#! the same config through topology_tag, so they cannot drift apart.
IN_PREFIX = "data/GEP_perturb_mix"
HORIZON   = 219

parser = argparse.ArgumentParser(
    description="Solve GEP instances with exact Benders and harvest ED training data."
)
add_path_args(parser)                       # --home-path / --data-root / --output-root
parser.add_argument(
    "-c", "--config",
    default="configs/config.json",
    help="Run config JSON. Relative paths resolve against the repository.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=None,
    help="Hours per instance, i.e. which stage 1 output to read. Left unset, "
         "the directory is discovered automatically; pass it only when several "
         "horizons exist for this topology.",
)
cli_args = parser.parse_args()

args = json.load(open(under_repo(cli_args.config), "r"))

roots = resolve_roots(args, cli_args)
data_root = roots["data_root"]

#! Stage 1's horizon is a flag now, so rather than assume a value, find the
#! directory it actually produced for this topology. Only ask for --horizon
#! when the answer is genuinely ambiguous.
_stem = under_root(f"{IN_PREFIX}_{topology_tag(args)}", data_root)
if cli_args.horizon is not None:
    DIR = f"{_stem}_H{cli_args.horizon}"
else:
    _candidates = sorted(
        d for d in glob.glob(f"{_stem}_H*") if glob.glob(os.path.join(d, "gep_instance_*.pkl"))
    )
    if len(_candidates) > 1:
        raise SystemExit(
            "Several horizons exist for this topology:\n  "
            + "\n  ".join(os.path.basename(d) for d in _candidates)
            + "\nPick one with --horizon <hours>."
        )
    #! No match: fall back to the default name so the error below can print it.
    DIR = _candidates[0] if _candidates else f"{_stem}_H{HORIZON}"

#! The horizon we are actually reading, taken from the directory name rather
#! than assumed, so the harvest is labelled with the horizon that produced it.
_match = re.search(r"_H(\d+)$", os.path.basename(DIR))
HORIZON = int(_match.group(1)) if _match else HORIZON

#! Name derived from topology + horizon so harvests never collide.
RELATIVE_OUT = harvest_path(args, HORIZON)
ED_OUT = under_root(RELATIVE_OUT, data_root)
ensure_dir(os.path.dirname(ED_OUT))

print(f"Run config: {under_repo(cli_args.config)}")
print(f"Instances:  {DIR}")
print(f"Harvest to: {ED_OUT}")

#! main.py reads ED_args.direct_data_path. If it does not name the file we are
#! about to write, training would load something else entirely -- say so now
#! rather than let it be discovered three stages later.
_configured = args["ED_args"].get("direct_data_path")
if _configured and os.path.normpath(_configured) != os.path.normpath(RELATIVE_OUT):
    print(
        f"[warn] this config's ED_args.direct_data_path is\n"
        f"           {_configured}\n"
        f"       but this harvest will be written to\n"
        f"           {RELATIVE_OUT}\n"
        f"       main.py reads the former, so set it to the latter "
        f"(or re-run stage 1 at horizon {args['Benders_args']['sample_duration']})."
    )

SPLIT_SEED = 42   # near your other knobs at the top, for reproducibility
N_HOUR_CLUSTERS   = 20    # representative hour-groups per instance
HOURS_PER_CLUSTER = 1     # hours kept per cluster
N_INV_CLUSTERS = 30       # Unique amount of Inv Clusters

CUT_SELECTION = "single"  # "single" or "full" (full = all cuts, single = one cut per iteration)

def build_operational_companion(gep_data, args):
    a = copy.deepcopy(args); a["ED_args"] = copy.deepcopy(args["ED_args"])
    a["ED_args"]["2n_synthetic_samples"] = 0
    a["ED_args"]["generate_capacity_sobol"]   = False
    a["ED_args"]["synthetic_demand_capacity"] = False
    a["ED_args"]["gen_data_constraint"]       = False
    dummy_inv = torch.zeros((1, gep_data.num_g), dtype=torch.float64)
    return GEPOperationalProblemSet(
        a, gep_data.T, gep_data.N, gep_data.G, gep_data.L,
        gep_data.pDemand, gep_data.pGenAva, gep_data.pVOLL, gep_data.pWeight,
        gep_data.pRamping, gep_data.pInvCost, gep_data.pVarCost,
        gep_data.pUnitCap, gep_data.pExpCap, gep_data.pImpCap,
        pUnitInvestment_Input=dummy_inv)

def solve_and_harvest(instance_path, args, cut_selection="single", cut_selection_k=1,
                      check_licence=False):
    with open(instance_path, "rb") as f:
        gep_data = pickle.load(f)
    op = build_operational_companion(gep_data, args)
    solver = BendersSolver(
        gep_data=gep_data, operational_data=op, sample=0,
        primal_net=None, dual_net=None, exact=True, exact_refinement=False,
        max_investment=args["Benders_args"]["max_investment"], init_investment="Zero",
        cut_selection=cut_selection, cut_selection_k=cut_selection_k,
        parallel_subproblems=False, n_workers=None)
    if check_licence:
        solver.check_licence(gep_data, label=f"{topology_tag(args)}, H{HORIZON}")
    t0 = time.time()
    y_direct, obj_direct = solver.solve_matrix_problem(gep_data, 0)
    ub, lb, cuts, inv_all, subobjs, iters = solver.solve_with_benders(gep_data, False, 0)
    wall = time.time() - t0
    traj  = np.asarray(solver.inv_hist, dtype=float)
    u_opt = np.asarray(inv_all[-1], dtype=float) if len(inv_all) > 0 else traj[-1]
    return {"instance_path": instance_path, "trajectory": traj, "u_opt": u_opt,
            "obj_direct": float(obj_direct), "gap_rel": float((ub-lb)/max(1.0, abs(ub))),
            "iterations": int(iters), "n_traj_points": int(traj.shape[0]),
            "wall_sec": wall, "horizon": len(gep_data.T)}

def cluster_investments(traj, n_clusters, seed=0):
    """Cluster an instance's trajectory investments; keep centroid-nearest per cluster."""
    traj = np.unique(traj, axis=0)                     # drop exact dups first (cheap)
    if traj.shape[0] <= n_clusters:
        return traj                                    # already fewer than target
    tz = (traj - traj.mean(0)) / (traj.std(0) + 1e-9)  # standardize: generators differ in scale
    km = KMeans(min(n_clusters, traj.shape[0]), random_state=seed, n_init=10).fit(tz)
    keep = []
    for c in range(km.n_clusters):
        idx = np.where(km.labels_ == c)[0]
        if len(idx) == 0:
            continue
        d = ((tz[idx] - km.cluster_centers_[c])**2).sum(1)
        keep.append(idx[d.argmin()])
    return traj[np.sort(keep)]

def select_representative_hours(gep_data, n_clusters, per_cluster_cap, seed=0):
    """Cluster hours by standardized (demand, A*UnitCap) — avail cap at unit investment."""
    T, N, G = list(gep_data.T), gep_data.N, gep_data.G
    rows = []
    for t in T:
        d = [gep_data.pDemand[(n, t)] for n in N]
        # A_(g,t) * UnitCap_g  — capacity each generator could supply this hour per unit built
        ac = [gep_data.pGenAva.get((*g, t), 1.0) * gep_data.pUnitCap[g] for g in G]
        rows.append(d + ac)
    H = np.asarray(rows, dtype=float)
    H = (H - H.mean(0)) / (H.std(0) + 1e-9)
    k = min(n_clusters, len(T))
    labels = KMeans(k, random_state=seed, n_init=10).fit_predict(H)
    rng, keep = np.random.default_rng(seed), []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        keep.extend(idx if len(idx) <= per_cluster_cap
                    else rng.choice(idx, per_cluster_cap, replace=False))
    return [T[i] for i in np.sort(np.asarray(keep))]

def restrict_to_hours(gep_data, kept_hours):
    remap   = {old: new for new, old in enumerate(kept_hours, start=1)}
    pDemand = {(n, remap[t]): gep_data.pDemand[(n, t)]
               for n in gep_data.N for t in kept_hours}
    pGenAva = {(nn, tech, remap[tt]): v
               for (nn, tech, tt), v in gep_data.pGenAva.items() if tt in remap}
    return pDemand, pGenAva, list(range(1, len(kept_hours) + 1))


def restrict_to_hours(gep_data, kept_hours):
    remap   = {old: new for new, old in enumerate(kept_hours, start=1)}
    pDemand = {(n, remap[t]): gep_data.pDemand[(n, t)]
               for n in gep_data.N for t in kept_hours}
    pGenAva = {(nn, tech, remap[tt]): v
               for (nn, tech, tt), v in gep_data.pGenAva.items() if tt in remap}
    return pDemand, pGenAva, list(range(1, len(kept_hours) + 1))

def build_instance_X(h, args, n_hour_clusters, hours_per_cluster):
    with open(h["instance_path"], "rb") as f:
        gep_data = pickle.load(f)

    kept = select_representative_hours(gep_data, n_hour_clusters, hours_per_cluster)
    pDemand, pGenAva, new_T = restrict_to_hours(gep_data, kept)

    traj = cluster_investments(h["trajectory"], N_INV_CLUSTERS)
    U, Tn = traj.shape[0], len(new_T)
    pUnitInvestment = torch.tensor(np.repeat(traj, Tn, axis=0), dtype=torch.float64)

    a = copy.deepcopy(args); a["ED_args"] = copy.deepcopy(args["ED_args"])
    a["ED_args"]["2n_synthetic_samples"] = 0
    op = GEPOperationalProblemSet(
        a, new_T, gep_data.N, gep_data.G, gep_data.L,
        pDemand, pGenAva, gep_data.pVOLL, gep_data.pWeight,
        gep_data.pRamping, gep_data.pInvCost, gep_data.pVarCost,
        gep_data.pUnitCap, gep_data.pExpCap, gep_data.pImpCap,
        pUnitInvestment_Input=pUnitInvestment)
    op.n_samples       = pUnitInvestment.shape[0]
    op.pUnitInvestment = pUnitInvestment
    X = op.build_X()
    return op, X, U

# ================= RUN =================
paths = sorted(glob.glob(os.path.join(DIR, "gep_instance_*.pkl")))
print(f"DIR PATH: {DIR}")
print(len(paths), "GEP instances found for training data harvest")

#! Say so here rather than crashing further down on an empty harvest list.
if not paths:
    raise SystemExit(
        f"No gep_instance_*.pkl in {DIR}\n"
        f"Run stage 1 first, with the same config and the same --home-path:\n"
        f"    python gen_GEP/create_GEP_for_training.py "
        f"-c {cli_args.config} --home-path {roots['home_path']}"
    )
harvest = []
for i, p in enumerate(paths):
    print(f"\n===== solving {os.path.basename(p)} =====")
    #! Only on the first instance: if the licence cannot take a model this
    #! size it never will, and every instance here is the same shape.
    h = solve_and_harvest(p, args, CUT_SELECTION, 1, check_licence=(i == 0))
    harvest.append(h)
    print(f"  iters={h['iterations']}  traj_pts={h['n_traj_points']}  "
          f"gap={h['gap_rel']:.2e}  wall={h['wall_sec']:.1f}s")



#! The diagnostic histograms that used to live here were removed: they called
#! plt.show() (useless in a batch job) and labelled the axes from a hardcoded
#! 3-node generator list, which is wrong for any other topology. The two counts
#! below were the only numbers worth keeping.
raw_inv = np.vstack([np.unique(h["trajectory"], axis=0) for h in harvest])
clust_inv = np.vstack([cluster_investments(h["trajectory"], N_INV_CLUSTERS) for h in harvest])

print(f"raw unique investments:      {raw_inv.shape[0]:,}")
print(f"after clustering (kept):     {clust_inv.shape[0]:,}")



# build features, dedup investments per instance
base, Xs, n_inv_total = None, [], 0
for h in harvest:
    op, X, U = build_instance_X(h, args, N_HOUR_CLUSTERS, HOURS_PER_CLUSTER)
    Xs.append(X); n_inv_total += U
    if base is None: base = op

X_all    = torch.cat(Xs, dim=0)
X_unique = torch.unique(X_all, dim=0)                     # <-- dedup states (repeated hours / cross-instance)

# shuffle out of sorted order, reproducibly, so downstream contiguous splits are representative
g = torch.Generator().manual_seed(SPLIT_SEED)
perm = torch.randperm(X_unique.shape[0], generator=g)
X_unique = X_unique[perm]

print(f"\nUnique investment datapoints: {n_inv_total:,}")
print(f"ED states before dedup:       {X_all.shape[0]:,}")
print(f"ED states after dedup:        {X_unique.shape[0]:,}   (labels solved on these)")


# labels once, on unique states only
base.X           = X_unique
base.xdim        = X_unique.shape[1]
base.n_samples   = X_unique.shape[0]
base.opt_targets = base.compute_opt_targets()
base.total_demands = torch.ones((X_unique.shape[0], 1))
base.pUnitInvestment = None      # stale after state-dedup; trainer doesn't use it

# keep heuristic labels consistent with final X (config has them on; harmless if unused)
if base.ED_args.get("precompute_heuristic_lambda_labels", False):
    (base.heuristic_lambda_soft_labels, base.heuristic_lambda_confidence,
     base.heuristic_lambda_tier, base.heuristic_lambda_classes) = \
        base.compute_heuristic_lambda_soft_labels_vectorized(X=base.X)

with open(ED_OUT, "wb") as f:
    pickle.dump(base, f)
print(f"\nSaved -> {ED_OUT}  |  X dim {base.xdim}  |  targets {list(base.opt_targets.keys())}")