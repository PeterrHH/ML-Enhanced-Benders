import argparse, os, glob, pickle, copy, re, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
import json
import math
from sklearn.cluster import KMeans
import sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from gep_benders import BendersSolver, get_shared_env
from gep_problem_operational import GEPOperationalProblemSet, solve_matrix_problem_simple
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

SPLIT_SEED = 42   # near your other knobs at the top, for reproducibility
N_HOUR_CLUSTERS   = 20    # representative hour-groups per instance
HOURS_PER_CLUSTER = 1     # hours kept per cluster


TARGET_STATES = 48000

CUT_SELECTION = "single"  # "single" or "full" (full = all cuts, single = one cut per iteration)

#! Label LPs are tiny and cheap; this many chunks per worker keeps every core
#! busy to the end without paying the pickling overhead of one task per LP.
LABEL_CHUNKS_PER_WORKER = 4


def parse_cli():
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
    parser.add_argument(
        "-j", "--workers",
        type=int,
        default=None,
        help="Worker processes. Defaults to the CPUs allotted to this job "
             "($SLURM_CPUS_PER_TASK, else the affinity mask). 1 runs serially "
             "in-process, which is easiest to debug.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads per worker, shared by torch, BLAS and Gurobi. Defaults to "
             "CPUs // workers so the pool never oversubscribes the node.",
    )
    parser.add_argument(
        "--check-direct",
        action="store_true",
        help="Also solve each instance as one monolithic MIP and print its "
             "objective next to Benders'. Off by default: it is the most "
             "expensive solve per instance and the harvest does not use it.",
    )
    return parser.parse_args()


def available_cpus():
    """CPUs this job may actually use, not the node's total.

    On a shared SLURM node os.cpu_count() reports every core on the machine;
    sizing the pool from it oversubscribes the cores we were allotted.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return int(slurm)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def resolve_instance_dir(args, data_root, horizon):
    """Stage 1's horizon is a flag now, so rather than assume a value, find the
    directory it actually produced for this topology. Only ask for --horizon
    when the answer is genuinely ambiguous."""
    stem = under_root(f"{IN_PREFIX}_{topology_tag(args)}", data_root)
    if horizon is not None:
        return f"{stem}_H{horizon}"
    candidates = sorted(
        d for d in glob.glob(f"{stem}_H*") if glob.glob(os.path.join(d, "gep_instance_*.pkl"))
    )
    if len(candidates) > 1:
        raise SystemExit(
            "Several horizons exist for this topology:\n  "
            + "\n  ".join(os.path.basename(d) for d in candidates)
            + "\nPick one with --horizon <hours>."
        )
    #! No match: fall back to the default name so the error below can print it.
    return candidates[0] if candidates else f"{stem}_H{HORIZON}"


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

def make_solver(gep_data, args, cut_selection="single", cut_selection_k=1):
    op = build_operational_companion(gep_data, args)
    return BendersSolver(
        gep_data=gep_data, operational_data=op, sample=0,
        primal_net=None, dual_net=None, exact=True, exact_refinement=False,
        max_investment=args["Benders_args"]["max_investment"], init_investment="Zero",
        cut_selection=cut_selection, cut_selection_k=cut_selection_k,
        parallel_subproblems=False, n_workers=None)

def check_licence(instance_path, args, horizon):
    """Fail once, in the parent, if the licence cannot take a model this size.

    Every instance here is the same shape, so one probe answers for all of
    them -- and doing it before the pool starts means one clear error rather
    than every worker dying at once with a Gurobi traceback.
    """
    with open(instance_path, "rb") as f:
        gep_data = pickle.load(f)
    make_solver(gep_data, args).check_licence(
        gep_data, label=f"{topology_tag(args)}, H{horizon}")

def solve_and_harvest(instance_path, args, cut_selection="single", cut_selection_k=1,
                      check_direct=False):
    with open(instance_path, "rb") as f:
        gep_data = pickle.load(f)
    solver = make_solver(gep_data, args, cut_selection, cut_selection_k)
    t0 = time.time()
    obj_direct = None
    if check_direct:
        _, obj_direct = solver.solve_matrix_problem(gep_data, 0)
    ub, lb, cuts, inv_all, subobjs, iters = solver.solve_with_benders(gep_data, False, 0)
    wall = time.time() - t0
    #! Round: the master's u is integer, but Gurobi returns it with ~1e-10
    #! noise (and -0.0) that depends on its thread count. Left in, np.unique
    #! keeps near-duplicates apart and KMeans then keeps different
    #! representatives -- so the harvest changed with the number of workers.
    #! (+ 0.0 folds -0.0 into 0.0.)
    traj  = np.rint(np.asarray(solver.inv_hist, dtype=float)) + 0.0
    u_opt = np.rint(np.asarray(inv_all[-1], dtype=float)) + 0.0 if len(inv_all) > 0 else traj[-1]
    return {"instance_path": instance_path, "trajectory": traj, "u_opt": u_opt,
            "obj_benders": float(ub),
            "obj_direct": None if obj_direct is None else float(obj_direct),
            "gap_rel": float((ub-lb)/max(1.0, abs(ub))),
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

def build_instance_X(h, args, n_hour_clusters, n_inv_clusters):
    with open(h["instance_path"], "rb") as f:
        gep_data = pickle.load(f)

    kept = select_representative_hours(gep_data, n_hour_clusters, HOURS_PER_CLUSTER)
    pDemand, pGenAva, new_T = restrict_to_hours(gep_data, kept)

    traj = cluster_investments(h["trajectory"], n_inv_clusters) 
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


# ================= WORKERS =================
#! Workers are separate processes (spawned, see main), so each one gets its own
#! Gurobi Env through get_shared_env -- Envs cannot cross a process boundary.
#! Under a WLS licence that is one licence session per worker.

def _init_worker(threads):
    #! Not 1: get_sample_matrices rebuilds the dense full-horizon matrices every
    #! Benders iteration, and single-threaded torch makes that the bottleneck.
    torch.set_num_threads(threads)
    get_shared_env().setParam("Threads", threads)

def _harvest_instance(i, path, args, check_direct, keep_op, n_inv_clusters, n_hour_clusters):
    h = solve_and_harvest(path, args, CUT_SELECTION, 1, check_direct=check_direct)
    op, X, U = build_instance_X(h, args, n_hour_clusters, n_inv_clusters)  
    h["n_raw_unique"] = int(np.unique(h["trajectory"], axis=0).shape[0])
    h["n_inv_kept"] = int(U)
    #! Only the first instance's op becomes the saved dataset object; shipping
    #! every one back would just be pickled and dropped.
    return i, h, X.numpy(), pickle.dumps(op) if keep_op else None

def _label_chunk(start, obj_coeff, eq_cm, ineq_cm, eq_rhs, ineq_rhs):
    """compute_opt_targets for rows [start, start + len(eq_rhs)).

    Same LP and same sign convention as GEPOperationalProblemSet.compute_opt_targets.
    """
    #! Labels are small LPs; extra Gurobi threads only add overhead here.
    get_shared_env().setParam("Threads", 1)
    y, mu, lamb, obj = [], [], [], []
    for i in range(len(eq_rhs)):
        y_op, obj_op, dual_eq, dual_ineq = solve_matrix_problem_simple(
            obj_coeff, eq_cm, ineq_cm, eq_rhs[i], ineq_rhs[i])
        y.append(np.asarray(y_op))
        #! Negate duals, these are flipped in Gurobi.
        mu.append(-np.asarray(dual_ineq))
        lamb.append(-np.asarray(dual_eq))
        obj.append(obj_op)
    return start, np.stack(y), np.stack(mu), np.stack(lamb), np.asarray(obj)

def _run(pool, fn, jobs):
    """Yield fn(*job) results as they finish; serially in-process when pool is None."""
    if pool is None:
        for job in jobs:
            yield fn(*job)
        return
    futures = {pool.submit(fn, *job): job for job in jobs}
    try:
        for fut in as_completed(futures):
            yield fut.result()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        raise

def compute_opt_targets_parallel(base, pool, n_workers):
    obj_coeff = base.obj_coeff.numpy()
    eq_cm = base.eq_cm.numpy()
    ineq_cm = base.ineq_cm.numpy()
    eq_rhs, ineq_rhs = base.split_X(base.X)
    eq_rhs, ineq_rhs = eq_rhs.numpy(), ineq_rhs.numpy()

    n = eq_rhs.shape[0]
    size = max(1, -(-n // (n_workers * LABEL_CHUNKS_PER_WORKER)))
    jobs = [(s, obj_coeff, eq_cm, ineq_cm, eq_rhs[s:s + size], ineq_rhs[s:s + size])
            for s in range(0, n, size)]

    parts, done = {}, 0
    for start, y, mu, lamb, obj in _run(pool, _label_chunk, jobs):
        parts[start] = (y, mu, lamb, obj)
        done += len(obj)
        print(f"  labels {done:,}/{n:,}", flush=True)

    order = sorted(parts)
    stack = lambda k: torch.tensor(np.concatenate([parts[s][k] for s in order]), dtype=base.DTYPE)
    return {
        "y_operational": stack(0),
        "mu_operational": stack(1),
        "lamb_operational": stack(2),
        "obj": stack(3),
    }


# ================= RUN =================
def main():
    cli_args = parse_cli()
    args = json.load(open(under_repo(cli_args.config), "r"))

    roots = resolve_roots(args, cli_args)
    data_root = roots["data_root"]

    DIR = resolve_instance_dir(args, data_root, cli_args.horizon)

    #! The horizon we are actually reading, taken from the directory name rather
    #! than assumed, so the harvest is labelled with the horizon that produced it.
    _match = re.search(r"_H(\d+)$", os.path.basename(DIR))
    horizon = int(_match.group(1)) if _match else HORIZON

    #! Name derived from topology + horizon so harvests never collide.
    RELATIVE_OUT = harvest_path(args, horizon)
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

    paths = sorted(glob.glob(os.path.join(DIR, "gep_instance_*.pkl")))
    
    print(f"DIR PATH: {DIR}")

    per_instance = TARGET_STATES / max(1, len(paths))
    n_hour_clusters = N_HOUR_CLUSTERS                     
    n_inv_clusters  = max(1, round(per_instance / n_hour_clusters))
    print(f"Targeting {TARGET_STATES:,} states: {len(paths)} instances "
          f"× {n_inv_clusters} inv × {n_hour_clusters} hours "
          f"= {len(paths) * n_inv_clusters * n_hour_clusters:,}")

    #! Say so here rather than crashing further down on an empty harvest list.
    if not paths:
        raise SystemExit(
            f"No gep_instance_*.pkl in {DIR}\n"
            f"Run stage 1 first, with the same config and the same --home-path:\n"
            f"    python gen_GEP/create_GEP_for_training.py "
            f"-c {cli_args.config} --home-path {roots['home_path']}"
        )

    cpus = available_cpus()
    n_workers = max(1, cli_args.workers or cpus)
    threads = cli_args.threads or max(1, cpus // n_workers)
    print(f"CPUs: {cpus}  |  workers: {n_workers}  |  threads/worker: {threads}")

    check_licence(paths[0], args, horizon)

    pool = None
    if n_workers > 1:
        #! BLAS/OpenMP pools (numpy, sklearn's KMeans) size themselves to the
        #! whole node at import. Spawned children read these at import, so the
        #! pool as a whole stays inside the CPUs we were given.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(threads)
        #! spawn, not fork: the parent already holds a started Gurobi Env from
        #! the licence check, and a forked copy of it is not safe to use.
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(threads,),
        )

    try:
        harvest, Xs, base_bytes = [None] * len(paths), [None] * len(paths), None
        jobs = [(i, p, args, cli_args.check_direct, i == 0, n_inv_clusters, n_hour_clusters)
        for i, p in enumerate(paths)]
        t0 = time.time()
        for done, (i, h, X, op_bytes) in enumerate(_run(pool, _harvest_instance, jobs), 1):
            harvest[i], Xs[i] = h, X
            if op_bytes is not None:
                base_bytes = op_bytes
            direct = "" if h["obj_direct"] is None else \
                f"  obj_direct={h['obj_direct']:.6g} obj_benders={h['obj_benders']:.6g}"
            print(f"[{done}/{len(paths)}] {os.path.basename(h['instance_path'])}  "
                  f"iters={h['iterations']}  traj_pts={h['n_traj_points']}  "
                  f"gap={h['gap_rel']:.2e}  wall={h['wall_sec']:.1f}s{direct}", flush=True)
        print(f"Benders + features for {len(paths)} instances: {time.time() - t0:.1f}s")

        #! The diagnostic histograms that used to live here were removed: they called
        #! plt.show() (useless in a batch job) and labelled the axes from a hardcoded
        #! 3-node generator list, which is wrong for any other topology. The two counts
        #! below were the only numbers worth keeping.
        n_inv_total = sum(h["n_inv_kept"] for h in harvest)
        print(f"raw unique investments:      {sum(h['n_raw_unique'] for h in harvest):,}")
        print(f"after clustering (kept):     {n_inv_total:,}")

        base = pickle.loads(base_bytes)
        X_all    = torch.cat([torch.from_numpy(X) for X in Xs], dim=0)
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
        t0 = time.time()
        base.opt_targets = compute_opt_targets_parallel(base, pool, n_workers)
        print(f"Labels: {time.time() - t0:.1f}s")
    finally:
        if pool is not None:
            pool.shutdown()

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


#! Required: spawned workers re-import this file, and without the guard each
#! one would rerun the whole harvest.
if __name__ == "__main__":
    main()
