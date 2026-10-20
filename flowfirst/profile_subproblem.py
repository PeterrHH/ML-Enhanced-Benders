"""Where does an inexact (flow-first) subproblem pass spend its time, next to the exact LPs?

Mirrors one Benders iteration on `--hours` hourly instances of a flow-first run's dataset:
  * inexact: exactly what gep_benders.solve_matrix_problem_PDL calls (FlowFirstPrimal,
    FlowFirstDual, obj_fn / dual_obj_fn twice), timed stage by stage;
  * exact: the persistent MatrixLP re-solved hour by hour, as solve_subproblems does.

Run from the repo root:
  python flowfirst/profile_subproblem.py --run-dir <flow-first run dir> --hours 2190
  python flowfirst/profile_subproblem.py --run-dir <run> --hours 2190 --threads 1   # single-thread
  python flowfirst/profile_subproblem.py --config flowfirst/configs/config-6node-x8.json --variant flowfirst-gnn
      (untrained net, timing only; builds the config's dataset first if it is not cached)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
torch.set_default_dtype(torch.float64)

import json                                                            # noqa: E402
from flowfirst.dual import (DualRecovery, FlowFirstDual, FlowFirstPrimal, build_net_for_data,  # noqa: E402
                            load_flowfirst_pair)
from flowfirst.dataset import load_or_create                           # noqa: E402
from flowfirst.train import build_net, make_scaler                     # noqa: E402
from flowfirst.train import split_inputs                               # noqa: E402
import solver_backend as sb                                           # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run-dir", help="trained flow-first run (args.json + weights)")
p.add_argument("--weights", default="model.pt", help="checkpoint inside --run-dir, e.g. model_best.pt")
p.add_argument("--config", help="instead of --run-dir: a flow-first config; builds an UNTRAINED net (timing only)")
p.add_argument("--variant", default="flowfirst-gnn", choices=["flowfirst", "flowfirst-gnn"])
p.add_argument("--hours", type=int, default=2190)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads; default: torch's own")
p.add_argument("--solver", default="gurobi")
p.add_argument("--data-root", default=None)
p.add_argument("--no-exact", action="store_true")
p.add_argument("--torch-profile", action="store_true", help="also print the top torch ops")
p.add_argument("--device", default=None,
               help="also run the Benders surrogate pass on this device (e.g. cuda) through load_flowfirst_pair, "
                    "time it, and check it against the CPU pass; needs --run-dir")
a = p.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
if a.run_dir:
    run_args = json.load(open(os.path.join(a.run_dir, "args.json")))
    data, _ = load_or_create(run_args["config"], data_root=a.data_root or run_args.get("data_root"))
    net = build_net_for_data(run_args, data, a.run_dir, weights=a.weights)
    a.variant = f"trained {run_args['variant']}, {a.weights}"
else:
    data, args = load_or_create(a.config, data_root=a.data_root)
    args.update(device="cpu", dtype="float64", hidden_size_factor=28, n_layers=2, body="plain",
                small_output_init=False)
    X_train = data.X[:20000]
    scaler = None if a.variant == "flowfirst-gnn" else make_scaler("layernorm", X_train)
    net = build_net(a.variant, args, data, scaler, X_train=X_train,
                    gnn_kwargs=dict(hidden=128, rounds=2, id_embed=8, layernorm=False, antisym=False)).eval()
X = data.X[: a.hours].clone()
if X.shape[0] < a.hours:                                   # tile a small dataset up to the horizon
    X = X.repeat(int(np.ceil(a.hours / X.shape[0])), 1)[: a.hours]
B = X.shape[0]
print(f"{a.run_dir or a.config} [{a.variant}]: N={data.num_n} G={data.num_g} L={data.num_l} hours={B} "
      f"torch threads={torch.get_num_threads()} (cpus visible: {os.cpu_count()}, "
      f"usable: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'n/a'}, "
      f"SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})")

# as load_flowfirst_pair builds them for gep_benders: the dual shares the primal and reuses its flows
primal = FlowFirstPrimal(net, data, polish_sweeps=1)
dual = FlowFirstDual(primal, data)


def timed(fn, reps=a.repeats):
    fn()                                                    # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


# ---- inexact: stage by stage -------------------------------------------------
D, cap = split_inputs(data, X)
rows = []
with torch.no_grad():
    t_net, raw = timed(lambda: net(X)[2])
    t_pol, (f, _) = timed(lambda: primal.polish(X, raw, 1))
    t_fill, _ = timed(lambda: primal.fill(D, cap, f))
    t_quot, lam0 = timed(lambda: dual.recovery.quotient_prices(X, f))
    t_asc, lam = timed(lambda: dual.recovery.node_ascent(X, lam0, dual.sweeps))
    t_comp, mu = timed(lambda: dual.recovery.completion(lam))
    y = primal(X)
    t_obj, _ = timed(lambda: data.obj_fn(X, y))
    t_dobj, _ = timed(lambda: data.dual_obj_fn(X, mu, -lam))
    # what solve_matrix_problem_PDL actually times (t_sub): primal_net(X) then dual_net(X) on a new X each
    # iteration, so the primal computes the flows and the dual reuses them
    def iteration():
        Xi = X.clone()
        t0 = time.perf_counter()
        primal(Xi)
        t1 = time.perf_counter()
        dual(Xi)
        return t1 - t0, time.perf_counter() - t1
    iteration()
    t_primal_call, t_dual_call = map(float, np.median([iteration() for _ in range(a.repeats)], axis=0))

stages = [
    ("network forward (net(X))", t_net),
    ("polish, 1 sweep", t_pol),
    ("merit-order fill", t_fill),
    ("quotient prices", t_quot),
    ("node ascent, 3 sweeps", t_asc),
    ("completion", t_comp),
]
print("\nInexact pass, stage by stage (median of %d):" % a.repeats)
for name, t in stages:
    print(f"  {name:32s} {t * 1e3:9.1f} ms")
print(f"  {'-' * 32} ---------")
print(f"  {'primal_net(X)  [net+polish+fill]':32s} {t_primal_call * 1e3:9.1f} ms")
print(f"  {'dual_net(X)    [prices, reuses flows]':32s} {t_dual_call * 1e3:9.1f} ms")
print(f"  {'= t_sub in the iteration log':32s} {(t_primal_call + t_dual_call) * 1e3:9.1f} ms")
dup = t_net + t_pol
print(f"  before the flows were reused, dual_net also ran net + polish: about "
      f"{(t_primal_call + t_dual_call + dup) * 1e3:.1f} ms in total")
print(f"  after the clock stops: obj_fn x2 {2 * t_obj * 1e3:.1f} ms, dual_obj_fn x2 {2 * t_dobj * 1e3:.1f} ms")
print(f"  per hour: {(t_primal_call + t_dual_call) / B * 1e6:.0f} us")

if a.torch_profile:
    from torch.profiler import ProfilerActivity, profile
    Xp = X.clone()
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        primal(Xp)
        dual(Xp)
    ev = prof.key_averages()
    n_calls = sum(e.count for e in ev)
    print(f"\ntorch ops launched by one primal+dual pass: {n_calls}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

# ---- the same pass on an accelerator, as gep_benders --device runs it ----------
if a.device and a.device != "cpu":
    assert a.run_dir, "--device needs --run-dir"
    P, Dd = load_flowfirst_pair(a.run_dir, data, weights=a.weights, device=a.device)
    def device_iteration():
        Xi = X.clone()                                      # a new X each Benders iteration
        t0 = time.perf_counter()
        y_d = P(Xi)
        mu_d, nl_d = Dd(Xi)                                 # served from the same pass
        return time.perf_counter() - t0, (y_d, mu_d, nl_d)
    t_first, _ = device_iteration()                         # includes the CUDA-graph capture
    runs = [device_iteration() for _ in range(a.repeats)]
    t_dev = float(np.median([r[0] for r in runs]))
    y_d, mu_d, nl_d = runs[-1][1]
    Xc = X.clone(); y_c = primal(Xc); mu_c, nl_c = dual(Xc)
    R = DualRecovery(data)
    b_c, b_d = R.dual_objective(X, -nl_c).sum().item(), R.dual_objective(X, -nl_d).sum().item()
    rel = lambda u, v: ((u - v).abs().max() / v.abs().max().clamp(min=1)).item()
    print(f"\nSurrogate pass on {a.device} (primal_net + dual_net, one graph replay, copy back included):")
    print(f"  first call (graph capture)   {t_first * 1e3:9.1f} ms")
    print(f"  per iteration, median        {t_dev * 1e3:9.1f} ms   vs {(t_primal_call + t_dual_call) * 1e3:.1f} ms on the CPU")
    print(f"  check against the CPU pass: y rel {rel(y_d, y_c):.1e}, prices rel {rel(nl_d, nl_c):.1e}, "
          f"multipliers rel {rel(mu_d, mu_c):.1e}, dual bound total rel {(b_d - b_c) / abs(b_c):+.1e}")
    print("  (expect all near 1e-12 or below in float64; a larger gap means the device pass differs)")

# ---- exact: persistent LP re-solved hour by hour -----------------------------
if not a.no_exact:
    eq_rhs, ineq_rhs = data.split_X(X)
    b_eq, b_ineq = eq_rhs.numpy(), ineq_rhs.numpy()
    opts = sb.solver_options(a.solver, threads=1, seed=0, lp_optimality_tol=1e-9, feasibility_tol=1e-9)
    lp = sb.MatrixLP(a.solver, data.obj_coeff.numpy(), data.ineq_cm.numpy(), data.eq_cm.numpy(), opts)
    lp.solve(b_ineq[0], b_eq[0])                            # first solve builds the model and the fast path
    t = time.perf_counter()
    objs, sum_solve = [], 0.0
    for h in range(B):
        obj, _, _, st = lp.solve(b_ineq[h], b_eq[h])
        objs.append(obj)
        sum_solve += st
    wall = time.perf_counter() - t
    print(f"\nExact LPs, {B} hours ({a.solver}, 1 thread, warm-started):")
    print(f"  sum of solver time  {sum_solve * 1e3:9.1f} ms   <- t_sub in the iteration log")
    print(f"  wall incl. Python   {wall * 1e3:9.1f} ms   ({wall / B * 1e6:.0f} us per hour)")
    with torch.no_grad():
        ub = float(data.obj_fn(X, y).sum())
    print(f"  sanity: exact total {sum(objs):.6g} vs surrogate primal {ub:.6g} "
          f"(surrogate {100 * (ub / sum(objs) - 1):+.3f} %)")
    print(f"\nRatio t_sub inexact / exact: {(t_primal_call + t_dual_call) / sum_solve:.1f}x "
          f"(on wall time: {(t_primal_call + t_dual_call) / wall:.1f}x)")
