"""Plan A offline on the labelled block dataset: the hourly flow-first network plus the ramp ladder.

Reads one split of the block problem set (gen_GEP/solve_for_train.py --ramp-block-hours 4; default the test split) and, for
every block, compares against its exact labels

  * level A:  the hourly pipeline as it stands (nu = 0, ramp rows unchecked; U may be infeasible)
  * plan A:   the ladder, once per --modes entry (r3/escalate), or "dp": Way 2-iii's no-LP pipeline
              (BlockLadder.run_dp, the node-block program instead of R2/R3)

reporting the cost gap (U - Q*)/Q*, the bound gap (Q* - L)/Q*, the cut-slope error, the rung each
block ends on and the time, split into the hard cases:
  binding       Q* > Q_free (the ramp rows change the optimum), bucketed by ramp impact (Q* - Q_free)/Q*
  violated      the hourly sorted dispatch breaks a ramp row (level A is infeasible)
  class         instance class of the manifest
and a list of the worst blocks.

Run from the repo root:
  venv/bin/python3 flowfirst/eval_block_ladder.py
  venv/bin/python3 flowfirst/eval_block_ladder.py --split valid --modes lp/unmet lp/infeasible bisect/unmet
  venv/bin/python3 flowfirst/eval_block_ladder.py --run-dir flowfirst/runs/flowfirst-spacetime-dp-e250 \
      --weights model_best.pt --modes dp lp/infeasible --out flowfirst/runs/ramp-ladder-eval/dp-e250
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
torch.set_default_dtype(torch.float64)

from flowfirst.block import (BlockLadder, BlockSystem, RUNG_FILL, RUNG_NODE_LP, RUNG_R3, RUNG_R4,  # noqa: E402
                             RUNG_SHRINK, block_inputs, network_flows)
from flowfirst.rampdp import RampDP  # noqa: E402
from flowfirst.dataset import load_or_create  # noqa: E402
from flowfirst.fill import MeritOrderFill  # noqa: E402
from flowfirst.spacetime import load_spacetime, ramp_frac, spacetime_dispatch  # noqa: E402
from flowfirst.dual import build_net_for_data  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/ED_data_gen/GEP_block_data_N3_G6_L3_H876_B4_single.pkl")
p.add_argument("--split", default="test", choices=("train", "valid", "test", "all"))
p.add_argument("--run-dir", default="flowfirst/runs/flowfirst-harvest", help="flow-first run whose network predicts the flows")
p.add_argument("--weights", default="model.pt")
p.add_argument("--modes", nargs="+", default=None,
               help="ladder modes r3/escalate, r3 in {lp, bisect}, escalate in {unmet, infeasible}, or dp (no LP); "
                    "default dp for a --dp run, else lp/infeasible lp/unmet")
p.add_argument("--solver", default="gurobi")
p.add_argument("--polish-sweeps", type=int, default=None, help="default 0 for a --dp run (as its validation), else 1")
p.add_argument("--worst", type=int, default=15, help="worst blocks to list")
p.add_argument("--out", default="flowfirst/runs/ramp-ladder-eval")
a = p.parse_args()

ds = pickle.load(open(a.data, "rb"))             # a GEPOperationalProblemSet after set_ramping(B)
bi, B, N = ds.block_info, ds.ramp_block_hours_, ds.num_n
sel = np.ones(len(bi["split"]), dtype=bool) if a.split == "all" else bi["split"] == a.split
K = int(sel.sum())
D = ds.X[torch.as_tensor(sel)][:, :B * N].numpy().reshape(K, B, N)
apmax, u = bi["apmax"][sel], bi["u"][sel]
Q = ds.opt_targets["obj"].numpy()[sel]
Q_free, bind, impact = bi["Q_free"][sel], bi["binding"][sel], bi["ramp_impact"][sel]
slope_x, cls = bi["slope"][sel], bi["class"][sel]
inst, start = bi["inst"][sel], bi["start_hour"][sel]

run_args = json.load(open(os.path.join(a.run_dir, "args.json")))
data, _ = load_or_create(run_args["config"])
#! A space-time run (flowfirst/train_block.py) reads whole block rows; every other run is an hourly network
#! on the blocks' hours. Both hand the ladder hourly flows [K*B, L]; a Way 2-ii run (--paths) also hands it
#! the ramp-limited units' clipped paths, which R1 and the fill keep fixed.
spacetime = run_args["variant"] == "flowfirst-spacetime"
if a.modes is None:
    a.modes = ["dp"] if run_args.get("dp") else ["lp/infeasible", "lp/unmet"]
if a.polish_sweeps is None:
    a.polish_sweeps = 0 if run_args.get("dp") else 1
if spacetime:
    net = load_spacetime(a.run_dir, data, ds, weights=a.weights)
else:
    net = build_net_for_data(run_args, data, a.run_dir, weights=a.weights)
X_blocks = ds.X[torch.as_tensor(sel)]


def flows(sweeps):
    """Hourly flows [K*B, L] and the fixed generation [K, B, G] of a Way 2-ii run (else None)."""
    if spacetime:
        fl, pf = spacetime_dispatch(net, data, X_blocks, sweeps)
        return fl, None if pf is None else pf.numpy()
    return network_flows(net, data, block_inputs(D, apmax, u), sweeps), None


system = BlockSystem.from_operational(ds, ds.ramp_limits(), B, solver=a.solver)
os.makedirs(a.out, exist_ok=True)

t0 = time.perf_counter()
f, p_fix = flows(a.polish_sweeps)
t_net = time.perf_counter() - t0
fixed_gens = net.slow.numpy() if p_fix is not None else None
res = {}
for m in a.modes:
    if m == "dp":
        if p_fix is not None:
            raise SystemExit("mode dp takes flows only; this run also gives paths (Way 2-ii)")
        res[m] = BlockLadder(system, data).run_dp(D, apmax, u, f, RampDP.from_data(data, ramp_frac(ds)))
    else:
        r3, esc = m.split("/")
        res[m] = BlockLadder(system, data, r3=r3, escalate=esc).run(D, apmax, u, f, p_fix, fixed_gens)
    print(f"{m}: {res[m].stats['times']['total']:.1f}s", flush=True)

# The hourly pipeline step by step, before R1: the merit-order fill on the network's clipped flows (R0, inside
# the network) and after the polish. Ramp rows are not checked here, so a dispatch may break them, or leave md
# outside [0, D] (what R1 removes); `feasible` says whether every row of the block LP holds. With paths the
# fill dispatches around them, as in training.
fill = MeritOrderFill(data)
X_h = block_inputs(D, apmax, u)


def stage(fl, pf):
    D_h, cap_h = X_h[:, :N], X_h[:, N:]
    pf_h = torch.zeros_like(cap_h)
    if pf is not None:
        pf_h = torch.tensor(pf.reshape(K * B, -1))
        cap_h = cap_h.clone()
        cap_h[:, net.slow] = 0.0
        D_h = D_h - pf_h @ data.node_to_gen_mask.T.to(D_h.dtype)
    with torch.no_grad():
        p_s, md_s, _ = fill(D_h, cap_h, fl)
    p_s, md_s = (p_s + pf_h).numpy().reshape(K, B, -1), md_s.numpy().reshape(K, B, N)
    U_s = (p_s @ system.V).sum(1) + system.voll * np.abs(md_s).sum((1, 2))
    return U_s, system.max_violation(p_s, fl.numpy().reshape(K, B, -1), md_s, D, apmax, u) <= 1e-6


stages = {"clip": stage(*flows(0)), "polish": stage(f, p_fix)}
st = res[a.modes[0]].stats
# the hourly sorted dispatch broke a ramp row (or md bounds); under dp every block runs the program, so read
# it off the polish stage
violated = ~stages["polish"][1] if a.modes[0] == "dp" else res[a.modes[0]].rung != RUNG_FILL

scale = np.maximum(np.abs(Q), 1e-9)


# floor: a block whose exact slope is ~0 (no capacity or ramp row binds) would divide by ~0
slope_floor = 1e-2 * np.median(np.abs(slope_x).sum(1)[np.abs(slope_x).sum(1) > 0])


def slope_err(sl):
    return np.abs(sl - slope_x).sum(1) / np.maximum(np.abs(slope_x).sum(1), slope_floor)


def measures(U, L, slopes, rung=None):
    out = dict(cost_gap=(U - Q) / scale, bound_gap=(Q - L) / scale, slope_err=slope_err(slopes))
    if rung is not None:
        out["rung"] = rung
    return out


methods = {"level A": measures(st["U_hourly"], st["L_hourly"], st["slopes_hourly"])}
for m in a.modes:
    methods[f"plan A {m}"] = measures(res[m].U, res[m].L, res[m].slopes, res[m].rung)

buckets = [("non-binding", ~bind),
           ("binding, impact <= 0.1%", bind & (impact <= 1e-3)),
           ("binding, 0.1% - 1%", bind & (impact > 1e-3) & (impact <= 1e-2)),
           ("binding, 1% - 5%", bind & (impact > 1e-2) & (impact <= 5e-2)),
           ("binding, > 5%", bind & (impact > 5e-2)),
           ("binding (all)", bind),
           ("violated", violated),
           ("violated, non-binding", violated & ~bind),
           ("all", np.ones(K, dtype=bool))]
buckets += [(f"class {c}", cls == c) for c in np.unique(cls)]
buckets += [(f"class {c}, binding", (cls == c) & bind) for c in np.unique(cls)]


def stats(x, mask):
    x = x[mask]
    if x.size == 0:
        return None
    return dict(mean=float(x.mean()), p50=float(np.median(x)), p90=float(np.quantile(x, 0.9)),
                p99=float(np.quantile(x, 0.99)), max=float(x.max()))


summary = dict(data=a.data, split=a.split, blocks=K, binding_share=float(bind.mean()),
               violated_share=float(violated.mean()), run_dir=a.run_dir,
               network_ms_per_block=1e3 * t_net / K, buckets={})
for name, mask in buckets:
    entry = dict(n=int(mask.sum()))
    for meth, ms in methods.items():
        entry[meth] = {k: stats(v, mask) for k, v in ms.items() if k != "rung"}
        entry[meth]["total_cost_gap"] = float(((ms["cost_gap"] * scale)[mask]).sum() / max(Q[mask].sum(), 1e-9))
        if "rung" in ms:
            entry[meth]["rung_share"] = {r: float((ms["rung"][mask] == v).mean()) if mask.any() else None
                                         for r, v in (("fill", RUNG_FILL), ("R2", RUNG_NODE_LP), ("R3", RUNG_R3),
                                                      ("R4", RUNG_R4), ("shrink", RUNG_SHRINK))}
    summary["buckets"][name] = entry
for m in a.modes:
    s = res[m].stats
    summary[f"plan A {m}"] = dict(ladder_ms_per_block=1e3 * s["times"]["total"] / K,
                                  times_s={k: float(v) for k, v in s["times"].items()},
                                  n_r2=s["n_r2"], n_r3=s["n_r3"], n_r4=s["n_r4"],
                                  share_dual_repaired=s["share_dual_repaired"],
                                  max_rel_violation=s["max_rel_violation"],
                                  md_added_share=float((res[m].md.sum((1, 2)) - st["md_hourly"]).sum()
                                                       / max(st["md_hourly"].sum(), 1e-9)))

json.dump(summary, open(os.path.join(a.out, f"summary_{a.split}.json"), "w"), indent=2)
np.savez_compressed(os.path.join(a.out, f"blocks_{a.split}.npz"), idx=np.flatnonzero(sel), Q=Q, Q_free=Q_free,
                    binding=bind, violated=violated, impact=impact,
                    **{f"{meth.replace(' ', '_').replace('/', '_')}_{k}": v for meth, ms in methods.items() for k, v in ms.items()},
                    **{f"stage_{s}_cost_gap": (U_s - Q) / scale for s, (U_s, _) in stages.items()},
                    **{f"stage_{s}_feasible": ok for s, (_, ok) in stages.items()})


def pct(x, w=8):
    return f"{'-':>{w}s}" if x is None else f"{100 * x:{w}.2f}%"


print(f"\n{K} {a.split} blocks: {100 * bind.mean():.2f}% binding ({int(bind.sum())}), "
      f"{100 * violated.mean():.2f}% violated by the hourly dispatch ({int(violated.sum())}); network {summary['network_ms_per_block']:.3f} ms/block")
for m in a.modes:
    s = summary[f"plan A {m}"]
    print(f"plan A {m}: {s['ladder_ms_per_block']:.3f} ms/block, R2 node-blocks {s['n_r2']}, R3 blocks {s['n_r3']}, "
          f"R4 {s['n_r4']}, max violation {s['max_rel_violation']:.1e}, unmet demand added {100 * s['md_added_share']:.1f}%")

for meas, title in (("cost_gap", "cost gap (U - Q*)/Q*"), ("bound_gap", "bound gap (Q* - L)/Q*"),
                    ("slope_err", "cut-slope error |s - s*|_1 / max(|s*|_1, 1% of median)")):
    print(f"\n{title}: mean / p50 / p99 / max")
    print(f"{'bucket':30s} {'n':>5s} " + " ".join(f"{meth:>40s}" for meth in methods))
    for name, _ in buckets:
        e = summary["buckets"][name]
        cells = []
        for meth in methods:
            sv = e[meth][meas]
            cells.append(f"{'-':>40s}" if sv is None else
                         f"{pct(sv['mean'], 7)} {pct(sv['p50'], 7)} {pct(sv['p99'], 7)} {pct(sv['max'], 7)}"[-40:].rjust(40))
        print(f"{name:30s} {e['n']:5d} " + " ".join(cells))

print(f"\naggregate cost gap sum(U - Q*)/sum(Q*)")
print(f"{'bucket':30s} {'n':>5s} " + " ".join(f"{meth:>24s}" for meth in methods))
for name, _ in buckets:
    e = summary["buckets"][name]
    print(f"{name:30s} {e['n']:5d} " + " ".join(f"{pct(e[meth]['total_cost_gap'], 23)}" for meth in methods))

print("\nrung each block ends on (fill / R2 / R3 / R4 / shrink; dp: fill = the program at the network's flows)")
for name in ("non-binding", "binding (all)", "binding, > 5%", "violated", "all"):
    e = summary["buckets"][name]
    print(f"{name:30s} " + "   ".join(f"{m}: " + " ".join(pct(e[f'plan A {m}']['rung_share'][r], 6) for r in ('fill', 'R2', 'R3', 'R4', 'shrink'))
                                       for m in a.modes))

m0 = f"plan A {a.modes[0]}"
cg = methods[m0]["cost_gap"]
for title, mask in (("all blocks", np.ones(K, dtype=bool)), ("binding blocks", bind)):
    print(f"\nworst {a.worst} {title} by {m0} cost gap")
    print(f"{'inst':>4s} {'class':16s} {'hour':>5s} {'bind':>5s} {'impact':>8s} {'viol':>5s} {'rung':>4s} "
          f"{'cost gap':>9s} {'bound gap':>9s} {'slope err':>9s} {'lvlA cost':>9s} {'Q*':>10s}")
    for k in np.flatnonzero(mask)[np.argsort(-cg[mask])][:a.worst]:
        print(f"{inst[k]:4d} {cls[k]:16s} {start[k]:5d} {str(bool(bind[k])):>5s} {pct(impact[k], 7)} {str(bool(violated[k])):>5s} "
              f"{methods[m0]['rung'][k]:4d} {pct(cg[k], 8)} {pct(methods[m0]['bound_gap'][k], 8)} "
              f"{pct(methods[m0]['slope_err'][k], 8)} {pct(methods['level A']['cost_gap'][k], 8)} {Q[k]:10.3f}")
print(f"\nsaved {a.out}/summary_{a.split}.json and blocks_{a.split}.npz")
