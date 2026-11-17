"""Train the space-time network (Way 2-i, Way 2-ii with --paths, Way 2-iii with --dp) on the ramping block set.

Unsupervised, like the hourly flow-first runs. The dispatch the loss scores is flowfirst/spacetime.py's
block_dispatch: the merit-order fill on the network's flows (2-i), or with --paths the ramp-limited
units on their clipped paths and the fill around them (2-ii). --loss picks the loss:

  objective    the subproblem objective of that dispatch: generation cost + VOLL x |unserved| over the
               block's hours. With --paths the ramps hold by construction, so nothing else is needed.
  lagrangian   the objective plus sum_s nu_s x (ramp violation of technology s), measured on the raw
               path before the clip (2-ii) or on the merit-order dispatch (2-i). nu >= 0, one per
               ramp-limited technology, starts at --lagrange-init and after every epoch moves by dual
               ascent: nu <- max(0, nu + --lagrange-step x mean violation per block in MW). With 2-ii
               this only shapes the raw path; the clip and the ladder are the same as for "objective".
               --lagrange-step 0 --lagrange-init 10 is the fixed VOLL penalty of the first 2-i run.

--dp (Way 2-iii) keeps flows as the only output and replaces both losses: the loss is the cost of the best
ramp-feasible dispatch given the flows, per node by flowfirst/rampdp.py's program (no LP), and its gradient
is the program's ramp-aware node prices.

The labels are only read for validation, which runs the deployed pipeline on the valid split every
evaluation: network, --polish-sweeps line-polish sweeps (default 1, 0 with --dp), then the ramp ladder
(r3="lp", escalate="infeasible"), or with --dp the no-LP pipeline BlockLadder.run_dp.
model_best.pt is the epoch with the lowest validation ladder gap total, sum(U - Q*) / sum(Q*).

Weights start at random. --init-from a flowfirst-gnn run instead loads its weights into the
per-hour part (the space-time paths start at zero, so epoch 0 is that hourly network's flows).

Run from the repo root:
  venv/bin/python3 -m flowfirst.train_block --paths --tag e250
  venv/bin/python3 -m flowfirst.train_block --paths --loss lagrangian --tag lagr-e250
  venv/bin/python3 -m flowfirst.train_block --dp --tag e250
Writes flowfirst/runs/flowfirst-spacetime[-paths|-dp][-tag]/ with the same files as flowfirst/train.py.
"""
import argparse
import csv
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from flowfirst.block import RUNG_FILL, RUNG_NODE_LP, RUNG_R3, RUNG_R4, RUNG_SHRINK, BlockLadder, BlockSystem
from flowfirst.dataset import load_or_create
from flowfirst.fill import MeritOrderFill
from flowfirst.rampdp import RampDP
from flowfirst.spacetime import block_loss, build_spacetime, dp_loss, ramp_frac, spacetime_dispatch
from flowfirst.train import RUNS_DIR, Tee, init_wandb, make_run_dir

torch.set_default_dtype(torch.float64)
VARIANT = "flowfirst-spacetime"
GNN_KEYS = ("gnn_hidden", "gnn_rounds", "gnn_no_id_embed", "gnn_layernorm", "gnn_antisym")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block-data", default="data/ED_data_gen/GEP_block_data_N3_G6_L3_H876_B4_single.pkl")
    ap.add_argument("--config", default="flowfirst/configs/config-3node-blockhours.json",
                    help="flow-first config of the same system, for the merit-order fill and the hourly GNN")
    ap.add_argument("--paths", action="store_true", help="Way 2-ii: also predict the ramp-limited units' output")
    ap.add_argument("--dp", action="store_true", help="Way 2-iii: flows only, loss = best ramp-feasible dispatch given the flows")
    ap.add_argument("--loss", choices=("objective", "lagrangian"), default="objective")
    ap.add_argument("--lagrange-init", type=float, default=0.0, help="starting nu per MW of ramp violation")
    ap.add_argument("--lagrange-step", type=float, default=1e-4, help="dual ascent step: nu change per MW of mean violation")
    ap.add_argument("--init-from", default=None, help="flowfirst-gnn run whose weights start the per-hour part; default random")
    ap.add_argument("--init-weights", default="model_best.pt")
    ap.add_argument("--gnn-hidden", type=int, default=None, help="default: the --init-from run's, else 128")
    ap.add_argument("--gnn-rounds", type=int, default=None, help="default: the --init-from run's, else 3")
    ap.add_argument("--gnn-no-id-embed", action="store_true")
    ap.add_argument("--gnn-layernorm", action="store_true")
    ap.add_argument("--gnn-antisym", action="store_true")
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=512, help="blocks per step (512 blocks = 2048 hours)")
    ap.add_argument("--polish-sweeps", type=int, default=None, help="validation pipeline; default 1, 0 with --dp")
    ap.add_argument("--solver", default="gurobi", help="LP solver of the validation ladder")
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--wandb-mode", default=None)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-group", default=None)
    cli = ap.parse_args()
    cli.variant = VARIANT
    if cli.dp and (cli.paths or cli.loss != "objective"):
        raise SystemExit("--dp is its own loss: no --paths, no --loss lagrangian")
    if cli.polish_sweeps is None:
        cli.polish_sweeps = 0 if cli.dp else 1

    data, args = load_or_create(cli.config)
    ds = pickle.load(open(cli.block_data, "rb"))           # GEPOperationalProblemSet after set_ramping(B)
    bi, B = ds.block_info, ds.ramp_block_hours_
    X, Q = ds.X, ds.opt_targets["obj"].numpy()
    split = bi["split"]
    train_idx = torch.as_tensor(np.flatnonzero(split == "train"))
    valid = split == "valid"
    X_train, X_valid = X[train_idx], X[torch.as_tensor(valid)]
    n_train = len(train_idx)

    init_args = None
    if cli.init_from is not None:
        init_args = json.load(open(Path(cli.init_from) / "args.json"))
        if init_args["variant"] != "flowfirst-gnn":
            raise SystemExit(f"--init-from needs a flowfirst-gnn run, got {init_args['variant']}")
    for k, default in (("gnn_hidden", 128), ("gnn_rounds", 3)):
        if getattr(cli, k) is None:
            setattr(cli, k, init_args[k] if init_args else default)
    if init_args:
        for k in ("gnn_no_id_embed", "gnn_layernorm", "gnn_antisym"):
            setattr(cli, k, bool(init_args.get(k, False)))
        if init_args["gnn_hidden"] != cli.gnn_hidden or init_args["gnn_rounds"] != cli.gnn_rounds:
            raise SystemExit("--gnn-hidden / --gnn-rounds must match the --init-from run")

    torch.manual_seed(cli.seed)
    a = {**{k: getattr(cli, k) for k in GNN_KEYS}, "paths": cli.paths}
    net = build_spacetime(a, data, ds, X_train)
    if init_args:
        net.hourly.load_state_dict(torch.load(Path(cli.init_from) / cli.init_weights, map_location="cpu", weights_only=True))
    opt = torch.optim.Adam(net.parameters(), lr=cli.lr)
    fill = MeritOrderFill(data)
    dp = RampDP.from_data(data, ramp_frac(ds))
    slow_names = ["_".join(ds.G[g]) for g in net.slow.tolist()]
    nu = torch.full((len(slow_names),), cli.lagrange_init if cli.loss == "lagrangian" else 0.0)
    fixed_gens = net.slow.numpy()

    # validation: the deployed pipeline on the valid split
    ladder = BlockLadder(BlockSystem.from_operational(ds, ds.ramp_limits(), B, solver=cli.solver), data,
                         r3="lp", escalate="infeasible")
    D_v = X_valid[:, :B * ds.num_n].numpy().reshape(-1, B, ds.num_n)
    apmax_v, u_v, Q_v, bind_v = bi["apmax"][valid], bi["u"][valid], Q[valid], bi["binding"][valid]
    scale_v = np.maximum(np.abs(Q_v), 1e-9)

    def validate_dp():
        net.eval()
        with torch.no_grad():
            dp_cost = dp_loss(dp, net, X_valid).numpy()
        f_v, _ = spacetime_dispatch(net, data, X_valid, cli.polish_sweeps)
        res = ladder.run_dp(D_v, apmax_v, u_v, f_v, dp)
        gap = (res.U - Q_v) / scale_v
        total = lambda x, m: float(((x - Q_v)[m]).sum() / np.abs(Q_v[m]).sum())
        every = np.ones_like(bind_v)
        return {"val/ladder_gap_total": total(res.U, every), "val/ladder_gap_total_binding": total(res.U, bind_v),
                "val/ladder_gap_total_nonbinding": total(res.U, ~bind_v), "val/ladder_gap_mean": float(gap.mean()),
                "val/ladder_gap_median": float(np.median(gap)),
                "val/dp_gap_total": total(dp_cost, every), "val/dp_gap_total_binding": total(dp_cost, bind_v),
                "val/r1_share": res.stats["n_blocks_r1"] / len(Q_v),
                "val/ramp_feasible_share": float((res.rung == RUNG_FILL).mean()),
                "val/shrink_share": float((res.rung == RUNG_SHRINK).mean()),
                "val/r4_share": float((res.rung == RUNG_R4).mean())}

    def validate():
        if cli.dp:
            return validate_dp()
        net.eval()
        with torch.no_grad():
            cost, voll, viol = block_loss(fill, net, X_valid)
        f_v, p_fix = spacetime_dispatch(net, data, X_valid, cli.polish_sweeps)
        res = ladder.run(D_v, apmax_v, u_v, f_v, None if p_fix is None else p_fix.numpy(), fixed_gens)
        gap = (res.U - Q_v) / scale_v
        total = lambda m: float(((res.U - Q_v)[m]).sum() / np.abs(Q_v[m]).sum())
        fill_gap = ((cost + voll).numpy() - Q_v)
        return {"val/ladder_gap_total": total(np.ones_like(bind_v)), "val/ladder_gap_total_binding": total(bind_v),
                "val/ladder_gap_total_nonbinding": total(~bind_v), "val/ladder_gap_mean": float(gap.mean()),
                "val/ladder_gap_median": float(np.median(gap)),
                "val/fill_gap_total": float(fill_gap.sum() / np.abs(Q_v).sum()),
                "val/fill_gap_total_binding": float(fill_gap[bind_v].sum() / np.abs(Q_v[bind_v]).sum()),
                "val/ramp_violation_mw": float(viol.sum(1).mean()),
                "val/ramp_feasible_share": float((res.rung == RUNG_FILL).mean()),
                "val/r2_share": float((res.rung == RUNG_NODE_LP).mean()),
                "val/r3_share": float((res.rung == RUNG_R3).mean())}

    run_dir = make_run_dir(Path(cli.runs_dir or RUNS_DIR), VARIANT + ("-paths" if cli.paths else "-dp" if cli.dp else ""), cli.tag)
    sys.stdout = Tee(run_dir / "train.log")
    print("command:", " ".join(sys.argv))
    writer = SummaryWriter(log_dir=str(run_dir))
    wandb_run = init_wandb(cli, args, run_dir, None)
    n_params = sum(p.numel() for p in net.parameters())
    with open(run_dir / "args.json", "w") as fh:
        json.dump({**vars(cli), "block_hours": B, "n_train": n_train,
                   "n_valid": int(valid.sum()), "n_params": n_params,
                   "slow_technologies": [list(ds.G[g]) for g in net.slow.tolist()]}, fh, indent=2)
    print(f"run dir: {run_dir}   params: {n_params}   train/valid blocks: {n_train}/{int(valid.sum())}   "
          f"ramp-limited: {slow_names}   way: {'2-ii (paths)' if cli.paths else '2-iii (dp)' if cli.dp else '2-i (flows)'}   "
          f"loss: {'dp' if cli.dp else cli.loss}   "
          f"init: {cli.init_from or 'random'}")

    LOSS_KEYS = (("loss/train_total",) if cli.dp else
                 ("loss/train_total", "loss/train_cost", "loss/train_voll", "loss/train_ramp", "train/ramp_violation_mw"))
    NU_KEYS = tuple(f"train/nu_{n}" for n in slow_names)
    csv_rows, best = [], {"epoch": -1, "val/ladder_gap_total": float("inf")}

    def log(epoch, train_stats, do_eval):
        row = {"epoch": epoch, **train_stats}
        if do_eval:
            row.update(validate())
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
        csv_rows.append(row)
        if do_eval and row["val/ladder_gap_total"] < best["val/ladder_gap_total"]:
            best.update(epoch=epoch, **{"val/ladder_gap_total": row["val/ladder_gap_total"]})
            torch.save(net.state_dict(), run_dir / "model_best.pt")
        if do_eval and (epoch % cli.log_every == 0 or epoch == cli.epochs):
            train = f"{row['loss/train_total']:12.2f}" if "loss/train_total" in row else " " * 12
            if cli.loss == "lagrangian":
                train += " nu " + " ".join(f"{float(v):.3g}" for v in nu)
            if cli.dp:
                print(f"epoch {epoch:4d} | train {train} | val ladder gap total {row['val/ladder_gap_total']:.5f} "
                      f"(binding {row['val/ladder_gap_total_binding']:.4f}) mean {row['val/ladder_gap_mean']:.4f} | "
                      f"dp gap total {row['val/dp_gap_total']:.5f} (binding {row['val/dp_gap_total_binding']:.4f}) | "
                      f"blocks changed by R1 {row['val/r1_share']:.4f}, shrunk {row['val/shrink_share']:.4f}, "
                      f"R4 {row['val/r4_share']:.4f}", flush=True)
                return
            print(f"epoch {epoch:4d} | train {train} | val ladder gap total {row['val/ladder_gap_total']:.5f} "
                  f"(binding {row['val/ladder_gap_total_binding']:.4f}) mean {row['val/ladder_gap_mean']:.4f} | "
                  f"fill gap total {row['val/fill_gap_total']:.5f} | ramp viol {row['val/ramp_violation_mw']:7.2f} MW, "
                  f"feasible {row['val/ramp_feasible_share']:.3f} R2 {row['val/r2_share']:.3f} R3 {row['val/r3_share']:.4f}",
                  flush=True)

    log(0, {}, True)
    shuffle_gen = torch.Generator().manual_seed(cli.seed)
    t0 = time.time()
    for epoch in range(1, cli.epochs + 1):
        net.train()
        sums, viol_sum, count = torch.zeros(len(LOSS_KEYS)), torch.zeros_like(nu), 0
        perm = torch.randperm(n_train, generator=shuffle_gen)
        for i in range(0, n_train, cli.batch_size):
            xb = X_train[perm[i:i + cli.batch_size]]
            opt.zero_grad()
            if cli.dp:
                loss = dp_loss(dp, net, xb).mean()
                parts = [loss]
            else:
                cost, voll, viol = block_loss(fill, net, xb)
                ramp = viol @ nu
                loss = (cost + voll + ramp).mean()
                parts = [loss, cost.mean(), voll.mean(), ramp.mean(), viol.sum(1).mean()]
                with torch.no_grad():
                    viol_sum += viol.sum(0)
            loss.backward()
            opt.step()
            with torch.no_grad():
                sums += torch.stack([x.detach() for x in parts]) * xb.shape[0]
            count += xb.shape[0]
        stats = {**dict(zip(LOSS_KEYS, (sums / count).tolist(), strict=True)),
                 **dict(zip(NU_KEYS, nu.tolist(), strict=True)), "train/lr": opt.param_groups[0]["lr"]}
        if cli.loss == "lagrangian":                                 # dual ascent on the epoch's mean violation
            nu = torch.relu(nu + cli.lagrange_step * viol_sum / count)
        do_eval = epoch % cli.eval_every == 0 or epoch == cli.epochs
        log(epoch, stats, do_eval)

    print(f"training time: {time.time() - t0:.1f}s")
    torch.save(net.state_dict(), run_dir / "model.pt")
    final = next(r for r in reversed(csv_rows) if "val/ladder_gap_total" in r)
    summary = {"final_epoch": final["epoch"], "final_ladder_gap_total": final["val/ladder_gap_total"],
               "best_epoch": best["epoch"], "best_ladder_gap_total": best["val/ladder_gap_total"],
               "final_nu": dict(zip(slow_names, nu.tolist(), strict=True))}
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    keys = list(dict.fromkeys(k for r in csv_rows for k in r))
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r.get(k, "") for k in keys})
    writer.close()
    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()
    print(f"final val ladder gap total {final['val/ladder_gap_total']:.5f} at epoch {final['epoch']}; best "
          f"{best['val/ladder_gap_total']:.5f} at epoch {best['epoch']} (model_best.pt); saved in {run_dir}")


if __name__ == "__main__":
    main()
