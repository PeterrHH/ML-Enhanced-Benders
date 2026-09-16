"""Print the tables behind FINDINGS F26 to F29 for a trained flow-first run.

Run from the repo root:

    .venv/bin/python -m flowfirst.dual_analysis flowfirst/runs/<run-dir> [--valid-size 8192] [--cold-sweeps 30]
"""
import argparse
import time

import torch

from flowfirst.dual import DualRecovery, PathPolish, Polish, load_run
from flowfirst.train import Reference

torch.set_default_dtype(torch.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--valid-size", type=int, default=8192)
    ap.add_argument("--cold-sweeps", type=int, default=30)
    cli = ap.parse_args()
    data, net, valid_start = load_run(cli.run_dir)
    ref = Reference(data, torch.arange(valid_start, valid_start + cli.valid_size))
    X, Q, s = ref.X, ref.obj, ref.shortage
    with torch.no_grad():
        f = torch.cat([net(X[i:i + 1024])[2] for i in range(0, cli.valid_size, 1024)])
    dual, polish = DualRecovery(data), Polish(data)
    U = polish.cost(X, f)
    assert torch.allclose(dual.dual_objective(X, ref.lam), Q), "Gurobi's prices must reproduce its optimum"

    def dual_row(name, lam, U, ms):
        fD = dual.dual_objective(X, lam)
        assert (fD <= Q * (1 + 1e-9)).all(), "weak duality"
        cert, slack = (U - fD) / Q, (Q - fD) / Q
        print(f"  {name:36s} | certified gap mean {cert.mean():8.5f} | <=1% {(cert <= 0.01).double().mean():.3f} | "
              f"dual exact {(slack <= 1e-6).double().mean():.3f} | no-shortage exact {(slack[~s] <= 1e-6).double().mean():.3f} | {ms:6.0f} ms")

    def primal_row(name, U, ms):
        g = (U - Q) / Q
        print(f"  {name:36s} | true gap mean {g.mean():8.5f} median {g.median():8.5f} | no-shortage {g[~s].mean():8.5f} | "
              f"within 1% {(g <= 0.01).double().mean():.3f} 0.1% {(g <= 0.001).double().mean():.3f} | {ms:6.0f} ms")

    print(f"{cli.valid_size} validation instances, shortage share {s.double().mean():.3f}, Gurobi label time {1000 * data.opt_label_runtime / data.X.shape[0]:.2f} ms per LP")
    print("F26 dual from the primal:")
    t0 = time.time(); lam = dual.local_prices(X, f); dual_row("raw fill prices", lam, U, 1000 * (time.time() - t0))
    t0 = time.time(); lam = dual.quotient_prices(X, f); dual_row("quotient fill", lam, U, 1000 * (time.time() - t0))
    t0 = time.time(); lam = dual.node_ascent(X, dual.quotient_prices(X, f)); dual_row("quotient fill + 3 node sweeps", lam, U, 1000 * (time.time() - t0))
    t0 = time.time(); lam_n = dual.node_ascent(X, dual.local_prices(X, f), sweeps=10); dual_row("node sweeps only (10)", lam_n, U, 1000 * (time.time() - t0))
    print("F27 inference polish:")
    primal_row("network flows", U, 0)
    for k in (1, 3):
        t0 = time.time(); f_k, U_k = polish(X, f, sweeps=k); ms = 1000 * (time.time() - t0)
        primal_row(f"+ line search, {k} sweep(s)", U_k, ms)
    cert = (U_k - dual.dual_objective(X, dual.node_ascent(X, dual.quotient_prices(X, f_k)))) / Q
    print(f"  after polish: certified <=1% {(cert <= 0.01).double().mean():.3f}, re-solve rate at 1% {(cert > 0.01).double().mean():.3f}")
    print(f"F28 cold start, zero flows, {cli.cold_sweeps} sweeps:")
    t0 = time.time(); _, U_cold = polish(X, torch.zeros_like(f), sweeps=cli.cold_sweeps)
    primal_row(f"zero flows + {cli.cold_sweeps} sweeps", U_cold, 1000 * (time.time() - t0))
    print("F29 exact min-cost flow (max-gain cycle cancelling):")
    path = PathPolish(data)
    for name, f0 in (("from network flows", f), ("from network + 1 polish sweep", polish(X, f, 1)[0]), ("from zero flows", torch.zeros_like(f))):
        t0 = time.time(); f_x, U_x, n = path(X, f0); ms = 1000 * (time.time() - t0)
        print(f"  {name:36s} | at optimum {(((U_x - Q) / Q) < 1e-9).double().mean():.4f} | augmentations mean {n.double().mean():.1f} "
              f"median {n.median().item()} max {n.max().item()} | {ms / cli.valid_size:6.3f} ms per instance")


if __name__ == "__main__":
    main()
