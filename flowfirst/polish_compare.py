"""Raw against polished gap for several saved runs: how much of a network's error one exact sweep removes.

Run from the repo root, e.g. every 6-node run:

    .venv/bin/python -m flowfirst.polish_compare flowfirst/runs/*6node* [--valid-size 8192] [--sweeps 1,3]

Per run, on the validation instances: the true gap of the network's flows, after each number of polish sweeps, and
the augmentations the exact min-cost flow needs from the one-sweep flows (the routing error the polish cannot fix).
"""
import argparse
import time
from pathlib import Path

import torch

from flowfirst.dataset import roots_from_cli
from flowfirst.dual import PathPolish, Polish, load_run
from flowfirst.train import Reference
from paths import add_path_args

torch.set_default_dtype(torch.float64)


def stats(U, Q, shortage):
    """One row of gap statistics, in the column order of `head` in `main`."""
    g = (U - Q) / Q
    return f"{g.mean():8.5f} {g.median():8.5f} {g[~shortage].mean():8.5f} {(g <= 0.01).double().mean():6.3f} {(g <= 0.001).double().mean():6.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_path_args(ap)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--valid-size", type=int, default=8192)
    ap.add_argument("--sweeps", default="1,3")
    cli = ap.parse_args()
    data_root, _ = roots_from_cli(cli)
    sweeps = [int(k) for k in cli.sweeps.split(",")]
    head = "gap mean   median no-short  <=1%  <=0.1%"
    print(f"{'run':52s} {'raw: ' + head:46s}" + "".join(f"{f'{k} sweep(s): ' + head:48s}" for k in sweeps) + "augmentations mean / p95 / max   s")
    for run_dir in cli.run_dirs:
        if not (Path(run_dir) / "model.pt").exists():
            continue
        t0 = time.time()
        data, net, valid_start = load_run(run_dir, data_root=data_root)
        ref = Reference(data, torch.arange(valid_start, valid_start + cli.valid_size))
        X, Q, s = ref.X, ref.obj, ref.shortage
        polish, path = Polish(data), PathPolish(data)
        with torch.no_grad():
            f = torch.cat([net(X[i:i + 1024])[2] for i in range(0, cli.valid_size, 1024)])
        cols = [stats(polish.cost(X, f), Q, s)]
        for k in sweeps:
            f_k, U_k = polish(X, f, sweeps=k)
            cols.append(stats(U_k, Q, s))
            if k == sweeps[0]:
                n_aug = path(X, f_k)[2].double()
        print(f"{Path(run_dir).name:52s}" + "".join(f"{c:48s}" for c in cols)
              + f"{n_aug.mean():5.1f} / {n_aug.quantile(0.95):3.0f} / {n_aug.max():3.0f}   {time.time() - t0:4.0f}", flush=True)


if __name__ == "__main__":
    main()
