"""Time the inference pipeline of a trained run on a Modal GPU: eager, CUDA graph, torch.compile, both (F30, F31).

    FLOWFIRST_GPU=A100 .venv/bin/modal run --detach flowfirst/modal/bench.py --run-dir <run directory on the runs volume>

For every precision in `--precisions` (see PRECISIONS; default float64 and the network cast to float32 and to
float16 with everything else in float64) and batches 1024 and 8192 it prints the ms per instance of `PrimalDual`
eager, replayed as a CUDA graph (`GraphedPrimalDual`), with the network under torch.compile, and both; the largest
deviation of every fast path from that precision's eager pipeline on 8192 + 1000 instances, so that the partial
chunk of the graph is exercised; then the accuracy of the deployed variant per precision against the Gurobi
labels, evaluated in float64; and the ratio of the best time to the roofline floor of the forward pass and to
Gurobi's marks. `--stages` adds the per-stage table of F29. `benchmark` also runs locally (no graph, no compile,
no TF32 without CUDA):

    .venv/bin/python -c "from flowfirst.modal.bench import benchmark; benchmark('flowfirst/runs/<run>', 'cpu', sizes=(128,))"
"""
import os
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import modal

# the package must be importable from wherever Modal loads this file: the checkout locally, /repo in the container
for root in (Path(__file__).resolve().parent.parent.parent, Path("/repo")):
    if (root / "flowfirst" / "__init__.py").exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
from flowfirst.modal.train import datasets, image, runs  # noqa: E402

app = modal.App("flowfirst-bench")

FLOOR = {"float64": 0.0035, "float32": 0.0018}   # ms per instance: roofline of the 20-node GNN forward pass on an A100 (F30)
GUROBI = {"persistent model, one core": 0.16, "eight cores": 0.02, "optimal basis after a reset": 0.05}   # ms per LP (F29)
VARIANTS = ("eager", "eager + CUDA graph", "compiled", "compiled + CUDA graph")
# name: data dtype, dtype of the network's parameters (`cast_net`), TF32 products (F31)
PRECISIONS = {
    "float64": ("float64", "float64", False),
    "float32": ("float32", "float32", False),     # everything float32, products at the float64 peak: for the record only
    "tf32": ("float32", "float32", True),         # everything float32, tensor-core products
    "net-fp32": ("float64", "float32", True),     # network in float32 with TF32 products; fill, polish, dual float64
    "net-fp16": ("float64", "float16", True),     # network in half precision, the rest float64
    "net-bf16": ("float64", "bfloat16", True),
}
DEFAULT_PRECISIONS = "float64,net-fp32,net-fp16"


def make_timer(device, reps):
    sync = torch_sync(device)

    def timeit(fn, n=reps):
        """Median wall time in ms of `fn`, after two untimed calls (first-call compilation, capture, allocation)."""
        for _ in range(2):
            fn()
        sync()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t0)
        return 1000 * sorted(ts)[len(ts) // 2]
    return timeit


def torch_sync(device):
    import torch
    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return lambda: None


def stage_table(run_dir, device, sizes, reps):
    """The per-stage table of F29, float64."""
    import torch

    from flowfirst.dual import DualRecovery, PathPolish, Polish, PrimalDual, load_run
    from flowfirst.train import Reference
    torch.set_default_dtype(torch.float64)
    data, net, valid_start = load_run(run_dir, device)
    ref = Reference(data, torch.arange(valid_start, valid_start + max(sizes)))
    dual, polish, path = DualRecovery(data, tol=1e-6), Polish(data), PathPolish(data)
    pipeline, pipeline_exact = PrimalDual(data, net), PrimalDual(data, net, exact=True)
    timeit = make_timer(device, reps)

    def prices_and_multipliers(X, f):
        return dual.completion(dual.prices(X, f))

    print(f"\nstages, float64, {device}: ms per instance (median of {reps}; batch in columns)", flush=True)
    print(f"{'stage':34s}" + "".join(f"{b:>12d}" for b in sizes), flush=True)
    rows = {}
    with torch.no_grad():
        for B in sizes:
            X = ref.X[:B]
            f = net(X)[2]
            f1 = polish(X, f, 1)[0]
            stages = {
                "forward (GNN + fill)": partial(net, X),
                "polish, 1 sweep": partial(polish, X, f, 1),
                "dual (prices, objective, mu)": partial(prices_and_multipliers, X, f1),
                "exact path step (from polished)": partial(path, X, f1),
                "PrimalDual, exact=False": partial(pipeline, X),
                "PrimalDual, exact=True": partial(pipeline_exact, X),
            }
            for name, fn in stages.items():
                rows.setdefault(name, []).append(timeit(fn, 3 if "path" in name or "exact=True" in name else reps) / B)
        for name, vals in rows.items():
            print(f"{name:34s}" + "".join(f"{v:12.4f}" for v in vals), flush=True)
        out = pipeline_exact(ref.X)
        gap = (out.primal - ref.obj) / ref.obj
        print(f"correctness on {max(sizes)} instances: max |gap| {gap.abs().max():.1e}, certificate max {out.certificate.max():.1e}, "
              f"dual <= optimum on all: {(out.dual <= ref.obj * (1 + 1e-9)).all().item()}", flush=True)


def deviation(out, ref, rows=slice(None)):
    """Per field: largest absolute difference and the number of instances that differ at all."""
    parts = []
    for k, v in vars(out).items():
        r = getattr(ref, k)[rows]
        diff = (v - r).abs()
        rows_diff = (diff > 0).any(dim=1) if diff.dim() > 1 else diff > 0
        parts.append(f"{k} {diff.max().item():.1e} ({rows_diff.sum().item()} inst.)")
    return ", ".join(parts)


def accuracy(out, ref, data64):
    """One line per precision: the returned dispatch scored in float64 against the Gurobi labels (F31)."""
    import torch
    y, X, Q = out.y.to(torch.float64), ref.X, ref.obj
    U = data64.obj_fn(X, y)                                   # true cost of the returned dispatch
    gap, cert, fD = (U - Q) / Q, out.certificate.to(torch.float64), out.dual.to(torch.float64)
    return (f"gap mean {gap.mean():.5f} median {gap.median():.5f} no-shortage {gap[~ref.shortage].mean():.5f} "
            f"totals {U.sum() / Q.sum() - 1:.5f} | within 1% {(gap <= 0.01).double().mean():.3f} 0.1% {(gap <= 0.001).double().mean():.3f} | "
            f"certificate mean {cert.mean():.5f} re-solve at 1% {(cert > 0.01).double().mean():.3f} | "
            f"max (dual - opt)/opt {((fD - Q) / Q).max():.1e} max (gap - certificate) {(gap - cert).max():.1e} | "
            f"max balance error {data64.eq_resid(X, y).abs().max():.1e} MW")


def benchmark(run_dir, device, sizes=(1024, 8192), reps=5, check=1000, precisions=DEFAULT_PRECISIONS):
    """ms per instance of the four pipeline variants per precision and batch, the deviation of each fast path from
    that precision's eager pipeline, the accuracy of the deployed variant per precision, and the ratios to the
    floor and to Gurobi."""
    import torch

    from flowfirst.dual import GraphedPrimalDual, PrimalDual, cast_net, load_run
    from flowfirst.train import Reference
    timeit, results, scores = make_timer(device, reps), {}, {}
    precisions = precisions.split(",") if isinstance(precisions, str) else list(precisions)
    print(f"{device}: torch {torch.__version__}, CUDA graphs {'on' if device == 'cuda' else 'off (eager chunks)'}; "
          f"ms per instance, median of {reps}", flush=True)
    torch.set_default_dtype(torch.float64)
    data64, net64, valid_start = load_run(run_dir, device)
    ref = Reference(data64, torch.arange(valid_start, valid_start + max(sizes)))
    for prec in precisions:
        data_dtype, net_dtype, tf32 = PRECISIONS[prec]
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.set_default_dtype(getattr(torch, data_dtype))      # the problem constants built by PrimalDual use it
        if data_dtype == "float64":
            data, net = data64, (net64 if net_dtype == "float64" else cast_net(net64, getattr(torch, net_dtype)))
        else:
            data, net, _ = load_run(run_dir, device, getattr(torch, data_dtype))   # everything built in that dtype
        print(f"\n{prec}: data {data_dtype}, network {net_dtype}, TF32 {tf32}", flush=True)
        X_all = data.X[valid_start:valid_start + max(sizes) + check]
        pipelines = {"eager": PrimalDual(data, net), "compiled": PrimalDual(data, torch.compile(net, dynamic=False))}
        outputs = {}
        with torch.no_grad():
            try:
                reference = pipelines["eager"](X_all)
            except Exception:
                print(f"  {prec} eager pipeline failed, precision skipped:\n{traceback.format_exc()}", flush=True)
                continue
            for B in sizes:
                X = X_all[:B]
                for name, pipeline in pipelines.items():
                    for graphed in (False, True):
                        label = name + (" + CUDA graph" if graphed else "")
                        try:
                            fn = GraphedPrimalDual(pipeline, B) if graphed else pipeline
                            results[(prec, label, B)] = timeit(partial(fn, X)) / B
                            print(f"  {prec} batch {B:5d} {label:24s} {results[(prec, label, B)]:.4f}", flush=True)
                            if B == max(sizes):   # the graph sees a full chunk and a partial one; eager compiled only its shape
                                outputs[label] = out = fn(X_all) if graphed else fn(X)
                                print(f"    deviation from eager: {deviation(out, reference, slice(None) if graphed else slice(0, B))}", flush=True)
                        except Exception:
                            print(f"  {prec} batch {B} {label} failed:\n{traceback.format_exc()}", flush=True)
                        fn = None   # a graph keeps a private memory pool alive: release it before the next capture
            deployed = next(label for label in reversed(VARIANTS) if label in outputs)   # the fastest variant that ran
            rows = slice(0, max(sizes))
            scored = type(reference)(**{k: v[rows] for k, v in vars(outputs[deployed]).items()})
            scores[prec] = f"{prec:12s} ({deployed:22s}) {accuracy(scored, ref, data64)}"
            print("  " + scores[prec], flush=True)
        del data, net, pipelines, reference, outputs, X_all, scored
        if device == "cuda":
            torch.cuda.empty_cache()
    torch.set_default_dtype(torch.float64)
    print(f"\n{'pipeline':26s}" + "".join(f"{prec + ' ' + str(B):>16s}" for prec in precisions for B in sizes), flush=True)
    for label in VARIANTS:
        cells = [results.get((prec, label, B)) for prec in precisions for B in sizes]
        print(f"{label:26s}" + "".join(f"{v:16.4f}" if v is not None else f"{'failed':>16s}" for v in cells), flush=True)
    print(f"\naccuracy of the deployed variant on {max(sizes)} validation instances, scored in float64:", flush=True)
    for prec in precisions:
        print(scores.get(prec, f"{prec:12s} no variant ran"), flush=True)
    for prec in precisions:
        best = min(((results[k], k[1]) for k in results if k[0] == prec and k[2] == max(sizes)), default=None)
        if best is None:
            continue
        ms, label = best
        floor = FLOOR["float64" if PRECISIONS[prec][1] == "float64" else "float32"]
        print(f"best {prec} at batch {max(sizes)}: {label} {ms:.4f} ms per instance = {ms / floor:.1f} x the roofline floor {floor}; "
              + "; ".join(f"{g / ms:.1f} x Gurobi {name} ({g})" for name, g in GUROBI.items()), flush=True)


@app.function(image=image, gpu=os.environ.get("FLOWFIRST_GPU", "A100"), timeout=3600,
              volumes={"/repo/flowfirst/datasets": datasets, "/repo/flowfirst/runs": runs})
def bench(run_dir: str, stages: bool, precisions: str):
    os.chdir("/repo")
    import torch
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    run_dir = f"/repo/flowfirst/runs/{run_dir}"
    if stages:
        stage_table(run_dir, "cuda", (128, 1024, 8192), 5)   # the CPU reference is measured locally, the container has few cores
    benchmark(run_dir, "cuda", precisions=precisions)


@app.local_entrypoint()
def main(run_dir: str = "flowfirst-gnn-20node-x8-b128-gnn128r6ln-step-clip-modal", stages: bool = False,
         precisions: str = DEFAULT_PRECISIONS):
    bench.remote(run_dir, stages, precisions)
