"""Checks for the dual recovery, the certificate and the polish against the Gurobi-labelled datasets.

Run from the repo root:  .venv/bin/pytest flowfirst
"""
import json
import os

import pytest
import torch

from flowfirst.dataset import CONFIG_DIR, dataset_path, load_or_create
from flowfirst.dual import DualRecovery, GraphedPrimalDual, PathPolish, Polish, PrimalDual, cast_net
from flowfirst.train import build_net, line_bounds

torch.set_default_dtype(torch.float64)
CONFIGS = [str(CONFIG_DIR / "config-3node.json"), str(CONFIG_DIR / "config-3node-3gen.json")]
B = 512


@pytest.fixture(scope="module", params=CONFIGS, ids=["2gen", "3gen"])
def problem(request):
    if not os.path.exists(dataset_path(json.load(open(request.param)))):
        pytest.skip(f"dataset not built; run python -m flowfirst.dataset {request.param} first")
    return load_or_create(request.param)


@pytest.fixture(scope="module")
def data(problem):
    return problem[0]


@pytest.fixture(scope="module")
def labels(data):
    """X, optimum, nodal price, multipliers and primal solution of the first B instances."""
    t = data.opt_targets
    return (data.X[:B], t["obj"][:B], -t["lamb_operational"][:B], t["mu_operational"][:B], t["y_operational"][:B])


def random_flows(data, seed=0):
    g = torch.Generator().manual_seed(seed)
    lb, ub = line_bounds(data)
    return lb + (ub - lb) * torch.rand(B, data.num_l, generator=g)


def small_net(problem, X):
    """An untrained 2-round GNN: enough to exercise every stage of the pipeline."""
    data, args = problem
    torch.manual_seed(0)
    return build_net("flowfirst-gnn", args, data, None, X_train=X,
                     gnn_kwargs=dict(hidden=16, rounds=2, id_embed=8, layernorm=False, antisym=False)).eval()


def assert_same(out, ref, rows=slice(None), tol=1e-12):
    """Every field of two solutions equal to `tol` relative to the field's scale."""
    for k, v in vars(out).items():
        r = getattr(ref, k)[rows]
        assert v.shape == r.shape and v.dtype == r.dtype, k
        assert (v - r).abs().max() <= tol * max(1.0, r.abs().max().item()), k


def test_gurobi_prices_reproduce_the_optimum(data, labels):
    X, obj, price, *_ = labels
    assert torch.allclose(DualRecovery(data).dual_objective(X, price), obj, rtol=1e-9)


def test_completion_agrees_with_the_problem_class_dual_objective(data, labels):
    X, _, price, *_ = labels
    dual = DualRecovery(data)
    for lam in (price, dual.prices(X, random_flows(data))):
        mu = dual.completion(lam)
        assert (mu >= 0).all()
        # the problem class stores the price negated; the multipliers keep their sign
        assert torch.allclose(data.dual_obj_fn(X, mu, -lam), dual.dual_objective(X, lam), rtol=1e-9)


def test_recovered_dual_is_a_lower_bound_for_any_flows(data, labels):
    X, obj, *_ = labels
    dual, polish = DualRecovery(data), Polish(data)
    for seed in range(3):
        f = random_flows(data, seed)
        fD = dual.dual_objective(X, dual.prices(X, f))
        assert (fD <= obj * (1 + 1e-9)).all()
        assert (fD <= polish.cost(X, f) * (1 + 1e-9)).all()


def test_node_ascent_never_lowers_the_bound(data, labels):
    X, *_ = labels
    dual = DualRecovery(data)
    lam_q = dual.quotient_prices(X, random_flows(data))
    assert (dual.dual_objective(X, dual.node_ascent(X, lam_q)) >= dual.dual_objective(X, lam_q) - 1e-9).all()


def test_optimal_flows_give_the_exact_dual_on_most_instances(data, labels):
    X, obj, _, _, y = labels
    f = data.split_dec_vars_from_Y(y)[1]
    fD = DualRecovery(data).dual_objective(X, DualRecovery(data).prices(X, f))
    assert ((obj - fD) / obj < 1e-6).double().mean() > 0.95


def test_polish_lowers_cost_within_limits_and_stays_an_upper_bound(data, labels):
    X, obj, *_ = labels
    polish, (lb, ub) = Polish(data), line_bounds(data)
    f0 = random_flows(data)
    f1, U1 = polish(X, f0, sweeps=2)
    assert (U1 <= polish.cost(X, f0) + 1e-9).all()
    assert (U1 < polish.cost(X, f0)).double().mean() > 0.5
    assert (f1 >= lb - 1e-9).all() and (f1 <= ub + 1e-9).all()
    assert (U1 >= obj * (1 - 1e-9)).all()
    assert torch.allclose(U1, polish.cost(X, f1))


def test_primal_dual_end_to_end(problem, labels):
    data, args = problem
    X, obj, *_ = labels
    out = PrimalDual(data, small_net(problem, X))(X)
    N, G, L = data.num_n, data.num_g, data.num_l
    assert out.y.shape == (B, G + L + N) and out.lam.shape == (B, N) and out.mu.shape == (B, 2 * (G + L + N))
    assert torch.allclose(out.primal, data.obj_fn(X, out.y))
    assert (data.eq_resid(X, out.y).abs() < 1e-6).all()
    assert (out.dual <= obj * (1 + 1e-9)).all() and (out.dual <= out.primal * (1 + 1e-9)).all()
    assert torch.allclose(out.dual, data.dual_obj_fn(X, out.mu, -out.lam))
    true_gap = (out.primal - obj) / obj
    assert (out.certificate >= true_gap - 1e-9).all()


def test_polish_keeps_improving_while_a_single_line_can(data, labels):
    """Exact line search never stalls where one line still lowers the cost (it may need many sweeps: a reroute
    through a kink node advances by that node's marginal unit per sweep)."""
    X, *_ = labels
    polish, (lb, ub) = Polish(data), line_bounds(data)
    f, U = polish(X, random_flows(data), sweeps=5)
    eps = 1e-6 * (ub - lb)
    descent = torch.zeros_like(U)
    for l in range(data.num_l):
        for sign in (1.0, -1.0):
            g = f.clone()
            g[:, l] = (f[:, l] + sign * eps[l]).clamp(lb[l], ub[l])
            descent = torch.maximum(descent, U - polish.cost(X, g))
    _, U_next = polish(X, f, sweeps=1)
    assert (U_next <= U + 1e-9 * U).all()
    improvable = descent > 1e-9 * U
    assert improvable.any()
    assert (U_next[improvable] < U[improvable]).all()


def test_polish_step_from_a_slightly_over_supplied_node_never_raises_cost(data):
    """The node cost has a breakpoint at zero residual: relieving 1 kW of over-supply must stop there, not run on
    into the node's own units when those are dearer than the neighbour's marginal unit."""
    polish, (lb, ub) = Polish(data), line_bounds(data)
    N, G, L = data.num_n, data.num_g, data.num_l
    M, c, A = data.node_to_gen_mask, data.cost_vec, data.lineflow_mask
    cheapest = torch.stack([c[M[n].bool()].min() for n in range(N)])
    dear, cheap = int(cheapest.argmax()), int(cheapest.argmin())
    l = int(torch.nonzero((A[dear] != 0) & (A[cheap] != 0))[0])          # a line joining the two nodes
    amount = min(1000.0, 0.9 * (ub[l] if A[dear, l] > 0 else -lb[l]).item())   # what the line can carry into the dear node
    D = torch.full((1, N), 1000.0)
    D[0, dear], D[0, cheap] = amount, 5000.0                               # the cheap node's first unit stays marginal
    X = torch.cat([D, torch.full((1, G), 10000.0)], dim=1)
    f = torch.zeros(1, L)
    f[0, l] = A[dear, l] * (amount + 1e-3)                                # 1 kW more into the dear node than it needs
    assert lb[l] <= f[0, l] <= ub[l]
    assert polish.fill.residual(D, f)[0, dear] < 0
    U0 = polish.cost(X, f)
    _, U1 = polish(X, f, sweeps=1)
    assert (U1 <= U0 + 1e-9 * U0).all()


def test_path_polish_is_exact_from_any_start(data, labels):
    X, obj, *_ = labels
    path, (lb, ub) = PathPolish(data), line_bounds(data)
    for f0 in (random_flows(data), torch.zeros(B, data.num_l)):
        f, U, n = path(X, f0)
        assert torch.allclose(U, obj, rtol=1e-9)
        assert (n < 200).all()
        assert (f >= lb - 1e-9).all() and (f <= ub + 1e-9).all()


def test_path_polish_leaves_optimal_flows_alone(data, labels):
    X, obj, _, _, y = labels
    f, U, n = PathPolish(data)(X, data.split_dec_vars_from_Y(y)[1])
    assert torch.allclose(U, obj, rtol=1e-9)
    assert (n <= 2).all()


def test_primal_dual_exact_mode_reaches_the_optimum(problem, labels):
    data, args = problem
    X, obj, *_ = labels
    out = PrimalDual(data, small_net(problem, X), exact=True)(X)
    assert torch.allclose(out.primal, obj, rtol=1e-9)
    assert torch.allclose(out.primal, data.obj_fn(X, out.y))
    assert (out.dual <= obj * (1 + 1e-9)).all()


def test_pipeline_has_no_data_dependent_ops(problem, labels):
    """Fixed shapes, no host-device syncs, no data-dependent control flow: the non-exact pipeline runs on fake
    tensors, which is exactly what a CUDA graph capture needs (F30). Fake tensors carry no values, so any
    `.item()`, `torch.equal`, `nonzero` or `if tensor:` on the way raises."""
    from torch._subclasses.fake_tensor import FakeTensorMode
    data, _ = problem
    X, *_ = labels
    pipeline = PrimalDual(data, small_net(problem, X))
    with FakeTensorMode(allow_non_fake_inputs=True):
        out = pipeline(X)
    assert out.certificate.shape == (B,)


def test_graphed_pipeline_matches_eager_for_any_number_of_instances(problem, labels):
    """Chunks of the capture batch, then a partial chunk that reuses the buffer's stale rows; on CPU the chunks run
    eagerly, the GPU capture itself is checked by modal/bench.py (F30)."""
    data, _ = problem
    X, *_ = labels
    pipeline = PrimalDual(data, small_net(problem, X))
    ref, graphed = pipeline(X), GraphedPrimalDual(pipeline, batch=200)
    for M in (B, 200, 37):        # two chunks and a partial one; exactly one chunk; less than one chunk
        assert_same(graphed(X[:M]), ref, rows=slice(0, M))
    with pytest.raises(ValueError):
        GraphedPrimalDual(PrimalDual(data, small_net(problem, X), exact=True), batch=200)


def test_network_compiles_as_one_graph_with_identical_flows(problem, labels):
    """torch.compile is the second switch of F30; it must not break the forward pass into pieces (fullgraph).
    Checked with the trace-only backend, inductor needs a GPU toolchain."""
    _, X = problem, labels[0]
    net = small_net(problem, X)
    compiled = torch.compile(net, backend="aot_eager", fullgraph=True, dynamic=False)
    with torch.no_grad():
        f, f_compiled = net(X)[2], compiled(X)[2]
    torch._dynamo.reset()
    assert (f - f_compiled).abs().max() <= 1e-12 * f.abs().max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["float32", "float16", "bfloat16"])
def test_reduced_precision_network_keeps_float64_bounds(problem, labels, dtype):
    """The network may hold its parameters in a cheaper dtype (F31): it returns flows in the data's dtype, so the
    dispatch balances to float64 precision and dual and certificate stay valid bounds."""
    data, _ = problem
    X, obj, *_ = labels
    net = cast_net(small_net(problem, X), dtype)
    assert next(net.parameters()).dtype == dtype and net.f_lb.dtype == torch.float64
    out = PrimalDual(data, net)(X)
    assert out.y.dtype == out.lam.dtype == out.certificate.dtype == torch.float64
    assert (data.eq_resid(X, out.y).abs() < 1e-6).all()
    assert torch.allclose(out.primal, data.obj_fn(X, out.y))
    assert (out.dual <= obj * (1 + 1e-9)).all()
    assert (out.certificate >= (out.primal - obj) / obj - 1e-9).all()
