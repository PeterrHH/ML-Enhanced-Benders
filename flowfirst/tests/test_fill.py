"""Checks for the merit-order fill against the Gurobi-labelled dataset.

Run from the repo root:  .venv/bin/pytest flowfirst
"""
import os

import pytest
import torch

from flowfirst.dataset import CONFIG_DIR, dataset_path, load_or_create
from flowfirst.fill import MeritOrderFill

torch.set_default_dtype(torch.float64)
CONFIGS = [str(CONFIG_DIR / "config-3node.json"), str(CONFIG_DIR / "config-3node-3gen.json")]


@pytest.fixture(scope="module", params=CONFIGS, ids=["2gen", "3gen"])
def data(request):
    import json
    if not os.path.exists(dataset_path(json.load(open(request.param)))):
        pytest.skip(f"dataset not built; run python -m flowfirst.dataset {request.param} first")
    return load_or_create(request.param)[0]


@pytest.fixture(scope="module")
def fill(data):
    return MeritOrderFill(data)


def split_X(data):
    N, G = data.num_n, data.num_g
    return data.X[:, :N], data.X[:, N:N + G]


def random_flows(data, n, scale=1.5, seed=0):
    """Uniform flows in scale times the line box, so some lie outside the limits."""
    g = torch.Generator().manual_seed(seed)
    lb = torch.tensor([-data.pImpCap[l] for l in data.L])
    ub = torch.tensor([data.pExpCap[l] for l in data.L])
    return lb * scale + (ub - lb) * scale * torch.rand(n, len(data.L), generator=g)


def test_fill_on_optimal_flows_reproduces_gurobi_objective(data, fill):
    D, cap = split_X(data)
    _, f_star, _ = data.split_dec_vars_from_Y(data.opt_targets["y_operational"])
    p, e, U = fill(D, cap, f_star)
    obj_star = data.opt_targets["obj"]
    assert torch.allclose(U, obj_star, rtol=1e-8, atol=1e-6)
    assert (e >= -1e-6).all() and (e <= D + 1e-6).all()
    assert (p >= 0).all() and (p <= cap + 1e-12).all()


def test_fill_cost_equals_codebase_objective(data, fill):
    D, cap = split_X(data)
    f = random_flows(data, 4096)
    p, e, U = fill(D[:4096], cap[:4096], f)
    y = torch.cat([p, f, e], dim=1)
    assert torch.allclose(U, data.obj_fn(data.X[:4096], y), rtol=1e-12, atol=1e-9)


def test_autograd_flow_gradient_equals_price_difference(data, fill):
    D, cap = split_X(data)
    n = 8192
    f = random_flows(data, n).requires_grad_(True)
    _, _, U = fill(D[:n], cap[:n], f)
    grad, = torch.autograd.grad(U.sum(), f)
    lo, hi = fill.price_interval(fill.residual(D[:n], f.detach()), cap[:n])
    not_kink = (lo == hi).all(dim=1)
    assert not_kink.float().mean() > 0.99            # kinks have measure zero on random flows
    expected = fill.flow_gradient(lo)
    assert torch.allclose(grad[not_kink], expected[not_kink], atol=1e-9)


def test_cost_is_convex_along_random_segments(data, fill):
    D, cap = split_X(data)
    n = 4096
    f1, f2 = random_flows(data, n, seed=1), random_flows(data, n, seed=2)
    a = torch.rand(n, 1, generator=torch.Generator().manual_seed(3))
    U1, U2 = fill(D[:n], cap[:n], f1)[2], fill(D[:n], cap[:n], f2)[2]
    Um = fill(D[:n], cap[:n], a * f1 + (1 - a) * f2)[2]
    assert (Um <= a[:, 0] * U1 + (1 - a[:, 0]) * U2 + 1e-6).all()


def test_price_regimes(data, fill):
    D, cap = split_X(data)
    node_cap = cap @ data.node_to_gen_mask.T
    B = D.shape[0]
    # shortage: residual above total capacity -> VOLL both sides
    lo, hi = fill.price_interval(node_cap + 1.0, cap)
    assert (lo == fill.voll).all() and (hi == fill.voll).all()
    # over-supply: negative residual -> -VOLL both sides
    lo, hi = fill.price_interval(-torch.ones(B, data.num_n), cap)
    assert (lo == -fill.voll).all() and (hi == -fill.voll).all()
    # kink: residual exactly at the cheapest unit's capacity -> [cheap cost, next cost]
    cheapest = torch.zeros(data.num_n, dtype=torch.long)
    for i in range(data.num_n):
        gens = torch.nonzero(data.node_to_gen_mask[i]).flatten()
        cheapest[i] = gens[torch.argmin(fill.cost[gens])]
    r = cap[:, cheapest]
    pos = (r > 0) & (r < node_cap)                    # only where the breakpoint is interior
    lo, hi = fill.price_interval(r, cap)
    assert (lo[pos] == fill.cost[cheapest].expand(B, -1)[pos]).all()
    assert (hi[pos] > lo[pos]).all()
