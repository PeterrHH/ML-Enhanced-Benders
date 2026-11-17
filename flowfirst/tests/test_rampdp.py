"""Checks of Way 2-iii's node-block program (flowfirst/rampdp.py) and the no-LP pipeline BlockLadder.run_dp.

Run from the repo root: venv/bin/python3 -m pytest flowfirst/tests/test_rampdp.py
"""
import os
import pickle
import sys

import numpy as np
import pytest
import torch
from scipy.optimize import linprog

sys.path.insert(0, os.getcwd())
torch.set_default_dtype(torch.float64)

from flowfirst.block import RUNG_FILL, RUNG_SHRINK, BlockLadder, BlockSystem  # noqa: E402
from flowfirst.dataset import load_or_create  # noqa: E402
from flowfirst.fill import MeritOrderFill  # noqa: E402
from flowfirst.rampdp import RampDP  # noqa: E402
from flowfirst.spacetime import build_spacetime, dp_loss, ramp_frac  # noqa: E402
from flowfirst.train import line_bounds  # noqa: E402

BLOCKS = "data/ED_data_gen/GEP_block_data_N3_G6_L3_H876_B4_single.pkl"
CONFIG = "flowfirst/configs/config-3node-blockhours.json"
GNN = {"gnn_hidden": 32, "gnn_rounds": 3, "gnn_no_id_embed": False, "gnn_layernorm": False, "gnn_antisym": False}


def toy():
    """The design page's toy: nuclear (cost 1, 60 MW, ramp 10) and solar (cost 0) at one node, demand 50."""
    dp = RampDP([[1, 1]], [1.0, 0.0], 100.0, [True, False])
    cap = torch.tensor([[[60, 0], [60, 20], [60, 20], [60, 0]]], dtype=torch.float64)
    return dp, torch.full((1, 4, 1), 50.0), cap, torch.tensor([[10.0, 1e9]])


def test_toy_by_hand():
    """Nuclear 50, 40, 40, 50 with 10 MW of solar curtailed in hours 2 and 3: cost 180. Prices 2, 0, 0, 2:
    one more MW in hour 1 lifts nuclear there and, by the ramp, in hour 2; solar is spilled in hours 2-3."""
    dp, r, cap, rho = toy()
    for hard in (False, True):
        out = dp(r, cap, rho, r, hard=hard)
        assert out["cost"].item() == pytest.approx(180.0, abs=1e-9)
        assert out["p"][0, :, 0].tolist() == pytest.approx([50, 40, 40, 50], abs=1e-9)
        assert out["md"].abs().max() <= 1e-9
        assert dp.prices(r, cap, rho, r, hard=hard).flatten().tolist() == pytest.approx([2, 0, 0, 2], abs=1e-5)


def node_lp(y, c, cap_s, rho_s, fcost, fcap, voll, D=None):
    """One node over B hours as an LP (scipy/HiGHS): k ramp-limited units, m others, unserved demand.
    Elastic (D None): md free in sign at VOLL |md|. Hard: md in [0, D]. Returns (value or None, duals)."""
    B, k = cap_s.shape
    m = fcap.shape[1]
    nh = k + m + 2
    obj = np.tile(np.concatenate([c, fcost, [voll, voll]]), B)
    A_eq = np.zeros((B, B * nh))
    bounds = []
    for t in range(B):
        A_eq[t, t * nh:t * nh + k + m + 1] = 1.0
        A_eq[t, t * nh + k + m + 1] = -1.0
        bounds += [(0, cap_s[t, i]) for i in range(k)] + [(0, fcap[t, i]) for i in range(m)]
        bounds += [(0, None), (0, None)] if D is None else [(0, D[t]), (0, 0)]
    rows, rhs = [], []
    for t in range(1, B):
        for i in range(k):
            for sgn in (1, -1):
                row = np.zeros(B * nh)
                row[t * nh + i], row[(t - 1) * nh + i] = sgn, -sgn
                rows.append(row)
                rhs.append(rho_s[i])
    res = linprog(obj, A_ub=np.array(rows) if rows else None, b_ub=np.array(rhs) if rows else None,
                  A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        return None, None
    return res.fun, res.eqlin.marginals


def random_node(rng, k, m, B=4):
    c = np.sort(rng.uniform(0.5, 3.0, k))
    fcost = np.sort(rng.uniform(0.0, 2.0, m))
    cap_s = rng.uniform(10, 60, (B, k)) * (rng.random((B, k)) > 0.1)
    fcap = rng.uniform(0, 40, (B, m)) * (rng.random((B, m)) > 0.3)
    rho_s = rng.uniform(2, 25, k)
    D = rng.uniform(20, 90, B)
    y = D + rng.uniform(-30, 30, B)                    # imports and exports
    return c, cap_s, rho_s, fcost, fcap, D, y


def dp_node(c, cap_s, rho_s, fcost, fcap, voll):
    k, m = len(c), len(fcost)
    dp = RampDP([[1] * (k + m)], np.concatenate([c, fcost]), voll, [True] * k + [False] * m)
    cap = torch.tensor(np.concatenate([cap_s, fcap], axis=1))[None]
    rho = torch.tensor(np.concatenate([rho_s, np.full(m, 1e9)]))[None]
    return dp, cap, rho


@pytest.mark.parametrize("k,m", [(0, 1), (1, 0), (1, 1), (1, 2), (1, 3)])
def test_matches_lp_at_most_one_ramp_limited_unit(k, m):
    """k <= 1: the program's cost and prices equal the node LP's optimum and balance duals, elastic and hard,
    and hard mode reports infeasible exactly where the LP is."""
    rng = np.random.default_rng(10 * k + m)
    voll = 10.0
    for _ in range(40):
        c, cap_s, rho_s, fcost, fcap, D, y = random_node(rng, k, m)
        dp, cap, rho = dp_node(c, cap_s, rho_s, fcost, fcap, voll)
        r, Dt = torch.tensor(y)[None, :, None], torch.tensor(D)[None, :, None]
        for hard in (False, True):
            v, duals = node_lp(y, c, cap_s, rho_s, fcost, fcap, voll, D if hard else None)
            out = dp(r, cap, rho, Dt, hard=hard)
            if hard:
                assert bool(out["ok"]) == (v is not None)
            if v is None:
                continue
            assert out["cost"].item() == pytest.approx(v, rel=1e-9, abs=1e-9)
            lam = dp.prices(r, cap, rho, Dt, hard=hard).flatten().numpy()
            assert np.allclose(lam, duals, atol=1e-4)


def test_two_ramp_limited_units_stay_feasible():
    """k = 2: one program per unit in turn. Whenever hard mode finds a dispatch it meets every row, and its
    cost is never below the LP's; elastic is never below the elastic LP. The prices are finite."""
    rng = np.random.default_rng(2)
    voll, found, gaps = 10.0, 0, []
    for _ in range(60):
        c, cap_s, rho_s, fcost, fcap, D, y = random_node(rng, 2, 1)
        dp, cap, rho = dp_node(c, cap_s, rho_s, fcost, fcap, voll)
        r, Dt = torch.tensor(y)[None, :, None], torch.tensor(D)[None, :, None]
        out = dp(r, cap, rho, Dt, hard=True)
        v, _ = node_lp(y, c, cap_s, rho_s, fcost, fcap, voll, D)
        if bool(out["ok"]):
            found += 1
            p = out["p"][0].numpy()
            md = out["md"][0, :, 0].numpy()
            assert (p >= -1e-9).all() and (p <= np.concatenate([cap_s, fcap], 1) + 1e-9).all()
            assert (np.abs(np.diff(p[:, :2], axis=0)) <= rho_s + 1e-9).all()
            assert (md >= -1e-7).all() and (md <= D + 1e-7).all()
            assert np.allclose(p.sum(1) + md, y)
            assert v is not None and out["cost"].item() >= v - 1e-7 * (1 + abs(v))
            gaps.append((out["cost"].item() - v) / max(abs(v), 1e-9))
        ve, _ = node_lp(y, c, cap_s, rho_s, fcost, fcap, voll)
        oute = dp(r, cap, rho)
        assert oute["cost"].item() >= ve - 1e-7 * (1 + abs(ve))
        assert torch.isfinite(dp.prices(r, cap, rho, Dt, hard=False)).all()
    assert found > 20
    print(f"k = 2: {found}/60 feasible, gap to the LP mean {np.mean(gaps):.4%}, max {np.max(gaps):.4%}")


# ---------------------------------------------------------------- the real system
@pytest.fixture(scope="module")
def setting():
    if not os.path.exists(BLOCKS):
        pytest.skip(f"{BLOCKS} not built")
    data, _ = load_or_create(CONFIG)
    ds = pickle.load(open(BLOCKS, "rb"))
    idx = torch.as_tensor(np.flatnonzero(ds.block_info["split"] == "test")[:500])
    return dict(data=data, ds=ds, idx=idx, dp=RampDP.from_data(data, ramp_frac(ds)))


def block_arrays(ds, idx):
    B, N, G, L = ds.ramp_block_hours_, ds.num_n, ds.num_g, ds.num_l
    X = ds.X[idx]
    K = X.shape[0]
    D = X[:, :B * N].reshape(K, B, N)
    cap = X[:, B * N:B * (N + G)].reshape(K, B, G)
    rho = torch.tensor(ds.ramp_limits()) * torch.as_tensor(ds.block_info["u"][idx.numpy()])
    f_opt = ds.opt_targets["y_operational"][idx].reshape(K, B, G + L + N)[..., G:G + L]
    return X, D, cap, rho, f_opt


def test_exact_flows_give_the_block_optimum(setting):
    """On the exact block LP's flows, the best ramp-feasible dispatch costs exactly Q*, and it exists."""
    data, ds, idx, dp = setting["data"], setting["ds"], setting["idx"], setting["dp"]
    _, D, cap, rho, f = block_arrays(ds, idx)
    out = dp(D - f @ data.lineflow_mask.T.to(f.dtype), cap, rho, D, hard=True)
    Q = ds.opt_targets["obj"][idx]
    assert out["ok"].all()
    assert torch.allclose(out["cost"], Q, rtol=1e-10, atol=1e-10)


def test_elastic_is_the_hourly_loss_without_ramps(setting):
    """With ramp limits out of reach, the program's elastic cost is the hourly merit-order loss of the flows."""
    data, ds, idx, dp = setting["data"], setting["ds"], setting["idx"], setting["dp"]
    _, D, cap, rho, f = block_arrays(ds, idx)
    lb, ub = line_bounds(data)
    torch.manual_seed(0)
    f = lb + (ub - lb) * torch.rand_like(f)
    K, B, N = D.shape
    cost = dp(D - f @ data.lineflow_mask.T.to(f.dtype), cap, torch.full_like(rho, 1e12))["cost"]
    hourly = MeritOrderFill(data)(D.reshape(K * B, N), cap.reshape(K * B, -1), f.reshape(K * B, -1))[2]
    assert torch.allclose(cost, hourly.reshape(K, B).sum(1), rtol=1e-10)


def test_run_dp_always_feasible(setting):
    """Way 2-iii's pipeline on garbage flows (uniform in the line limits): every block ends meeting every row,
    never cheaper than Q*, and some blocks need the shrink."""
    data, ds, idx, dp = setting["data"], setting["ds"], setting["idx"], setting["dp"]
    _, D, cap, rho, f = block_arrays(ds, idx)
    lb, ub = line_bounds(data)
    torch.manual_seed(1)
    f = lb + (ub - lb) * torch.rand_like(f)
    bi, n = ds.block_info, idx.numpy()
    system = BlockSystem.from_operational(ds, ds.ramp_limits(), ds.ramp_block_hours_)
    res = BlockLadder(system, data).run_dp(D.numpy(), bi["apmax"][n], bi["u"][n], f.reshape(-1, f.shape[-1]), dp)
    Q = ds.opt_targets["obj"].numpy()[n]
    assert system.max_violation(res.p, res.f, res.md, D.numpy(), bi["apmax"][n], bi["u"][n]).max() <= 1e-6
    assert (res.U >= Q - 1e-6 * (1 + np.abs(Q))).all()
    assert (res.L <= Q + 1e-6 * (1 + np.abs(Q))).all()
    assert (res.rung == RUNG_SHRINK).any() and (res.rung == RUNG_FILL).any()


def test_dp_loss_trains_the_flows(setting):
    """The loss reaches the network's readout, and a small step of the flows along minus the program's
    gradient lowers the cost."""
    data, ds, idx, dp = setting["data"], setting["ds"], setting["idx"], setting["dp"]
    X, D, cap, rho, _ = block_arrays(ds, idx[:64])
    net = build_spacetime(GNN, data, ds, X)
    dp_loss(dp, net, X).sum().backward()
    assert net.hourly.readout[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        f = net(X)[0]
    f = f.clone().requires_grad_(True)
    A = data.lineflow_mask.T.to(f.dtype)
    cost = dp(D - f @ A, cap, rho)["cost"]
    g, = torch.autograd.grad(cost.sum(), f)
    step = 1e-3 * g / g.abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    with torch.no_grad():
        after = dp(D - (f - step) @ A, cap, rho)["cost"]
    moved = g.abs().amax(dim=(1, 2)) > 0
    assert (after[moved] <= cost.detach()[moved] + 1e-9).all()
    assert (after[moved] < cost.detach()[moved]).float().mean() > 0.5
