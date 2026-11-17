"""Checks of the space-time network (networks.FlowFirstSpaceTime, Ways 2-i and 2-ii) and flowfirst/spacetime.py.

Run from the repo root: venv/bin/python3 -m pytest flowfirst/tests/test_spacetime.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.getcwd())
torch.set_default_dtype(torch.float64)

from flowfirst.dataset import load_or_create  # noqa: E402
from flowfirst.fill import MeritOrderFill  # noqa: E402
from flowfirst.block import BlockLadder, BlockSystem, RUNG_FILL  # noqa: E402
from flowfirst.spacetime import (block_dispatch, block_loss, build_spacetime, load_spacetime,  # noqa: E402
                                 ramp_clip, ramp_frac, ramp_violation, spacetime_dispatch)
from networks import FlowFirstGNN  # noqa: E402

BLOCKS = "data/ED_data_gen/GEP_block_data_N3_G6_L3_H876_B4_single.pkl"
CONFIG = "flowfirst/configs/config-3node-blockhours.json"
GNN = {"gnn_hidden": 32, "gnn_rounds": 3, "gnn_no_id_embed": False, "gnn_layernorm": False, "gnn_antisym": False}
PATHS = {**GNN, "paths": True}


@pytest.fixture(scope="module")
def setting():
    if not os.path.exists(BLOCKS):
        pytest.skip(f"{BLOCKS} not built")
    data, _ = load_or_create(CONFIG)
    ds = pickle.load(open(BLOCKS, "rb"))
    torch.manual_seed(0)
    idx = torch.randperm(ds.X.shape[0])[:64]
    return dict(data=data, ds=ds, X=ds.X[idx], idx=idx.numpy())


def perturbed(net, scale=0.05):
    """A trained-looking network: every weight moved off its initial value (zero layers included)."""
    with torch.no_grad():
        for p in net.parameters():
            p.add_(scale * torch.randn_like(p))
    return net


def test_ramp_limited_technologies(setting):
    """On 3 nodes the ramp-limited technologies are Belgian gas, German gas and French nuclear."""
    net = build_spacetime(GNN, setting["data"], setting["ds"], setting["X"])
    assert [tuple(setting["ds"].G[g]) for g in net.slow.tolist()] == [("BEL", "Gas"), ("GER", "Gas"), ("FRA", "Nuclear")]
    assert (ramp_frac(setting["ds"])[net.slow] < 1).all()


def test_starts_as_the_hourly_gnn(setting):
    """With an hourly checkpoint loaded, the untrained space-time paths change nothing."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = build_spacetime(GNN, data, ds, X)
    D, cap, _ = net.split(X)
    hourly = FlowFirstGNN(None, data, net.hours(D, cap), hidden=32, rounds=3)
    with torch.no_grad():                 # a trained-looking checkpoint, not the init the space-time net happens to have
        for p in hourly.parameters():
            p.add_(0.05 * torch.randn_like(p))
    net.hourly.load_state_dict(hourly.state_dict())
    with torch.no_grad():
        f_st = net(X)[0]
        f_h = hourly(net.hours(D, cap))[2]
    assert torch.equal(f_st.reshape(-1, data.num_l), f_h)


def test_ramp_paths_learn(setting):
    """One gradient step on the block loss reaches the technology-to-node update, and changes the flows."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = build_spacetime(GNN, data, ds, X)
    fill = MeritOrderFill(data)
    cost, voll, viol = block_loss(fill, net, X)
    (cost + voll + float(data.pVOLL) * viol.sum(1)).mean().backward()
    assert net.tech_to_node[0][-1].weight.grad.abs().sum() > 0
    assert net.hour_emb.grad.abs().sum() > 0


def test_block_loss_by_hand(setting):
    """Way 2-i: block_loss is the merit-order cost of the four hours, and the MW of ramp violation."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = build_spacetime(GNN, data, ds, X)
    fill = MeritOrderFill(data)
    with torch.no_grad():
        cost, voll, viol = block_loss(fill, net, X)
        D, cap, rho = net.split(X)
        f = net(X)[0]
        for k in range(4):
            U, v = 0.0, 0.0
            p_prev = None
            for t in range(ds.ramp_block_hours_):
                p, e, u_t = fill(D[k, t][None], cap[k, t][None], f[k, t][None])
                U += float(u_t)
                if p_prev is not None:
                    v += float(torch.relu((p - p_prev).abs() - rho[k]).sum())
                p_prev = p
            assert np.isclose(float(cost[k] + voll[k]), U, rtol=1e-12)
            assert np.isclose(float(viol[k].sum()), v, rtol=1e-12, atol=1e-9)


def test_load_and_flows(setting, tmp_path):
    """A saved run loads back to the same flows; spacetime_dispatch without polish is the forward pass."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = perturbed(build_spacetime(GNN, data, ds, X).eval())
    torch.save(net.state_dict(), tmp_path / "model.pt")
    json.dump({"variant": "flowfirst-spacetime", **GNN}, open(tmp_path / "args.json", "w"))
    back = load_spacetime(tmp_path, data, ds)
    f0, p_fix = spacetime_dispatch(net, data, X, 0)
    assert p_fix is None
    assert torch.equal(spacetime_dispatch(back, data, X, 0)[0], f0)
    with torch.no_grad():
        assert torch.equal(f0, net(X)[0].reshape(-1, data.num_l))
    f1 = spacetime_dispatch(net, data, X, 1)[0]                 # the polish only lowers the fill cost
    D, cap, _ = net.split(X)
    Xh = net.hours(D, cap)
    fill = MeritOrderFill(data)
    assert (fill(Xh[:, :data.num_n], Xh[:, data.num_n:], f1)[2] <= fill(Xh[:, :data.num_n], Xh[:, data.num_n:], f0)[2] + 1e-9).all()


# ---------------------------------------------------------------- Way 2-ii: the network also gives the paths
def test_ramp_clip_by_hand():
    """The toy of the design page: merit order 50, 30, 30, 50 with ramp 10 clips to 50, 40, 30, 40;
    a feasible path is left alone; a capacity drop faster than the ramp is met by starting lower."""
    rho = torch.tensor([[10.0]])
    cap = torch.full((1, 4, 1), 60.0)
    clip = lambda t, c=cap: ramp_clip(torch.tensor(t)[None, :, None], c, rho)[0, :, 0].tolist()
    assert clip([50.0, 30.0, 30.0, 50.0]) == [50.0, 40.0, 30.0, 40.0]
    assert clip([50.0, 40.0, 40.0, 50.0]) == [50.0, 40.0, 40.0, 50.0]
    drop = torch.tensor([60.0, 60.0, 60.0, 5.0])[None, :, None]
    assert clip([60.0, 60.0, 60.0, 5.0], drop) == [35.0, 25.0, 15.0, 5.0]


def test_ramp_clip_always_feasible():
    """Any raw path, any capacities: the clipped path lies in [0, cap] and keeps every ramp limit."""
    g = torch.Generator().manual_seed(1)
    K, B, S = 2000, 4, 3
    cap = 100 * torch.rand(K, B, S, generator=g) * (torch.rand(K, B, S, generator=g) > 0.1)
    rho = 40 * torch.rand(K, S, generator=g)
    p = ramp_clip(150 * torch.randn(K, B, S, generator=g), cap, rho)
    assert (p >= 0).all() and (p <= cap + 1e-12).all()
    assert ramp_violation(p, rho).max() <= 1e-12


def test_paths_start_at_clipped_merit_order(setting):
    """Before training the correction is 0: the paths are the merit-order output pushed into the ramp
    limits, the other units fill around them, every node balances, and no ramp limit breaks."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = build_spacetime(PATHS, data, ds, X)
    fill = MeritOrderFill(data)
    with torch.no_grad():
        p, md, f, raw = block_dispatch(fill, net, X)
        D, cap, rho = net.split(X)
        K, B = D.shape[:2]
        p_mo = fill(D.reshape(K * B, -1), cap.reshape(K * B, -1), f.reshape(K * B, -1))[0].reshape(K, B, -1)
    s = net.slow
    assert torch.equal(raw, p_mo[..., s])
    assert torch.equal(p[..., s], ramp_clip(raw, cap[..., s], rho[:, s]))
    assert ramp_violation(p[..., s], rho[:, s]).max() <= 1e-9
    imp = f @ data.lineflow_mask.T.to(f.dtype)
    balance = p @ data.node_to_gen_mask.T.to(p.dtype) + md + imp - D
    assert balance.abs().max() <= 1e-8 * (1 + D.abs().max())
    free = [g for g in range(data.num_g) if g not in s.tolist()]
    assert (p[..., free] >= -1e-12).all() and (p[..., free] <= cap[..., free] + 1e-9).all()


def test_ramp_price_trains_the_corrections_only(setting):
    """The Lagrangian term sees the raw path, whose merit-order part is detached: its gradient reaches
    the path head but not the flow readout, so it cannot push flows toward ramp-feasible merit order.
    The objective reaches both."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = perturbed(build_spacetime(PATHS, data, ds, X))
    fill = MeritOrderFill(data)
    cost, voll, viol = block_loss(fill, net, X)
    assert viol.sum() > 0
    viol.sum().backward(retain_graph=True)
    assert net.path_head[-1].weight.grad.abs().sum() > 0
    assert net.hourly.readout[-1].weight.grad is None or net.hourly.readout[-1].weight.grad.abs().sum() == 0
    net.zero_grad()
    (cost + voll).sum().backward()
    assert net.hourly.readout[-1].weight.grad.abs().sum() > 0
    assert net.path_head[-1].weight.grad.abs().sum() > 0


def test_paths_follow_the_flows(setting):
    """The clipped paths keep their gradient through merit order: moving a flow moves the ramp-limited
    units it displaces, so the flows learn the hourly network's price signal (nuclear replacing gas)."""
    data, ds, X = setting["data"], setting["ds"], setting["X"]
    net = perturbed(build_spacetime(PATHS, data, ds, X))
    p, _, _, _ = block_dispatch(MeritOrderFill(data), net, X)
    p[..., net.slow].sum().backward()
    assert net.hourly.readout[-1].weight.grad.abs().sum() > 0


def test_ladder_with_paths_ends_feasible(setting):
    """Way 2-ii through the deployed pipeline: every block ends feasible and no cheaper than Q*, and
    blocks that stop at the fill keep the network's paths."""
    data, ds, X, idx = setting["data"], setting["ds"], setting["X"], setting["idx"]
    net = perturbed(build_spacetime(PATHS, data, ds, X).eval(), 0.2)
    B, N = ds.ramp_block_hours_, ds.num_n
    bi = ds.block_info
    D, apmax, u = X[:, :B * N].numpy().reshape(-1, B, N), bi["apmax"][idx], bi["u"][idx]
    Q = ds.opt_targets["obj"].numpy()[idx]
    system = BlockSystem.from_operational(ds, ds.ramp_limits(), B)
    f, p_fix = spacetime_dispatch(net, data, X, 1)
    res = BlockLadder(system, data).run(D, apmax, u, f, p_fix.numpy(), net.slow.numpy())
    assert system.max_violation(res.p, res.f, res.md, D, apmax, u).max() <= 1e-6
    assert (res.U >= Q - 1e-6 * (1 + np.abs(Q))).all()
    kept = res.rung == RUNG_FILL
    assert kept.any()
    assert np.allclose(res.p[kept][..., net.slow.numpy()], p_fix.numpy()[kept][..., net.slow.numpy()])
