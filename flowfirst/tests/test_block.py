"""Checks for the ramp-block ladder (flowfirst/block.py) on the harvested 3-node B = 4 instances.

Each test states one guarantee of docs/ramping-surrogate-design.tex and checks it against the
exact block LP. Run from the repo root:  venv/bin/python3 -m pytest flowfirst/tests/test_block.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from flowfirst.block import (BlockLadder, BlockSystem, RUNG_R3, block_inputs,  # noqa: E402
                             network_flows, sample_blocks)
from flowfirst.dataset import load_or_create  # noqa: E402
from flowfirst.dual import build_net_for_data  # noqa: E402
from flowfirst.train import line_bounds  # noqa: E402

sys.path.insert(0, os.getcwd())
from gep_benders import SampleStructure  # noqa: E402

TRAJ = "data/ED_data_gen/GEP_trajectories_N3_G6_L3_H876_B4_single.pkl"
RUN = "flowfirst/runs/flowfirst-harvest"
B = 4


@pytest.fixture(scope="module")
def setting():
    if not (os.path.exists(TRAJ) and os.path.exists(os.path.join(RUN, "args.json"))):
        pytest.skip("harvested 3-node trajectories or the flowfirst-harvest run are not available")
    trajs = pickle.load(open(TRAJ, "rb"))
    run_args = json.load(open(os.path.join(RUN, "args.json")))
    data, _ = load_or_create(run_args["config"])
    gep = pickle.load(open(trajs[0]["instance_path"], "rb"))
    gep.set_ramping(B)
    struct = SampleStructure(gep, 0)
    system = BlockSystem.from_gep(gep, B)
    ladders = {(r3, esc): BlockLadder(system, data, r3=r3, escalate=esc)
               for r3 in ("lp", "bisect") for esc in ("infeasible", "unmet")}
    U = np.asarray(trajs[0]["trajectory"])
    # each point: every block of the sample at one trajectory investment
    points = [sample_blocks(struct, U[i]) for i in (len(U) // 3, 2 * len(U) // 3, len(U) - 1)]
    # plus a batch with a different investment per block
    rng = np.random.default_rng(7)
    D, apmax, _ = points[0]
    points.append((D, apmax, U[rng.integers(0, len(U), D.shape[0])]))
    return dict(data=data, sys=system, struct=struct, ladders=ladders, run_args=run_args, points=points)


def rel(a, b):
    return np.abs(a - b) / np.maximum(1.0, np.abs(b))


def test_system_matches_sample_structure(setting):
    """BlockSystem builds the same block matrices and rhs as gep_benders.SampleStructure."""
    s, st = setting["sys"], setting["struct"]
    assert np.array_equal(s.A_ineq_unit, st.A_ineq_unit) and np.array_equal(s.A_eq_unit, st.A_eq_unit)
    u = setting["points"][0][2][0]
    D, apmax, uu = sample_blocks(st, u)
    assert np.allclose(s.b_ineq(D, apmax * uu[:, None, :]).reshape(st.T, -1), st.b_ineq_at(u))
    assert np.allclose(s.ramp_rhs(s.rho(uu))[0], st.ramp_rhs_at(u))


def test_exact_duals_give_the_exact_bound(setting):
    """Proposition 1 with the LP's own duals: the completion recovers Q_b exactly (strong duality)."""
    s = setting["sys"]
    for D, apmax, u in setting["points"]:
        ex = s.exact(D, apmax, u)
        Lb, _ = s.dual_bound(ex["lam"], ex["nu_up"], ex["nu_dn"], D, apmax, u)
        assert rel(Lb, ex["Q"]).max() < 1e-6


def test_slopes_match_finite_differences(setting):
    """The cut slope is the gradient of L in u at fixed (lam, nu): L(u + e_g) - L(u) = slope_g."""
    s = setting["sys"]
    for D, apmax, u in setting["points"][2:]:
        ex = s.exact(D, apmax, u)
        bound = lambda v: s.dual_bound(ex["lam"], ex["nu_up"], ex["nu_dn"], D, apmax, v)
        L0, slopes = bound(u)
        for g in range(s.G):
            e = np.zeros_like(u)
            e[:, g] = 1.0
            L1, _ = bound(u + e)
            assert rel(L1 - L0, slopes[:, g]).max() < 1e-6


def test_random_duals_are_lower_bounds(setting):
    """Proposition 1: any prices and any nu >= 0 give L_b <= Q_b."""
    s = setting["sys"]
    rng = np.random.default_rng(0)
    D, apmax, u = setting["points"][3]
    Q = s.exact(D, apmax, u)["Q"]
    K = D.shape[0]
    for _ in range(5):
        lam = rng.uniform(-2 * s.voll, 2 * s.voll, size=(K, B, s.N))
        nu_up = rng.exponential(s.voll, size=(K, B - 1, s.G))
        nu_dn = rng.exponential(s.voll, size=(K, B - 1, s.G))
        Lb, _ = s.dual_bound(lam, nu_up, nu_dn, D, apmax, u)
        assert np.all(Lb <= Q + 1e-6 * np.maximum(1.0, np.abs(Q)))


def test_ramp_ascent_only_raises_valid_bounds(setting):
    """ramp_ascent: for fixed prices the bound never drops and stays a lower bound."""
    s = setting["sys"]
    rng = np.random.default_rng(3)
    for D, apmax, u in setting["points"]:
        ex = s.exact(D, apmax, u)
        zero = np.zeros_like(ex["nu_up"])
        for lam in (ex["lam"], ex["lam"] + rng.normal(0, 0.05, ex["lam"].shape)):
            L0, _ = s.dual_bound(lam, zero, zero, D, apmax, u)
            nu_u, nu_d = s.ramp_ascent(lam, zero.copy(), zero.copy(), apmax, u)
            L1, _ = s.dual_bound(lam, nu_u, nu_d, D, apmax, u)
            assert np.all(nu_u >= 0) and np.all(nu_d >= 0)
            assert np.all(L1 >= L0 - 1e-9 * np.maximum(1.0, np.abs(L0)))
            assert np.all(L1 <= ex["Q"] + 1e-6 * np.maximum(1.0, np.abs(ex["Q"])))


def test_window_scaling_makes_every_hour_feasible(setting):
    """Lemma 3: after R1 each node's residual lies in [0, C + D] and flows stay within limits."""
    lad = setting["ladders"][("lp", "infeasible")]
    data = lad.data
    f_lb, f_ub = line_bounds(data)
    D, apmax, u = setting["points"][0]
    X = block_inputs(D, apmax, u)
    Dt, cap = X[:, :data.num_n], X[:, data.num_n:]
    g = torch.Generator().manual_seed(1)
    f = f_lb + (f_ub - f_lb) * torch.rand(X.shape[0], lad.sys.L, generator=g)
    alpha, f1 = lad.window_scaling(Dt, cap, f)
    r = Dt - f1 @ data.lineflow_mask.T
    C = cap @ data.node_to_gen_mask.T
    tol = 1e-9 * (1 + Dt.abs())
    assert torch.all(r >= -tol) and torch.all(r <= C + Dt + tol)
    assert torch.all(f1 >= f_lb - 1e-9) and torch.all(f1 <= f_ub + 1e-9)
    assert torch.all((alpha >= 0) & (alpha <= 1))


def test_window_scaling_keeps_useful_flows(setting):
    """R1 removes only the infeasible part of an hour's flows. A node with no capacity that exports
    (free under the hourly fill, which prices unmet demand at VOLL wherever it sits) loses that
    export; the useful flows out of the node with capacity stay as they are. Scaling the whole
    hour by one factor would zero them as well."""
    lad = setting["ladders"][("lp", "unmet")]
    data = lad.data
    A = data.lineflow_mask
    fr, to = (A == -1).int().argmax(0), (A == 1).int().argmax(0)
    f_lb, f_ub = line_bounds(data)
    D = torch.full((1, data.num_n), 20000.0)
    cap = torch.zeros(1, data.num_g)
    cap[0, data.node_to_gen_mask[0].bool()] = 1e6                  # only node 0 can generate
    f = torch.zeros(1, data.num_l)
    bad = next(l for l in range(data.num_l) if 0 not in (int(fr[l]), int(to[l])))
    sender = int(fr[bad])
    f[0, bad] = 1000.0                                             # an empty node exports 1000 MW
    useful = []
    for l in range(data.num_l):
        a, b = int(fr[l]), int(to[l])
        if 0 in (a, b) and sender not in (a, b):                   # node 0 to the other node, half the limit
            f[0, l] = 0.5 * f_ub[l] if a == 0 else 0.5 * f_lb[l]
            useful.append(l)
    assert useful
    alpha, f1 = lad.window_scaling(D, cap, f)
    assert torch.equal(f1[0, useful], f[0, useful]) and float(alpha) == 1.0
    assert abs(float(f1[0, bad])) < 1e-9
    r = D - f1 @ A.T
    C = cap @ data.node_to_gen_mask.T
    assert torch.all(r >= -1e-9) and torch.all(r <= C + D + 1e-9)


def test_window_scaling_leaves_feasible_flows_alone(setting):
    """Flows whose hours are already feasible (the exact block LP's) pass R1 unchanged."""
    lad = setting["ladders"][("lp", "unmet")]
    D, apmax, u = setting["points"][0]
    ex = lad.sys.exact(D, apmax, u)
    X = block_inputs(D, apmax, u)
    f = torch.tensor(ex["f"].reshape(-1, lad.sys.L))
    alpha, f1 = lad.window_scaling(X[:, :lad.sys.N], X[:, lad.sys.N:], f)
    assert torch.equal(f1, f) and torch.all(alpha == 1)


def test_exact_flows_give_the_exact_cost(setting):
    """With the block LP's optimal flows the ladder's dispatch is optimal: U_b = Q_b on every block."""
    lad = setting["ladders"][("lp", "infeasible")]
    for D, apmax, u in setting["points"]:
        ex = lad.sys.exact(D, apmax, u)
        res = lad.run(D, apmax, u, torch.tensor(ex["f"].reshape(-1, lad.sys.L)))
        assert res.stats["max_rel_violation"] < 1e-6
        assert rel(res.U, ex["Q"]).max() < 1e-6
        assert np.all(res.L <= ex["Q"] + 1e-6 * np.maximum(1.0, np.abs(ex["Q"])))


@pytest.mark.parametrize("mode", [("lp", "infeasible"), ("lp", "unmet"), ("bisect", "infeasible"), ("bisect", "unmet")])
def test_random_flows_always_end_feasible(setting, mode):
    """Proposition 2: whatever the flows, every block leaves the ladder feasible, and U >= Q >= L."""
    lad = setting["ladders"][mode]
    f_lb, f_ub = line_bounds(lad.data)
    n_r3 = 0
    for i, (D, apmax, u) in enumerate(setting["points"]):
        Q = lad.sys.exact(D, apmax, u)["Q"]
        g = torch.Generator().manual_seed(10 + i)
        f = f_lb + (f_ub - f_lb) * torch.rand(D.shape[0] * B, lad.sys.L, generator=g)
        res = lad.run(D, apmax, u, f)
        assert res.stats["max_rel_violation"] < 1e-6
        tol = 1e-6 * np.maximum(1.0, np.abs(Q))
        assert np.all(res.U >= Q - tol)
        assert np.all(res.L <= Q + tol)
        n_r3 += int((res.rung == RUNG_R3).sum())
    assert n_r3 > 0, "random flows never reached R3, so this test did not exercise it"


def test_network_flows_end_feasible(setting):
    """The deployed path: flowfirst-harvest flows through the ladder end feasible with valid bounds."""
    lad, data = setting["ladders"][("lp", "unmet")], setting["data"]
    net = build_net_for_data(setting["run_args"], data, RUN, weights="model.pt")
    for D, apmax, u in setting["points"]:
        Q = lad.sys.exact(D, apmax, u)["Q"]
        res = lad.run(D, apmax, u, network_flows(net, data, block_inputs(D, apmax, u)))
        assert res.stats["max_rel_violation"] < 1e-6
        tol = 1e-6 * np.maximum(1.0, np.abs(Q))
        assert np.all(res.U >= Q - tol) and np.all(res.L <= Q + tol)


# ---------------------------------------------------------------- GEPOperationalProblemSet with ramping
@pytest.fixture(scope="module")
def op_sets(setting):
    """The operational companion of instance 0 (as solve_for_train builds it), hourly and as B-hour blocks."""
    sys.path.insert(0, os.path.join(os.getcwd(), "gen_GEP"))
    from solve_for_train import build_operational_companion
    gep = pickle.load(open(pickle.load(open(TRAJ, "rb"))[0]["instance_path"], "rb"))
    args = json.load(open("configs/config.json"))
    hourly = build_operational_companion(gep, args)
    ref = {k: getattr(hourly, k).clone() for k in ("ineq_cm", "ineq_rhs", "eq_cm", "obj_coeff", "cost_vec")}
    block = build_operational_companion(gep, args).set_ramping(B)
    return hourly, ref, block


def block_X(op, D, apmax, u):
    K = D.shape[0]
    return torch.cat([torch.tensor(D.reshape(K, -1)), torch.tensor((apmax * u[:, None, :]).reshape(K, -1)),
                      op.ramp_room(u)], dim=1)


def test_operational_set_ramping_off_is_unchanged(op_sets):
    """set_ramping(1) leaves every matrix and index of the hourly set exactly as it was."""
    hourly, ref, _ = op_sets
    idx = (list(hourly.capacity_ub_indices), list(hourly.missed_demand_ub_indices))
    hourly.set_ramping(1)
    assert all(torch.equal(getattr(hourly, k), v) for k, v in ref.items())
    assert (hourly.capacity_ub_indices, hourly.missed_demand_ub_indices) == idx


def test_operational_blocks_match_benders(setting, op_sets):
    """With set_ramping(B) the operational set holds Benders' block LP: same matrices as
    SampleStructure, and split_X gives the same right-hand sides, ramp rows included."""
    _, _, op = op_sets
    st = setting["struct"]
    assert np.array_equal(op.ineq_cm.numpy(), st.A_ineq_unit) and np.array_equal(op.eq_cm.numpy(), st.A_eq_unit)
    for D, apmax, u in setting["points"]:
        eq, ineq = op.split_X(block_X(op, D, apmax, u))
        sysm = setting["sys"]
        bI = np.concatenate([sysm.b_ineq(D, apmax * u[:, None, :]).reshape(D.shape[0], -1), sysm.ramp_rhs(sysm.rho(u))], 1)
        assert np.allclose(ineq.numpy(), bI) and np.allclose(eq.numpy(), D.reshape(D.shape[0], -1))


def test_operational_block_labels(setting, op_sets):
    """The set's own label LP (compute_opt_targets) gives Q*, a zero duality gap, and the ramp
    multipliers of the block LP; obj_fn of the optimal dispatch is Q* and it breaks no row."""
    _, _, op = op_sets
    D, apmax, u = (z[:12] for z in setting["points"][3])
    ex = setting["sys"].exact(D, apmax, u)
    op.X = block_X(op, D, apmax, u)
    t = op.compute_opt_targets()
    Q = torch.tensor(ex["Q"])
    tol = 1e-6 * Q.abs().clamp(min=1)
    assert torch.all((t["obj"] - Q).abs() < tol)
    assert torch.all((op.dual_obj_fn(op.X, t["mu_operational"], t["lamb_operational"]) - Q).abs() < tol)
    assert torch.all((op.obj_fn(op.X, t["y_operational"]) - Q).abs() < tol)
    assert op.ineq_resid(op.X, t["y_operational"]).max() < 1e-6
    up, dn = op.split_ramp_constraints(t["mu_operational"])
    assert np.allclose(up.numpy().reshape(ex["nu_up"].shape), ex["nu_up"], atol=1e-6)
    assert np.allclose(dn.numpy().reshape(ex["nu_dn"].shape), ex["nu_dn"], atol=1e-6)
    assert torch.equal(op.hourly_X(op.X), block_inputs(D, apmax, u))
