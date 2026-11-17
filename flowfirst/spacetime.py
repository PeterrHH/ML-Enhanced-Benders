"""Ways 2-i, 2-ii and 2-iii of docs/ramping-surrogate-design.tex: the space-time network on ramping blocks.

The network is networks.FlowFirstSpaceTime; this module holds what the block trainer
(flowfirst/train_block.py) and the block evaluation (flowfirst/eval_block_ladder.py) share:

  * `ramp_clip`: Way 2-ii's output layer. A raw 4-hour path per ramp-limited technology is pushed
    hour by hour into [0, cap] and the ramp limits, so the result never breaks a ramp row.
  * `block_dispatch`: the dispatch the loss scores. Way 2-i: the merit-order fill on the network's
    flows. Way 2-ii: the ramp-limited technologies follow their clipped paths, and the fill
    dispatches the rest of each node around them.
  * `block_loss`: the subproblem objective (generation cost + VOLL x |unserved|) of that dispatch
    over the block's hours, and the ramp violation per technology of the raw path (2-ii, before
    the clip) or of the merit-order dispatch (2-i). The trainer prices the violation: 0 for the
    objective-only loss, nu by dual ascent for the Lagrangian loss.
  * `dp_loss`: Way 2-iii's loss, the cost of the best ramp-feasible dispatch given the network's flows
    (flowfirst/rampdp.py), with the node prices as its gradient in the flows.
  * `build_spacetime` / `load_spacetime`: build the network from a run's args.json, and rebuild a
    trained run against a block set.
  * `spacetime_dispatch`: hourly flows [K*B, L] of a trained network on block rows, polished as the
    deployed pipeline does, and for Way 2-ii the clipped paths, which the ladder takes as fixed.

Block rows are the block set's X (gen_GEP/solve_for_train.py --ramp-block-hours B):
[demand per hour, A*UnitCap*u per hour, rho = R*u per technology].
"""
import json
from pathlib import Path

import torch

from networks import FlowFirstSpaceTime


def ramp_frac(block_set):
    """Ramp limit as a share of unit size per technology [G]; below 1 means ramp-limited."""
    return torch.tensor([float(block_set.pRamping[g]) for g in block_set.G])


def ramp_clip(target, cap, rho):
    """Clip raw paths into the ramp limits, hour by hour. target, cap [K, B, S]; rho [K, S].

    ub_t is the highest output in hour t from which the unit can still get under every later
    capacity: ub_B = cap_B, ub_t = min(cap_t, ub_{t+1} + rho). Then
      p_1 = clip(target_1, 0, ub_1),   p_t = clip(target_t, max(0, p_{t-1} - rho), min(ub_t, p_{t-1} + rho)).
    Each interval is non-empty since p_{t-1} <= ub_{t-1} <= ub_t + rho. Where the target already
    fits, p = target; where an hour sits on its ramp bound, the gradient reaches the previous hour.
    """
    B = target.shape[1]
    ub = [cap[:, B - 1]]
    for t in range(B - 2, -1, -1):
        ub.insert(0, torch.minimum(cap[:, t], ub[0] + rho))
    p = [torch.minimum(torch.relu(target[:, 0]), ub[0])]
    for t in range(1, B):
        lo, hi = torch.relu(p[-1] - rho), torch.minimum(ub[t], p[-1] + rho)
        p.append(torch.minimum(torch.maximum(target[:, t], lo), hi))
    return torch.stack(p, dim=1)


def ramp_violation(p, rho):
    """MW by which consecutive hours differ by more than rho, per block and technology [K, S]."""
    return torch.relu((p[:, 1:] - p[:, :-1]).abs() - rho[:, None, :]).sum(dim=1)


def fix_paths(net, D, cap, p_s):
    """Generation fixed at the paths p_s [K, B, S]: p_fix [K, B, G] (zero for the other units), the
    capacity left to the fill (ramp-limited units at 0), and the demand left once the paths run."""
    p_fix = torch.zeros_like(cap).index_copy(2, net.slow, p_s)
    free = torch.ones(cap.shape[2], dtype=cap.dtype, device=cap.device).index_fill(0, net.slow, 0.0)
    return p_fix, cap * free, D - p_fix @ net.data.node_to_gen_mask.T.to(cap.dtype)


def block_dispatch(fill, net, X):
    """The dispatch the loss scores, differentiable in the network's parameters.

    Returns p [K, B, G], md [K, B, N], f [K, B, L], and the raw output [K, B, S] of the ramp-limited
    technologies: the merit-order output for Way 2-i, the path before ramp_clip for Way 2-ii.
    In Way 2-ii the clipped path keeps its gradient through the merit-order part, so a flow sees the
    ramp-limited units it displaces (nuclear replacing gas) as the hourly network does. The raw path
    returned for the ramp price has that part detached: the price then trains the corrections only,
    and does not push the flows toward ramp-feasible merit order.
    """
    D, cap, rho = net.split(X)
    f, _, corr = net(X)
    K, B, N = D.shape
    p, e, _ = fill(D.reshape(K * B, -1), cap.reshape(K * B, -1), f.reshape(K * B, -1))
    p, e = p.reshape(K, B, -1), e.reshape(K, B, -1)
    if corr is None:
        return p, e, f, p[..., net.slow]
    mo = p[..., net.slow]
    p_s = ramp_clip(mo + corr, cap[..., net.slow], rho[:, net.slow])
    raw = mo.detach() + corr
    p_fix, cap_free, D_free = fix_paths(net, D, cap, p_s)
    p, e, _ = fill(D_free.reshape(K * B, -1), cap_free.reshape(K * B, -1), f.reshape(K * B, -1))
    return p.reshape(K, B, -1) + p_fix, e.reshape(K, B, N), f, raw


def block_loss(fill, net, X):
    """Per-block generation cost and VOLL part of block_dispatch, and the ramp violation per
    technology [K, S] of its raw output (see block_dispatch). The trainer adds the price of the
    violation; the objective-only loss is cost + voll."""
    _, _, rho = net.split(X)
    p, e, _, raw = block_dispatch(fill, net, X)
    cost = (p.abs() @ fill.cost).sum(dim=1)
    voll = fill.voll * e.abs().sum(dim=(1, 2))
    return cost, voll, ramp_violation(raw, rho[:, net.slow])


def dp_loss(dp, net, X):
    """Way 2-iii: per-block cost [K] of the best ramp-feasible dispatch given the network's flows (dp a
    flowfirst.rampdp.RampDP, elastic mode: over-supply and unserved demand at VOLL, as in the hourly loss).
    Where no ramp limit binds it is the hourly merit-order cost; its gradient in a flow is
    lambda_from - lambda_to with the ramp-aware node prices of the program."""
    D, cap, rho = net.split(X)
    f = net(X)[0]
    return dp(D - f @ net.data.lineflow_mask.T.to(f.dtype), cap, rho)["cost"]


def build_spacetime(a, data, block_set, X_train):
    """The network a run's args `a` describe. X_train (block rows) only sets the feature statistics,
    which are buffers: a checkpoint overwrites them."""
    return FlowFirstSpaceTime(data, ramp_frac(block_set), block_set.ramp_block_hours_, X_train,
                              hidden=a["gnn_hidden"], rounds=a["gnn_rounds"],
                              id_embed=0 if a.get("gnn_no_id_embed") else 8,
                              layernorm=a.get("gnn_layernorm", False), antisym=a.get("gnn_antisym", False),
                              paths=a.get("paths", False))


def load_spacetime(run_dir, data, block_set, weights="model.pt", device="cpu"):
    """Rebuild a trained space-time run against `block_set`."""
    a = json.load(open(Path(run_dir) / "args.json"))
    net = build_spacetime(a, data, block_set, block_set.X[:2]).to(device)
    net.load_state_dict(torch.load(Path(run_dir) / weights, map_location="cpu", weights_only=True))
    return net.eval()


def spacetime_dispatch(net, data, X, polish_sweeps=1):
    """What a space-time network hands the ladder for block rows X: hourly flows [K*B, L] after
    `polish_sweeps` line-polish sweeps per hour (flowfirst.dual.Polish), and for Way 2-ii the clipped
    paths as fixed generation p_fix [K, B, G] (None for Way 2-i). With paths, the polish sees only
    the units the fill dispatches: the demand left after the paths, and the other units' capacity."""
    from flowfirst.dual import Polish
    from flowfirst.fill import MeritOrderFill
    with torch.no_grad():
        D, cap, rho = net.split(X)
        f, _, corr = net(X)
        p_fix = None
        if corr is not None:
            K, B = D.shape[:2]
            p, _, _ = MeritOrderFill(data)(D.reshape(K * B, -1), cap.reshape(K * B, -1), f.reshape(K * B, -1))
            p_s = ramp_clip(p.reshape(K, B, -1)[..., net.slow] + corr, cap[..., net.slow], rho[:, net.slow])
            p_fix, cap, D = fix_paths(net, D, cap, p_s)
        f = f.reshape(-1, data.num_l)
        if polish_sweeps:
            f, _ = Polish(data)(net.hours(D, cap), f, polish_sweeps)
    return f, p_fix
