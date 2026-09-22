"""Train a single primal network on the economic-dispatch objective.

No dual network, no augmented Lagrangian: plain Adam on

    mean_i [ sum_g c_g p_g + VOLL * sum_n |md_n| ]   (+ L1 balance penalty for
                                                     the no-completion variant)

Three ways to parameterize the output y = [p, f, md]:

  old-prioritized   Peter's PrimalNetEndToEnd as configured today: predict p and
                    f, sigmoid-repair both into their boxes, move every generator
                    at a node by the same fraction toward its bound until the node
                    covers its residual demand (the generation-prioritized layer),
                    complete md from the balance.
  old-nocompletion  predict p, f and md, sigmoid-repair all three into their
                    boxes, nothing else. The nodal balance is only encouraged by
                    an L1 penalty, so its gradient reaches every variable but the
                    network has to learn the balance itself.
  flowfirst         predict f only, sigmoid-repair into the line limits, then
                    dispatch every node by merit order (flowfirst.fill).
  flowfirst-gnn     as flowfirst, but the flows come from a message-passing
                    network over the grid graph instead of one MLP on the flat
                    input (see FlowFirstGNN).

All variants share the same feed-forward body, optimizer and data split and are
scored by the same objective. ``--loss-norm`` reweights instances: ``opt``
divides each instance's objective by Gurobi's optimum (the training loss then
equals the mean relative gap), ``log`` takes the log of the objective (same
equalization near the optimum, no label needed), ``none`` is the raw mean, in
which shortage instances dominate because their objectives are far larger.
``--input-scale`` chooses how the body sees the inputs: ``layernorm`` is Peter's
default (scale-invariant: it cannot tell an instance from the same instance
with all demands and capacities doubled), ``fixed`` divides all inputs by one
constant, ``zscore`` standardizes each feature with training-set mean and std.
Only the body sees scaled inputs; bounds, residuals and the fill use raw MW.
``--train-voll`` sets the lost-load penalty used in the training loss only;
evaluation always uses the dataset's VOLL. Any value above the most expensive
generator leaves the optimal dispatch unchanged and only reconditions the loss. Per epoch the script logs to TensorBoard and to
metrics.csv: the training loss split into cost / VOLL / penalty parts, the
validation gap to Gurobi, constraint violations, flow and price errors, and a
census of the gradient that reaches the flows (see gradient_census). Epoch 0 is
logged before any training step.

Run from the repo root:
    .venv/bin/python -m flowfirst.train --variant flowfirst
    flowfirst/run_all.sh [extra args]        # all three variants concurrently
    tensorboard --logdir flowfirst/runs

Each run writes train.log, metrics.csv, args.json, model.pt and the TensorBoard
event file to flowfirst/runs/<variant>[-<tag>]/.
"""
import argparse
import copy
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from devices import KNOWN_DEVICES, resolve_device_name
from flowfirst.dataset import DEFAULT_CONFIG, load_or_create, roots_from_cli
from networks import (BoundRepairLayer, FlowFirst, FlowFirstGNN, MeritOrderFill, PrimalNetEndToEnd,
                      hidden_sizes, line_bounds, make_body, make_scaler, split_inputs)
from paths import add_path_args, under_root

torch.set_default_dtype(torch.float64)
RUNS_DIR = Path(__file__).resolve().parent / "runs"
VARIANTS = ("old-prioritized", "old-nocompletion", "flowfirst", "flowfirst-gnn")
EPS = 1e-7        # a flow gradient below this counts as zero
SAT_TOL = 1e-3    # a flow within this fraction of its range from a bound counts as saturated


# ----------------------------------------------------------------------------
# the baseline variants. forward(X) -> (y, f_raw, f_repaired)
# (FlowFirst and FlowFirstGNN live in networks.py)
# ----------------------------------------------------------------------------
class OldPrioritized(nn.Module):
    """PrimalNetEndToEnd with bounds repair, prioritized rescale and completion."""

    def __init__(self, args, data, scaler=None):
        super().__init__()
        a = copy.deepcopy(args)
        a.update(repair=True, repair_bounds=True, repair_power_balance=True,
                 repair_completion=True, use_blend_repair=False, flow_scale_by_prod=False)
        self.net = PrimalNetEndToEnd(a, data)
        if args.get("body", "plain") == "residual":
            self.net.feed_forward = make_body(args, data, self.net.out_dim, scaler)
        elif scaler is not None:
            # swap the input LayerNorm for the scaling module, leaving the rest of Peter's net intact
            ff = self.net.feed_forward
            assert isinstance(ff.net[0], nn.LayerNorm)
            ff.net[0] = nn.Identity()
            self.net.feed_forward = nn.Sequential(scaler, ff)
        self.num_l = data.num_l
        self._repair_calls = []
        # the repair layer is called for p first, then for f; capture both
        self.net.bound_repair_layer.register_forward_hook(
            lambda module, inputs, output: self._repair_calls.append((inputs[0], output)))

    def forward(self, X):
        self._repair_calls.clear()
        y = self.net(X)
        f_raw, f_rep = self._repair_calls[1]
        assert f_rep.shape[1] == self.num_l
        return y, f_raw, f_rep


class OldNoCompletion(nn.Module):
    """p, f and md all predicted and sigmoid-repaired; balance not enforced."""

    def __init__(self, args, data, scaler=None):
        super().__init__()
        self.data = data
        self.body = make_body(args, data, data.num_g + data.num_l + data.num_n, scaler)
        self.repair = BoundRepairLayer(repair_scaler="Sigmoid")
        f_lb, f_ub = line_bounds(data)
        self.register_buffer("f_lb", f_lb)
        self.register_buffer("f_ub", f_ub)

    def forward(self, X):
        D, cap = split_inputs(self.data, X)
        p_raw, f_raw, md_raw = torch.split(self.body(X), [self.data.num_g, self.data.num_l, self.data.num_n], dim=1)
        p = self.repair(p_raw, torch.zeros_like(cap), cap)
        f = self.repair(f_raw, self.f_lb, self.f_ub)
        md = self.repair(md_raw, torch.zeros_like(D), D)
        return torch.cat([p, f, md], dim=1), f_raw, f


def build_net(variant, args, data, scaler=None, X_train=None, gnn_kwargs=None):
    """scaler None keeps the input LayerNorm; otherwise the module (see make_scaler) replaces it.
    flowfirst-gnn needs X_train for its feature statistics and takes gnn_kwargs (hidden, rounds, id_embed)."""
    if variant == "flowfirst-gnn":
        return FlowFirstGNN(args, data, X_train, **(gnn_kwargs or {}))
    return {"old-prioritized": OldPrioritized,
            "old-nocompletion": OldNoCompletion,
            "flowfirst": FlowFirst}[variant](args, data, scaler)


# ----------------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------------
def loss_terms(data, X, y, penalty_weight, voll_price=None):
    """Per-instance cost, lost-load and balance-penalty parts.

    With voll_price None (the dataset's VOLL) cost + voll == data.obj_fn.
    """
    p, f, md = data.split_dec_vars_from_Y(y)
    cost = p.abs() @ data.cost_vec
    voll = (data.pVOLL if voll_price is None else voll_price) * md.abs().sum(dim=1)
    if penalty_weight > 0:
        pen = penalty_weight * data.eq_resid(X, y).abs().sum(dim=1)
    else:
        pen = torch.zeros_like(cost)
    return cost, voll, pen


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
class Reference:
    """Gurobi labels and regime masks for one split."""

    def __init__(self, data, idx):
        self.X = data.X[idx]
        self.obj = data.opt_targets["obj"][idx]
        self.y = data.opt_targets["y_operational"][idx]
        self.p, self.f, self.md = data.split_dec_vars_from_Y(self.y)
        self.lam = -data.opt_targets["lamb_operational"][idx]   # stored sign-flipped; this is the price
        self.shortage = self.md.sum(dim=1) > 1e-6


@torch.no_grad()
def evaluate(net, data, fill, ref, f_lb, f_ub):
    y, _, f = net(ref.X)
    D, cap = split_inputs(data, ref.X)
    obj = data.obj_fn(ref.X, y)
    gap = (obj - ref.obj) / ref.obj
    p, _, md = data.split_dec_vars_from_Y(y)
    # split the excess cost into the dispatch part and the VOLL part; they sum to the gap
    gap_dispatch = ((p.abs() @ data.cost_vec) - (ref.p @ data.cost_vec)) / ref.obj
    gap_voll = data.pVOLL * (md.abs().sum(dim=1) - ref.md.sum(dim=1)) / ref.obj
    # residual error is the right flow metric on a meshed network: a circulating flow changes nothing
    resid_err = (fill.residual(D, f) - fill.residual(D, ref.f)).abs().sum(dim=1) / D.sum(dim=1)
    box = data.ineq_dist(ref.X, y)
    bal = data.eq_resid(ref.X, y).abs().sum(dim=1) / D.sum(dim=1)
    rng = f_ub - f_lb
    sat = lambda flows: ((flows - f_lb) / rng < SAT_TOL) | ((f_ub - flows) / rng < SAT_TOL)
    lo, hi = fill.price_interval(fill.residual(D, f), cap)
    price_ok = (ref.lam >= torch.minimum(lo, hi) - 1e-9) & (ref.lam <= torch.maximum(lo, hi) + 1e-9)
    s = ref.shortage
    total_gap = lambda m: ((obj[m].sum() - ref.obj[m].sum()) / ref.obj[m].sum()).item()
    return {
        "val/gap_total": total_gap(torch.ones_like(s)),            # ratio of summed objectives: what the raw loss encodes
        "val/gap_total_shortage": total_gap(s),
        "val/gap_total_noshortage": total_gap(~s),
        "val/gap_mean": gap.mean().item(),
        "val/gap_median": gap.median().item(),
        "val/within_1pct": (gap < 0.01).float().mean().item(),
        "val/gap_shortage": gap[s].mean().item(),
        "val/gap_noshortage": gap[~s].mean().item(),
        "val/gap_dispatch_part": gap_dispatch.mean().item(),
        "val/gap_voll_part": gap_voll.mean().item(),
        "val/gap_dispatch_part_noshortage": gap_dispatch[~s].mean().item(),
        "val/residual_err_rel": resid_err.mean().item(),
        "val/box_violation_max": box.max(dim=1).values.mean().item(),
        "val/box_violation_mean": box.mean().item(),
        "val/balance_violation_rel": bal.mean().item(),
        "val/flow_err_rel": ((f - ref.f).abs() / rng).mean().item(),
        "val/line_sat_acc": (sat(f) == sat(ref.f)).float().mean().item(),
        "val/price_acc": price_ok.float().mean().item(),
        "val/price_acc_noshortage": price_ok[~s].float().mean().item(),
    }


def gradient_census(net, data, fill, ref, f_lb, f_ub, penalty_weight, voll_price=None):
    """What gradient do the flows actually receive, versus the true price difference?

    The true subgradient of the dispatch cost w.r.t. a flow is the price
    difference between its ends, computed by the merit-order fill from the
    network's own flows. That is what an ideal gradient would be for any
    variant; the census compares the autograd gradient against it.
    """
    y, f_raw, f_rep = net(ref.X)
    D, cap = split_inputs(data, ref.X)
    cost, voll, pen = loss_terms(data, ref.X, y, penalty_weight, voll_price)
    g_rep, g_raw = torch.autograd.grad((cost + voll + pen).sum(), [f_rep, f_raw], retain_graph=True)
    g_cost = torch.autograd.grad(cost.sum(), f_rep, retain_graph=True, allow_unused=True)[0]
    g_voll = torch.autograd.grad(voll.sum(), f_rep, retain_graph=True, allow_unused=True)[0]
    g_pen = torch.autograd.grad(pen.sum(), f_rep, allow_unused=True)[0] if penalty_weight > 0 else None
    zeros = torch.zeros_like(g_rep)
    g_cost = zeros if g_cost is None else g_cost
    g_voll = zeros if g_voll is None else g_voll
    g_pen = zeros if g_pen is None else g_pen

    with torch.no_grad():
        lo, hi = fill.price_interval(fill.residual(D, f_rep), cap)
        g_true = fill.flow_gradient(lo)
        from_node = (fill.lineflow == -1).to(lo.dtype)   # [N, L]
        to_node = (fill.lineflow == 1).to(lo.dtype)
        lam_from, lam_to = lo @ from_node, lo @ to_node   # [B, L]
        true_zero = g_true.abs() < EPS
        both_voll = (lam_from == fill.voll) & (lam_to == fill.voll)
        both_over = (lam_from == -fill.voll) & (lam_to == -fill.voll)
        act_zero = g_rep.abs() < EPS
        raw_zero = g_raw.abs() < EPS
        rng = f_ub - f_lb
        sat = ((f_rep - f_lb) / rng < SAT_TOL) | ((f_ub - f_rep) / rng < SAT_TOL)
        both_nonzero = ~true_zero & ~act_zero
        mismatch = both_nonzero & (g_rep * g_true < 0)
        kink = (lo != hi).any(dim=1)
        ns = ~ref.shortage
        frac = lambda m: m.float().mean().item()
        return {
            "grad/true_zero": frac(true_zero),
            "grad/true_zero_both_voll": frac(true_zero & both_voll),
            "grad/true_zero_both_oversupply": frac(true_zero & both_over),
            "grad/true_zero_cost_tie": frac(true_zero & ~both_voll & ~both_over),   # equal finite prices at both ends
            "grad/actual_zero_repaired": frac(act_zero),
            "grad/actual_zero_raw": frac(raw_zero),
            "grad/missing": frac(~true_zero & act_zero),
            "grad/spurious": frac(true_zero & ~act_zero),
            "grad/sign_mismatch": frac(mismatch),
            "grad/sign_mismatch_of_nonzero": (mismatch.float().sum() / both_nonzero.float().sum().clamp_min(1)).item(),
            "grad/flow_saturated": frac(sat),
            "grad/kink_instances": frac(kink),
            "grad/true_zero_noshortage": frac(true_zero[ns]),
            "grad/actual_zero_repaired_noshortage": frac(act_zero[ns]),
            "grad/missing_noshortage": frac((~true_zero & act_zero)[ns]),
            "grad/sign_mismatch_noshortage": frac(mismatch[ns]),
            "grad/abs_mean_actual": g_rep.abs().mean().item(),
            "grad/abs_mean_true": g_true.abs().mean().item(),
            "grad/abs_mean_cost_part": g_cost.abs().mean().item(),
            "grad/abs_mean_voll_part": g_voll.abs().mean().item(),
            "grad/abs_mean_penalty_part": g_pen.abs().mean().item(),
        }


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def resolve_runs_dir(runs_dir, output_root, args):
    """--runs-dir wins; then the output root, grouped by problem size as main.py groups its runs; then the package."""
    if runs_dir is not None:
        return runs_dir
    if output_root is None:
        return RUNS_DIR
    ed = args["ED_args"]
    return under_root(os.path.join("outputs", "FlowFirst", "ED", f"N{len(ed['N'])}_G{len(ed['G'])}"), output_root)


def make_run_dir(runs_dir, variant, tag):
    """runs/<variant>[-tag], with -2, -3, ... appended if taken. Safe for concurrent starts."""
    name = variant if not tag else f"{variant}-{tag}"
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir, k = runs_dir / name, 2
    while True:
        try:
            run_dir.mkdir()
            return run_dir
        except FileExistsError:
            run_dir, k = runs_dir / f"{name}-{k}", k + 1


class Tee:
    """Write everything printed to stdout into a log file as well."""

    def __init__(self, path):
        self.file, self.stdout = open(path, "a"), sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    #! --home-path / --data-root / --output-root, the same flags main.py takes: on a cluster they move the
    #! dataset and the run directories under $SCRATCH. Without them nothing leaves the package (roots_from_cli).
    add_path_args(ap)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--loss-norm", choices=("none", "opt", "log"), default="none",
                    help="per-instance loss weighting: none (raw objective), opt (divide by Gurobi optimum), log")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-schedule", choices=("none", "plateau", "step", "cosine"), default="none",
                    help="plateau: multiply the lr by --lr-decay when the validation gap has not improved for "
                         "--lr-patience epochs (the thesis trainer's ReduceLROnPlateau); step: multiply by --lr-decay "
                         "every --lr-step epochs; cosine: anneal to --lr-min-frac of --lr")
    ap.add_argument("--lr-decay", type=float, default=0.99, help="factor for plateau and step (thesis config: 0.99)")
    ap.add_argument("--lr-step", type=int, default=50, help="step schedule: epochs between decays")
    ap.add_argument("--lr-min-frac", type=float, default=0.01, help="cosine schedule: final lr as a fraction of --lr")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="plateau patience in epochs (thesis config: 10); rounded to whole --eval-every intervals")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--flow-reg", type=float, default=0.0,
                    help="quadratic flow regularizer: add eps * sum_l (f_l / F_l)^2 to the training objective (F_l = larger "
                         "line limit). Selects the minimum-norm flow on degenerate faces (F25). Keep eps below "
                         "dc_min * F_min / (2 * diameter); 0.3 on the 6-node set. Training only")
    ap.add_argument("--flow-reg-final", type=float, default=None,
                    help="if set, the flow regularizer decays geometrically from --flow-reg at epoch 1 to this value at "
                         "the last epoch (strong tie-breaking early, bound-respecting late)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="decay of an exponential moving average of the weights (e.g. 0.999); the averaged model is evaluated "
                         "alongside the raw one under ema/ and saved as model_ema.pt / model_ema_best.pt. 0 = off")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="clip the global gradient norm to this value before each step (0 = off); guards against "
                         "batches where a few instances crossing the VOLL cliff produce an outsized gradient")
    ap.add_argument("--hidden-factor", type=int, default=28, help="hidden width = factor * input dim")
    ap.add_argument("--layers", type=int, default=2, help="hidden layers (plain body) or 2 x residual blocks (residual body)")
    ap.add_argument("--small-output-init", action="store_true",
                    help="plain body: initialize the output layer near zero so flows start mid-range (the residual body always does)")
    ap.add_argument("--body", choices=("plain", "residual"), default="plain",
                    help="plain: Peter's ReLU stack; residual: input projection + skip-connected blocks (needs --input-scale zscore/fixed)")
    ap.add_argument("--penalty-weight", type=float, default=None,
                    help="L1 balance penalty for old-nocompletion; default 2 * VOLL")
    ap.add_argument("--input-scale", choices=("layernorm", "fixed", "zscore"), default="layernorm",
                    help="layernorm: the body's input LayerNorm; fixed: divide inputs by the training-set maximum; "
                         "zscore: per-feature standardization with training-set mean and std")
    ap.add_argument("--train-voll", type=float, default=None,
                    help="lost-load price in the training loss only (default: the dataset's VOLL)")
    ap.add_argument("--train-subset", choices=("all", "noshortage", "shortage"), default="all",
                    help="restrict training instances by whether Gurobi's optimum sheds load; validation stays complete")
    ap.add_argument("--train-size", type=int, default=None,
                    help="use only the first N training instances (after --train-subset); for overfit tests")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate on validation every K epochs (always at the last); the training loss is logged every epoch")
    ap.add_argument("--gnn-hidden", type=int, default=128, help="flowfirst-gnn: hidden width of node and edge states")
    ap.add_argument("--gnn-rounds", type=int, default=2, help="flowfirst-gnn: message-passing rounds (3 is ~1.5x slower)")
    ap.add_argument("--gnn-no-id-embed", action="store_true", help="flowfirst-gnn: no learnable node/edge identity embeddings")
    ap.add_argument("--gnn-antisym", action="store_true",
                    help="flowfirst-gnn: antisymmetric readout z = g(e, h_from, h_to) - g(e, h_to, h_from) (potential-difference bias)")
    ap.add_argument("--gnn-layernorm", action="store_true",
                    help="flowfirst-gnn: pre-LayerNorm inside every message-passing block (recommended from ~6 rounds)")
    ap.add_argument("--device", default="cpu", choices=KNOWN_DEVICES,
                    help="cpu (float64), mps (Apple GPU, float32 only), cuda (float64 by default, see --dtype), or "
                         "auto (cuda when present, else cpu). Only the flowfirst-gnn variant runs on cuda; "
                         "Peter's networks know cpu and mps")
    ap.add_argument("--dtype", choices=("float64", "float32"), default=None,
                    help="tensor precision; default float64 on cpu/cuda, float32 on mps (which has no float64)")
    ap.add_argument("--valid-size", type=int, default=None,
                    help="evaluate on only the first N validation instances (and N training instances); for large datasets")
    ap.add_argument("--census-size", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--runs-dir", default=None,
                    help="where run directories go; default flowfirst/runs, or "
                         "<output-root>/outputs/FlowFirst/ED/N<nodes>_G<generators> when a root is given")
    ap.add_argument("--log-every", type=int, default=10)
    cli = ap.parse_args()
    #! Resolve "auto" once, here, so every later check reads a concrete device name, as main.py does.
    cli.device = resolve_device_name(cli.device)
    data_root, output_root = roots_from_cli(cli)

    data, args = load_or_create(cli.config, data_root=data_root)
    if cli.device == "mps":
        assert torch.backends.mps.is_available(), "MPS is not available on this machine"
        assert cli.dtype in (None, "float32"), "MPS has no float64"
    dtype = torch.float32 if (cli.device == "mps" or cli.dtype == "float32") else torch.float64
    if cli.device != "cpu" or dtype != torch.float64:
        # move every tensor the training and evaluation touch; the networks are built on the data's device below.
        # No torch.set_default_device: it installs a Python hook on every torch call, hundreds per training step
        torch.set_default_dtype(dtype)
        to = lambda t: t.to(dtype=dtype, device=cli.device) if t.is_floating_point() else t.to(device=cli.device)
        data.X = to(data.X)
        data.opt_targets = {k: to(v) for k, v in data.opt_targets.items()}
        for name in ("node_to_gen_mask", "lineflow_mask", "cost_vec", "obj_coeff", "ineq_rhs", "ineq_cm", "eq_cm"):
            setattr(data, name, to(getattr(data, name)))
        args["device"] = cli.device   # Peter's networks read this ("mps" -> float32 on the Apple GPU)
    args["hidden_size_factor"], args["n_layers"], args["body"] = cli.hidden_factor, cli.layers, cli.body
    args["small_output_init"] = cli.small_output_init
    penalty_weight = 0.0
    if cli.variant == "old-nocompletion":
        penalty_weight = 2.0 * data.pVOLL if cli.penalty_weight is None else cli.penalty_weight

    torch.manual_seed(cli.seed)
    n = data.X.shape[0]
    n_train, n_valid = int(0.8 * n), int(0.1 * n)
    train_idx = torch.arange(0, n_train)
    valid_idx = torch.arange(n_train, n_train + n_valid)
    if cli.train_subset != "all":
        sheds = data.split_dec_vars_from_Y(data.opt_targets["y_operational"])[2].sum(dim=1) > 1e-6
        keep = sheds if cli.train_subset == "shortage" else ~sheds
        train_idx = train_idx[keep[train_idx]]
    if cli.train_size is not None:
        train_idx = train_idx[:cli.train_size]
    n_train = len(train_idx)
    if cli.valid_size is not None:
        valid_idx = valid_idx[:cli.valid_size]
        n_valid = len(valid_idx)
    ref_valid = Reference(data, valid_idx)
    ref_census = Reference(data, valid_idx[:cli.census_size])
    ref_train_eval = Reference(data, train_idx[:n_valid])       # fit on the training instances themselves
    fill = MeritOrderFill(data)                       # evaluation: true VOLL
    train_voll = data.pVOLL if cli.train_voll is None else cli.train_voll
    fill_train = MeritOrderFill(data, voll=train_voll)  # census: what training sees
    f_lb, f_ub = line_bounds(data)
    scaler = make_scaler(cli.input_scale, data.X[train_idx])

    gnn_kwargs = dict(hidden=cli.gnn_hidden, rounds=cli.gnn_rounds, id_embed=0 if cli.gnn_no_id_embed else 8,
                      layernorm=cli.gnn_layernorm, antisym=cli.gnn_antisym)
    net = build_net(cli.variant, args, data, scaler, X_train=data.X[train_idx], gnn_kwargs=gnn_kwargs).to(cli.device)
    opt = torch.optim.Adam(net.parameters(), lr=cli.lr)
    ema_net = None
    if cli.ema > 0:
        # a second, identically built network holds the averaged weights (a fresh build rather than a deepcopy,
        # because OldPrioritized registers a forward hook bound to its own instance)
        ema_net = build_net(cli.variant, args, data, scaler, X_train=data.X[train_idx], gnn_kwargs=gnn_kwargs).to(cli.device)
        ema_net.load_state_dict(net.state_dict())
        ema_net.eval()
    ema_steps = 0
    F_scale = torch.maximum(f_ub, -f_lb)
    if cli.lr_schedule == "plateau":
        # the scheduler only sees the validation gap at evaluation epochs, so convert the patience to evaluations
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cli.lr_decay,
                                                               patience=max(1, cli.lr_patience // cli.eval_every))
    elif cli.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cli.lr_step, gamma=cli.lr_decay)
    elif cli.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cli.epochs, eta_min=cli.lr_min_frac * cli.lr)
    else:
        scheduler = None
    # one index_select per batch; a DataLoader over a TensorDataset makes two Python-level __getitem__ calls per instance
    X_train, obj_train = data.X[train_idx], data.opt_targets["obj"][train_idx]
    shuffle_gen = torch.Generator(device=X_train.device).manual_seed(cli.seed)
    LOSS_KEYS = ("loss/train_total", "loss/train_objective", "loss/train_cost", "loss/train_voll",
                 "loss/train_penalty", "loss/train_flowreg")

    run_dir = make_run_dir(Path(resolve_runs_dir(cli.runs_dir, output_root, args)), cli.variant, cli.tag)
    sys.stdout = Tee(run_dir / "train.log")
    print("command:", " ".join(sys.argv))
    writer = SummaryWriter(log_dir=str(run_dir))
    with open(run_dir / "args.json", "w") as fh:
        #! The resolved roots overwrite the raw CLI strings: --home-path leaves cli.data_root None, and
        #! load_run reads this key to find the dataset again from an analysis script.
        json.dump({**vars(cli), "data_root": data_root, "output_root": output_root,
                   "penalty_weight_used": penalty_weight, "train_voll_used": train_voll,
                   "n_train": n_train, "n_valid": n_valid,
                   "hidden_sizes": hidden_sizes(args, data), "n_params": sum(p.numel() for p in net.parameters())},
                  fh, indent=2)
    print(f"run dir: {run_dir}   params: {sum(p.numel() for p in net.parameters())}   "
          f"train/valid: {n_train}/{n_valid}   penalty weight: {penalty_weight}   "
          f"train VOLL: {train_voll}   input scale: {cli.input_scale}")

    csv_rows = []
    best = {"epoch": -1, "val/gap_mean": float("inf")}
    best_ema = {"epoch": -1, "ema/gap_mean": float("inf")}

    def log(epoch, train_stats, do_eval=True):
        """Training statistics every epoch; validation, training-set fit and census when do_eval."""
        net.eval()
        row = {"epoch": epoch, **train_stats}
        if do_eval:
            row.update(evaluate(net, data, fill, ref_valid, f_lb, f_ub))
            row.update({k.replace("val/", "trainset/"): v for k, v in evaluate(net, data, fill, ref_train_eval, f_lb, f_ub).items()})
            row.update(gradient_census(net, data, fill_train, ref_census, f_lb, f_ub, penalty_weight, train_voll))
            if ema_net is not None:
                row.update({k.replace("val/", "ema/"): v for k, v in evaluate(ema_net, data, fill, ref_valid, f_lb, f_ub).items()})
                row.update({k.replace("val/", "ema_trainset/"): v for k, v in evaluate(ema_net, data, fill, ref_train_eval, f_lb, f_ub).items()})
        for k, v in row.items():
            if k != "epoch":
                writer.add_scalar(k, v, epoch)
        csv_rows.append(row)
        if do_eval and row["val/gap_mean"] < best["val/gap_mean"]:
            best.update(epoch=epoch, **{"val/gap_mean": row["val/gap_mean"]})
            torch.save(net.state_dict(), run_dir / "model_best.pt")
        if do_eval and ema_net is not None and row["ema/gap_mean"] < best_ema["ema/gap_mean"]:
            best_ema.update(epoch=epoch, **{"ema/gap_mean": row["ema/gap_mean"]})
            torch.save(ema_net.state_dict(), run_dir / "model_ema_best.pt")
        if epoch % cli.log_every == 0 or epoch == cli.epochs:
            train = f"{row['loss/train_total']:12.2f}" if 'loss/train_total' in row else " " * 12
            if not do_eval:
                print(f"epoch {epoch:4d} | train {train} | (no evaluation this epoch)")
                return
            ema_txt = f"ema total {row['ema/gap_total']:.4f} mean {row['ema/gap_mean']:.4f} | " if ema_net is not None else ""
            print(f"epoch {epoch:4d} | train {train} | trainset gap total {row['trainset/gap_total']:.4f} mean {row['trainset/gap_mean']:.4f} | {ema_txt}"
                  f"val gap total {row['val/gap_total']:.4f} mean {row['val/gap_mean']:8.4f} (median {row['val/gap_median']:8.4f}, <1% {row['val/within_1pct']:.3f}, "
                  f"dispatch {row['val/gap_dispatch_part']:.4f} voll {row['val/gap_voll_part']:.4f}) | "
                  f"box {row['val/box_violation_max']:9.2f} bal {row['val/balance_violation_rel']:.4f} | "
                  f"true0 {row['grad/true_zero']:.3f} act0 {row['grad/actual_zero_repaired']:.3f} "
                  f"miss {row['grad/missing']:.3f} sign {row['grad/sign_mismatch']:.3f} sat {row['grad/flow_saturated']:.3f}")

    log(0, {})
    t0 = time.time()
    for epoch in range(1, cli.epochs + 1):
        net.train()
        if cli.flow_reg > 0 and cli.flow_reg_final is not None and cli.epochs > 1:
            flow_reg = cli.flow_reg * (cli.flow_reg_final / cli.flow_reg) ** ((epoch - 1) / (cli.epochs - 1))
        else:
            flow_reg = cli.flow_reg
        sums, count = torch.zeros(len(LOSS_KEYS), device=X_train.device), 0
        perm = torch.randperm(n_train, generator=shuffle_gen, device=X_train.device)
        for i in range(0, n_train, cli.batch_size):
            j = perm[i:i + cli.batch_size]
            xb, ob = X_train[j], obj_train[j]
            opt.zero_grad()
            y, _, f_rep = net(xb)
            cost, voll, pen = loss_terms(data, xb, y, penalty_weight, train_voll)
            reg = flow_reg * ((f_rep / F_scale) ** 2).sum(dim=1) if flow_reg > 0 else torch.zeros_like(cost)
            objective = cost + voll + pen + reg
            if cli.loss_norm == "opt":
                per_instance = objective / ob
            elif cli.loss_norm == "log":
                per_instance = torch.log(objective + 1e-9)
            else:
                per_instance = objective
            loss = per_instance.mean()
            loss.backward()
            if cli.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cli.clip_grad)
            opt.step()
            if ema_net is not None:
                ema_steps += 1
                d = min(cli.ema, (1.0 + ema_steps) / (10.0 + ema_steps))   # short ramp so the average is usable early
                with torch.no_grad():
                    for p_ema, p_raw in zip(ema_net.parameters(), net.parameters(), strict=True):
                        p_ema.mul_(d).add_(p_raw, alpha=1.0 - d)
            b = xb.shape[0]
            # accumulate on the device: every .item() here would stall the CPU until the GPU has caught up
            with torch.no_grad():
                sums += torch.stack([loss, objective.mean(), cost.mean(), voll.mean(), pen.mean(), reg.mean()]) * b
            count += b
        do_eval = epoch % cli.eval_every == 0 or epoch == cli.epochs
        train_stats = dict(zip(LOSS_KEYS, (sums / count).tolist(), strict=True))
        log(epoch, {**train_stats, "train/lr": opt.param_groups[0]["lr"], "train/flow_reg": flow_reg}, do_eval)
        if cli.lr_schedule == "plateau" and do_eval:
            scheduler.step(csv_rows[-1]["val/gap_mean"])
        elif cli.lr_schedule in ("step", "cosine"):
            scheduler.step()

    print(f"training time: {time.time() - t0:.1f}s")
    torch.save(net.state_dict(), run_dir / "model.pt")
    if ema_net is not None:
        torch.save(ema_net.state_dict(), run_dir / "model_ema.pt")
    final = next(r for r in reversed(csv_rows) if "val/gap_mean" in r)
    summary = {"final_epoch": final["epoch"], "final_gap_mean": final["val/gap_mean"],
               "best_epoch": best["epoch"], "best_gap_mean": best["val/gap_mean"]}
    if ema_net is not None:
        summary.update({"final_ema_gap_mean": final["ema/gap_mean"], "final_ema_gap_total": final["ema/gap_total"],
                        "best_ema_epoch": best_ema["epoch"], "best_ema_gap_mean": best_ema["ema/gap_mean"]})
    with open(run_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"final val gap {final['val/gap_mean']:.4f} at epoch {final['epoch']}; "
          f"best val gap {best['val/gap_mean']:.4f} at epoch {best['epoch']} (model_best.pt)")
    keys = list(dict.fromkeys(k for r in csv_rows for k in r))
    with open(run_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r.get(k, "") for k in keys})
    writer.close()
    print(f"saved model.pt, model_best.pt, metrics.csv, summary.json in {run_dir}")


if __name__ == "__main__":
    main()
