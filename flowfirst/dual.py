"""Dual solution, certificate and inference polish from the flows of a primal network.

Section 2 of flow_first_summary.md, measured in FINDINGS F26 to F31. Prices are
positive (the problem class stores them negated: pass `-lam` to its
`dual_obj_fn`); multipliers follow `split_ineq_constraints` row order.

Everything except `PathPolish` has fixed shapes, no host-device syncs and no
data-dependent control flow, so `PrimalDual(exact=False)` can be replayed as
one CUDA graph (`GraphedPrimalDual`, F30).
"""
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from flowfirst.dataset import load_or_create, resolve_config
from flowfirst.fill import MeritOrderFill
from flowfirst.train import build_net, line_bounds, make_scaler, split_inputs

INF = float("inf")
DATA_TENSORS = ("node_to_gen_mask", "lineflow_mask", "cost_vec", "obj_coeff", "ineq_rhs", "ineq_cm", "eq_cm")


class DualRecovery:
    """Equilibrium prices, box multipliers and the dual objective from predicted flows.

    Every price vector in [0, VOLL] completes to a feasible dual, so
    `dual_objective` is a valid lower bound on the optimum for any flows; the
    flows' congestion pattern only decides how tight it is.

    Args:
        tol: a flow within this fraction of its range counts as congested.
            Predicted flows hedge just inside their limit (F14) and need 1e-2;
            exact flows need 1e-6.
    """

    def __init__(self, data, tol=1e-2):
        self.data, self.fill, self.voll = data, MeritOrderFill(data), float(data.pVOLL)
        self.N, self.tol = data.num_n, tol
        A = data.lineflow_mask
        self.fr, self.to = torch.nonzero(A.T == -1)[:, 1], torch.nonzero(A.T == 1)[:, 1]
        self.f_lb, self.f_ub = line_bounds(data)
        self.imp, self.exp, self.rng = -self.f_lb, self.f_ub, self.f_ub - self.f_lb
        self.M, self.c = data.node_to_gen_mask.to(torch.get_default_dtype()), data.cost_vec
        self.gen_node = data.node_to_gen_mask.T.argmax(dim=1)
        # the dual objective is concave piecewise-linear in each price with kinks only at these values
        self.alphabet = torch.unique(torch.cat([self.c, torch.tensor([0.0, self.voll], device=self.c.device)]))
        self.below = (self.c[None, :] < self.alphabet[:, None]).to(self.M.dtype)
        self.upto = (self.c[None, :] <= self.alphabet[:, None]).to(self.M.dtype)
        # lines incident to each node, padded to the maximum degree: line index, +1 at the from-node / -1 at the
        # to-node / 0 for padding, and the node at the other end (padding points at node 0 and gets weight 0)
        per_node = [[] for _ in range(self.N)]
        for l, (a, b) in enumerate(zip(self.fr.tolist(), self.to.tolist(), strict=True)):
            per_node[a].append((l, 1.0, b))
            per_node[b].append((l, -1.0, a))
        deg = max(len(lines) for lines in per_node)
        self.inc = torch.zeros(self.N, deg, dtype=torch.long, device=self.c.device)
        self.sgn = torch.zeros(self.N, deg, dtype=self.c.dtype, device=self.c.device)
        self.other = torch.zeros(self.N, deg, dtype=torch.long, device=self.c.device)
        for n, lines in enumerate(per_node):
            for j, (l, sign, o) in enumerate(lines):
                self.inc[n, j], self.sgn[n, j], self.other[n, j] = l, sign, o

    def prices(self, X, f, sweeps=3):
        """Equilibrium prices: regional merit order, then node-wise ascent."""
        return self.node_ascent(X, self.quotient_prices(X, f), sweeps)

    def local_prices(self, X, f):
        D, cap = split_inputs(self.data, X)
        lo, hi = self.fill.price_interval(self.fill.residual(D, f), cap)
        return (0.5 * (lo + hi)).clamp(0, self.voll)

    def regions(self, f):
        """Label nodes by the smallest node index reachable over uncongested lines.

        Transitive closure by repeated squaring of the adjacency matrix: a fixed
        number of steps, ceil(log2(N - 1)), covers every path, so there is no
        data-dependent exit and the call can be captured in a CUDA graph.
        """
        B, N = f.shape[0], self.N
        unsat = ((f - self.f_lb) / self.rng >= self.tol) & ((self.f_ub - f) / self.rng >= self.tol)
        reach = torch.eye(N, dtype=torch.bool, device=f.device).expand(B, N, N).clone()
        reach[:, self.fr, self.to] = unsat
        reach[:, self.to, self.fr] |= unsat
        for _ in range((N - 2).bit_length()):
            r = reach.to(f.dtype)
            reach = reach | ((r @ r) > 0)
        return torch.where(reach, torch.arange(N, device=f.device)[None, None, :], N).min(dim=2).values

    def quotient_prices(self, X, f):
        """Merit-order price of every region on its aggregate residual: the equilibrium price in one shot."""
        D, cap = split_inputs(self.data, X)
        lab = self.regions(f)
        onehot = (lab[:, :, None] == torch.arange(self.N, device=f.device)).to(f.dtype)
        R = torch.einsum("bn,bnr->br", self.fill.residual(D, f), onehot)
        gen_region = torch.einsum("bnr,ng->bgr", onehot, self.M)
        cap_below = torch.einsum("kg,bg,bgr->bkr", self.below, cap, gen_region)
        cap_upto = torch.einsum("kg,bg,bgr->bkr", self.upto, cap, gen_region)
        Rk, al = R[:, None, :], self.alphabet[None, :, None].expand_as(cap_below)
        big = torch.full_like(cap_below, self.voll)
        lo = torch.where((cap_below < Rk) & (Rk <= cap_upto), al, big).min(dim=1).values
        hi = torch.where((cap_below <= Rk) & (Rk < cap_upto), al, big).min(dim=1).values
        lo = torch.where(R <= 0, torch.zeros_like(lo), lo)
        hi = torch.where(R < 0, torch.zeros_like(hi), hi)
        return torch.gather(0.5 * (lo + hi), 1, lab)

    def node_ascent(self, X, lam, sweeps=3):
        """Coordinate ascent on the dual objective, one node at a time, exact over the alphabet.

        Repairs nodes that `regions` tied to the wrong neighbours. On its own it
        stalls at corners of the piecewise-linear objective (F26): start it from
        `quotient_prices`.
        """
        D, cap = split_inputs(self.data, X)
        v = self.alphabet[None, None, :]
        t_node = v * D[:, :, None] - torch.einsum("ng,bgk->bnk", self.M, (v - self.c[None, :, None]).clamp(min=0) * cap[:, :, None])
        for _ in range(sweeps):
            for n in range(self.N):
                # price difference across each incident line, oriented so that a positive value exports from n
                dl = self.sgn[n][None, :, None] * (lam[:, self.other[n]][:, :, None] - v)
                line = -(dl.clamp(min=0) * self.exp[self.inc[n]][None, :, None] + (-dl).clamp(min=0) * self.imp[self.inc[n]][None, :, None])
                lam = lam.clone()
                lam[:, n] = self.alphabet[(t_node[:, n] + line.sum(1)).argmax(dim=1)]
        return lam

    def dual_objective(self, X, lam):
        D, cap = split_inputs(self.data, X)
        mu_up = (lam[:, self.gen_node] - self.c).clamp(min=0)
        dl = lam[:, self.to] - lam[:, self.fr]
        return (lam * D).sum(1) - (mu_up * cap).sum(1) - (dl.clamp(min=0) * self.exp).sum(1) - ((-dl).clamp(min=0) * self.imp).sum(1)

    def completion(self, lam):
        """Box multipliers for the rows p >= 0, p <= cap, f >= -imp, f <= exp, e >= 0, e <= D."""
        lam_g, dl = lam[:, self.gen_node], lam[:, self.to] - lam[:, self.fr]
        return torch.cat([(self.c - lam_g).clamp(min=0), (lam_g - self.c).clamp(min=0),
                          (-dl).clamp(min=0), dl.clamp(min=0),
                          self.voll - lam, torch.zeros_like(lam)], dim=1)


class Polish:
    """Exact coordinate descent on the fill cost, one line at a time.

    The cost is linear in a flow until the residual at either end reaches a
    merit-order breakpoint, so each step is a closed-form line search and only
    the two end nodes change. Lines that share no node move in the same round,
    so a sweep costs O(batch x lines x units per node) and its sequential depth
    is the number of colour classes, about the maximum node degree. From a
    network's flows one sweep removes most of the kink hedge (F27); from zero
    flows it stalls, because routing power along a path needs two flows to move
    together (F28).
    """

    def __init__(self, data):
        self.data, self.fill, self.voll = data, MeritOrderFill(data), float(data.pVOLL)
        A, M, c = data.lineflow_mask, data.node_to_gen_mask, data.cost_vec
        self.fr, self.to = torch.nonzero(A.T == -1)[:, 1], torch.nonzero(A.T == 1)[:, 1]
        self.f_lb, self.f_ub = line_bounds(data)
        K = int(M.sum(dim=1).max().item())
        slot = torch.full((data.num_n, K), -1, dtype=torch.long, device=c.device)
        for n in range(data.num_n):
            gens = torch.nonzero(M[n]).flatten()
            slot[n, :len(gens)] = gens[torch.argsort(c[gens])]
        self.valid, self.slot = slot >= 0, slot.clamp(min=0)
        self.slot_cost = torch.where(self.valid, c[self.slot], torch.full_like(c[self.slot], self.voll))
        self.fr_list, self.to_list = self.fr.tolist(), self.to.tolist()   # for Python loops over lines, no device syncs
        rounds = []  # greedy edge colouring: a round holds lines with pairwise disjoint end nodes
        for l in range(data.num_l):
            ends = {self.fr_list[l], self.to_list[l]}
            for r in rounds:
                if ends.isdisjoint(r["nodes"]):
                    r["lines"].append(l)
                    r["nodes"] |= ends
                    break
            else:
                rounds.append({"lines": [l], "nodes": set(ends)})
        self.rounds = [torch.tensor(r["lines"], device=c.device) for r in rounds]

    def cost(self, X, f):
        D, cap = split_inputs(self.data, X)
        p, e, _ = self.fill(D, cap, f)
        return (p.abs() @ self.data.cost_vec) + self.voll * e.abs().sum(1)

    def _node_state(self, r, cum, cost):
        """Left/right price and the nearest breakpoint below/above the residual of each given node.

        r [B, C], cum [B, C, K] cumulative capacities in merit order (inf past
        the last unit), cost [C, K] unit costs (VOLL past the last unit).
        """
        K = cum.shape[-1]
        idx = torch.arange(K, device=r.device)
        first = lambda cond: torch.where(cond, idx, K).min(dim=-1).values          # first slot meeting cond, K if none
        pick = lambda j: torch.where(j < K, torch.gather(cost.expand(r.shape[0], -1, -1), 2, j.clamp(max=K - 1)[..., None])[..., 0],
                                     torch.full_like(r, self.voll))
        j_hi, j_lo = first(cum > r[..., None] + 1e-9), first(cum >= r[..., None] - 1e-9)
        hi = torch.where(r < -1e-9, torch.full_like(r, -self.voll), pick(j_hi))
        lo = torch.where(r <= 1e-9, torch.full_like(r, -self.voll), pick(j_lo))
        up = torch.where(cum > r[..., None] + 1e-9, cum, torch.full_like(cum, INF)).min(dim=-1).values
        up = torch.where(r < -1e-9, torch.zeros_like(r), up)          # zero residual is a breakpoint too: -VOLL below, c_1 above
        down = torch.where(cum < r[..., None] - 1e-9, cum, torch.full_like(cum, -INF)).max(dim=-1).values
        down = torch.maximum(down, torch.where(r > 1e-9, torch.zeros_like(r), torch.full_like(r, -INF)))
        return lo, hi, up, down

    @torch.no_grad()
    def __call__(self, X, f, sweeps=1):
        """Return improved flows and their cost."""
        D, cap = split_inputs(self.data, X)
        f, r = f.clone(), self.fill.residual(D, f)
        cum = torch.cumsum(cap[:, self.slot] * self.valid, dim=2)
        cum = torch.where(self.valid, cum, torch.full_like(cum, INF))
        for _ in range(sweeps):
            for lines in self.rounds:
                a, b = self.fr[lines], self.to[lines]
                r_a, r_b, fl = r[:, a], r[:, b], f[:, lines]
                lo_a, hi_a, up_a, down_a = self._node_state(r_a, cum[:, a], self.slot_cost[a])
                lo_b, hi_b, up_b, down_b = self._node_state(r_b, cum[:, b], self.slot_cost[b])
                d_p = torch.minimum(torch.minimum(up_a - r_a, r_b - down_b), self.f_ub[lines] - fl).clamp(min=0)
                d_m = torch.minimum(torch.minimum(up_b - r_b, r_a - down_a), fl - self.f_lb[lines]).clamp(min=0)
                step = torch.where(lo_b - hi_a > 1e-9, d_p, torch.where(lo_a - hi_b > 1e-9, -d_m, torch.zeros_like(d_p)))
                step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
                f[:, lines] = fl + step
                r[:, a], r[:, b] = r_a + step, r_b - step
        return f, self.cost(X, f)


class PathPolish(Polish):
    """Exact min-cost flow from any starting flows by max-gain cycle cancelling.

    Lines cost nothing, so every negative cycle of the residual graph passes
    through the source: a path from a node whose marginal unit is cheap to a
    node whose marginal unit is dear, over lines with spare capacity in the
    travelling direction, gaining lo_b - hi_a per MW. When no such path is
    left there is no negative cycle, so the flows are optimal. Transit nodes
    keep their residual, which is the move coordinate descent cannot make
    (F28).
    """

    @torch.no_grad()
    def __call__(self, X, f, max_iter=200, tol=1e-9):
        """Return optimal flows, their cost, and augmentations per instance (`max_iter` marks non-convergence)."""
        D, cap = split_inputs(self.data, X)
        f, r = f.clone(), self.fill.residual(D, f)
        cum = torch.cumsum(cap[:, self.slot] * self.valid, dim=2)
        cum = torch.where(self.valid, cum, torch.full_like(cum, INF))
        B, N, L, dev = f.shape[0], self.data.num_n, self.data.num_l, f.device
        fr, eye = self.fr, torch.eye(self.data.num_n, dtype=torch.bool, device=dev)
        n_aug, active = torch.zeros(B, dtype=torch.long, device=dev), torch.arange(B, device=dev)
        for _ in range(max_iter):
            fa, ra = f[active], r[active]
            lo, hi, up, down = self._node_state(ra, cum[active], self.slot_cost)
            spare = torch.zeros(len(active), N, N, device=dev)     # spare capacity per ordered node pair, and the line giving it
            line = torch.full((len(active), N, N), -1, dtype=torch.long, device=dev)
            for l in range(L):
                for i, j, c in ((self.fr_list[l], self.to_list[l], self.f_ub[l] - fa[:, l]), (self.to_list[l], self.fr_list[l], fa[:, l] - self.f_lb[l])):
                    better = c > spare[:, i, j]
                    spare[:, i, j] = torch.where(better, c, spare[:, i, j])
                    line[:, i, j] = torch.where(better, torch.full_like(line[:, i, j], l), line[:, i, j])
            usable = spare > tol
            reach = usable | eye
            for _ in range((N - 1).bit_length()):
                reach = reach | ((reach.to(f.dtype) @ reach.to(f.dtype)) > 0)
            room_a, room_b = up - ra, ra - down
            gain = lo[:, None, :] - hi[:, :, None]                 # [b, a, b']: turn b' down, a up
            ok = reach & ~eye & (room_a > tol)[:, :, None] & (room_b > tol)[:, None, :]
            best, arg = torch.where(ok, gain, torch.full_like(gain, -INF)).flatten(1).max(dim=1)
            still = best > tol
            if not still.any():
                break
            active, fa, ra, room_a, room_b, spare, line, usable, arg = (t[still] for t in (active, fa, ra, room_a, room_b, spare, line, usable, arg))
            rows = torch.arange(len(active), device=dev)
            src, dst = arg // N, arg % N
            visited = torch.nn.functional.one_hot(src, N).bool()  # breadth-first search from src, parents recorded
            frontier, parent = visited.clone(), torch.full((len(active), N), -1, dtype=torch.long, device=dev)
            for _ in range(N):
                step = frontier[:, :, None] & usable
                new = step.any(dim=1) & ~visited
                parent = torch.where(new, step.to(f.dtype).argmax(dim=1), parent)
                visited, frontier = visited | new, new
                if visited[rows, dst].all():
                    break
            delta = torch.minimum(room_a[rows, src], room_b[rows, dst])
            node, done, arcs = dst, torch.zeros(len(active), dtype=torch.bool, device=dev), []
            for _ in range(N):                                      # walk back to src: bottleneck and the arcs to augment
                prev = parent[rows, node].clamp(min=0)
                delta = torch.where(done, delta, torch.minimum(delta, spare[rows, prev, node]))
                l = line[rows, prev, node].clamp(min=0)
                arcs.append((l, torch.where(fr[l] == prev, 1.0, -1.0), ~done))
                done = done | (prev == src)
                node = torch.where(done, node, prev)
                if done.all():
                    break
            for l, sign, use in arcs:
                f.index_put_((active[use], l[use]), sign[use] * delta[use], accumulate=True)
            r[active, src] += delta
            r[active, dst] -= delta
            n_aug[active] += 1
        return f, self.cost(X, f), n_aug


@dataclass
class PrimalDualSolution:
    y: torch.Tensor            # [B, G + L + N]: production, flows, unmet demand
    lam: torch.Tensor          # [B, N] nodal prices
    mu: torch.Tensor           # [B, 2 (G + L + N)] box multipliers, see DualRecovery.completion
    primal: torch.Tensor       # fill cost of y, an upper bound on the optimum
    dual: torch.Tensor         # dual objective, a lower bound on the optimum
    certificate: torch.Tensor  # (primal - dual) / dual, an upper bound on the relative gap of y


class PrimalDual:
    """Input to primal and dual solution in one call: network, polish, prices, completion, certificate.

    `exact` finishes the flows with `PathPolish`, so the primal is optimal and
    the certificate only reports the dual's slack; about four times the cost
    of the polish alone on the 20-node grid (F29). The network may hold its
    parameters in a cheaper dtype (`cast_net`): it then runs its message
    passing in that dtype and still returns flows in the data's, so the fill,
    polish, prices and certificate keep the data's precision (F31).
    """

    def __init__(self, data, net, polish_sweeps=1, ascent_sweeps=3, exact=False, tol=1e-2):
        self.data, self.net = data, net
        self.dual, self.polish, self.path = DualRecovery(data, tol=1e-6 if exact else tol), Polish(data), PathPolish(data)
        self.polish_sweeps, self.ascent_sweeps, self.exact = polish_sweeps, ascent_sweeps, exact

    @torch.no_grad()
    def __call__(self, X):
        f = self.net(X)[2]
        if self.polish_sweeps > 0:
            f, _ = self.polish(X, f, self.polish_sweeps)
        if self.exact:
            f, _, _ = self.path(X, f)
        D, cap = split_inputs(self.data, X)
        p, e, U = self.dual.fill(D, cap, f)
        lam = self.dual.prices(X, f, self.ascent_sweeps)
        fD = self.dual.dual_objective(X, lam)
        return PrimalDualSolution(torch.cat([p, f, e], dim=1), lam, self.dual.completion(lam), U, fD,
                                  (U - fD) / fD.clamp(min=torch.finfo(fD.dtype).tiny))


class GraphedPrimalDual:
    """`PrimalDual` replayed as one CUDA graph at a fixed batch size (F30).

    Replay removes the Python dispatch of the pipeline's few thousand small
    kernels, which is most of the time below a few thousand instances per
    batch and about a tenth of it at 8192 (F30). Calls take any number of
    instances, in chunks of `batch`. Without CUDA the same chunked path runs
    eagerly, so the class works everywhere.

    A final partial chunk is replayed on a buffer whose unused rows still hold
    the previous chunk's inputs. That is deliberate: rows are independent and
    every input is valid, so the stale rows are simply discarded, and it costs
    less than an eager pass on the remainder.

    Raises:
        ValueError: if the pipeline is exact; `PathPolish` loops until
            convergence and cannot be recorded.
    """

    def __init__(self, pipeline, batch, warmup=3):
        if pipeline.exact:
            raise ValueError("PathPolish has a data-dependent loop; capture the exact=False pipeline")
        self.pipeline, self.batch = pipeline, batch
        c = pipeline.dual.c
        self.x = torch.zeros(batch, pipeline.data.xdim, dtype=c.dtype, device=c.device)
        self.graph = None
        if c.device.type == "cuda":
            # warm up on a side stream (lazy cuBLAS / kernel loads happen outside the capture), then record
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup):
                    self.out = pipeline(self.x)
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = pipeline(self.x)

    def _replay(self):
        if self.graph is None:
            return self.pipeline(self.x)
        self.graph.replay()
        return self.out

    def __call__(self, X):
        M, fields = X.shape[0], None
        for i in range(0, M, self.batch):
            n = min(self.batch, M - i)
            self.x[:n].copy_(X[i:i + n])
            out = vars(self._replay())
            if fields is None:
                fields = {k: torch.empty((M, *v.shape[1:]), dtype=v.dtype, device=v.device) for k, v in out.items()}
            for k, v in out.items():
                fields[k][i:i + n].copy_(v[:n])
        return PrimalDualSolution(**fields)


def cast_data(data, dtype, device=None):
    """A shallow copy of the problem set with every float tensor in `dtype` (and on `device`)."""
    out = copy.copy(data)

    def move(t):
        return t.to(device, dtype) if t.is_floating_point() else t.to(device)
    out.X = move(data.X)
    out.opt_targets = {k: move(v) for k, v in data.opt_targets.items()}
    for name in DATA_TENSORS:
        setattr(out, name, move(getattr(data, name)))
    return out


def cast_net(net, dtype):
    """Copy the GNN with its parameters and incidence buffers in `dtype`, sharing the problem data.

    Only the encoders, the message-passing rounds and the readout run in
    `dtype`: the network standardizes its features and squashes its logit in
    the input's dtype, so line limits, feature statistics and the fill's
    constants stay in the data's precision. That split is what keeps float16
    usable, since raw inputs in MW would overflow it and standardized features
    do not (F31).
    """
    net = copy.deepcopy(net, memo={id(net.data): net.data})
    for p in net.parameters():
        p.data = p.data.to(dtype)
    for name in ("A_in", "A_out"):
        setattr(net, name, getattr(net, name).to(dtype))
    return net


def load_run(run_dir, device="cpu", dtype=None, data_root=None):
    """Rebuild a training run's network from its `args.json` and `model.pt`.

    Returns `(data, net, validation_start)`, the last being the index where
    the validation split begins. The data is moved to `device` before the
    network is built, because the fill inside the network holds its constants
    as plain tensors that `Module.to` would not move.

    Args:
        dtype: casts every float tensor of the data, and so of the network
            built from it; set the default dtype to match before calling,
            because the problem constants built later follow it.
        data_root: where the run's dataset lives, for a run trained with
            --data-root / --home-path. Without it the package default is
            used, and a dataset that is not there would be rebuilt and
            relabelled by Gurobi rather than found.
    """
    a = json.load(open(Path(run_dir) / "args.json"))
    data, args = load_or_create(a["config"], data_root=data_root or a.get("data_root"))
    data = cast_data(data, dtype, device)
    # keys missing from older runs take the training script's defaults
    args.update(hidden_size_factor=a["hidden_factor"], n_layers=a["layers"], body=a.get("body", "plain"),
                small_output_init=a.get("small_output_init", False))
    #! Where this run's validation rows begin. Runs made with --split pdl put them right after the
    #! training rows instead of at 80 %, so read the recorded value; runs from before it fall back.
    n_valid_start = a.get("valid_start", int(0.8 * data.X.shape[0]))
    X_train = data.X[:a.get("n_train", n_valid_start)]               # --train-size may have truncated the training set
    kw = dict(hidden=a.get("gnn_hidden", 128), rounds=a.get("gnn_rounds", 2), id_embed=0 if a.get("gnn_no_id_embed") else 8,
              layernorm=a.get("gnn_layernorm", False), antisym=a.get("gnn_antisym", False))
    scaler = None if a["variant"] == "flowfirst-gnn" else make_scaler(a.get("input_scale", "layernorm"), X_train)
    net = build_net(a["variant"], args, data, scaler, X_train=X_train, gnn_kwargs=kw).to(device)
    net.load_state_dict(torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=True))
    return data, net.eval(), n_valid_start


# ----------------------------------------------------------------------------
# Using a trained run on someone else's problem set: gep_benders.py builds its own
# operational data per experiment, and the evaluation notebook scores every model on
# one set of instances. Both need the network rebuilt against *given* data, and both
# need the pair of interfaces the Benders solver calls, primal_net(X) -> y and
# dual_net(X) -> (mu, lamb). flow-first has no dual network: the prices follow from
# the flows (F26), and the multipliers from the prices.
# ----------------------------------------------------------------------------

def build_net_for_data(a, data, run_dir, weights="model.pt", device="cpu"):
    """Rebuild the network of the run in `run_dir` against `data`.

    Like `load_run`, but the problem set is supplied instead of loaded from the run's
    config. Only the architecture keys of that config are read; the scaler and the GNN
    feature statistics are buffers, so the checkpoint overwrites whatever they are
    constructed with here.
    """
    with open(resolve_config(a["config"])) as fh:
        args = json.load(fh)
    args.update(device=device, dtype=str(torch.get_default_dtype()).rsplit(".", 1)[-1],
                hidden_size_factor=a["hidden_factor"], n_layers=a["layers"],
                body=a.get("body", "plain"), small_output_init=a.get("small_output_init", False))
    X_train = data.X[:a.get("n_train", int(0.8 * data.X.shape[0]))]
    kw = dict(hidden=a.get("gnn_hidden", 128), rounds=a.get("gnn_rounds", 2),
              id_embed=0 if a.get("gnn_no_id_embed") else 8,
              layernorm=a.get("gnn_layernorm", False), antisym=a.get("gnn_antisym", False))
    scaler = None if a["variant"] == "flowfirst-gnn" else make_scaler(a.get("input_scale", "layernorm"), X_train)
    net = build_net(a["variant"], args, data, scaler, X_train=X_train, gnn_kwargs=kw).to(device)
    net.load_state_dict(torch.load(Path(run_dir) / weights, map_location=device, weights_only=True))
    return net.eval()


class FlowFirstPrimal:
    """A flow-first network behind the `primal_net(X) -> y` interface.

    The network returns (y, f_raw, f); `polish_sweeps` exact line-search sweeps are
    applied to the flows first, which is what the deployed pipeline does and what cuts
    the gap by about five (F27). 0 scores the network alone.
    """

    def __init__(self, net, data, polish_sweeps=1):
        self.net, self.data, self.sweeps = net, data, polish_sweeps
        self.fill = MeritOrderFill(data)
        self.polish = Polish(data) if polish_sweeps else None

    def flows(self, X):
        with torch.no_grad():
            f = self.net(X)[2]
            if self.polish is not None:
                f, _ = self.polish(X, f, self.sweeps)
        return f

    def __call__(self, X, total_demands=None):
        D, cap = split_inputs(self.data, X)
        f = self.flows(X)
        p, e, _ = self.fill(D, cap, f)
        return torch.cat([p, f, e], dim=1)

    def eval(self):
        return self


class FlowFirstDual:
    """The constructed dual behind the `dual_net(X) -> (mu, lamb)` interface.

    Prices from the quotient fill over regions of uncongested lines plus node-wise
    ascent (F26); `completion` turns them into the box multipliers, in the row order
    the problem class and the Benders cut builder use. The price is returned negated,
    the convention the problem class stores and `dual_obj_fn` expects.
    """

    def __init__(self, primal, data, ascent_sweeps=3):
        self.primal, self.recovery, self.sweeps = primal, DualRecovery(data), ascent_sweeps

    def __call__(self, X):
        with torch.no_grad():
            lam = self.recovery.prices(X, self.primal.flows(X), self.sweeps)
        return self.recovery.completion(lam), -lam

    def eval(self):
        return self


def load_flowfirst_pair(run_dir, data, weights="model.pt", polish_sweeps=1, device="cpu"):
    """(primal, dual) adapters for the flow-first run in `run_dir`, on `data`.

    Drop-in for the `(primal_net, dual_net)` pair `gep_benders.py` loads from two PDL
    checkpoint directories; here one directory holds the only network there is.
    """
    a = json.load(open(Path(run_dir) / "args.json"))
    net = build_net_for_data(a, data, run_dir, weights=weights, device=device)
    primal = FlowFirstPrimal(net, data, polish_sweeps)
    return primal, FlowFirstDual(primal, data)


def is_flowfirst_run(run_dir, weights="model.pt"):
    """A flow-first run directory holds one network and no dual checkpoint."""
    return (Path(run_dir) / weights).exists()
