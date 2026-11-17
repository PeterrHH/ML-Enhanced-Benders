"""Way 2-iii: the best ramp-feasible dispatch of every node over a block, flows fixed, with no LP.

With the flows fixed, node n has residual r_t = D_t - net import_t in each hour t of the block. Its
ramp-limited units run at p_t; its other units (no ramp limit, R = 1) cover the rest in merit order, hour by
hour, and what is left is unserved demand md. For one ramp-limited unit the cost of hour t depends on p_t
alone,

    g_t(p) = c p + F_t(r_t - p),   F_t(y) = merit-order cost of the other units for y + VOLL |unserved|,

convex and piecewise linear, and the node's block problem is a chain

    min sum_t g_t(p_t)   over  lo_t <= p_t <= hi_t,  |p_t - p_(t-1)| <= rho,

solved exactly by dynamic programming: V_1 = g_1, V_t(p) = g_t(p) + min_{|q - p| <= rho} V_(t-1)(q). This is
R2's node-block LP (flowfirst/block.py) without a solver.

A convex piecewise-linear function on [lo, lo + sum(L)] is held as (lo, value at lo, slopes S, lengths L), its
segments in order of position, so S is non-decreasing. The windowed minimum is then the merge of the segments
with one more segment of slope 0 and length 2 rho, sorted by slope, starting at lo - rho. A sum is the merge of
the two functions' breakpoints on the common domain. Everything is batched over rows (node-blocks).

The gradient. The optimal cost is convex and piecewise linear in the residuals, and its derivative in r_t is the
node's price lambda_t (the dual of its balance row): what the flows need, since d cost / d f = lambda_from -
lambda_to. It is computed by central differences of the program itself, all 2B perturbed problems in one batch:
exact away from a kink, between the left and right slope at one. Two shortcuts are wrong. Holding the optimal
path fixed and differentiating only through the merit-order fill: an optimum sits on a kink, where it gives VOLL
instead of the ramp-coupled price. Autograd through the program: coinciding breakpoints (a unit with zero
capacity, say) split its gradient at ties into meaningless fractions.

Two modes:
  elastic (training)  0 <= p_t <= cap_t; over-supply (md < 0) and exports beyond the node's means are charged
                      VOLL |md|, as in the hourly loss. p = 0 is always allowed, so the program never fails.
  hard (evaluation)   md in [0, D]: p_t <= min(cap_t, r_t) and p_t >= r_t - (other units' capacity) - D_t.
                      A node-block with no such path is reported infeasible; BlockLadder.run_dp shrinks the flows.

Several ramp-limited units at one node (k >= 2): one program per unit, cheapest first, each on the residual
the others leave (the units not yet planned count at 0, and in hard mode as able to give up to their
capacity), then `refine` passes re-planning each unit with the others fixed. Every pass keeps the dispatch
feasible and can only lower its cost, but the result is not guaranteed optimal. The gradient is then the
cheapest one-unit response: per hour, the lowest of the units' prices with the others fixed. With k = 0 or 1
everything is exact.
"""
from dataclasses import dataclass

import torch


@dataclass
class PWL:
    """Convex piecewise-linear function on [lo, lo + L.sum(-1)], per row. S, L: [R, M]."""
    lo: torch.Tensor
    val: torch.Tensor
    S: torch.Tensor
    L: torch.Tensor

    def ends(self):
        return self.lo[:, None] + torch.cumsum(self.L, dim=-1)

    def hi(self):
        return self.lo + self.L.sum(dim=-1)

    def at(self, x):
        """Value at x [R], x inside the domain."""
        start = self.ends() - self.L
        return self.val + (self.S * torch.minimum(torch.relu(x[:, None] - start), self.L)).sum(dim=-1)

    def window(self, rho):
        """min over |q - p| <= rho of f(q): one more segment of slope 0 and length 2 rho, by slope."""
        S = torch.cat([self.S, torch.zeros_like(self.S[:, :1])], dim=-1)
        L = torch.cat([self.L, 2 * rho[:, None]], dim=-1)
        order = torch.argsort(S.detach(), dim=-1, stable=True)
        return PWL(self.lo - rho, self.val, S.gather(-1, order), L.gather(-1, order))

    def argmin(self):
        """Left end of the minimizing set, and the minimum."""
        neg = (self.S < 0).to(self.L.dtype)
        return self.lo + (self.L * neg).sum(dim=-1), self.val + (self.S * self.L * neg).sum(dim=-1)


def add(f, g):
    """f + g on the common domain, and a flag per row where that domain is empty."""
    lo, hi = torch.maximum(f.lo, g.lo), torch.minimum(f.hi(), g.hi())
    empty = lo > hi + 1e-9 * (1 + lo.abs())
    hi = torch.maximum(hi, lo)
    ef, eg = f.ends(), g.ends()
    clip = lambda e: torch.minimum(torch.maximum(e, lo[:, None]), hi[:, None])
    Q = torch.sort(torch.cat([lo[:, None], clip(ef), clip(eg)], dim=-1), dim=-1).values
    L = Q[:, 1:] - Q[:, :-1]
    mid = (0.5 * (Q[:, 1:] + Q[:, :-1])).detach()
    slope = lambda h, e: h.S.gather(-1, torch.searchsorted(e.detach().contiguous(), mid.contiguous()).clamp(max=h.S.shape[1] - 1))
    return PWL(lo, f.at(lo) + g.at(lo), slope(f, ef) + slope(g, eg), L), empty


class RampDP:
    """The node-block dispatch of Way 2-iii for one network.

    Args:
        node_to_gen: [N, G] 0/1, the node of each generator.
        cost: [G] variable cost.
        voll: value of lost load.
        ramp_limited: [G] bool, units with a ramp limit below their size (R < 1).
    """

    def __init__(self, node_to_gen, cost, voll, ramp_limited):
        M = torch.as_tensor(node_to_gen, dtype=torch.get_default_dtype())
        self.cost = torch.as_tensor(cost, dtype=M.dtype)
        self.voll = float(voll)
        self.M = M
        self.N, self.G = M.shape
        slow = torch.as_tensor(ramp_limited, dtype=torch.bool)
        self.ramp_limited = slow

        def slots(mask):
            per = [[g for g in torch.argsort(self.cost).tolist() if M[n, g] > 0 and mask[g]] for n in range(self.N)]
            width = max(1, max(len(s) for s in per))
            idx = torch.zeros(self.N, width, dtype=torch.long)
            valid = torch.zeros(self.N, width, dtype=torch.bool)
            for n, s in enumerate(per):
                idx[n, :len(s)], valid[n, :len(s)] = torch.tensor(s, dtype=torch.long), True
            return idx, valid
        self.slow_idx, self.slow_ok = slots(slow)          # [N, k], merit order
        self.free_idx, self.free_ok = slots(~slow)         # [N, m], merit order
        self.k = int(self.slow_ok.sum(1).max())

    @classmethod
    def from_data(cls, data, ramp_frac):
        return cls(data.node_to_gen_mask, data.cost_vec, data.pVOLL, torch.as_tensor(ramp_frac) < 1.0)

    # ------------------------------------------------------------------ one hour, one unit
    def _hour(self, c, y_r, fcum0, fcap, fcost, lo, hi):
        """g(p) = c p + F(y_r - p) on [lo, hi] as a PWL. c, y_r, lo, hi: [R]; fcum0 [R, m+1] (0 first);
        fcap, fcost [R, m] in merit order."""
        R, m = fcap.shape
        bp = y_r[:, None] - fcum0.flip(-1)                                    # ascending: y_r - cum_m, ..., y_r
        inf = torch.full((R, 1), float("inf"), dtype=y_r.dtype)
        start = torch.cat([-inf, bp], dim=-1)
        end = torch.cat([bp, inf], dim=-1)
        clip = lambda e: torch.minimum(torch.maximum(e, lo[:, None]), hi[:, None])
        L = clip(end) - clip(start)
        v = torch.full((R, 1), self.voll, dtype=y_r.dtype)
        S = c[:, None] + torch.cat([-v, -fcost.flip(-1), v], dim=-1)
        return PWL(lo, c * lo + self._F(y_r - lo, fcum0, fcap, fcost), S, L)

    def _F(self, y, fcum0, fcap, fcost):
        """Merit-order cost of the other units for y, plus VOLL |unserved|. y [..]; fcum0 [.., m+1]."""
        served = torch.minimum(torch.relu(y[..., None] - fcum0[..., :-1]), fcap)
        return (fcost * served).sum(-1) + self.voll * (torch.relu(-y) + torch.relu(y - fcum0[..., -1]))

    # ------------------------------------------------------------------ one unit over the block
    def chain(self, c, rho, y_r, fcap, fcost, lo, hi):
        """The program for one unit. c, rho [R]; y_r, lo, hi [R, B]; fcap [R, B, m]; fcost [R, m].
        Returns (min cost [R], path p [R, B] (no gradient), empty [R])."""
        B = y_r.shape[1]
        fcum0 = torch.cat([torch.zeros_like(fcap[..., :1]), torch.cumsum(fcap, dim=-1)], dim=-1)
        empty = lo > hi + 1e-9 * (1 + lo.abs())
        hi = torch.maximum(hi, lo)
        hour = lambda t: self._hour(c, y_r[:, t], fcum0[:, t], fcap[:, t], fcost, lo[:, t], hi[:, t])
        V = hour(0)
        left = [V.argmin()[0]]
        bad = empty.any(dim=1)
        for t in range(1, B):
            V, e = add(V.window(rho), hour(t))
            bad = bad | e
            left.append(V.argmin()[0])
        x, value = V.argmin()
        with torch.no_grad():
            p = [x]
            for t in range(B - 2, -1, -1):
                p.insert(0, torch.minimum(torch.maximum(left[t], p[0] - rho), p[0] + rho))
            p = torch.stack(p, dim=1)
        return value, p, bad

    # ------------------------------------------------------------------ every node of every block
    def __call__(self, r, cap, rho, D=None, hard=False, refine=2, prices=None):
        """Dispatch K blocks. r, D [K, B, N]; cap [K, B, G]; rho [K, G].

        Returns a dict with p [K, B, G], md [K, B, N], cost [K], lam [K, B, N] (or None), node_ok [K, N] and
        ok [K] (hard mode: a dispatch with md in [0, D] was found for every node). With `prices` (default:
        when r carries a gradient) cost has the gradient lam in r, else none."""
        if prices is None:
            prices = torch.is_grad_enabled() and r.requires_grad
        K, B, N = r.shape
        Rn = K * N
        rows = lambda z: z.permute(0, 2, 1).reshape(Rn, B)                  # [K, B, N] -> [K*N, B]
        y_g = rows(r)
        y = y_g.detach()
        fcap = (cap[:, :, self.free_idx] * self.free_ok).permute(0, 2, 1, 3).reshape(Rn, B, -1)
        fcost = torch.where(self.free_ok, self.cost[self.free_idx], torch.full_like(self.cost[self.free_idx], self.voll))
        fcost = fcost.repeat(K, 1)
        ftot = fcap.sum(-1)
        Dr = rows(D) if D is not None else None
        k = self.slow_idx.shape[1]
        scap = (cap[:, :, self.slow_idx] * self.slow_ok).permute(0, 2, 1, 3).reshape(Rn, B, k)
        srho = (rho[:, self.slow_idx] * self.slow_ok).reshape(Rn, k)
        sc = self.cost[self.slow_idx].repeat(K, 1)
        has = self.slow_ok.repeat(K, 1)                                     # [Rn, k]

        def unit(j, y_j, later, reps=1):
            """The program for unit j on residual y_j [reps*Rn, B]; `later` [Rn, B] is the capacity of the
            units not yet planned (hard mode). Returns (cost of unit j and the other units, path, bad)."""
            rep = lambda z: z.repeat((reps,) + (1,) * (z.dim() - 1))
            if hard:
                lo = torch.relu(y_j - rep(ftot) - rep(Dr) - rep(later))
                hi = torch.minimum(rep(scap[..., j]), y_j)
            else:
                lo, hi = torch.zeros_like(y_j), rep(scap[..., j])
            return self.chain(rep(sc[:, j]), rep(srho[:, j]), y_j, rep(fcap), rep(fcost), lo, hi)

        with torch.no_grad():
            p = torch.zeros(Rn, B, k, dtype=r.dtype)
            planned = torch.zeros(k, dtype=torch.bool)
            ok = torch.ones(Rn, dtype=torch.bool)
            for i_pass in range(1 + (refine if self.k >= 2 else 0)):
                for j in range(k):
                    later = (scap * (~planned & (torch.arange(k) != j))).sum(-1)
                    _, pj, bad = unit(j, y - (p.sum(-1) - p[..., j]), later)
                    p[..., j] = pj * has[:, j, None]                          # a node without unit j: 0
                    planned[j] = True
                    if hard and i_pass == 0:
                        ok = ok & ~(bad & has[:, j])
            # the other units in merit order on what the ramp-limited units leave
            yl = y - p.sum(-1)
            fcum0 = torch.cat([torch.zeros_like(fcap[..., :1]), torch.cumsum(fcap, dim=-1)], dim=-1)
            q = torch.minimum(torch.relu(yl[..., None] - fcum0[..., :-1]), fcap)
            md = yl - q.sum(-1)
            if hard:
                tol = 1e-7 * (1 + Dr.abs())
                ok = ok & ((md >= -tol) & (md <= Dr + tol)).all(dim=1)
            cost_rows = (sc[:, None, :] * p).sum((1, 2)) + (fcost[:, None, :] * q).sum((1, 2)) + self.voll * md.abs().sum(1)
            P = torch.zeros(K, N, B, self.G, dtype=r.dtype)
            P.scatter_add_(3, self.free_idx[None, :, None, :].expand(K, N, B, -1), q.reshape(K, N, B, -1) * self.free_ok[None, :, None, :])
            P.scatter_add_(3, self.slow_idx[None, :, None, :].expand(K, N, B, -1), p.reshape(K, N, B, k) * self.slow_ok[None, :, None, :])
            lam = self._prices(unit, y, p, has, B) if prices else None

        cost = cost_rows
        if lam is not None:
            cost = cost + (lam * (y_g - y)).sum(-1)                         # value unchanged, gradient lam in r
        unrows = lambda z: z.reshape(K, N, B).permute(0, 2, 1)
        return dict(p=P.sum(1), md=unrows(md), cost=cost.reshape(K, N).sum(1),
                    lam=None if lam is None else unrows(lam), node_ok=ok.reshape(K, N), ok=ok.reshape(K, N).all(dim=1))

    def _prices(self, unit, y, p, has, B):
        """lambda [Rn, B]: per hour, the central difference of a unit's program value in the node's residual,
        with the other ramp-limited units fixed; the lowest over the node's units (the cheapest one-unit
        response; exact for k = 1). The program's value is convex and piecewise linear in the residual, so the
        difference is exact away from a kink, and between the left and right slope at one. A side whose
        perturbed problem is infeasible (hard mode) is left out."""
        Rn, k = y.shape[0], p.shape[-1]
        eps = 1e-7 * (1 + y.abs().amax(dim=1, keepdim=True))                # [Rn, 1]
        E = torch.eye(B, dtype=y.dtype)
        lam = torch.full((Rn, B, k), float("inf"), dtype=y.dtype)
        zero = torch.zeros_like(y)
        for j in range(k):
            yj = y - (p.sum(-1) - p[..., j])
            pert = torch.cat([yj] + [yj + sgn * eps * E[t] for sgn in (1, -1) for t in range(B)], dim=0)
            v, _, bad = unit(j, pert, zero, reps=2 * B + 1)
            v, bad = v.reshape(2 * B + 1, Rn), bad.reshape(2 * B + 1, Rn)
            up = (v[1:B + 1] - v[0]) / eps[:, 0]                             # [B, Rn]
            dn = (v[0] - v[B + 1:]) / eps[:, 0]
            ok_up, ok_dn = ~bad[1:B + 1], ~bad[B + 1:]
            both = torch.where(ok_up & ok_dn, 0.5 * (up + dn), torch.where(ok_up, up, dn))
            use = has[:, j] | (j == 0)                                       # a node with no unit: its F alone
            lam[:, :, j] = torch.where(use[:, None], both.T, lam[:, :, j])
        return lam.min(dim=-1).values

    def prices(self, r, cap, rho, D=None, hard=True):
        """lambda [K, B, N]: the derivative of the dispatch cost in each node's residual."""
        return self(r.detach(), cap, rho, D, hard, prices=True)["lam"]
