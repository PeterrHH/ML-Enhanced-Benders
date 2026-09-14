"""Merit-order fill: complete generation and unmet demand from the flows.

Given flows f, the residual demand at node n is r_n = D_n - net inflow_n.
Each node's generators are dispatched in increasing cost order up to their
available capacity; whatever is left is unmet demand e_n, which is negative
when the node is over-supplied. This is the exact solution of the one-row
box LP at every node, so

    U(f) = sum_g c_g p_g + VOLL * sum_n |e_n|

is convex piecewise-linear in f and dU/df_l = price(from(l)) - price(to(l)),
where the price of a node is the cost of its marginal generator, VOLL when
it is short, and -VOLL when it is over-supplied. Autograd through the fill
produces exactly that gradient; ``price_interval`` gives it explicitly,
including the left/right pair at a kink.

Cost ties within a node are broken by generator index. That is the place
where a deterministic cost perturbation would go later.
"""
import torch


class MeritOrderFill:
    def __init__(self, data, voll=None):
        """voll overrides the dataset's value of lost load, e.g. for a training-time penalty."""
        self.cost = data.cost_vec.clone()                  # [G]
        self.voll = float(data.pVOLL if voll is None else voll)
        self.node_to_gen = data.node_to_gen_mask.clone()   # [N, G]
        self.lineflow = data.lineflow_mask.clone()         # [N, L], -1 at from-node, +1 at to-node

        same_node = (self.node_to_gen.T @ self.node_to_gen).bool()          # [G, G]
        c, idx = self.cost, torch.arange(len(self.cost), device=self.cost.device)
        precedes = (c[None, :] < c[:, None]) | ((c[None, :] == c[:, None]) & (idx[None, :] < idx[:, None]))
        # [g, g'] = 1 if g' is dispatched before g at the same node
        self.precedes_same_node = (same_node & precedes).to(self.cost.dtype)

    # ---- pieces -----------------------------------------------------------
    def residual(self, D, f):
        """r_n = D_n - net inflow_n, shape [B, N]."""
        return D - f @ self.lineflow.T

    def dispatch(self, r, cap):
        """Merit-order fill. r: [B, N], cap: [B, G]. Returns p [B, G], e [B, N]."""
        r_g = r @ self.node_to_gen                       # residual of each generator's node
        before = cap @ self.precedes_same_node.T         # capacity of cheaper units at that node
        p = torch.minimum(torch.relu(r_g - before), cap)
        e = r - p @ self.node_to_gen.T
        return p, e

    def cost_fn(self, p, e):
        return p @ self.cost + self.voll * e.abs().sum(dim=1)

    def __call__(self, D, cap, f):
        """Returns p, e, U for flows f. Differentiable in f."""
        p, e = self.dispatch(self.residual(D, f), cap)
        return p, e, self.cost_fn(p, e)

    # ---- explicit prices --------------------------------------------------
    def price_interval(self, r, cap):
        """Left and right derivative of the node cost in r, shape [B, N] each.

        They coincide except at a kink, where r sits exactly on a capacity
        breakpoint. Lo is the price just below r, hi the price just above.
        """
        r_g = r @ self.node_to_gen
        before = cap @ self.precedes_same_node.T
        after = before + cap
        lo_mask = (before < r_g) & (r_g <= after)        # unique generator per node, or none
        hi_mask = (before <= r_g) & (r_g < after)
        cost = self.cost[None, :].expand_as(r_g)
        lo_g = torch.where(lo_mask, cost, torch.zeros_like(cost)) @ self.node_to_gen.T
        hi_g = torch.where(hi_mask, cost, torch.zeros_like(cost)) @ self.node_to_gen.T
        has_lo = (lo_mask.to(cost.dtype) @ self.node_to_gen.T) > 0
        has_hi = (hi_mask.to(cost.dtype) @ self.node_to_gen.T) > 0
        voll = torch.full_like(r, self.voll)
        lo = torch.where(has_lo, lo_g, torch.where(r <= 0, -voll, voll))
        hi = torch.where(has_hi, hi_g, torch.where(r < 0, -voll, voll))
        return lo, hi

    def flow_gradient(self, lam):
        """dU/df_l = lam_from - lam_to for nodal prices lam [B, N]. Shape [B, L]."""
        return -(lam @ self.lineflow)
