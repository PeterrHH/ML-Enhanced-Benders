"""Playground for flowfirst/rampdp.py: one node, a few hours, hand-made units.

Runs the real RampDP on a single node-block and records what RampDP.chain builds hour by hour (g_t, the
windowed V_(t-1), V_t, and the read-back), so the exact piecewise-linear program can be read as tables and
plots. An LP (scipy, HiGHS) solves the same node-block for comparison: cost, dispatch and prices. Used by
RampDP_Playground.ipynb.
"""
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

from flowfirst.rampdp import RampDP, add

torch.set_default_dtype(torch.float64)


@dataclass
class Unit:
    name: str
    cost: float
    cap: object          # MW available: one number, or one per hour
    ramp: float = None   # MW per hour; None = no ramp limit


class NodeBlock:
    """One node over B hours: residual r_t (= demand - net import, what the node's own units must cover)."""

    def __init__(self, residual, units, voll=100.0, demand=None):
        self.r = np.asarray(residual, dtype=float)
        self.B, self.G = len(self.r), len(units)
        self.units, self.voll = units, float(voll)
        self.D = self.r.copy() if demand is None else np.asarray(demand, dtype=float)
        self.cap = np.stack([np.broadcast_to(np.asarray(u.cap, dtype=float), (self.B,)) for u in units], axis=1)
        self.limited = np.array([u.ramp is not None for u in units])
        self.rho = np.array([u.ramp if u.ramp is not None else 0.0 for u in units])
        self.dp = RampDP(np.ones((1, self.G)), [u.cost for u in units], self.voll, self.limited)
        dp = self.dp
        self.slow = [int(g) for g, ok in zip(dp.slow_idx[0], dp.slow_ok[0]) if ok]   # ramp-limited, cheapest first
        self.free = [int(g) for g, ok in zip(dp.free_idx[0], dp.free_ok[0]) if ok]   # no ramp limit, merit order
        cap_t = torch.as_tensor(self.cap)
        self.fcap = (cap_t[:, dp.free_idx[0]] * dp.free_ok[0])[None]                  # [1, B, m]
        self.fcost = torch.where(dp.free_ok[0], dp.cost[dp.free_idx[0]],
                                 torch.full_like(dp.cost[dp.free_idx[0]], self.voll))[None]
        self.fcum0 = torch.cat([torch.zeros_like(self.fcap[..., :1]), torch.cumsum(self.fcap, dim=-1)], dim=-1)

    # ------------------------------------------------------------------ the real program, as used in training
    def solve(self, hard=False, refine=2):
        """RampDP itself on this node-block: dispatch table with prices."""
        r = torch.as_tensor(self.r)[None, :, None]
        out = self.dp(r, torch.as_tensor(self.cap)[None], torch.as_tensor(self.rho)[None],
                      D=torch.as_tensor(self.D)[None, :, None], hard=hard, refine=refine, prices=True)
        P, md = out["p"][0].numpy(), out["md"][0, :, 0].numpy()
        return self._table(P, md, out["lam"][0, :, 0].numpy(), "price λ (program)"), bool(out["ok"][0])

    # ------------------------------------------------------------------ one ramp-limited unit, step by step
    def _bounds(self, j_unit, y, later=None, hard=False):
        cap = torch.as_tensor(self.cap[:, j_unit])[None]
        if not hard:
            return torch.zeros_like(y), cap
        later = torch.zeros_like(y) if later is None else later
        lo = torch.relu(y - self.fcap.sum(-1) - torch.as_tensor(self.D)[None] - later)
        return lo, torch.minimum(cap, y)

    def trace(self, hard=False):
        """Hour by hour what RampDP.chain builds for the node's one ramp-limited unit.
        Returns a list of steps {t, g, W, V} (PWLs; W = windowed V_(t-1)) and the path read back."""
        assert len(self.slow) == 1, "trace() is for exactly one ramp-limited unit; use passes() for more"
        s = self.slow[0]
        y = torch.as_tensor(self.r)[None]
        c, rho = torch.tensor([self.units[s].cost]), torch.tensor([self.rho[s]])
        lo, hi = self._bounds(s, y, hard=hard)
        hour = lambda t: self.dp._hour(c, y[:, t], self.fcum0[:, t], self.fcap[:, t], self.fcost, lo[:, t], hi[:, t])
        V = hour(0)
        steps = [dict(t=1, g=V, W=None, V=V)]
        for t in range(1, self.B):
            W, g = V.window(rho), hour(t)
            V, empty = add(W, g)
            steps.append(dict(t=t + 1, g=g, W=W, V=V, empty=bool(empty[0])))
        # read back: the last hour at the minimum of V_B, then each earlier hour as close as the ramp allows
        # to the left end of its own minimum (V_t is convex, so that is the best point inside the window)
        path = [float(V.argmin()[0][0])]
        for st in reversed(steps[:-1]):
            left = float(st["V"].argmin()[0][0])
            path.insert(0, min(max(left, path[0] - self.rho[s]), path[0] + self.rho[s]))
        _, p_chain, _ = self.dp.chain(c, rho, y, self.fcap, self.fcost, lo, hi)
        assert np.allclose(path, p_chain[0].numpy()), "trace differs from RampDP.chain"
        return steps, np.array(path)

    def g_labels(self):
        """What each segment of g_t means, in the order _hour builds them (before empty ones are dropped)."""
        names = [self.units[g].name for g in self.free]
        m = self.fcap.shape[-1]
        pad = ["(no unit)"] * (m - len(names))
        return (["p too low: unserved demand"] + [f"p displaces {n}" for n in (names + pad)[::-1]]
                + ["p too high: over-supply"])

    # ------------------------------------------------------------------ several ramp-limited units
    def passes(self, hard=False, refine=2):
        """RampDP's passes for k ramp-limited units, unit by unit (cheapest first, then `refine` re-plans).
        Returns a table: after each unit's program, the paths of all ramp-limited units and the node's cost."""
        k, B = len(self.slow), self.B
        y = torch.as_tensor(self.r)[None]
        scap = torch.as_tensor(self.cap[:, self.slow])[None]
        p = torch.zeros(1, B, k)
        planned = torch.zeros(k, dtype=torch.bool)
        rows = []
        for i_pass in range(1 + (refine if k >= 2 else 0)):
            for j, s in enumerate(self.slow):
                later = (scap * (~planned & (torch.arange(k) != j))).sum(-1)
                yj = y - (p.sum(-1) - p[..., j])
                lo, hi = self._bounds(s, yj, later, hard)
                _, pj, bad = self.dp.chain(torch.tensor([self.units[s].cost]), torch.tensor([self.rho[s]]),
                                           yj, self.fcap, self.fcost, lo, hi)
                p[..., j], planned[j] = pj, True
                row = {"pass": "first" if i_pass == 0 else f"refine {i_pass}", "planned": self.units[s].name}
                row.update({self.units[g].name: np.round(p[0, :, i].numpy(), 3).tolist() for i, g in enumerate(self.slow)})
                row["node cost"] = self.cost_of(p[0].numpy())
                rows.append(row)
        return pd.DataFrame(rows)

    def cost_of(self, p_slow):
        """Cost of the node when its ramp-limited units run at p_slow [B, k]: the others fill in merit order."""
        yl = self.r - p_slow.sum(-1)
        fcap, fcum0 = self.fcap[0].numpy(), self.fcum0[0].numpy()
        q = np.minimum(np.maximum(yl[:, None] - fcum0[:, :-1], 0), fcap)
        md = yl - q.sum(-1)
        cs = np.array([self.units[g].cost for g in self.slow])
        return float((p_slow * cs).sum() + (q * self.fcost[0].numpy()).sum() + self.voll * np.abs(md).sum())

    # ------------------------------------------------------------------ the same node-block as an LP
    def lp(self, hard=False):
        """min sum c p + VOLL (md+ + md-)  s.t. sum_g p + md+ - md- = r_t, 0 <= p <= cap, |p_t - p_(t-1)| <= rho.
        hard: md- = 0, md+ <= D. Returns (cost, dispatch table with the LP's balance prices)."""
        B, G = self.B, self.G
        nv = B * (G + 2)
        ix = lambda t, g: t * (G + 2) + g                                    # p_(t,g); g = G: md+, G+1: md-
        cost = np.zeros(nv)
        bounds = []
        for t in range(B):
            for g, u in enumerate(self.units):
                cost[ix(t, g)] = u.cost
                bounds.append((0, self.cap[t, g]))
            cost[ix(t, G)] = cost[ix(t, G + 1)] = self.voll
            bounds += [(0, self.D[t] if hard else None), (0, 0 if hard else None)]
        A_eq = np.zeros((B, nv))
        for t in range(B):
            A_eq[t, [ix(t, g) for g in range(G)]] = 1
            A_eq[t, ix(t, G)], A_eq[t, ix(t, G + 1)] = 1, -1
        A_ub, b_ub = [], []
        for g in np.flatnonzero(self.limited):
            for t in range(1, B):
                for sgn in (1, -1):
                    row = np.zeros(nv)
                    row[ix(t, g)], row[ix(t - 1, g)] = sgn, -sgn
                    A_ub.append(row)
                    b_ub.append(self.rho[g])
        res = linprog(cost, A_ub=np.array(A_ub) if A_ub else None, b_ub=b_ub or None, A_eq=A_eq, b_eq=self.r,
                      bounds=bounds, method="highs")
        if res.status != 0:
            return None, None
        x = res.x.reshape(B, G + 2)
        return float(res.fun), self._table(x[:, :G], x[:, G] - x[:, G + 1], res.eqlin.marginals, "price λ (LP)")

    # ------------------------------------------------------------------ display
    def _table(self, P, md, lam, lam_name):
        df = pd.DataFrame({"residual r_t": self.r})
        for g, u in enumerate(self.units):
            tag = f" (ramp {u.ramp:g}/h)" if u.ramp is not None else ""
            df[f"{u.name}{tag}"] = P[:, g]
        df["unserved"] = md
        df["curtailed / unused"] = self.cap.sum(1) - P.sum(1)
        df["hour cost"] = P @ np.array([u.cost for u in self.units]) + self.voll * np.abs(md)
        df[lam_name] = lam
        df.index = [f"hour {t + 1}" for t in range(self.B)]
        return df.round(3)


def pwl_table(f, labels=None):
    """A PWL (row 0) as a table of its pieces: from, to, slope, value at both ends. Empty pieces dropped."""
    lo, val = float(f.lo[0]), float(f.val[0])
    S, L = f.S[0].numpy(), f.L[0].numpy()
    rows, x, v = [], lo, val
    for i, (s, l) in enumerate(zip(S, L)):
        if l > 1e-12:
            row = {"p from": x, "p to": x + l, "slope": s, "value at from": v, "value at to": v + s * l}
            if labels is not None:
                row["meaning"] = labels[i]
            rows.append(row)
        x, v = x + l, v + s * l
    return pd.DataFrame(rows).round(3)


def _xy(f):
    xs = [float(f.lo[0])] + list(float(f.lo[0]) + np.cumsum(f.L[0].numpy()))
    ys = [float(f.val[0])] + list(float(f.val[0]) + np.cumsum((f.S[0] * f.L[0]).numpy()))
    return np.array(xs), np.array(ys)


def plot_trace(nb, steps, path, ylim=None):
    """One panel per hour: g_t, the windowed V_(t-1), V_t = their sum, its minimum, the chosen p_t, and the
    window the next hour's choice allows."""
    s = nb.slow[0]
    rho, unit = nb.rho[s], nb.units[s].name
    lo_y, top = ylim if ylim is not None else (0, 2.2 * max(float(st["V"].argmin()[1][0]) for st in steps) + 1)
    # x range: where some curve is in view (the curves themselves are drawn exactly, from their breakpoints)
    xs_in = []
    for st in steps:
        for f in (st["g"], st["W"], st["V"]):
            if f is not None:
                x, y = _xy(f)
                fine = np.linspace(x[0], x[-1], 2000)
                xs_in.append(fine[np.interp(fine, x, y) <= top])
    xs_in = np.concatenate(xs_in + [path])
    pad = 0.08 * (xs_in.max() - xs_in.min() + 1)
    fig, axes = plt.subplots(1, nb.B, figsize=(4.2 * nb.B, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, st, p_t in zip(axes, steps, path):
        t = st["t"]
        if t < nb.B:
            nxt = path[t]
            ax.axvspan(nxt - rho, nxt + rho, color="grey", alpha=0.15, lw=0, label="allowed by the next hour (±ρ)")
        ax.plot(*_xy(st["g"]), "--", color="tab:orange", lw=1.6, label="g_t(p): this hour alone")
        if st["W"] is not None:
            ax.plot(*_xy(st["W"]), ":", color="tab:green", lw=1.9, label="min of V_(t-1) within ±ρ")
        ax.plot(*_xy(st["V"]), "-", color="tab:blue", lw=2.2, label="V_t(p) = sum: hours 1..t")
        xm, vm = float(st["V"].argmin()[0][0]), float(st["V"].argmin()[1][0])
        ax.plot([xm], [vm], "o", color="tab:blue", ms=6, label="bottom of V_t")
        ax.axvline(p_t, color="black", lw=1.2, label="chosen p_t")
        ax.text(0.02, 0.97, f"p_{t} = {p_t:.2f}", transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_title(f"hour {t}: residual {nb.r[t - 1]:g} MW", fontsize=11)
        ax.set_xlabel(f"{unit} output p (MW)")
        ax.set_xlim(xs_in.min() - pad, xs_in.max() + pad)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("cost")
    axes[0].set_ylim(lo_y, top)
    handles, labels = {}, {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


def show_steps(nb, steps):
    """The tables behind plot_trace, hour by hour."""
    from IPython.display import display, Markdown
    for st in steps:
        t = st["t"]
        display(Markdown(f"**Hour {t}** · g_{t}(p), the cost of hour {t} alone"))
        display(pwl_table(st["g"], nb.g_labels()))
        if st["W"] is not None:
            display(Markdown(f"min of V_{t - 1} within ±ρ: the bottom of V_{t - 1} stretched flat by 2ρ"))
            display(pwl_table(st["W"]))
        display(Markdown(f"V_{t}(p) = g_{t}(p) + that: the cheapest cost of hours 1..{t} ending at p"))
        display(pwl_table(st["V"]))
