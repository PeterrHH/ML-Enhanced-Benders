"""Ramping blocks for the flow-first surrogate: the block LP, the repair ladder and the block dual.

docs/ramping-surrogate-design.tex, Sections 3-5. With ramping the subproblem unit is a
block of B hours, and the hourly flow-first pipeline knows nothing about ramp rows. This
module works on a batch of K blocks, each with its own investment u_k (a dataset row, or
every block of one sample at one investment), and returns per block

  * a dispatch (p, f, md) that meets every row of the block LP, ramp rows included
    (Proposition 2 of the design note: the ladder always ends feasible), and its cost U;
  * prices and ramp multipliers, completed in closed form into a dual-feasible point,
    so the bound L and the cut slope are valid whatever the network predicted
    (Proposition 1).

The ladder (Section 4.2):
  R1  shrink flows so every node's generation window is non-empty: first only the lines out of an
      over-exporting or into an over-importing node, then Lemma 3's one factor per hour if needed
  --  merit-order fill and hourly prices, as today (nu = 0)
  R2  node-blocks whose sorted dispatch breaks a ramp row (or leaves md outside [0, D]):
      that node's dispatch over the block re-solved as a small LP with the flows fixed
  R3  blocks R2 cannot repair: the exact block LP (r3="lp") or bisection on one scaling
      factor of the block's flows (r3="bisect")
  R4  every row of every block checked; a violation sends the block to the exact LP

With `p_fixed` (Way 2-ii: the network also gives the ramp-limited units' output) those units are
held at the given output through R1 and the fill, which dispatch the other units around them; R2
onwards re-plans every unit of a node it touches, as without.

`run_dp` is Way 2-iii's pipeline, with no LP: R1, then flowfirst/rampdp.py's node-block program (R2's LP
solved exactly by dynamic programming) on every node; a block where some node has no ramp-feasible dispatch
at the given flows gets its flows scaled by the largest factor that works (bisection, zero flows always work).

`escalate` decides which blocks reach R3: "infeasible" only those where fixed flows admit
no ramp-feasible dispatch (the node-block LP needs slack); "unmet" also those where the
R2 repair leaves more unserved demand than the hourly fill had. The default is "infeasible",
so the exact block LP runs only where no dispatch fits the flows.

Shapes: D [K, B, N] demand; apmax [K, B, G] = A * UnitCap (capacity per unit built);
u [K, G]; cap = apmax * u; rho = R_g * u [K, G] the ramp room. Prices are positive (the
economic nodal price); nu_up / nu_dn [K, B-1, G] are >= 0, row j being the edge from hour
j-1 to hour j of the block. Costs are unweighted (pWeight is applied by the Benders code).
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import solver_backend as sb
from gep_problem import ramp_block_rows

R3_MODES = ("lp", "bisect")
ESCALATE_MODES = ("infeasible", "unmet")

# rungs a block can end on; run_dp ends on FILL (the program at the network's flows), SHRINK or R4
RUNG_FILL, RUNG_NODE_LP, RUNG_R3, RUNG_R4, RUNG_SHRINK = 1, 2, 3, 4, 5


def _np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


class BlockSystem:
    """Everything about one network that the block LP and the block dual need, as numpy.

    Build it from the GEP instance (`from_gep`) or from the flow-first operational problem
    plus the ramp limits (`from_operational`); both give the same arrays for the same system.
    """

    def __init__(self, var_cost, voll, node_to_gen, lineflow, f_lb, f_ub, ramp_limits, B, solver="gurobi"):
        self.V = np.asarray(var_cost, dtype=float)
        self.voll = float(voll)
        self.M = np.asarray(node_to_gen, dtype=float)                         # [N, G]
        self.A = np.asarray(lineflow, dtype=float)                            # [N, L], -1 from, +1 to
        self.f_lb, self.f_ub = np.asarray(f_lb, dtype=float), np.asarray(f_ub, dtype=float)
        self.R_g = np.asarray(ramp_limits, dtype=float)
        self.B, self.solver = int(B), solver
        self.N, self.G = self.M.shape
        self.L = self.A.shape[1]
        self.gen_node = self.M.argmax(axis=0)
        self.fr = np.argmax(self.A == -1, axis=0)
        self.to = np.argmax(self.A == 1, axis=0)
        self.gens_at = [np.flatnonzero(self.M[n]) for n in range(self.N)]
        G, L, N, B = self.G, self.L, self.N, self.B
        C = G + L + N
        self.obj_h = np.concatenate([self.V, np.zeros(L), np.full(N, self.voll)])
        # one hour's rows, exactly as gep_benders.SampleStructure lays them out
        Ah = np.zeros((2 * C, C))
        Ah[0:G, 0:G], Ah[G:2 * G, 0:G] = -np.eye(G), np.eye(G)
        Ah[2 * G:2 * G + L, G:G + L], Ah[2 * G + L:2 * G + 2 * L, G:G + L] = -np.eye(L), np.eye(L)
        Ah[2 * G + 2 * L:2 * G + 2 * L + N, G + L:] = -np.eye(N)
        Ah[2 * G + 2 * L + N:, G + L:] = np.eye(N)
        Ae = np.concatenate([self.M, self.A, np.eye(N)], axis=1)
        r, c, v, gen = ramp_block_rows(B, G, C)
        self.n_ramp = 2 * G * (B - 1)
        A_ramp = np.zeros((self.n_ramp, B * C))
        A_ramp[r, c] = v
        self.ramp_gen = np.asarray(gen)
        self.A_ineq_unit = np.vstack([np.kron(np.eye(B), Ah), A_ramp])
        self.A_eq_unit = np.kron(np.eye(B), Ae)
        self.R_rows = 2 * C
        self.opts = sb.solver_options(solver, threads=1, seed=0, lp_optimality_tol=1e-9, feasibility_tol=1e-9)
        self._block_lp = None

    @classmethod
    def from_gep(cls, gep, B, solver="gurobi"):
        return cls([float(gep.pVarCost[g]) for g in gep.G], gep.pVOLL, _np(gep.node_to_gen_mask),
                   _np(gep.lineflow_mask), [-float(gep.pImpCap[l]) for l in gep.L],
                   [float(gep.pExpCap[l]) for l in gep.L], gep.ramp_limits(), B, solver)

    @classmethod
    def from_operational(cls, data, ramp_limits, B, solver="gurobi"):
        from flowfirst.train import line_bounds
        f_lb, f_ub = line_bounds(data)
        return cls(_np(data.cost_vec), data.pVOLL, _np(data.node_to_gen_mask), _np(data.lineflow_mask),
                   _np(f_lb), _np(f_ub), ramp_limits, B, solver)

    # ------------------------------------------------------------------ right-hand sides
    def rho(self, u):
        """Ramp room R_g * u_g, [K, G] for u [K, G] (or [G] -> [G])."""
        return self.R_g * np.asarray(u, dtype=float)

    def b_ineq(self, D, cap):
        """Per-hour inequality rhs [K, B, 2C]: p >= 0, p <= cap, line limits, md >= 0, md <= D."""
        K, B = D.shape[:2]
        z = lambda d: np.zeros((K, B, d))
        lines = lambda x: np.broadcast_to(x, (K, B, self.L))
        return np.concatenate([z(self.G), cap, lines(-self.f_lb), lines(self.f_ub), z(self.N), D], axis=2)

    def ramp_rhs(self, rho):
        """Ramp-row rhs [K, n_ramp] in the row order of ramp_block_rows."""
        return np.asarray(rho)[:, self.ramp_gen]

    # ------------------------------------------------------------------ exact block LP
    def solve_block(self, b_ineq_hours, b_eq_hours, ramp_rhs):
        """The exact block LP, flows free. Returns (Q, p, f, md, lam, nu_up, nu_dn), per hour arrays [B, .]."""
        if self._block_lp is None:
            self._block_lp = sb.MatrixLP(self.solver, np.tile(self.obj_h, self.B), self.A_ineq_unit,
                                         self.A_eq_unit, self.opts)
        B, G, L, N = self.B, self.G, self.L, self.N
        Q, x, duals, _ = self._block_lp.solve(np.concatenate([np.ravel(b_ineq_hours), ramp_rhs]), np.ravel(b_eq_hours))
        x = x.reshape(B, G + L + N)
        R = self.R_rows
        ramp = -duals[B * R:B * R + self.n_ramp]                 # Pi <= 0 on <= rows -> nu >= 0
        half = G * (B - 1)
        lam = duals[B * R + self.n_ramp:].reshape(B, N)          # d(cost)/d(demand): positive price
        return (Q, x[:, :G], x[:, G:G + L], x[:, G + L:], lam,
                np.maximum(ramp[:half], 0).reshape(B - 1, G), np.maximum(ramp[half:], 0).reshape(B - 1, G))

    def exact(self, D, apmax, u, relax_ramps=False):
        """Exact block LP for every block: Q [K], lam [K,B,N], nu_up/nu_dn [K,B-1,G], p, f, md.
        With relax_ramps the ramp rows get a huge rhs: the ramp-free optimum of each block."""
        K, B = D.shape[:2]
        cap = apmax * u[:, None, :]
        bI, rr = self.b_ineq(D, cap), self.ramp_rhs(self.rho(u))
        if relax_ramps:
            rr = np.full_like(rr, 1e9)
        out = dict(Q=np.empty(K), lam=np.empty((K, B, self.N)), nu_up=np.empty((K, B - 1, self.G)),
                   nu_dn=np.empty((K, B - 1, self.G)), p=np.empty((K, B, self.G)), f=np.empty((K, B, self.L)),
                   md=np.empty((K, B, self.N)))
        for k in range(K):
            (out["Q"][k], out["p"][k], out["f"][k], out["md"][k], out["lam"][k],
             out["nu_up"][k], out["nu_dn"][k]) = self.solve_block(bI[k], D[k], rr[k])
        return out

    # ------------------------------------------------------------------ dual
    def dual_bound(self, lam, nu_up, nu_dn, D, apmax, u):
        """Closed-form completion with ramp-adjusted costs (Proposition 1): L [K] and slopes [K, G].

        Valid for any lam and any nu >= 0. The cut is theta_b >= L + slopes @ (u' - u).
        """
        u = np.asarray(u, dtype=float)
        K = lam.shape[0]
        cap, rho = apmax * u[:, None, :], self.rho(u)
        w = nu_up - nu_dn                                                   # row j = edge j-1 -> j
        w_in = np.concatenate([np.zeros((K, 1, self.G)), w], axis=1)        # w of the edge into hour t
        w_out = np.concatenate([w, np.zeros((K, 1, self.G))], axis=1)       # w of the edge out of hour t
        a = self.V[None, None, :] + w_in - w_out                            # adjusted cost
        mu_cap = np.maximum(lam[:, :, self.gen_node] - a, 0.0)
        dl = lam[:, :, self.to] - lam[:, :, self.fr]
        line = np.maximum(dl, 0) * self.f_ub + np.maximum(-dl, 0) * (-self.f_lb)
        loss = np.maximum(lam - self.voll, 0) * D
        ramp_mult = (nu_up + nu_dn).sum(axis=1)                             # [K, G]
        Lb = ((lam * D).sum((1, 2)) - line.sum((1, 2)) - loss.sum((1, 2))
              - (mu_cap * cap).sum((1, 2)) - (ramp_mult * rho).sum(1))
        slopes = -(mu_cap * apmax).sum(axis=1) - ramp_mult * self.R_g[None, :]
        return Lb, slopes

    def ramp_ascent(self, lam, nu_up, nu_dn, apmax, u, sweeps=3):
        """Improve the ramp multipliers for fixed prices, one ramp edge at a time, exactly.

        With lam fixed, L depends on the net multiplier w of edge j -> j+1 of unit g only through
          -cap_j (c1 + w)^+  -  cap_{j+1} (c2 - w)^+  -  rho |w|,
          c1 = lam_j - V - w_into_j,   c2 = lam_{j+1} - V + w_out_of_{j+1},
        a concave piecewise-linear function whose maximum sits at one of w in {0, -c1, c2}.
        Each step can only raise L, and every iterate is a valid (lam, nu). Batched.
        """
        u = np.asarray(u, dtype=float)
        cap, rho = apmax * u[:, None, :], self.rho(u)
        K, B = lam.shape[:2]
        w = nu_up - nu_dn
        pi = lam[:, :, self.gen_node]                                           # [K, B, G]
        V = self.V[None, :]
        for _ in range(sweeps):
            for j in range(B - 1):
                w_in = w[:, j - 1] if j >= 1 else np.zeros((K, self.G))
                w_out_next = w[:, j + 1] if j + 1 < B - 1 else np.zeros((K, self.G))
                c1 = pi[:, j] - V - w_in
                c2 = pi[:, j + 1] - V + w_out_next
                cand = np.stack([np.zeros_like(c1), -c1, c2])                    # [3, K, G]
                val = (-cap[None, :, j] * np.maximum(c1[None] + cand, 0)
                       - cap[None, :, j + 1] * np.maximum(c2[None] - cand, 0) - rho[None] * np.abs(cand))
                best = np.take_along_axis(cand, val.argmax(axis=0)[None], axis=0)[0]
                cur = (-cap[:, j] * np.maximum(c1 + w[:, j], 0) - cap[:, j + 1] * np.maximum(c2 - w[:, j], 0)
                       - rho * np.abs(w[:, j]))
                w[:, j] = np.where(val.max(axis=0) > cur, best, w[:, j])
        return np.maximum(w, 0.0), np.maximum(-w, 0.0)

    def max_violation(self, p, f, md, D, apmax, u):
        """Largest relative violation of any row of the block LP, per block [K]."""
        K = D.shape[0]
        cap = apmax * u[:, None, :]
        y = np.concatenate([p, f, md], axis=2).reshape(K, -1)
        bI = np.concatenate([self.b_ineq(D, cap).reshape(K, -1), self.ramp_rhs(self.rho(u))], axis=1)
        bE = D.reshape(K, -1)
        vI = (y @ self.A_ineq_unit.T - bI) / (1.0 + np.abs(bI))
        vE = np.abs(y @ self.A_eq_unit.T - bE) / (1.0 + np.abs(bE))
        return np.maximum(np.maximum(vI.max(axis=1), vE.max(axis=1)), 0.0)


def block_inputs(D, apmax, u):
    """Hourly network input X [K*B, N+G] = [demand, A*UnitCap*u] for a batch of blocks."""
    K, B = D.shape[:2]
    cap = apmax * np.asarray(u, dtype=float)[:, None, :]
    return torch.tensor(np.concatenate([D, cap], axis=2).reshape(K * B, -1))


def sample_blocks(struct, u):
    """Every block of one gep_benders.SampleStructure sample at one investment u [G]:
    returns D [K, B, N], apmax [K, B, G], u [K, G]."""
    K, B = struct.K, struct.B
    return (struct.b_eq_t.reshape(K, B, -1), struct.apmax.reshape(K, B, -1),
            np.broadcast_to(np.asarray(u, dtype=float), (K, struct.G)).copy())


def network_flows(net, data, X, polish_sweeps=1):
    """Hourly flows of a flow-first network, as the deployed pipeline produces them:
    network, bound repair (inside the network), then `polish_sweeps` line-polish sweeps."""
    from flowfirst.dual import Polish
    with torch.no_grad():
        f = net(X)[2]
        if polish_sweeps:
            f, _ = Polish(data)(X, f, polish_sweeps)
    return f


@dataclass
class BlockResult:
    """Per-block output of the ladder. Arrays are [K, B, .] per hour, [K, B-1, G] per ramp row."""
    p: np.ndarray
    f: np.ndarray
    md: np.ndarray
    lam: np.ndarray
    nu_up: np.ndarray
    nu_dn: np.ndarray
    U: np.ndarray            # [K] cost of the dispatch, an upper bound on Q_b
    L: np.ndarray            # [K] dual objective at u, a lower bound on Q_b
    slopes: np.ndarray       # [K, G] cut slope in u; the cut is theta_b >= L + slopes @ (u' - u)
    rung: np.ndarray         # [K] RUNG_* the block ended on
    broken: np.ndarray       # [K, N] node-blocks sent to R2
    stats: dict = field(default_factory=dict)


class BlockLadder:
    """The repair ladder on top of a BlockSystem.

    Args:
        system: the BlockSystem (costs, matrices, ramp limits, block length).
        data: the flow-first operational problem, for the merit-order fill and the hourly
            price recovery (the same network as `system`).
        r3: "lp" or "bisect", see the module docstring.
        escalate: "infeasible" or "unmet", see the module docstring.
        price_tol: congestion tolerance of the hourly price recovery (DualRecovery).
    """

    def __init__(self, system, data, r3="lp", escalate="infeasible", price_tol=1e-2, bisect_steps=20):
        from flowfirst.dual import DualRecovery
        from flowfirst.fill import MeritOrderFill
        if r3 not in R3_MODES:
            raise ValueError(f"r3 must be one of {R3_MODES}, got {r3!r}")
        if escalate not in ESCALATE_MODES:
            raise ValueError(f"escalate must be one of {ESCALATE_MODES}, got {escalate!r}")
        self.sys, self.data = system, data
        self.r3, self.escalate, self.bisect_steps = r3, escalate, bisect_steps
        self.fill = MeritOrderFill(data)
        self.recovery = DualRecovery(data, tol=price_tol)
        self._node_lp = {}
        self.big = 1e3 * system.voll        # cost of the node-block LP's slack; >> VOLL

    # ------------------------------------------------------------------ node-block LP
    def _node_template(self, n):
        """Node-block LP of node n over B hours, flows fixed (eq. (9) of the design note).

        Variables per hour: p of n's units, md, s_plus, s_minus. Rows per hour: p >= 0,
        p <= cap, md >= 0, md <= D, s_plus >= 0, s_minus >= 0; then the ramp rows (all up
        rows, then all down rows, each ordered (hour, unit)); balance per hour.
        """
        if n in self._node_lp:
            return self._node_lp[n]
        s = self.sys
        gens, B = s.gens_at[n], s.B
        m = len(gens)
        c = m + 3
        Ah = np.zeros((2 * m + 4, c))
        Ah[:m, :m], Ah[m:2 * m, :m] = -np.eye(m), np.eye(m)
        Ah[2 * m, m], Ah[2 * m + 1, m] = -1.0, 1.0
        Ah[2 * m + 2, m + 1], Ah[2 * m + 3, m + 2] = -1.0, -1.0
        A_ramp = np.zeros((2 * m * (B - 1), B * c))
        half = m * (B - 1)
        for j in range(1, B):
            for i in range(m):
                r = (j - 1) * m + i
                A_ramp[r, j * c + i], A_ramp[r, (j - 1) * c + i] = 1.0, -1.0
                A_ramp[half + r, j * c + i], A_ramp[half + r, (j - 1) * c + i] = -1.0, 1.0
        A_ineq = np.vstack([np.kron(np.eye(B), Ah), A_ramp])
        Ae = np.zeros((1, c))
        Ae[0, :m], Ae[0, m], Ae[0, m + 1], Ae[0, m + 2] = 1.0, 1.0, 1.0, -1.0
        A_eq = np.kron(np.eye(B), Ae)
        obj = np.tile(np.concatenate([s.V[gens], [s.voll, self.big, self.big]]), B)
        lp = sb.MatrixLP(s.solver, obj, A_ineq, A_eq, s.opts)
        self._node_lp[n] = (lp, m, c)
        return self._node_lp[n]

    def solve_node_block(self, n, r, cap_n, D, rho_n):
        """Node n over one block with the flows fixed. r, D: [B]; cap_n: [B, |G_n|]; rho_n: [|G_n|].

        Returns (feasible, p [B, |G_n|], md [B], lam [B], nu_up [B-1, |G_n|], nu_dn [B-1, |G_n|]).
        """
        lp, m, c = self._node_template(n)
        B = self.sys.B
        rows_h = np.concatenate([np.zeros((B, m)), cap_n, np.zeros((B, 1)), D[:, None], np.zeros((B, 2))], axis=1)
        b_ineq = np.concatenate([rows_h.ravel(), np.tile(rho_n, 2 * (B - 1))])
        _, x, duals, _ = lp.solve(b_ineq, r)
        x = x.reshape(B, c)
        n_ineq = B * (2 * m + 4)
        half = m * (B - 1)
        ramp = -duals[n_ineq:n_ineq + 2 * half]
        slack = np.abs(x[:, m + 1]) + np.abs(x[:, m + 2])
        feasible = bool(np.all(slack <= 1e-7 * (1.0 + np.abs(r))))
        return (feasible, x[:, :m], x[:, m], duals[n_ineq + 2 * half:],
                np.maximum(ramp[:half], 0).reshape(B - 1, m), np.maximum(ramp[half:], 0).reshape(B - 1, m))

    # ------------------------------------------------------------------ ladder
    def window_scaling(self, D, cap, f, sweeps=10):
        """R1 (Lemma 3): shrink flows until every node's net import lies in [-capacity, demand], so
        its generation window is non-empty. Torch, [T, .] in, (alpha [T], f [T, L]) out.

        First, targeted: a node exporting more than its capacity shrinks the lines carrying flow
        out of it, a node importing more than its demand the lines carrying flow into it, each by
        the factor that removes its excess; a line takes the smaller factor of its two ends.
        Repeated for `sweeps` sweeps, since shrinking a line moves its other end. The hourly fill
        prices unmet demand at VOLL wherever it sits, so a network may route power out of a node
        with no capacity (moving unmet demand around for free); scaling every line of the hour by
        one alpha, as Lemma 3 does, would then throw away the useful flows too.
        Then, for hours still outside their windows, Lemma 3's single alpha <= 1 for the hour,
        which always exists (alpha = 0). Every factor lies in [0, 1], so flows stay within their
        line limits. alpha is that last factor (1 where the targeted pass sufficed).
        """
        A = self.data.lineflow_mask.to(f.dtype)                             # [N, L], -1 from, +1 to
        C = cap @ self.data.node_to_gen_mask.T.to(f.dtype)                  # node capacity [T, N]
        frm, to = (A == -1).to(f.dtype), (A == 1).to(f.dtype)               # [N, L]
        for _ in range(sweeps):
            imp = f @ A.T                                                   # net import [T, N]
            ex_out = torch.relu(-imp - C)                                   # export beyond capacity
            ex_in = torch.relu(imp - D)                                     # import beyond demand
            if not (ex_out.gt(1e-9 * (1 + C)).any() or ex_in.gt(1e-9 * (1 + D)).any()):
                break
            pos, neg = torch.relu(f), torch.relu(-f)                        # flow along / against each line
            out_n = pos @ frm.T + neg @ to.T                                # flow leaving each node [T, N]
            in_n = pos @ to.T + neg @ frm.T                                 # flow entering each node
            k_out = 1 - (ex_out / out_n.clamp(min=1e-300)).clamp(max=1)     # factor for lines leaving n
            k_in = 1 - (ex_in / in_n.clamp(min=1e-300)).clamp(max=1)        # factor for lines entering n
            # sender and receiver of each line, by the sign of its flow
            send = torch.where(f >= 0, k_out @ frm, k_out @ to)
            recv = torch.where(f >= 0, k_in @ to, k_in @ frm)
            f = f * torch.minimum(send, recv)
        imp = f @ A.T
        ratio = torch.full_like(imp, float("inf"))
        ratio = torch.where(imp > 0, D / imp.clamp(min=1e-300), ratio)
        ratio = torch.where(imp < 0, C / (-imp).clamp(min=1e-300), ratio)
        alpha = ratio.min(dim=1).values.clamp(max=1.0)
        return alpha, f * alpha[:, None]

    def run(self, D, apmax, u, f, p_fixed=None, fixed_gens=None, tol=1e-7):
        """Run the ladder on K blocks. D [K, B, N], apmax [K, B, G], u [K, G]; f [K*B, L] hourly flows.

        p_fixed [K, B, G] (zero outside `fixed_gens`) holds those units at that output in R1 and the
        fill: a node's demand left to the fill is D minus its fixed output, and R1 counts the fixed
        output as generation the node has (it can export it), so it shrinks only imports above the
        demand left, floored at 0. A fixed output above that demand gives md < 0, which the check
        sends to R2. The ramp check covers the fixed units too.
        """
        t_start = time.perf_counter()
        s = self.sys
        B, N, G, L = s.B, s.N, s.G, s.L
        K = D.shape[0]
        u = np.asarray(u, dtype=float)
        rho = s.rho(u)
        cap = apmax * u[:, None, :]
        times = {}

        # R1
        t0 = time.perf_counter()
        D_t = torch.tensor(D.reshape(K * B, N))
        cap_t = torch.tensor(cap.reshape(K * B, G))
        X = torch.cat([D_t, cap_t], dim=1)
        f0 = torch.as_tensor(f, dtype=D_t.dtype)
        if p_fixed is None:
            p_fix, D_fill, cap_fill = torch.zeros_like(cap_t), D_t, cap_t
        else:
            p_fix = torch.tensor(np.asarray(p_fixed, dtype=float).reshape(K * B, G))
            free = torch.ones(G, dtype=D_t.dtype)
            free[torch.as_tensor(fixed_gens)] = 0.0
            cap_fill = cap_t * free
            D_fill = D_t - p_fix @ self.data.node_to_gen_mask.T.to(D_t.dtype)
        alpha, f1 = self.window_scaling(D_fill.clamp(min=0.0), cap_fill + p_fix, f0)
        r1_hours = int((f1 != f0).any(dim=1).sum())
        times["r1"] = time.perf_counter() - t0

        # fill and hourly prices, nu = 0 (today's pass)
        t0 = time.perf_counter()
        with torch.no_grad():
            p_t, md_t, _ = self.fill(D_fill, cap_fill, f1)
            p_t = p_t + p_fix
            lam_t = self.recovery.prices(X, f1)
        times["fill_prices"] = time.perf_counter() - t0
        rs = lambda z, d: _np(z).reshape(K, B, d).astype(float)
        p, md, lam, ff = rs(p_t, G), rs(md_t, N), rs(lam_t, N), rs(f1, L)
        imp_k = rs(f1 @ self.data.lineflow_mask.T.to(f1.dtype), N)
        nu_up = np.zeros((K, B - 1, G))
        nu_dn = np.zeros((K, B - 1, G))
        md_hourly, lam_hourly = md.copy(), lam.copy()

        # level A for reference: the hourly answer as it stands, nu = 0 (may break ramp rows)
        U_hourly = (p @ s.V).sum(1) + s.voll * np.abs(md).sum((1, 2))
        L_hourly, slopes_hourly = s.dual_bound(lam, nu_up, nu_dn, D, apmax, u)

        # ramp check
        t0 = time.perf_counter()
        dp = np.abs(np.diff(p, axis=1)) - rho[:, None, :]
        gen_bad = (dp > tol * (1.0 + rho[:, None, :])).any(axis=1)                   # [K, G]
        md_bad = ((md < -tol * (1 + D)) | (md > D + tol * (1 + D))).any(axis=1)     # [K, N]
        broken = ((gen_bad.astype(float) @ s.M.T) > 0) | md_bad
        rung = np.full(K, RUNG_FILL)
        times["check"] = time.perf_counter() - t0

        # R2: node-block LPs, flows fixed
        t0 = time.perf_counter()
        to_r3 = np.zeros(K, dtype=bool)
        n_r2 = 0
        for k, n in zip(*np.nonzero(broken)):
            n_r2 += 1
            gens = s.gens_at[n]
            ok, pn, mdn, lamn, nuu, nud = self.solve_node_block(
                n, D[k, :, n] - imp_k[k, :, n], cap[k][:, gens], D[k, :, n], rho[k, gens])
            if not ok or (self.escalate == "unmet" and mdn.sum() > md_hourly[k, :, n].sum() + tol * (1 + D[k, :, n].sum())):
                to_r3[k] = True
                continue
            p[k][:, gens], md[k, :, n], lam[k, :, n] = pn, mdn, lamn
            nu_up[k][:, gens], nu_dn[k][:, gens] = nuu, nud
            rung[k] = max(rung[k], RUNG_NODE_LP)
        times["r2"] = time.perf_counter() - t0

        # R3: free the flows
        t0 = time.perf_counter()
        bI = s.b_ineq(D, cap)
        rr = s.ramp_rhs(rho)
        alphas_r3 = []
        for k in np.flatnonzero(to_r3):
            if self.r3 == "lp":
                _, p[k], ff[k], md[k], lam[k], nu_up[k], nu_dn[k] = s.solve_block(bI[k], D[k], rr[k])
            else:
                a_k = self._bisect_block(k, D, cap, imp_k, rho, p, md, lam, nu_up, nu_dn)
                ff[k] *= a_k
                alphas_r3.append(a_k)
            rung[k] = RUNG_R3
        times["r3"] = time.perf_counter() - t0

        # R4: every row of every block
        t0 = time.perf_counter()
        viol = s.max_violation(p, ff, md, D, apmax, u)
        bad = viol > 1e-6
        for k in np.flatnonzero(bad):
            _, p[k], ff[k], md[k], lam[k], nu_up[k], nu_dn[k] = s.solve_block(bI[k], D[k], rr[k])
            rung[k] = RUNG_R4
        if bad.any():
            viol = s.max_violation(p, ff, md, D, apmax, u)
        times["r4"] = time.perf_counter() - t0

        # cost, and the best of two valid dual points per block (Proposition 1 holds for any
        # prices and nu >= 0, so the larger bound is valid too):
        #   "hourly":   the hourly prices from the flows, nu by ramp_ascent from 0
        #   "repaired": prices and nu after R2-R4 (LP duals where an LP ran), nu improved by ramp_ascent
        t0 = time.perf_counter()
        U = (p @ s.V).sum(1) + s.voll * md.sum((1, 2))
        zero = np.zeros((K, B - 1, G))
        cands = []
        for lam_c, nu_u, nu_d in ((lam_hourly, zero, zero), (lam, nu_up, nu_dn)):
            nu_u, nu_d = s.ramp_ascent(lam_c, nu_u.copy(), nu_d.copy(), apmax, u)
            Lc, sc = s.dual_bound(lam_c, nu_u, nu_d, D, apmax, u)
            cands.append((Lc, sc, lam_c, nu_u, nu_d))
        pick = cands[1][0] > cands[0][0]                                    # True: the repaired point
        sel = lambda i: np.where(pick.reshape((-1,) + (1,) * (cands[0][i].ndim - 1)), cands[1][i], cands[0][i])
        Lb, slopes, lam_out, nu_up_out, nu_dn_out = (sel(i) for i in range(5))
        times["dual"] = time.perf_counter() - t0
        times["total"] = time.perf_counter() - t_start
        stats = dict(times=times, n_node_blocks=K * N, n_r2=n_r2, n_r3=int(to_r3.sum()), n_r4=int(bad.sum()),
                     alpha_r1_min=float(alpha.min()), n_hours_scaled=int((alpha < 1).sum()), n_hours_r1=r1_hours,
                     alpha_r3=alphas_r3, max_rel_violation=float(viol.max()), share_dual_repaired=float(pick.mean()),
                     L_hourly_ascent=cands[0][0], U_hourly=U_hourly, L_hourly=L_hourly,
                     slopes_hourly=slopes_hourly, md_hourly=md_hourly.sum((1, 2)))
        return BlockResult(p, ff, md, lam_out, nu_up_out, nu_dn_out, U, Lb, slopes, rung, broken, stats)

    def run_dp(self, D, apmax, u, f, dp, shrink_steps=20, tol=1e-6):
        """Way 2-iii's pipeline on K blocks, with no LP. D [K, B, N], apmax [K, B, G], u [K, G]; f [K*B, L]
        hourly flows; dp a flowfirst.rampdp.RampDP of the same network.

          R1      as in `run`
          dp      every node's best ramp-feasible dispatch over the block at these flows (hard mode: md in
                  [0, D]), exact for at most one ramp-limited unit per node
          shrink  blocks where a node has none: the block's flows times the largest alpha in [0, 1] (to
                  shrink_steps halvings) at which every node has one; alpha = 0 always works (each node an
                  island with r = D: every ramp-limited unit at 0 is a valid path)
          R4      every row checked; a violation (numerical only) falls back to zero flows
          cut     the better of two dual points, as in `run`: the hourly prices from the flows, and the
                  program's prices; each with ramp multipliers by ramp_ascent
        """
        t_start = time.perf_counter()
        s = self.sys
        B, N, G, L = s.B, s.N, s.G, s.L
        K = D.shape[0]
        u = np.asarray(u, dtype=float)
        rho = s.rho(u)
        cap = apmax * u[:, None, :]
        times = {}
        A = self.data.lineflow_mask
        tt = lambda z: torch.as_tensor(np.asarray(z, dtype=float))
        D_k, cap_k, rho_k = tt(D), tt(cap), tt(rho)

        t0 = time.perf_counter()
        D_t, cap_t = D_k.reshape(K * B, N), cap_k.reshape(K * B, G)
        X = torch.cat([D_t, cap_t], dim=1)
        f0 = torch.as_tensor(f, dtype=D_t.dtype)
        alpha_r1, f1 = self.window_scaling(D_t, cap_t, f0)
        r1_changed = (f1 != f0).any(dim=1)
        r1_hours, r1_blocks = int(r1_changed.sum()), int(r1_changed.reshape(K, B).any(dim=1).sum())
        times["r1"] = time.perf_counter() - t0

        # level A for reference, as in `run`: the hourly fill and prices on the flows (may break ramp rows)
        with torch.no_grad():
            p_h, md_h, _ = self.fill(D_t, cap_t, f1)
            lam_hourly = _np(self.recovery.prices(X, f1)).reshape(K, B, N)
        p_h, md_h = _np(p_h).reshape(K, B, G), _np(md_h).reshape(K, B, N)
        zero = np.zeros((K, B - 1, G))
        U_hourly = (p_h @ s.V).sum(1) + s.voll * np.abs(md_h).sum((1, 2))
        L_hourly, slopes_hourly = s.dual_bound(lam_hourly, zero, zero, D, apmax, u)

        t0 = time.perf_counter()
        fk = f1.reshape(K, B, L)
        imp = fk @ A.T.to(fk.dtype)                                          # [K, B, N]
        with torch.no_grad():
            out = dp(D_k - imp, cap_k, rho_k, D_k, hard=True)
        p, md = out["p"].clone(), out["md"].clone()
        broken = ~out["node_ok"].numpy()
        rung = np.full(K, RUNG_FILL)
        times["dp"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        bad = torch.nonzero(~out["ok"]).flatten()
        alpha = torch.ones(K, dtype=fk.dtype)
        if len(bad):
            Db, cb, rb, ib = D_k[bad], cap_k[bad], rho_k[bad], imp[bad]
            ok_at = lambda a: dp(Db - a[:, None, None] * ib, cb, rb, Db, hard=True)
            lo, hi = torch.zeros(len(bad), dtype=fk.dtype), torch.ones(len(bad), dtype=fk.dtype)
            with torch.no_grad():
                if not bool(ok_at(lo)["ok"].all()):
                    raise RuntimeError("node-block programs infeasible with zero flows; this contradicts Lemma 1")
                for _ in range(shrink_steps):
                    mid = 0.5 * (lo + hi)
                    good = ok_at(mid)["ok"]
                    lo, hi = torch.where(good, mid, lo), torch.where(good, hi, mid)
                res = ok_at(lo)
            p[bad], md[bad], alpha[bad] = res["p"], res["md"], lo
            rung[bad.numpy()] = RUNG_SHRINK
        ff = (fk * alpha[:, None, None]).numpy()
        times["shrink"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        p, md = p.numpy(), md.numpy()
        viol = s.max_violation(p, ff, md, D, apmax, u)
        bad4 = np.flatnonzero(viol > tol)
        if len(bad4):
            with torch.no_grad():
                z = dp(D_k[bad4], cap_k[bad4], rho_k[bad4], D_k[bad4], hard=True)
            p[bad4], md[bad4], ff[bad4] = z["p"].numpy(), z["md"].numpy(), 0.0
            rung[bad4] = RUNG_R4
            viol = s.max_violation(p, ff, md, D, apmax, u)
        times["r4"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        U = (p @ s.V).sum(1) + s.voll * md.sum((1, 2))
        f_fin = torch.as_tensor(ff)
        lam_flows = _np(self.recovery.prices(X, f_fin.reshape(K * B, L))).reshape(K, B, N)
        lam_dp = dp.prices(D_k - f_fin @ A.T.to(f_fin.dtype), cap_k, rho_k, D_k).numpy()
        cands = []
        for lam_c in (lam_flows, lam_dp):
            nu_u, nu_d = s.ramp_ascent(lam_c, zero.copy(), zero.copy(), apmax, u)
            Lc, sc = s.dual_bound(lam_c, nu_u, nu_d, D, apmax, u)
            cands.append((Lc, sc, lam_c, nu_u, nu_d))
        pick = cands[1][0] > cands[0][0]                                    # True: the program's prices
        sel = lambda i: np.where(pick.reshape((-1,) + (1,) * (cands[0][i].ndim - 1)), cands[1][i], cands[0][i])
        Lb, slopes, lam_out, nu_up_out, nu_dn_out = (sel(i) for i in range(5))
        times["dual"] = time.perf_counter() - t0
        times["total"] = time.perf_counter() - t_start
        stats = dict(times=times, n_node_blocks=K * N, n_r2=0, n_r3=int(len(bad)), n_r4=int(len(bad4)),
                     n_shrink=int(len(bad)), alpha_shrink=alpha[bad].tolist() if len(bad) else [],
                     alpha_r1_min=float(alpha_r1.min()), n_hours_scaled=int((alpha_r1 < 1).sum()), n_hours_r1=r1_hours,
                     n_blocks_r1=r1_blocks, max_rel_violation=float(viol.max()), share_dual_repaired=float(pick.mean()),
                     L_hourly_ascent=cands[0][0], U_hourly=U_hourly, L_hourly=L_hourly,
                     slopes_hourly=slopes_hourly, md_hourly=md_h.sum((1, 2)))
        return BlockResult(p, ff, md, lam_out, nu_up_out, nu_dn_out, U, Lb, slopes, rung, broken, stats)

    def _bisect_block(self, k, D, cap, imp_k, rho, p, md, lam, nu_up, nu_dn):
        """R3 without a full LP: the largest alpha in [0, 1] (to bisect_steps halvings) at which
        every node of block k has a ramp-feasible dispatch with the block's flows scaled by
        alpha. alpha = 0 always works (proof of Proposition 2). Writes that dispatch and its
        duals into the arrays and returns alpha."""
        s = self.sys

        def solve_all(a):
            out = []
            for n in range(s.N):
                gens = s.gens_at[n]
                res = self.solve_node_block(n, D[k, :, n] - a * imp_k[k, :, n], cap[k][:, gens], D[k, :, n], rho[k, gens])
                if not res[0]:
                    return None
                out.append((n, gens) + res[1:])
            return out

        lo, hi = 0.0, 1.0
        best = solve_all(0.0)
        if best is None:
            raise RuntimeError("node-block LPs infeasible at alpha = 0; this contradicts Lemma 1")
        for _ in range(self.bisect_steps):
            mid = 0.5 * (lo + hi)
            res = solve_all(mid)
            if res is None:
                hi = mid
            else:
                lo, best = mid, res
        for n, gens, pn, mdn, lamn, nuu, nud in best:
            p[k][:, gens], md[k, :, n], lam[k, :, n] = pn, mdn, lamn
            nu_up[k][:, gens], nu_dn[k][:, gens] = nuu, nud
        return lo
