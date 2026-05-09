import os
import time
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from data_wrangling import dataframe_to_dict
from gep_config_parser import parse_config
from gep_problem import GEPProblemSet
from scipy.stats import qmc

class GEPOperationalProblemSet():

    def __init__(self, args, T, N, G, L, pDemand, pGenAva, pVOLL, pWeight, pRamping, pInvCost, pVarCost, pUnitCap, pExpCap, pImpCap, pUnitInvestment_Input = None):
        
        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("cpu")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Args:
        self.args = args
        self.ED_args = args["ED_args"]
        # Operation sample is always of length 1 for benders!
        self.sample_duration = 1

        # Number of samples to generate (through randomly chosen investment variables)
        self.n_samples = 2 ** self.ED_args["2n_synthetic_samples"]

        # Input Sets
        self.T = T
        self.G = G
        self.L = L
        self.N = N

        self.num_g = len(G)
        self.num_l = len(L)
        self.num_n = len(N)

        # Input Parameters (Dictionaries)
        self.pDemand = pDemand # MW
        self.pGenAva = pGenAva # [0,1]
        self.pVOLL = pVOLL # $/MW
        self.pWeight = pWeight # Integer
        self.pRamping = pRamping # [0,1]
        self.pInvCost = pInvCost # $/MW
        self.pVarCost = pVarCost # $/MW
        self.pUnitCap = pUnitCap # MW
        self.pExpCap = pExpCap # MW
        self.pImpCap = pImpCap # MW 


        # Value can be None (we don't perturb) or a percentage (we perturb by that percentage)
        if self.ED_args['perturb_operating_costs']:
            self.pVarCost = self.perturb_operating_costs(self.ED_args['perturb_operating_costs'])
        
        # Number variables, variables are [p_g, f_l, md_n]
        self.n_vars = self.num_g + self.num_l + self.num_n

        # Number of timesteps per data sample
        assert (float(len(T)) / self.sample_duration).is_integer() # Total number of timesteps should be divisible by the number of timesteps per data sample.

        # Time slices, each slice makes a different sample.
        self.time_ranges = [range(i, i + self.sample_duration, 1) for i in range(1, len(T), self.sample_duration)]

        self.neq = self.num_n
        self.nineq = 2 * (self.num_g + self.num_l + self.num_n) # 2 inequality each for power_loss per node, flow per line, production per tech
        self.n_prod_vars = self.num_g * self.sample_duration
        self.n_line_vars = self.num_l * self.sample_duration
        self.n_md_vars = self.num_n * self.sample_duration
        self.ydim = self.n_prod_vars + self.n_line_vars + self.n_md_vars

        # Generate unit investment data:
        if pUnitInvestment_Input is not None:
            print("Using provided unit investment data with shape:", pUnitInvestment_Input.shape)
            self.pUnitInvestment = pUnitInvestment_Input
        else:
            if self.ED_args.get("generate_capacity_sobol", False):
                self.pUnitInvestment = None
            else:
                if args["ED_args"].get("gen_data_constraint", False):
                    if args["ED_args"].get("gen_data_node_constraint", False):
                        self.pUnitInvestment = self.generate_ui_data_node_constraint(
                            m=self.ED_args["2n_synthetic_samples"],
                            max_inv=self.ED_args["max_investment"],
                            renewable_ava_percentile=self.args["ED_args"].get("gen_data_renew_availability", 90),
                        )
                    else:
                        self.pUnitInvestment = self.generate_ui_data_constraint(
                            m=self.ED_args["2n_synthetic_samples"],
                            max_inv=self.ED_args["max_investment"],
                            lb_activated=self.args["ED_args"].get("gen_data_constraint_lb", False),
                            alpha=self.args["ED_args"].get("gen_data_constraint_lb_alpha", 0.2),
                            renewable_lb_zero=self.args["ED_args"].get("gen_data_constraint_lb_renewable_zero", True),
                            renewable_ub_max_inv=self.args["ED_args"].get("gen_data_constraint_ub_renewable_max_inv", True),
                            renewable_ava_percentile=self.args["ED_args"].get("gen_data_renew_availability", 100),
                        )
                else:
                    self.pUnitInvestment = self.generate_ui_data(
                        m=self.ED_args["2n_synthetic_samples"],
                        max_inv=self.ED_args["max_investment"]
                    )

        # Masks for node balance
        # Initialize mask, to store which generator belong to which node
        self.node_to_gen_mask = torch.zeros((len(N), len(G)), dtype=self.DTYPE)

        # Populate mask
        for g_idx, (node, _) in enumerate(G):
            node_idx = N.index(node)
            self.node_to_gen_mask[node_idx, g_idx] = 1
        
        # Initialize mask
        self.lineflow_mask = torch.zeros((len(N), len(L)), dtype=self.DTYPE)

        # Populate mask (directed adjacency matrix), -1 where line starts, 1 where line stops
        for l_idx, (start_node, end_node) in enumerate(L):
            start_idx = N.index(start_node)
            end_idx = N.index(end_node)
            self.lineflow_mask[start_idx, l_idx] = -1
            self.lineflow_mask[end_idx, l_idx] = 1
        
        # Variable within an example data is [G,L,N] per timestep
        self.md_indices = self.get_md_nt_indices()
        self.f_lt_indices = self.get_f_lt_indices()

        self.capacity_ub_indices = self.get_capacity_ub_indices()
        self.missed_demand_ub_indices = self.get_missed_demand_ub_indices()

        # Create constraint matrices, rhs and obj_fns
        print("Populating ineq constraints")
        self.ineq_cm, self.ineq_rhs = self.build_ineq_cm_rhs()
        print("Populating eq constraints")
        self.eq_cm = self.build_eq_cm()
        print("Creating objective coefficients")
        self.obj_coeff, self.cost_vec = self.build_obj_coeff()
        if self.ED_args["synthetic_demand_capacity"]:
            self.X = self.build_synthetic_X()
        elif self.ED_args.get("generate_capacity_sobol", False):
            self.X = self.build_X_capacity_sobol()
        else:
            self.X = self.build_X()
        
        if self.ED_args["normalize_input"]:
            self.total_demands = self.X[:, :self.data.num_n].sum(dim=1).unsqueeze(1)
            self.X /= self.total_demands
        else:
            self.total_demands = torch.ones((self.X.shape[0], 1))

        self.xdim = self.X.shape[1]

        self.opt_targets = self.compute_opt_targets() # Solve for the opt target

        # if self.args["device"] == 'mps':
            # self.obj_coeff = self.obj_coeff.to(torch.float32).to(torch.device('mps'))
            # self.cost_vec = self.cost_vec.to(torch.float32).to(torch.device('mps'))
            # self.node_to_gen_mask = self.node_to_gen_mask.to(torch.float32).to(torch.device('mps'))
            # self.lineflow_mask = self.lineflow_mask.to(torch.float32).to(torch.device('mps'))
            
    def to_mps(self):
        print("Converting to mps")
        self.obj_coeff = self.obj_coeff.to(torch.float32).to(torch.device('mps'))
        self.cost_vec = self.cost_vec.to(torch.float32).to(torch.device('mps'))
        self.node_to_gen_mask = self.node_to_gen_mask.to(torch.float32).to(torch.device('mps'))
        self.lineflow_mask = self.lineflow_mask.to(torch.float32).to(torch.device('mps'))
        self.ineq_rhs = self.ineq_rhs.to(torch.float32).to(torch.device('mps'))
        self.ineq_cm = self.ineq_cm.to(torch.float32).to(torch.device('mps'))
        self.eq_cm = self.eq_cm.to(torch.float32).to(torch.device('mps'))

    def load_targets(self, target_path):
        with open(target_path, 'rb') as file:
            return pickle.load(file)
    
    def save_data(self, save_path):
        with open(save_path, 'wb') as file:
            pickle.dump(self, file)

    def build_X_capacity_sobol(self):
        """
        Build X using:
        - historical demand vectors (Choice A)
        - Sobol-sampled generator upper bounds directly

        X = [D_1, ..., D_N, cap_ub_1, ..., cap_ub_G]
        """
        self.capacity_samples = self.generate_capacity_data_constraint(
            m=self.ED_args["2n_synthetic_samples"],
            beta=self.ED_args.get("gen_data_beta", 1.0),
        )

        X = []
        for i in range(self.n_samples):
            t = i % len(self.T) + 1

            Xi = []
            # historical nodal demand vector
            for n in self.N:
                Xi.append(float(self.pDemand[(n, t)]))

            # sampled ED capacity upper bounds directly
            Xi.extend(self.capacity_samples[i].tolist())
            X.append(Xi)

        X = torch.tensor(np.array(X), dtype=self.DTYPE)

        perm = torch.randperm(X.shape[0])
        X = X[perm]
        self.capacity_samples = self.capacity_samples[perm]

        return X

    def generate_capacity_data_constraint(
        self,
        m=15,
        beta=1.2,
    ):
        """
        Generate Sobol samples directly over ED generator upper bounds (MW),
        using the same constraint-aware node-based logic.

        Returns
        -------
        cap_samples : torch.Tensor [n_samples, num_g]
            Direct generator production upper bounds for ED.
        """
        dimensions = self.num_g
        sobol_sampler = qmc.Sobol(d=dimensions)
        base_points = sobol_sampler.random_base2(m=m)
        np.random.shuffle(base_points)

        dtype_np = np.float32 if self.args["device"] == "mps" else np.float64
        points01 = torch.tensor(base_points.astype(dtype_np), dtype=self.DTYPE)

        lb_g = torch.zeros(self.num_g, dtype=self.DTYPE)
        ub_g = torch.zeros(self.num_g, dtype=self.DTYPE)

        # max demand per node
        node_max_demand = {}
        for n in self.N:
            node_max_demand[n] = max(float(self.pDemand[(n, t)]) for t in self.T)

        # export/import capability proxy per node
        node_export_cap = {n: 0.0 for n in self.N}
        for (start_node, end_node) in self.L:
            node_export_cap[start_node] += float(self.pExpCap[(start_node, end_node)])
            node_export_cap[end_node] += float(self.pImpCap[(start_node, end_node)])

        for g_idx, g in enumerate(self.G):
            n, _ = g
            pmax = float(self.pUnitCap[g])

            # same constraint-aware logic, now directly in MW cap-space
            cap_target = (float(node_max_demand[n]) + float(node_export_cap[n])) * beta

            # hard cap from max investment times unit size
            hard_cap = float(self.ED_args["max_investment"]) * pmax

            lb_g[g_idx] = 0.0
            if self.ED_args.get("capacity_sobol_use_hard_cap", False):
                ub_g[g_idx] = min(cap_target, hard_cap)
            else:
                ub_g[g_idx] = cap_target

        cap_samples = lb_g.unsqueeze(0) + points01 * (ub_g - lb_g).unsqueeze(0)
        cap_samples = torch.clamp(cap_samples, min=lb_g.unsqueeze(0), max=ub_g.unsqueeze(0))

        return cap_samples

    def generate_ui_data_node_constraint(
        self,
        m=15,
        max_inv=1000.0,
        renewable_ava_percentile=90,
        r_choices=None,
        r_probs=None,
        alpha_choices=(0.2, 1.0, 5.0),
    ):
        """
        Node-level effective-capacity investment sampling.

        For each node n:
            sum_g q_g * pUnitCap_g * u_g ~= r_n * C_n

        where:
            C_n = max_t D_n,t + export capacity from node n
            q_g = 1 for dispatchable generators
            q_g = renewable availability percentile for renewables

        The node budget is then split across local generators using Dirichlet sampling.
        """

        def _is_renewable(g):
            _, tech = g
            return str(tech).lower() in {"sunpv", "windon", "windoff"}

        if r_choices is None:
            r_choices = np.array([0.5, 0.8, 0.95, 1.0, 1.1, 1.4, 1.8])

        if r_probs is None:
            r_probs = np.array([0.05, 0.10, 0.20, 0.25, 0.20, 0.12, 0.08])

        # Number of investment samples
        n_samples = 2 ** m
        all_points = torch.zeros((n_samples, self.num_g), dtype=self.DTYPE)

        # Node capacity bound C_n
        node_C = {}
        for n in self.N:
            max_demand = max(float(self.pDemand[(n, t)]) for t in self.T)

            export_cap = 0.0
            for l in self.L:
                start_node, end_node = l
                if start_node == n:
                    export_cap += float(self.pExpCap[l])
                elif end_node == n:
                    export_cap += float(self.pImpCap[l])

            node_C[n] = max_demand + export_cap

        # Effective availability q_g
        q_g = {}
        for g in self.G:
            if _is_renewable(g):
                vals = [float(self.pGenAva.get((*g, t), 1.0)) for t in self.T]
                nonzero_vals = [v for v in vals if v > 0]

                if len(nonzero_vals) > 0:
                    q_g[g] = max(float(np.percentile(nonzero_vals, renewable_ava_percentile)), 1e-6)
                else:
                    q_g[g] = 1.0
            else:
                q_g[g] = 1.0

        # Sample node budgets and split locally
        for i in range(n_samples):
            inv = {}

            for n in self.N:
                gens_at_node = [g for g in self.G if g[0] == n]

                if len(gens_at_node) == 0:
                    continue

                r_n = np.random.choice(r_choices, p=r_probs)
                E_n = r_n * node_C[n]

                alpha = np.random.choice(alpha_choices)
                shares = np.random.dirichlet(alpha * np.ones(len(gens_at_node)))

                for share, g in zip(shares, gens_at_node):
                    p_unit = float(self.pUnitCap[g])
                    q_eff = max(float(q_g[g]), 1e-8)

                    u_g = share * E_n / (q_eff * p_unit)
                    u_g = min(max(u_g, 0.0), float(max_inv))

                    inv[g] = u_g

            all_points[i] = torch.tensor([inv[g] for g in self.G], dtype=self.DTYPE)

        return all_points

    def generate_ui_data_constraint(
        self,
        m=15,
        max_inv=1000.0,
        lb_activated=False,
        alpha=0.1,
        renewable_lb_zero=True,
        renewable_ub_max_inv=True,
        beta=1.2,                       
        renewable_ava_percentile=90,    
    ):
        """
        Constraint-aware unit investment sampling with optional lower bounds.
    
        Current design:
        - Sobol sampling in [0,1]^|G|
        - Non-renewables: ub = min(computed_ub, max_inv)
            where computed_ub = (maxD + export) * beta / P_max
        - Renewables (if renewable_ub_max_inv=False):
            ub = max(max_inv, computed_ub)
            where computed_ub = (maxD + export) * beta / (P_max * ava_p90)
            and ava_p90 is the p-th percentile of non-zero availability
        - Optional lower bound for non-renewables:
            lb_g = alpha * max_t(D_n,t) / Pmax_g
        - Renewables can be forced to have lb = 0
        """
    
        def _is_renewable(g):
            _, tech = g
            tech = str(tech).lower()
            return tech in {"sunpv", "windon", "windoff"}
    
        def _compute_lb_ub_per_generator():
            lb_g = torch.zeros(self.num_g, dtype=self.DTYPE)
            ub_g = torch.full((self.num_g,), float(max_inv), dtype=self.DTYPE)
    
            # max demand per node
            node_max_demand = {}
            for n in self.N:
                node_max_demand[n] = max(self.pDemand[(n, t)] for t in self.T)
    
            # total export per node
            node_export_cap = {n: 0.0 for n in self.N}
            for (start_node, end_node) in self.L:
                node_export_cap[start_node] += float(self.pExpCap[(start_node, end_node)])
    
            for g_idx, g in enumerate(self.G):
                n, _ = g
                pmax = float(self.pUnitCap[g])
                is_ren = _is_renewable(g)
    
                # -------------------------------------------------------
                # Compute effective availability for the denominator
                # For renewables: use the p-th percentile of non-zero hours
                # For non-renewables: availability = 1.0 (always available)
                # -------------------------------------------------------
                if is_ren:
                    ava_values = [self.pGenAva.get((*g, t), 1.0) for t in self.T]
                    ava_nonzero = [a for a in ava_values if a > 0]
                    if ava_nonzero:
                        ava_eff = float(np.percentile(ava_nonzero, renewable_ava_percentile))
                    else:
                        ava_eff = 1.0
                else:
                    ava_eff = 1.0
    
                # -------------------------------------------------------
                # Computed upper bound: (maxD + export) * beta / (P_max * ava)
                # -------------------------------------------------------
                computed_ub = (
                    float(node_max_demand[n]) + float(node_export_cap[n])
                ) * beta / (pmax * ava_eff) if pmax > 0 else 0.0
                # print(f"AVA EFF IS {ava_eff:.3f} FOR {g}, computed_ub: {computed_ub:.1f} is ren {is_ren}")
                # Upper bound
                if is_ren:
                    if renewable_ub_max_inv:
                        ub = float(max_inv)
                    else:
                        ub = min(float(max_inv), float(computed_ub))
                        print(f"COMPUTED UB FOR {g}: {computed_ub:.1f} "
                            f"(ava_p{renewable_ava_percentile}={ava_eff:.3f}), "
                            f"MAX_INV: {max_inv}, ub: {ub:.1f}")
                else:
                    ub = min(float(computed_ub), float(max_inv))
    
                # Lower bounds
                if lb_activated:
                    if is_ren and renewable_lb_zero:
                        lb = 0.0
                    else:
                        lb = float(alpha * node_max_demand[n] / pmax) if pmax > 0 else 0.0
                else:
                    lb = 0.0
    
                # Safety
                lb = min(lb, ub)
    
                lb_g[g_idx] = lb
                ub_g[g_idx] = ub
    
            return lb_g, ub_g
    
        dimensions = self.num_g
        sobol_sampler = qmc.Sobol(d=dimensions)
        base_points = sobol_sampler.random_base2(m=m)
        np.random.shuffle(base_points)
    
        dtype_np = np.float32 if self.args["device"] == "mps" else np.float64
        points01 = torch.tensor(base_points.astype(dtype_np))
    
        lb_g, ub_g = _compute_lb_ub_per_generator()
    
        points = lb_g.unsqueeze(0) + points01 * (ub_g - lb_g).unsqueeze(0)
        points = torch.clamp(points, min=lb_g.unsqueeze(0), max=ub_g.unsqueeze(0))
    
        return points

    def generate_ui_data(self, m=15, max_inv=100000, zero_fraction=0.15):
        dimensions = self.num_g
        sobol_sampler = qmc.Sobol(d=dimensions)
        
        # Generate base Sobol points (2^m points) -- [0, 1]^d
        base_points = sobol_sampler.random_base2(m=m)
        np.random.shuffle(base_points)
        # Scale to max_inv
        if self.args["device"] == "mps":
            points = torch.tensor(np.float32(base_points)) * max_inv
        else:
            points = torch.tensor(np.float64(base_points)) * max_inv
        
       #! TODO: Do we need samples to be exactly 0?

        return points

    def sparsify_batch(self, X):
        """
        Given a batch of decision vectors X (shape: [batch_size, dim]),
        zero out elements to make each row k-sparse, where k ∈ num_active_range.
        """
        batch_size, dim = X.shape
        num_active_range=(1, self.num_g) #! TODO: This should be a parameter, in larger spaces we probably want more investment decisions to be zero.
        # Number of active (non-zero) entries per row
        num_active = torch.randint(num_active_range[0], num_active_range[1] + 1, (batch_size,))

        # Generate random scores to select top-k indices
        scores = torch.rand(batch_size, dim)

        # Sort and zero out all but top-k entries per row
        sorted_scores, sorted_indices = torch.sort(scores, dim=1, descending=True)

        # Create binary mask with top-k indices set to 1
        row_indices = torch.arange(batch_size).unsqueeze(1)
        mask = torch.zeros_like(X, dtype=torch.bool)
        for i in range(num_active_range[0], num_active_range[1] + 1):
            sel = (num_active == i)
            if sel.any():
                topk_indices = sorted_indices[sel][:, :i]
                batch_rows = torch.arange(topk_indices.size(0)).unsqueeze(1)
                mask[sel] = mask[sel].scatter(1, topk_indices, True)

        # Apply the mask
        X_sparse = X * mask

        return X_sparse
    
    def perturb_operating_costs(self, noise=0.01):
        """Perturbs the operating costs to remove symmetry in the solution space -- This stagnates the learning of the neural network.

        Args:
            noise (_type_, optional): _description_. Defaults to 1e-6.

        Returns:
            _type_: _description_
        """
        # Group keys by their cost value
        groups = {}
        for key, cost in self.pVarCost.items():
            groups.setdefault(cost, []).append(key)

        perturbed_pVarCost = {}
        for cost, keys in groups.items():
            if len(keys) > 1:
                n = len(keys)
                # Generate evenly spaced noise values between -noise and noise.
                noise_values = [ -noise + (2 * noise * i) / (n - 1) for i in range(n) ]
                for key, noise_val in zip(keys, noise_values):
                    perturbed_pVarCost[key] = cost + noise_val*cost
            else:
                perturbed_pVarCost[keys[0]] = cost

        return perturbed_pVarCost

    def get_md_nt_indices(self):
        indices = []
        for i in range(self.sample_duration):
            # We have p_g and f_l first, then come the md_n.
            offset = i * self.n_vars + self.num_g + self.num_l
            indices_this_t = list(range(offset, offset + self.num_n))
            indices.extend(indices_this_t)
        
        return indices
    
    def get_f_lt_indices(self):
        indices = []
        for i in range(self.sample_duration):
            # We have p_g first, then f_l.
            offset = i * self.n_vars + self.num_g
            indices_this_t = list(range(offset, offset + self.num_l))
            indices.extend(indices_this_t)
        
        return indices
    
    def get_capacity_ub_indices(self):
        #! Second |G| constraints are capacity upper bounds.
        indices = list(range(self.num_g, 2*self.num_g))
        return indices
    
    def get_missed_demand_ub_indices(self):
        #! Last |N| constraints are demand upper bounds.
        indices = list(range(2*self.num_g + 2*self.num_l + self.num_n, 2*self.num_g + 2*self.num_l + 2*self.num_n))
        return indices
    
    # TODO: Change for compact form!
    def split_ineq_constraints(self, ineq):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1h, 3.1b, 3.1d, 3.1e, 3.1i, 3,1j] (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints]

            Returns h, b, d, e, i, j
        """
        batch_size = ineq.shape[0]  # Get batch size
        
        # Reshape ineq to [batch_size, n_ineq_per_t]
        ineq = ineq.view(batch_size, self.nineq)

        # Define segment sizes based on multiple constraints per type
        sizes = [self.num_g, self.num_g, self.num_l, self.num_l, self.num_n, self.num_n]  # Num constraints per type

        # Extract constraint tensors along the first dimension (constraint type)
        h, b, d, e, i, j = torch.split(ineq, sizes, dim=1)

        return h, b, d, e, i, j
        
    # TODO: Change for compact form!
    def split_eq_constraints(self, eq, log=False):
        """Groups the inequality [residuals or RHS] into constraints by constraint type.
            Assume ineq = [3.1c] (per sample in batch)

            returns each constraint type in the form [Batchsize, nr_of_constraints]

            Returns c
        """
        # # If we are using compact form, we have ui_g = UI_G constraints prepended.
        # if self.benders_compact_form and not log:
        #     ui_g = eq[:, :self.num_g]
        #     c = eq[:, self.num_g:]
        # else:
        ui_g = None
        c = eq
        return ui_g, c

    # TODO: Change for compact form
    def split_dec_vars_from_Y_raw(self, Y):
        """Groups the decision variables from the NN output BEFORE REPAIRS by type.
            Assume y =  [p_{g}, f_{l}] (per sample in batch)

            Returns p_{g}, f_{l}
        """
        batch_size = Y.shape[0]  # Get batch size

        # if self.benders_compact_form:
        #     ui_g = Y[:, :self.num_g]
        #     Y = Y[:, self.num_g:]
        # else:
        ui_g = None

        # Reshape Y for efficient slicing
        Y = Y.view(batch_size, self.n_vars - self.num_n)

        # Define segment sizes
        sizes = [self.num_g, self.num_l]

        # Extract constraint tensors
        p_gt, f_lt = torch.split(Y, sizes, dim=1)

        return ui_g, p_gt, f_lt

    # TODO: Change for compact form
    def split_dec_vars_from_Y(self, Y, log=False):
        """Groups the decision variables from the NN output AFTER REPAIRS by type.
            Assume y = [p_g, f_l, md_n]
            Returns p_g, f_l, md_n in the shape [B, G|L|N]
        """
        batch_size = Y.shape[0]  # Get batch size

        # Remove benders compact form handling if needed
        # if self.benders_compact_form and not log:
        #     Y = Y[:, self.num_g:]

        # Define segment sizes
        sizes = [self.num_g, self.num_l, self.num_n]
        # print(Y.shape)

        # Extract constraint tensors
        p_gt, f_lt, md_nt = torch.split(Y, sizes, dim=1)

        return p_gt, f_lt, md_nt
    


    def ineq_resid(self, X, Y):
        eq_rhs, ineq_rhs = self.split_X(X)
        
        return Y @ self.ineq_cm.T - ineq_rhs

    def ineq_dist(self, X, Y):
        resids = self.ineq_resid(X, Y)
        return torch.clamp(resids, 0)

    def eq_resid(self, X, Y):
        eq_rhs, ineq_rhs = self.split_X(X)
        return Y @ self.eq_cm.T - eq_rhs
    
    def dual_ineq_resid(self, mu, lamb):
        """Dual inequality Residual, takes on the form:
            -mu <= 0
        """
        return -mu

    def dual_eq_resid(self, mu, lamb):
        """Dual equality Residual, takes on the form:
            G^T \mu + H^T \lambda + c = 0"""
        # return self.obj_coeff - (lamb @ self.eq_cm - mu @ self.ineq_cm)
        return mu @ self.ineq_cm + lamb @ self.eq_cm + self.obj_coeff

    def obj_fn(self, X, Y):
        # obj_coeff does not need batching, objective is the same over different samples.
        p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)

        p_gt = torch.abs(p_gt)
        
        cost = self.cost_vec @ p_gt.T

        load_shedding = self.pVOLL * torch.norm(md_nt, p=1, dim=1)

        return  cost + load_shedding
    
    def dual_obj_fn(self, X, mu, lamb):
        eq_rhs, ineq_rhs = self.split_X(X)
        # Batched dot product
        ineq_term = torch.sum(mu * ineq_rhs, dim=1)
        # Batched dot product
        eq_term = torch.sum(lamb * eq_rhs, dim=1)
        return -eq_term - ineq_term

    def relative_ineq_resid(self, X, Y):
        """Relative inequality residuals. Based on Ch.4.4.3, metrics
            
        """
        ineq_resid = self.ineq_resid(X, Y)
        _, ineq_rhs = self.split_X(X)
        p_gt_lb_resid, p_gt_ub_resid, f_lt_lb_resid, f_lt_ub_resid, md_nt_lb_resid, md_nt_ub_resid = self.split_ineq_constraints(ineq_resid)

        p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self.split_ineq_constraints(ineq_rhs)

        p_gt = p_gt_ub - p_gt_lb
        # Production can have lower bound and upper bound = 0, resulting in division by zero.
        # Default to 1 to avoid division by zero.
        p_gt = torch.where(p_gt == 0, torch.ones_like(p_gt), p_gt)

        f_lt = f_lt_ub + f_lt_lb # For flows, add them, because lb is actually negative, but positive in the RHS.
        md_nt = md_nt_ub - md_nt_lb
        

        p_gt_lb_resid = p_gt_lb_resid / p_gt
        p_gt_ub_resid = p_gt_ub_resid / p_gt
        f_lt_lb_resid = f_lt_lb_resid / f_lt
        f_lt_ub_resid = f_lt_ub_resid / f_lt
        md_nt_lb_resid = md_nt_lb_resid / md_nt
        md_nt_ub_resid = md_nt_ub_resid / md_nt

        return torch.concat([p_gt_lb_resid, p_gt_ub_resid, f_lt_lb_resid, f_lt_ub_resid, md_nt_lb_resid, md_nt_ub_resid], dim=1)
    
    def relative_ineq_dist(self, X, Y):
        ineq_resid = self.relative_ineq_resid(X, Y)
        ineq_resid = torch.clamp(ineq_resid, 0)
        return ineq_resid

    def relative_eq_resid(self, X, Y):
        """Relative equality residuals. Based on Ch.4.4.3, metrics"""
        eq_resid = self.eq_resid(X, Y)
        eq_rhs, _ = self.split_X(X)

        _, demand = self.split_eq_constraints(eq_rhs)
        
        eq_resid = eq_resid.abs() / demand.abs()    

        return eq_resid
    
    def build_obj_coeff(self,):
        """ Builds the objective function coefficients (only the operating costs)
            Assume y =  [p_{g,t0}, f_{l,t0}, md_{n,t0}, 
                         p_{g,t1}, f_{l,t1}, md_{n,t1}, ..., 
                         p_{g,tn}, f_{l,tn}, md_{n,tn}] """
        # coeff vector of zero's to be filled
        obj_coeff = torch.zeros(self.n_vars * self.sample_duration, dtype=self.DTYPE)
        cost_vec = torch.zeros(self.num_g * self.sample_duration, dtype=self.DTYPE)

        for t in range(self.sample_duration):
            # Sum over G,T (PC_g * p_{g,t}) --> Generation costs
            for idx_g, g in enumerate(self.G):
                # Generator variables are always at first |G| indices per timestep
                coeff_idx = t * self.n_vars + idx_g
                obj_coeff[coeff_idx] = self.pVarCost[g]
                cost_vec[coeff_idx] = self.pVarCost[g]

            # Sum over N,T (MDC * md_{n,t}) --> Cost missed demand
            for idx_n in range(self.num_n):
                # Missed demand variables come after generator and lineflow variables
                coeff_idx = t * self.n_vars + self.num_g + self.num_l + idx_n
                obj_coeff[coeff_idx] = self.pVOLL

        return obj_coeff, cost_vec

    def split_X(self, X):
        
        #print(f"NEQ IS {self.neq}, N is {len(self.N)} G is {len(self.G)}")
        ''' 
        eq_rhs = X[:,:len(self.N)]
        capacity_ub = X[:, len(self.N):len(self.N)+len(self.G)]
        '''
        # eq_rhs = X[:, :self.neq]
        # capacity_ub = X[:, self.neq:]
        eq_rhs = X[:,:len(self.N)]
        capacity_ub = X[:, len(self.N):len(self.N)+len(self.G)]
        ineq_rhs = self.ineq_rhs.clone().repeat(X.size(0), 1)
        #print(f"Capacity UB Index: {self.capacity_ub_indices}, Missed Demand UB Index: {self.missed_demand_ub_indices}")
        #! Second |G| constraints are capacity upper bounds.
        ineq_rhs[:, self.capacity_ub_indices] = capacity_ub
        #! Last |N| constraints are demand upper bounds.
        ineq_rhs[:, self.missed_demand_ub_indices] = eq_rhs

        return eq_rhs, ineq_rhs

    def build_X(self):
        """Builds a tensor containing the features that vary across problem instances.
        """
        # For ineq RHS, only 3.1b varies across instances --> first |G| constraints are 3.1h, second |G| constraints are 3.1b.
        # ineq_rhs_varying_indices = [t * self.n_ineq_per_t + self.num_g + i for t in range(self.sample_duration) for i in range(self.num_g)]

        # The entire RHS changes for equality constraints
        # X = torch.concat([self.eq_rhs, self.ineq_rhs[:, ineq_rhs_varying_indices]], dim=1)
        X = []
        total_demands = []
        for i in range(self.n_samples):
            t = i % len(self.T) + 1
            Xi = []
            for n in self.N:
                Xi.append(self.pDemand[(n, t)])
            for g_idx, g in enumerate(self.G):
                p_gt_ub = self.pUnitInvestment[i, g_idx] * self.pUnitCap[g] * self.pGenAva.get((*g, t), 1.0)
                Xi.append(p_gt_ub)
            X.append(Xi)
        X = np.array(X)
        # np.random.shuffle(X)

        X = torch.tensor(np.array(X), dtype=self.DTYPE)
        perm = torch.randperm(X.shape[0])

        X = X[perm]
        self.pUnitInvestment = self.pUnitInvestment[perm]
        X = torch.tensor(X)
        
        return X
    
    def build_synthetic_X(self):
        """Builds synthetic samples, which follow an easy distribution.

        Args:
            num_samples (_type_): _description_
        """
        X = np.random.uniform(0, 1, size=(self.n_samples, self.num_n + self.num_g))

        return torch.tensor(X)

    
    def build_X_alternative(self,):
        X = []
        for t in range(1, self.n_samples - 1):
            x = []
            for idx, g in enumerate(self.G):
                x.append(self.pUnitInvestment[t, idx])
            for n in self.N:
                x.append(self.pDemand[(n, t)])
            for g in self.G:
                x.append(self.pGenAva.get((*g, t), 1.0))
            X.append(x)
        return torch.tensor(X)

    
    # Bounds are always identity, -1 for lower bounds, +1 for upper bounds.
    def assign_identity_or_scalar(self, matrix, row_start, col_start, size, value=1):
        """ Assign identity matrix if size > 1, otherwise assign a scalar value. """
        if size > 1:
            # Torch.eye returns a 2-D tensor with ones on the diagonal and zeros elsewhere.
            matrix[row_start:row_start + size, col_start:col_start + size] = value*torch.eye(size)
        else:
            matrix[row_start, col_start] = value

    def build_ineq_cm_rhs(self):
        """For a single data sample characterised by the time_range, 
        build the full inequality constraint matrix for all timesteps using Kronecker product.
        """

        # num_timesteps = len(time_range)
        num_columns_per_t = self.n_vars

        # Compute number of constraint rows per timestep
        num_rows_per_t = 2 * (self.num_g + self.num_l + self.num_n)

        # Create a single block for one timestep
        block_matrix = torch.zeros((num_rows_per_t, num_columns_per_t))

        # Apply constraints
        # Bounds are always identity, -1 for lower bounds, +1 for upper bounds.
        self.assign_identity_or_scalar(block_matrix, 0, 0, self.num_g, -1) # 3.1h: Production lower bound
        self.assign_identity_or_scalar(block_matrix, self.num_g, 0, self.num_g, 1) # 3.1b: Production upper bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g, self.num_g, self.num_l, -1) # 3.1d: Lineflow lower bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g + self.num_l, self.num_g, self.num_l, 1) # 3.1e: Lineflow upper bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g + 2*self.num_l, self.num_g + self.num_l, self.num_n, -1) # 3.1i: Missed demand lower bound
        self.assign_identity_or_scalar(block_matrix, 2*self.num_g + 2*self.num_l + self.num_n, self.num_g + self.num_l, self.num_n, 1)  # 3.1j: Missed demand upper bound

        # Use Kronecker product to copy the block matrix diagonally for all timesteps
        #! Sample duration is always 1
        # ineq_cm = torch.kron(torch.eye(self.sample_duration), block_matrix)
        ineq_cm = block_matrix

        ineq_rhs = []

        # If we have more synthetic samples than timesteps, keep it within bounds through modulo (start counting again from 0 once we go out of bounds)
        # t = (sample_idx % len(self.T)) + 1

        # 3.1h: Production lower bound
        ineq_rhs += [0.0 for _ in range(self.num_g)]
        # 3.1b: Production upper bound
        # ineq_rhs += [self.pGenAva.get((*g, t), 1.0) * self.pUnitCap[g] * self.pUnitInvestment[sample_idx, idx] for idx, g in enumerate(self.G)]
        #! This is a placeholder, where we will substitute the capacity upper bound taken from X.
        ineq_rhs += [0.0 for _ in range(self.num_g)] # We wanto add max capacity = INvestment*UnitCapacity*Availablity
        # 3.1d: Lineflow lower bound
        ineq_rhs += [self.pImpCap[l] for l in self.L]
        # 3.1e: Lineflow upper bound
        ineq_rhs += [self.pExpCap[l] for l in self.L]
        # 3.1i: Missed demand lower bound
        ineq_rhs += [0.0 for _ in range(self.num_n)] 
        # 3.1j: Missed demand upper bound
        # ineq_rhs += [self.pDemand[(n, t)] for n in self.N]
        #! This is a placeholder, where we will substitute the missed demand upper bound (Demand) taken from X.
        ineq_rhs += [0.0 for _ in range(self.num_n)] # Upper bound is Demand.
        
        return ineq_cm, torch.tensor(ineq_rhs)

    def build_eq_cm(self):
        """Build the constraint matrix for the equality constraints
        """

        # p_{g}, f_in - f_out, md_n
        eq_cm = torch.concat([self.node_to_gen_mask, self.lineflow_mask, torch.eye(self.num_n)], dim=1)

        # Use Kronecker product to copy the block matrix diagonally for all timesteps
        #! Not necessary, sample duration is always 1.
        # eq_cm = torch.kron(torch.eye(self.sample_duration), block_matrix)
                
        return eq_cm

    def _split_X_in_sets(self, train=0.8, valid=0.1, test=0.1):
        # Ensure the split ratios sum to 1
        assert train + valid + test == 1.0

        # Total number of samples
        B = self.X.size(0)
        indices = torch.arange(B)

        # Compute sizes for each set
        train_size = int(train * B)
        valid_size = int(valid * B)

        # Split the indices
        self._train_indices = indices[:train_size]
        self._valid_indices = indices[train_size:train_size+valid_size]
        self._test_indices = indices[train_size+valid_size:]

        print(f"Size of train set: {train_size}")
        print(f"Size of val set: {valid_size}")
        print(f"Size of test set: {B - train_size - valid_size}")
    
    def total_prod(self, p_gt):
        """
        p_gt: [batch_size, num_generators]
        node_to_gen_mask: [num_nodes, num_generators]
        returns: [batch_size, num_nodes]
        """
        p_nt = torch.matmul(p_gt, self.node_to_gen_mask.to(dtype=self.DTYPE).T)
        return p_nt

    def net_flow(self, f_lt):
        """
        f_lt: [batch_size, num_lines]
        lineflow_mask: [num_nodes, num_lines]
        returns: [batch_size, num_nodes]
        """
        net_flow_nt = torch.matmul(f_lt, self.lineflow_mask.to(dtype=self.DTYPE).T)
        return net_flow_nt

    def net_balance(self, p_gt, f_lt):
        p_nt = self.total_prod(p_gt)
        net_flow_nt = self.net_flow(f_lt)

        net_balance = p_nt + net_flow_nt
        return net_balance
    
    def missed_demand(self, p_gt, f_lt, D_nt):
        net_balance = self.net_balance(p_gt, f_lt)
        md_nt = D_nt - net_balance

        return md_nt

    def plot_balance(self, primal_net, dual_net):
        sample = 0
        with torch.no_grad():
            # Predictions for primal variables
            Y = primal_net(self.X)
            eq_rhs, ineq_rhs = self.split_X(self.X)
            # Extract decision variables
            p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)  # [B, (G|L|N)]
            UI_g, D_nt = self.split_eq_constraints(eq_rhs[sample:sample+1])  # [1, N]
            
            # Convert to numpy and ensure correct shape
            total_prod = self.total_prod(p_gt[sample:sample+1])[0].cpu().numpy()  # Shape [N], always positive
            net_flow = self.net_flow(f_lt[sample:sample+1])[0].cpu().numpy()  # Shape [N], can be positive or negative
            missed_demand = md_nt[sample:sample+1][0].cpu().numpy()  # Shape [N], can be positive or negative
            demand = D_nt[0].cpu().numpy()  # Shape [N], total sum should match this
            num_nodes = total_prod.shape[0]
            
            fig, ax = plt.subplots(num_nodes, 1, figsize=(10, 6 * num_nodes), sharex=True)
            
            if num_nodes == 1:
                ax = [ax]
            
            for node in range(num_nodes):
                # Get data for the current node
                tp = total_prod[node]  # Always positive
                nf = net_flow[node]  # Can be positive or negative
                md = missed_demand[node]  # Can be positive or negative
                d = demand[node]  # Total sum of all components
                
                # Define positions
                timestep = 0  # Single timestep
                
                # Separate positive and negative components
                md_pos = max(md, 0)
                md_neg = min(md, 0)
                nf_pos = max(nf, 0)
                nf_neg = min(nf, 0)
                
                # Compute stacking order
                base_pos = 0
                base_neg = 0
                
                # Plot positive values above zero
                ax[node].bar(timestep, tp, bottom=base_pos, label='Total Production', color='blue')
                base_pos += tp
                
                ax[node].bar(timestep, nf_pos, bottom=base_pos, label='Net Flow (+)', color='green')
                base_pos += nf_pos
                
                ax[node].bar(timestep, md_pos, bottom=base_pos, label='Missed Demand (+)', color='orange')
                base_pos += md_pos
                
                # Plot negative values below zero
                ax[node].bar(timestep, md_neg, bottom=base_neg, label='Missed Demand (-)', color='darkorange')
                base_neg += md_neg
                
                ax[node].bar(timestep, nf_neg, bottom=base_neg, label='Net Flow (-)', color='darkgreen')
                base_neg += nf_neg
                
                # Add demand as a red dot
                ax[node].scatter(timestep, d, color='red', label='Demand', zorder=3)
                
                # Make the 0 line thick
                ax[node].axhline(0, color='black', linewidth=2)
                
                ax[node].set_ylabel('Values')
                ax[node].set_title(f'Node {node}')
                ax[node].legend()
                ax[node].grid(True)
            
            ax[-1].set_xlabel('Timestep')
            plt.tight_layout()
            plt.show()
        
    def plot_decision_variable_diffs(self, primal_net, dual_net):
        sample = 0
        with torch.no_grad():
            Y = primal_net(self.X)
            Y_target = self.opt_targets["y_operational"]

            mu, lamb = dual_net(self.X)
            mu_target = self.opt_targets["mu_operational"]
            lamb_target = self.opt_targets["lamb_operational"]

            eq_rhs, ineq_rhs = self.split_X(self.X)

            _, p_gt_ub, _, _, _, _ = self.split_ineq_constraints(ineq_rhs)
            
            p_gt, f_lt, md_nt = self.split_dec_vars_from_Y(Y)  
            p_gt_target, f_lt_target, md_nt_target = self.split_dec_vars_from_Y(Y_target, log=True)  

            dual_lb_p, dual_ub_p, dual_lb_f, dual_ub_f, dual_lb_md, dual_ub_md = self.split_ineq_constraints(mu)
            dual_target_lb_p, dual_target_ub_p, dual_target_lb_f, dual_target_ub_f, dual_target_lb_md, dual_target_ub_md = self.split_ineq_constraints(mu_target)

            ui_g, dual_node_balance = self.split_eq_constraints(lamb)
            ui_g, dual_node_balance_target = self.split_eq_constraints(lamb_target, log=True)

            num_gens, num_lines, num_nodes = p_gt.shape[1], f_lt.shape[1], md_nt.shape[1]
            
            fig, axes = plt.subplots(2, 4, figsize=(24, 12), sharex=False)
            
            x_pos_g = np.arange(num_gens)
            x_labels_g = [f"G{g}" for g in range(num_gens)]
            x_pos_l = np.arange(num_lines)
            x_labels_l = [f"L{l}" for l in range(num_lines)]
            x_pos_n = np.arange(num_nodes)
            x_labels_n = [f"N{n}" for n in range(num_nodes)]
            
            column_titles = ["Generation Production", "Line Flows", "Missed Demand", "Node Balance Duals"]
            for col, title in enumerate(column_titles):
                axes[0, col].set_title(title, fontsize=14)
            
            def plot_variable(ax, x_pos, labels, pred, target, ylabel, color, ub=None):
                ax.bar(x_pos, pred, label='Predicted', color=color, alpha=0.5)
                ax.scatter(x_pos, target, color='red', label='Target', zorder=3)
                if ub is not None:
                    ax.scatter(x_pos, ub, color='purple', marker='x', label='Max capacity', zorder=4)
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=90)
                ax.legend()
                ax.axhline(0, color='black', linewidth=2)
                ax.grid(True)
            
            plot_variable(axes[0, 0], x_pos_g, x_labels_g, p_gt[sample].reshape(-1), p_gt_target[sample].reshape(-1), 'Primal Generation', 'blue', ub=p_gt_ub[sample].reshape(-1))
            plot_variable(axes[0, 1], x_pos_l, x_labels_l, f_lt[sample].reshape(-1), f_lt_target[sample].reshape(-1), 'Primal Line Flow', 'green')
            plot_variable(axes[0, 2], x_pos_n, x_labels_n, md_nt[sample].reshape(-1), md_nt_target[sample].reshape(-1), 'Primal Missed Demand', 'orange')
            
            def plot_dual_variable(ax, x_pos, labels, lb, ub, lb_target, ub_target, ylabel):
                ax.bar(x_pos, lb, label='Dual LB', color='purple', alpha=0.5)
                ax.bar(x_pos, ub, label='Dual UB', color='cyan', alpha=0.5)
                ax.scatter(x_pos, lb_target, color='purple', marker='o', label='Target LB', zorder=3, edgecolors='black')
                ax.scatter(x_pos, ub_target, color='cyan', marker='o', label='Target UB', zorder=2, edgecolors='black')
                ax.set_ylabel(ylabel)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=90)
                ax.legend()
                ax.axhline(0, color='black', linewidth=2)
                ax.grid(True)
            
            plot_dual_variable(axes[1, 0], x_pos_g, x_labels_g, dual_lb_p[sample].reshape(-1), dual_ub_p[sample].reshape(-1), dual_target_lb_p[sample].reshape(-1), dual_target_ub_p[sample].reshape(-1), 'Dual (Gen)')
            plot_dual_variable(axes[1, 1], x_pos_l, x_labels_l, dual_lb_f[sample].reshape(-1), dual_ub_f[sample].reshape(-1), dual_target_lb_f[sample].reshape(-1), dual_target_ub_f[sample].reshape(-1), 'Dual (Flow)')
            plot_dual_variable(axes[1, 2], x_pos_n, x_labels_n, dual_lb_md[sample].reshape(-1), dual_ub_md[sample].reshape(-1), dual_target_lb_md[sample].reshape(-1), dual_target_ub_md[sample].reshape(-1), 'Dual (Missed Demand)')
            plot_variable(axes[1, 3], x_pos_n, x_labels_n, dual_node_balance[sample].reshape(-1), dual_node_balance_target[sample].reshape(-1), 'Dual Node Balance', 'brown')
            
            plt.tight_layout()
            plt.show()
    
    def compute_opt_targets(self):
        y = []
        mu = []
        lamb = []
        obj = []

        obj_coeff = self.obj_coeff.numpy()
        eq_cm = self.eq_cm.numpy()
        ineq_cm = self.ineq_cm.numpy()
        eq_rhs, ineq_rhs = self.split_X(self.X)
        eq_rhs = eq_rhs.numpy()
        ineq_rhs = ineq_rhs.numpy()

        for i in range(len(self.X)):
            y_operational, obj_operational, dual_eq, dual_ineq = solve_matrix_problem_simple(obj_coeff, eq_cm, ineq_cm, eq_rhs[i], ineq_rhs[i])
            y.append(torch.tensor(y_operational, dtype=self.DTYPE))
            #! Negate duals, these are flipped in Gurobi.
            mu.append(torch.tensor(-dual_ineq, dtype=self.DTYPE))
            lamb.append(torch.tensor(-dual_eq, dtype=self.DTYPE))
            obj.append(torch.tensor(obj_operational, dtype=self.DTYPE))
        
        return {
            "y_operational": torch.stack(y),
            "mu_operational": torch.stack(mu), 
            "lamb_operational": torch.stack(lamb),
            "obj": torch.stack(obj)
        }

def solve_matrix_problem_simple(obj_coeff, eq_cm, ineq_cm, eq_rhs, ineq_rhs, verbose=False):
    import gurobipy as gp
    from gurobipy import GRB

    # Create a new model
    m = gp.Model("matrix_problem")
    
    # Important: Set method to dual simplex to get dual values
    m.setParam('Method', 1)  # Use dual simplex
    
    # Create variables
    x = m.addMVar(shape=len(obj_coeff), lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    
    # Set objective
    m.setObjective(obj_coeff @ x, GRB.MINIMIZE)
    
    # Add constraints and store their references
    eq_constrs = m.addConstr(eq_cm @ x == eq_rhs, name="eq")
    ineq_constrs = m.addConstr(ineq_cm @ x <= ineq_rhs, name="ineq")
    
    # Optimize
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        # Get dual values after confirming optimal solution
        dual_eq = eq_constrs.Pi
        dual_ineq = ineq_constrs.Pi
        
        if verbose:
            print(f"Optimal objective: {m.ObjVal}")
            print(f"Dual values (eq): {dual_eq}")
            print(f"Dual values (ineq): {dual_ineq}")
            
        return x.X, m.ObjVal, dual_eq, dual_ineq
    else:
        raise RuntimeError(f"Optimization failed with status {m.Status}")


