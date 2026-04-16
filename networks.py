import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler


class PrimalNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        if args["hidden_size"]:
            self.hidden_sizes = [int(args["hidden_size"])] * args["n_layers"]
        else:
            self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        
        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self.hidden_sizes + [data.ydim]
        layers = []

        # layers.append(nn.LayerNorm(data.xdim))

        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            if out_size != data.ydim:  # Add ReLU activation for hidden layers only
                layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x, total_demands=None):
        #! If we are training, we do not scale the output by the total demands. Only for logging metrics.

        if not self.training and (total_demands != None):
            return self.net(x) * total_demands
        else:
            return self.net(x)

class DualNet(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        if args["hidden_size"]:
            self.hidden_sizes = [int(args["hidden_size"])] * args["n_layers"]
        else:
            self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.mu_size = self.data.nineq
        self.lamb_size = self.data.neq

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Create the list of layer sizes
        layer_sizes = [data.xdim] + self.hidden_sizes
        # layer_sizes = [2*data.xdim + 1000] + self.hidden_sizes
        layers = []
        # Create layers dynamically based on the provided hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())

        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        # Add the output layer
        self.out_layer = nn.Linear(self.hidden_sizes[-1], self.mu_size + self.lamb_size)
        nn.init.zeros_(self.out_layer.weight)  # Initialize output layer weights to 0
        nn.init.zeros_(self.out_layer.bias)    # Initialize output layer biases to 0
        layers.append(self.out_layer)

        self.net = nn.Sequential(*layers)

        self.normalize = args.get("normalize", None)
        if self.normalize == "z_score":
            N = data.num_n
            G = data.num_g
            self.register_buffer("d_mean", torch.zeros(N, dtype=self.DTYPE))
            self.register_buffer("d_std",  torch.ones(N, dtype=self.DTYPE))

            # capacity stats
            self.register_buffer("p_mean", torch.zeros(G, dtype=self.DTYPE))
            self.register_buffer("p_std",  torch.ones(G, dtype=self.DTYPE))

    
    def forward(self, x, *args):
        # print(f"DualNet input x shape: {x.shape} 0: {x[0,:]}")
        out = self.net(x)
        out_mu = out[:, :self.mu_size]
        out_lamb = out[:, self.mu_size:]
        # print(f"DualNet output mu shape: {out_mu.shape} lamb shape: {out_lamb.shape} MU: {out_mu[0,:]} LAMB: {out_lamb[0,:]}")
        return out_mu, out_lamb


class DualNetTwoOutputLayers(nn.Module):
    def __init__(self, data, hidden_size):
        super().__init__()
        self.data = data
        self.hidden_size = hidden_size
        layer_sizes = [data.xdim, self.hidden_size, self.hidden_size]
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ReLU())
        for layer in layers:
            if type(layer) == nn.Linear:
                nn.init.kaiming_normal_(layer.weight)
        self.out_layer_mu = nn.Linear(self.hidden_size, data.nineq)
        self.out_layer_lamb = nn.Linear(self.hidden_size, data.neq)
        # Init last layers as 0, like in the paper
        nn.init.zeros_(self.out_layer_mu.weight)
        nn.init.zeros_(self.out_layer_mu.bias)
        nn.init.zeros_(self.out_layer_lamb.weight)
        nn.init.zeros_(self.out_layer_lamb.bias)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.net(x)
        out_mu = self.out_layer_mu(out)
        out_lamb = self.out_layer_lamb(out)
        return out_mu, out_lamb

class FeedForwardNet(nn.Module):
    def __init__(self, args, input_dim, hidden_sizes, output_dim, layernorm=None):
        """_summary_

        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output features
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.output_dim = output_dim

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # Create the list of layer sizes
        layer_sizes = [self.input_dim] + self.hidden_sizes + [self.output_dim]
        layers = []
        
        if layernorm is None:
            if args["layernorm"]:
                layers.append(nn.LayerNorm(input_dim)) #! This is necessary to prevent gradient saturation in the sigmoids.
        elif layernorm is True:
            layers.append(nn.LayerNorm(input_dim))

        # Create layers dynamically based on the provided hidden_sizes
        for idx, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            # Add ReLU only if it is not the last layer
            if idx < len(layer_sizes) - 2:  # The last layer does not need ReLU
                layers.append(nn.ReLU())
        
        # Initialize all layers
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_uniform_(layer.weight)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x_out = self.net(x)
        return x_out
    

class FeedForwardNetSeparateHead(nn.Module):
    """
    Shared MLP backbone with independent classification head per dual variable.
    
    Instead of one flat Linear(hidden -> n_dual_vars * n_classes) output in FeedForwardNet,
    each dual variable gets its own Linear(hidden -> n_classes) head.
    This prevents gradient interference between nodes during training.
    """
    def __init__(self, args, input_dim, hidden_sizes, n_heads, n_classes):
        """
        Args:
            input_dim:    Number of input features
            hidden_sizes: List of hidden layer sizes for shared backbone
            n_heads:      Number of independent heads (= n_dual_vars = n nodes)
            n_classes:    Number of classes per head (= number of cost classes)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.n_heads = n_heads
        self.n_classes = n_classes

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # --- Shared backbone (no output layer) ---
        backbone_layers = []
        backbone_layers.append(nn.LayerNorm(input_dim))  # always on for dual net

        layer_sizes = [input_dim] + hidden_sizes
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            backbone_layers.append(nn.Linear(in_size, out_size))
            backbone_layers.append(nn.ReLU())

        # Kaiming init on backbone linears
        for layer in backbone_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

        self.backbone = nn.Sequential(*backbone_layers)

        # --- Per-node classification heads ---
        # Each head: Linear(hidden -> n_classes), no activation (softmax applied outside)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], n_classes)
            for _ in range(n_heads)
        ])

        # Kaiming init on heads
        for head in self.heads:
            nn.init.kaiming_normal_(head.weight)

    def forward(self, x):
        h = self.backbone(x)  # [B, hidden]

        # Each head independently classifies its node
        logits = torch.stack(
            [head(h) for head in self.heads], dim=1
        )  # [B, n_heads, n_classes]

        return logits.view(x.shape[0], -1)  # [B, n_heads*n_classes]


class BoundRepairLayer(nn.Module):
    def __init__(self, scale_sigmoid = False, shifted_sigmoid_scale = False, k = 1.0, repair_scaler = "Sigmoid", softplus_k = 10.0):
        super().__init__()
        self.scale_sigmoid = scale_sigmoid
        self.shifted_sigmoid_scale = shifted_sigmoid_scale
        self.k = k
        self.softplus_k = softplus_k
        self.repair_scaler = repair_scaler
        self.s = 100
    
    def forward(self, x, lb, ub):
        """_summary_

        Args:
            x (_type_): Decision variables, shape [B, N, T]
            lb (_type_): Lower bounds of decision variables, shape [B, N, T]
            ub (_type_): Upper bounds of decision variables, shape [B, N, T]

        Returns:
            _type_: _description_
        """
        if self.repair_scaler == "ScaledSigmoid":
            sig_input = x*1/self.k

            scaled = torch.sigmoid(sig_input)
            
            repaired = lb + (ub - lb) * scaled
        elif self.repair_scaler == "ShiftedScaledSigmoid":

            sig_input = (x - (lb + ub)/2)*1/self.k

            scaled = torch.sigmoid(sig_input)
            
            repaired = lb + (ub - lb) * scaled
        elif self.repair_scaler == "Sigmoid":
            sig_input = x
            scaled = torch.sigmoid(sig_input)
            
            repaired = lb + (ub - lb) * scaled
        elif self.repair_scaler == "SoftPlus":
            repaired = self.smooth_clip(x, lb, ub, k=self.softplus_k)
        if False: #! Set to True to attach hooks to inspect gradients
            def print_grad(name):
                return lambda grad: print(f"Gradient for {name}: {grad.norm():.4e}")
            x.register_hook(print_grad("x"))
            scaled.register_hook(print_grad("sigmoid(kx)"))
            repaired.register_hook(print_grad("repaired output"))

        # Softplus



        return repaired
    def softplusK(self, x, k=1.0):
        return F.softplus(k * x) / k

    def s_max(self,x, M, k=1.0):
        return M - self.softplusK(M - x, k)

    def s_min(self,x, M, k=1.0):
        return self.softplusK(x - M, k) + M

    def smooth_clip(self,x, L, U, k=1.0):
        return self.s_max(self.s_min(x, L, k), U, k)
    
class EstimateSlackLayer(nn.Module):
    def __init__(self, node_to_gen_mask, lineflow_mask):
        super().__init__()

        self.node_to_gen_mask = node_to_gen_mask    # [N, G]
        self.lineflow_mask = lineflow_mask          # [N, L]


    def forward(self, p_gt, f_lt, D_nt):
        """Compute md_n,t

        Args:
            p_gt (_type_): Generator production, shape [B, G]
            f_lt (_type_): Line flow, shape [B, L]
            D_nt (_type_): Demand, shape [B, N]

            Line Flow Mask.T: like this
                    Node 0  Node 1  Node 2
            Line 0    -1     1        0      Line 0 connects Node 0 to Node 1
            Line 1    -1     0        1      Line 1 connects Node 0 to Node 2
            Line 2     0     -1       1      Line 2 connects Node 1 to Node 2
        """


        combined_flow = torch.matmul(p_gt, self.node_to_gen_mask.T) + \
                        torch.matmul(f_lt, self.lineflow_mask.T)
        # print(f"Production Location: {torch.matmul(p_gt, self.node_to_gen_mask.T)}")
        #print("########")
        net_prod = torch.matmul(p_gt, self.node_to_gen_mask.T)
        net_flow = torch.matmul(f_lt, self.lineflow_mask.T)

        
        # print("########")
        md_nt = D_nt - combined_flow
        
        net_flow_demand_mask = md_nt > D_nt

        return md_nt

class PrimalNetEndToEnd(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(args["hidden_size_factor"]*data.xdim)] * args["n_layers"]
        self.args = args

        if not self.args["repair"]:
            assert not self.args["repair_bounds"], "If repair is disabled, bounds repair should not be enabled."
            assert not self.args["repair_completion"], "If repair is disabled, completion repair should not be enabled."
            assert not self.args["repair_power_balance"], "If repair is disabled, power balance repair should not be enabled."

        if self.args["repair_power_balance"]:
            # Power balance repair requires bounds repair.
            assert self.args["repair_bounds"], "Power balance repair requires bounds repair."
        
        if self.args["repair_bounds"]:
            # Bound repairs need layernorm to prevent gradient saturation in sigmoids.
            assert self.args["layernorm"], "Bounds repair requires layernorm."

        if self.args["device"] == "mps":
                self.DTYPE = torch.float32
                self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        # TODO: Implement compact benders form.
        # if self.data.args["benders_compact"]:
            # self.out_dim = data.num_g + data.n_prod_vars + data.n_line_vars
        # else:
        if args["repair_completion"]:
            self.out_dim = data.n_prod_vars + data.n_line_vars
        else:
            self.out_dim = data.n_prod_vars + data.n_line_vars + data.n_md_vars

        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim).to(self.DTYPE).to(self.DEVICE)
        
        if "scale_sigmoid" not in self.args:
            self.args["scale_sigmoid"] = False
        if "shifted_sigmoid_scale" not in self.args:
            self.args["shifted_sigmoid_scale"] = False
        if "k" not in self.args:
            self.args["k"] = 1.0
        if "use_blend_repair" not in self.args:
            self.args["use_blend_repair"] = False

        if self.args["repair_bounds"]:
            # if "scale_sigmoid" in self.args and self.args["scale_sigmoid"]:
            #     self.bound_repair_layer = BoundRepairLayer(scale_sigmoid = True)
            # else:
            #     self.bound_repair_layer = BoundRepairLayer()
            # # self.ramping_repair_layer = RampingRepairLayer()
            if "repair_scaler" not in self.args:
                if self.args["scale_sigmoid"]:
                    self.args["repair_scaler"] = "ScaledSigmoid"
                elif self.args["shifted_sigmoid_scale"]:
                    self.args["repair_scaler"] = "ShiftedScaledSigmoid"
                else:
                    self.args["repair_scaler"] = "Sigmoid"
            self.bound_repair_layer = BoundRepairLayer(scale_sigmoid= self.args["scale_sigmoid"], shifted_sigmoid_scale=self.args["shifted_sigmoid_scale"], 
                                                       k = self.args["k"], repair_scaler= self.args["repair_scaler"], softplus_k = self.args.get("soft_plus_k", 10.0))
        if self.args["repair_completion"]:
            self.estimate_slack_layer = EstimateSlackLayer(data.node_to_gen_mask.to(self.DTYPE).to(self.DEVICE), data.lineflow_mask.to(self.DTYPE).to(self.DEVICE))
    
        self.normalize = args.get("normalize", None)
        if self.normalize == "z_score":
            N = data.num_n
            G = data.num_g
            L = data.num_l
            self.register_buffer("d_mean", torch.zeros(N, dtype=self.DTYPE))
            self.register_buffer("d_std",  torch.ones(N, dtype=self.DTYPE))

            # capacity stats
            self.register_buffer("p_mean", torch.zeros(G, dtype=self.DTYPE))
            self.register_buffer("p_std",  torch.ones(G, dtype=self.DTYPE))

            self.register_buffer("f_mean", torch.zeros(L, dtype=self.DTYPE))
            self.register_buffer("f_std",  torch.ones(L, dtype=self.DTYPE))

    def scale(self, x):
        if self.normalize == "z_score":
            d = x[:, :self.data.num_n]
            p = x[:, self.data.num_n:self.data.num_n + self.data.num_g]

            d = (d - self.d_mean) / self.d_std
            p = (p - self.p_mean) / self.p_std
            x_scale = torch.cat([d, p], dim=1)
        else:
            x_scale = x
        return x_scale
    
    def inverse_scale(self, p, f):
        ''' 
        if repair_completion:
            x_out is [md, p, f]
        else:
            x_out is [p, f, md]

        Here we only worry about p, f
        '''
        p = p * self.p_std + self.p_mean
        return p, f
    
    def _box_slacks(self, y, lb, ub):
        """
        Box constraints lb <= y <= ub
        Return two slacks:
        s_lb = y - lb     (>=0 if y >= lb, i.e. slack is positive when y is feasible, negative when y is infeasible)
        s_up  = ub - y     (>=0 if y <= ub, i.e. slack is positive when y is feasible, negative when y is infeasible)
        """
        s_lb = y - lb
        s_up  = ub - y
        return s_lb, s_up

    def _compute_alpha_from_slacks(self, slack_pairs_TN, slack_pairs_SN, tol=1e-12):
        """
        slack_pairs_*: list of tuples [(s_lb, s_up), ...] for different variable blocks.
        Each s_* is [B, dim].
        Returns alpha: [B, 1] in [0,1].
        """
        alpha = None

        for (sL_TN, sU_TN), (sL_SN, sU_SN) in zip(slack_pairs_TN, slack_pairs_SN):
            # flatten per sample
            # violations where slack < 0
            for sTN, sSN in [(sL_TN, sL_SN), (sU_TN, sU_SN)]:
                # sTN, sSN are [B, dim]
                viol = sTN < 0.0
                if viol.any():
                    denom = (sSN - sTN).clamp_min(tol)  # should be positive if SN is feasible
                    a = torch.where(viol, (-sTN) / denom, torch.zeros_like(sTN))
                    a_max = a.max(dim=1, keepdim=True).values  # [B,1]
                    alpha = a_max if alpha is None else torch.maximum(alpha, a_max)

        if alpha is None:
            # already feasible
            return torch.zeros((slack_pairs_TN[0][0].shape[0], 1), dtype=self.DTYPE, device=self.DEVICE)
        return alpha.clamp(0.0, 1.0)

    def _blend_repair(self, p, f, md, p_lb, p_ub, f_lb, f_ub, md_lb, md_ub, D_nt):
        """
        Apply alpha-blend repair with safe point (p=0,f=0,md=D).
        Returns repaired (p,f,md) and alpha [B,1].
        """
        # --- Safe point ---
        pSN  = torch.zeros_like(p)
        fSN  = torch.zeros_like(f)
        # safest is md = D
        mdSN = D_nt  # [B,N]

        # --- Slacks for TN ---
        s_p_TN  = self._box_slacks(p,  p_lb,  p_ub)
        s_f_TN  = self._box_slacks(f,  f_lb,  f_ub)
        s_md_TN = self._box_slacks(md, md_lb, md_ub)

        # --- Slacks for SN ---
        s_p_SN  = self._box_slacks(pSN,  p_lb,  p_ub)
        s_f_SN  = self._box_slacks(fSN,  f_lb,  f_ub)
        s_md_SN = self._box_slacks(mdSN, md_lb, md_ub)

        # --- Alpha ---
        alpha = self._compute_alpha_from_slacks(
            slack_pairs_TN=[s_p_TN, s_f_TN, s_md_TN],
            slack_pairs_SN=[s_p_SN, s_f_SN, s_md_SN],
            tol=1e-12
        )  # [B,1]

        # --- Blend (broadcast alpha) ---
        p_new  = (1 - alpha) * p  + alpha * pSN
        f_new  = (1 - alpha) * f  + alpha * fSN
        md_new = (1 - alpha) * md + alpha * mdSN

        return p_new, f_new, md_new, alpha

    def forward(self, x, total_demands=None):

        eq_rhs, ineq_rhs = self.data.split_X(x)  # Eq_rhs: [B, N_eqrhs], Ineq_rhs: [B, N_ineqrhs]
        # print(f"In Primal Eq Rhs shape: {eq_rhs[0,:]}, Ineq Rhs shape: {ineq_rhs[0,:]} X SHape {x[0,:]} ")
        if self.normalize:
            x = self.scale(x)
            pass


        x_out = self.feed_forward(x)

        if not self.args["repair"]:
            # TODO: if normalize, rescale it.
            return x_out
        # [B, G, T], [B, L, T]
        if self.args["repair_completion"]:
            ui_g, p_gt, f_lt = self.data.split_dec_vars_from_Y_raw(x_out)
        else:
            p_gt, f_lt, md_nt = self.data.split_dec_vars_from_Y(x_out)

        # [B, bounds, T]
        p_gt_lb, p_gt_ub, f_lt_lb, f_lt_ub, md_nt_lb, md_nt_ub = self.data.split_ineq_constraints(ineq_rhs)


        if self.normalize:
            p_gt, f_lt = self.inverse_scale(p_gt, f_lt)

        if self.args["repair_bounds"]:
            p_gt = self.bound_repair_layer(p_gt, p_gt_lb, p_gt_ub)
                # Lineflow lower bound is negative.
                #! Note: Lineflows cannot be repaired more, since they depend on other lineflows. 
                #! For example, if we repair a lineflow such that it cannot export more than (imports + generation),
                #! then it will affect other lineflows, and we would need to repair them again.
            if "flow_scale_by_prod" in self.args and self.args["flow_scale_by_prod"]:
                # More restrictive way to repair line flows based on production capacity at nodes.  
                total_prod = torch.matmul(p_gt, self.data.node_to_gen_mask.T)  # [B, N]
 
                source_mask = (self.data.lineflow_mask == -1).to(self.DTYPE).to(self.DEVICE)  # [N, L]
                
                count_line_per_node = torch.count_nonzero(self.data.lineflow_mask, dim = 0)

                max_outflow_by_prod = torch.matmul(total_prod, source_mask) / count_line_per_node  # [B, L]

                source_mask = (self.data.lineflow_mask == 1).to(self.DTYPE).to(self.DEVICE)  # [N, L]
       
                '''
                Total Prod: (B,N) Mask: (N,L) --> (B,L) 
                '''
                max_inflow_by_prod = torch.matmul(total_prod, source_mask) / count_line_per_node # [B, L]

                f_lt_ub_adjusted = torch.min(f_lt_ub, max_outflow_by_prod)
                f_lt_lb_adjusted = torch.min(f_lt_lb, max_inflow_by_prod)
               
                f_lt = self.bound_repair_layer(f_lt, -f_lt_lb_adjusted, f_lt_ub_adjusted)
            else:
                f_lt = self.bound_repair_layer(f_lt, -f_lt_lb, f_lt_ub) #! Bounds need to be repaired for this to work!

    


        if self.args["repair_power_balance"]:
            net_flow = torch.matmul(f_lt, self.data.lineflow_mask.T)  # [B, N]
            updated_demand = eq_rhs - net_flow                                    # [B, N]
            # Clamp demand between lower and upper bound of total capacity, since generators must produce between that.
            total_capacity_lb = torch.matmul(p_gt_lb, self.data.node_to_gen_mask.T)
            total_capacity_ub = torch.matmul(p_gt_ub, self.data.node_to_gen_mask.T)
            demand_clamped = torch.clamp(updated_demand, min=total_capacity_lb, max=total_capacity_ub)

            # Repair generation.
            total_generation = torch.matmul(p_gt, self.data.node_to_gen_mask.T)
            # Calculate zeta_up and zeta_down using vectorized operations
            mask_up = (total_generation < demand_clamped).to(self.DTYPE)
            # Calculate zeta_up and zeta_down based on conditions
            zeta_up = (demand_clamped - total_generation) / ((total_capacity_ub - total_generation) + 1e-12)
            zeta_down = (total_generation - demand_clamped) / ((total_generation - total_capacity_lb) + 1e-12)

            # Expand masks to generator dimension.
            mask_up = torch.matmul(mask_up, self.data.node_to_gen_mask)
            zeta_up = torch.matmul(zeta_up, self.data.node_to_gen_mask)
            zeta_down = torch.matmul(zeta_down, self.data.node_to_gen_mask)
            # Apply the updates to p_gt based on the condition
            p_gt_repaired = torch.where(mask_up.bool(), (1 - zeta_up) * p_gt + zeta_up * p_gt_ub, (1 - zeta_down) * p_gt + zeta_down * p_gt_lb)
            # p_gt = torch.where(mask_down, (1 - zeta_down) * p_gt + zeta_down * p_gt_lb, p_gt)

            p_gt = p_gt_repaired


        if self.args["repair_completion"]:
            UI_g, D_nt = self.data.split_eq_constraints(eq_rhs)
            
            md_nt = self.estimate_slack_layer(p_gt, f_lt, D_nt)

        if self.args.get("use_blend_repair", False):
            # True bounds for f:
            f_lb_true = -f_lt_lb
            f_ub_true =  f_lt_ub

            # md bounds are already md_nt_lb, md_nt_ub (typically [0, D])
            # D_nt available from split_eq_constraints above
            p_gt, f_lt, md_nt, alpha = self._blend_repair(
                p=p_gt, f=f_lt, md=md_nt,
                p_lb=p_gt_lb, p_ub=p_gt_ub,
                f_lb=f_lb_true, f_ub=f_ub_true,
                md_lb=md_nt_lb, md_ub=md_nt_ub,
                D_nt=D_nt
            )
            # Store for logging potentially
            self.last_blend_alpha = alpha.detach()


        y = torch.cat([p_gt, f_lt, md_nt], dim=1)

        # Only scale if we are not training.
        if not self.training and (total_demands != None):
            # print(f"Multiplying by total demands: {total_demands} in eval mode")
            return y * total_demands
            # return y
        else:

            return y

class DualNetEndToEnd(nn.Module):
    def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
        
        self.args = args
        self.ED_args = args["ED_args"]

        if self.ED_args["benders_compact"]:
            self.out_dim = data.num_g + data.neq
        else:
            self.out_dim = data.neq

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        self.normalize = args.get("normalize", None)
        if self.normalize == "z_score":
            N = data.num_n
            G = data.num_g
            self.register_buffer("d_mean", torch.zeros(N, dtype=self.DTYPE))
            self.register_buffer("d_std",  torch.ones(N, dtype=self.DTYPE))

            # capacity stats
            self.register_buffer("p_mean", torch.zeros(G, dtype=self.DTYPE))
            self.register_buffer("p_std",  torch.ones(G, dtype=self.DTYPE))


        #! Only predict lambda, we infer mu from it.
        self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim, layernorm=True).to(self.DTYPE).to(self.DEVICE)

        self.register_buffer('f_l_lb', torch.tensor(
            [-data.pImpCap[l] for l in data.L], dtype=self.DTYPE
        ))
        self.register_buffer('f_l_ub', torch.tensor(
            [data.pExpCap[l] for l in data.L], dtype=self.DTYPE
        ))

    
    
    def scale(self, x):
        if self.normalize == "z_score":
            d = x[:, :self.data.n_md_vars]
            p = x[:, self.data.n_md_vars:]

            d = (d - self.d_mean) / self.d_std
            p = (p - self.p_mean) / self.p_std
            x_scale = torch.cat([d, p], dim=1)
        else:
            x_scale = x
        return x_scale
    
    def complete_duals(self, lamb):
        
        eq_cm_D_nt = self.data.eq_cm
        lamb_D_nt = lamb
        obj_coeff = self.data.obj_coeff

        mu = obj_coeff + torch.matmul(lamb_D_nt, eq_cm_D_nt)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-

        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :self.data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :self.data.num_g]  # Upper bounds for p_g

        f_l_lb = mu_lb[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Upper bounds for f_l

        md_n_lb = mu_lb[:, self.data.num_g + self.data.num_l:]  # Lower bounds for md_n
        md_n_ub = mu_ub[:, self.data.num_g + self.data.num_l:]  # Upper bounds for md_n

        # Concatenate while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=1)

        return out_mu
    
    def complete_duals_smooth(self, lamb, X, mu_barrier):
        """
        S3L smooth dual completion . Implemented from paper Dual Interior Point Optimization Learning
        When mu_barrier=0, reduces exactly to relu completion (Theorem 1).
        
        Args:
            lamb:        [B, neq]  — predicted equality duals (lambda)
            X:           [B, xdim] — input features (contains p_g_max and D_n)
            mu_barrier:  float     — barrier parameter, annealed toward 0 during training
        """
        # Compute reduced costs 
        z = self.data.obj_coeff + torch.matmul(lamb, self.data.eq_cm)  # [B, n_vars]

        # Step 2: extract bounds for each variable type
        n_g = self.data.num_g
        n_l = self.data.num_l
        n_n = self.data.num_n

        # Generator bounds: l=0, u=p_g_max (from input X)
        p_g_max = X[:, n_n:n_n + n_g]                    # [B, n_g]
        l_pg = torch.zeros_like(p_g_max)
        u_pg = p_g_max

        # Flow bounds: l=-f_l_max, u=f_l_max (fixed, from data)
        l_fl = self.f_l_lb.unsqueeze(0).expand(X.shape[0], -1)
        u_fl = self.f_l_ub.unsqueeze(0).expand(X.shape[0], -1)

        # Unmet demand bounds: l=0, u=D_n (from input X)
        D_n = X[:, :n_n]                                  # [B, n_n]
        l_md = torch.zeros_like(D_n)
        u_md = D_n

        # Concatenate bounds in same order as z
        l = torch.cat([l_pg, l_fl, l_md], dim=1)          # [B, n_vars]
        u = torch.cat([u_pg, u_fl, u_md], dim=1)          # [B, n_vars]

        # Step 3: smooth completion
        if mu_barrier > 0:
            # S3L Theorem 2 — smooth, always strictly positive
            v = mu_barrier / (u - l).clamp(min=1e-8)      # [B, n_vars]
            w = z / 2                                       # [B, n_vars]
            sq = torch.sqrt(v**2 + w**2)                   # [B, n_vars]

            mu_lb = v + w + sq                             # always > 0
            mu_ub = v - w + sq                             # always > 0
        else:
            # S3L Theorem 1 — reduces to relu when barrier=0
            mu_lb = torch.relu(z)
            mu_ub = torch.relu(-z)

        # Step 4: split into variable groups — same structure as before
        p_g_lb = mu_lb[:, :n_g]
        p_g_ub = mu_ub[:, :n_g]

        f_l_lb = mu_lb[:, n_g:n_g + n_l]
        f_l_ub = mu_ub[:, n_g:n_g + n_l]

        md_n_lb = mu_lb[:, n_g + n_l:]
        md_n_ub = mu_ub[:, n_g + n_l:]

        out_mu = torch.cat([
            p_g_lb, p_g_ub,
            f_l_lb, f_l_ub,
            md_n_lb, md_n_ub
        ], dim=1)

        return out_mu
    
        
    def forward(self, x, mu = None):
        x_raw = x
        if self.normalize:
            x = self.scale(x)

        
        out_lamb = self.feed_forward(x)

        if self.args.get("clamp_lambda", False):
            lambda_min = -float(self.data.pVOLL)                    # -10.0
            lambda_max = -float(self.data.cost_vec.min().item())    # -0.0001
            out_lamb = lambda_min + (lambda_max - lambda_min) * torch.sigmoid(out_lamb)

        if self.args.get("dual_regularization", "NA") == "S3L" and mu is not None:
            out_mu = self.complete_duals_smooth(out_lamb, x_raw, mu_barrier=mu)
        else:
            out_mu = self.complete_duals(out_lamb)
        # print(out_lamb)
        return out_mu, out_lamb
    



    
class DualClassificationNetEndToEnd(nn.Module):
    def __init__(self, args, data, hidden_size_factor=5.0, n_layers=4):
        super().__init__()
        self.data = data
        self.hidden_sizes = [int(hidden_size_factor*data.xdim)] * n_layers
        self.args = args
        self.ED_args = args["ED_args"]

        # Objective coefficients contain all costs for all generators and unmet demand.
        self.classes = -1 * torch.concat([self.data.cost_vec.unique(), torch.tensor([self.data.pVOLL])])
        self.n_classes = self.classes.numel()
        self.n_dual_vars = data.neq
        
        #! For each dual variable, We now predict probabilities for each class
        self.out_dim = self.n_classes * self.n_dual_vars

        if args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        self.normalize = args.get("normalize", None)
        if self.normalize == "z_score":
            N = data.num_n
            G = data.num_g
            self.register_buffer("d_mean", torch.zeros(N, dtype=self.DTYPE))
            self.register_buffer("d_std",  torch.ones(N, dtype=self.DTYPE))

            # capacity stats
            self.register_buffer("p_mean", torch.zeros(G, dtype=self.DTYPE))
            self.register_buffer("p_std",  torch.ones(G, dtype=self.DTYPE))

        #! Only predict lambda, we infer mu from it.
        #! Softmax requires layer norm.
        if args.get("SeperatePredicationHead", False):
            self.feed_forward = FeedForwardNetSeparateHead(
                args,
                input_dim=data.xdim,
                hidden_sizes=self.hidden_sizes,
                n_heads=self.n_dual_vars,
                n_classes=self.n_classes
            ).to(self.DTYPE).to(self.DEVICE)
            # print("======= Using separate prediction heads for each dual variable. ========== ")
        else:
            self.feed_forward = FeedForwardNet(args, data.xdim, self.hidden_sizes, output_dim=self.out_dim, layernorm=True).to(self.DTYPE).to(self.DEVICE)


        self.register_buffer('f_l_lb', torch.tensor(
            [-data.pImpCap[l] for l in data.L], dtype=self.DTYPE
        ))
        self.register_buffer('f_l_ub', torch.tensor(
            [data.pExpCap[l] for l in data.L], dtype=self.DTYPE
        ))

        self.tau = 1.0
            
    def scale(self, x):
        if self.normalize == "z_score":
            d = x[:, :self.data.num_n]
            p = x[:, self.data.num_n:]

            d = (d - self.d_mean) / self.d_std
            p = (p - self.p_mean) / self.p_std
            x_scale = torch.cat([d, p], dim=1)
        else:
            x_scale = x
        return x_scale
    
    def complete_duals(self, lamb):
        eq_cm_D_nt = self.data.eq_cm
        lamb_D_nt = lamb
        obj_coeff = self.data.obj_coeff

        mu = obj_coeff + torch.matmul(lamb_D_nt, eq_cm_D_nt)

        # Compute lower and upper bound multipliers
        mu_lb = torch.relu(mu)   # Lower bound multipliers |mu|^+
        mu_ub = torch.relu(-mu)  # Upper bound multipliers |mu|^-

        # Split into groups, following the exact structure of mu
        p_g_lb = mu_lb[:, :self.data.num_g]  # Lower bounds for p_g
        p_g_ub = mu_ub[:, :self.data.num_g]  # Upper bounds for p_g

        f_l_lb = mu_lb[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Lower bounds for f_l
        f_l_ub = mu_ub[:, self.data.num_g:self.data.num_g + self.data.num_l]  # Upper bounds for f_l

        md_n_lb = mu_lb[:, self.data.num_g + self.data.num_l:]  # Lower bounds for md_n
        md_n_ub = mu_ub[:, self.data.num_g + self.data.num_l:]  # Upper bounds for md_n

        # Concatenate while maintaining order
        out_mu = torch.cat([
            p_g_lb, p_g_ub,  # Lower and Upper bounds for p_g
            f_l_lb, f_l_ub,  # Lower and Upper bounds for f_l
            md_n_lb, md_n_ub  # Lower and Upper bounds for md_n
        ], dim=1)

        return out_mu
        
    def complete_duals_smooth(self, lamb, X, mu_barrier):
        """
        S3L smooth dual completion . Implemented from paper Dual Interior Point Optimization Learning
        When mu_barrier=0, reduces exactly to relu completion (Theorem 1).
        
        Args:
            lamb:        [B, neq]  — predicted equality duals (lambda)
            X:           [B, xdim] — input features (contains p_g_max and D_n)
            mu_barrier:  float     — barrier parameter, annealed toward 0 during training
        """
        # Compute reduced costs 
        z = self.data.obj_coeff + torch.matmul(lamb, self.data.eq_cm)  # [B, n_vars]

        # Step 2: extract bounds for each variable type
        n_g = self.data.num_g
        n_l = self.data.num_l
        n_n = self.data.num_n

        # Generator bounds: l=0, u=p_g_max (from input X)
        p_g_max = X[:, n_n:n_n + n_g]                    # [B, n_g]
        l_pg = torch.zeros_like(p_g_max)
        u_pg = p_g_max

        # Flow bounds: l=-f_l_max, u=f_l_max (fixed, from data)
        l_fl = self.f_l_lb.unsqueeze(0).expand(X.shape[0], -1)
        u_fl = self.f_l_ub.unsqueeze(0).expand(X.shape[0], -1)

        # Unmet demand bounds: l=0, u=D_n (from input X)
        D_n = X[:, :n_n]                                  # [B, n_n]
        l_md = torch.zeros_like(D_n)
        u_md = D_n

        # Concatenate bounds in same order as z
        l = torch.cat([l_pg, l_fl, l_md], dim=1)          # [B, n_vars]
        u = torch.cat([u_pg, u_fl, u_md], dim=1)          # [B, n_vars]

        # Step 3: smooth completion
        if mu_barrier > 0:
            # S3L Theorem 2 — smooth, always strictly positive
            v = mu_barrier / (u - l).clamp(min=1e-8)      # [B, n_vars]
            w = z / 2                                       # [B, n_vars]
            sq = torch.sqrt(v**2 + w**2)                   # [B, n_vars]

            mu_lb = v + w + sq                             # always > 0
            mu_ub = v - w + sq                             # always > 0
        else:
            # S3L Theorem 1 — reduces to relu when barrier=0
            mu_lb = torch.relu(z)
            mu_ub = torch.relu(-z)

        # Step 4: split into variable groups — same structure as before
        p_g_lb = mu_lb[:, :n_g]
        p_g_ub = mu_ub[:, :n_g]

        f_l_lb = mu_lb[:, n_g:n_g + n_l]
        f_l_ub = mu_ub[:, n_g:n_g + n_l]

        md_n_lb = mu_lb[:, n_g + n_l:]
        md_n_ub = mu_ub[:, n_g + n_l:]

        out_mu = torch.cat([
            p_g_lb, p_g_ub,
            f_l_lb, f_l_ub,
            md_n_lb, md_n_ub
        ], dim=1)

        return out_mu
    
    def forward(self, x, mu=None):
        x_raw = x
        if self.normalize:
            x = self.scale(x)
        out_lamb_raw_probas = self.feed_forward(x)
        out_lamb_raw_probas = out_lamb_raw_probas.view(-1, self.n_dual_vars, self.n_classes)

        if self.training:
            if self.args.get("dual_gumbel", False):
                # tau = getattr(self, 'tau', 1.0)
                # y_hard = F.gumbel_softmax(
                #     out_lamb_raw_probas,
                #     tau=tau,
                #     hard=True,
                #     dim=-1
                # )
                # out_lamb = (y_hard * self.classes).sum(dim=-1)
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(out_lamb_raw_probas) + 1e-20) + 1e-20)
                noisy_logits = (out_lamb_raw_probas + gumbel_noise) / self.tau

                # Soft probabilities (differentiable)
                probs = torch.softmax(noisy_logits, dim=-1)

                # Straight-through for hard selection
                hard_idx = probs.argmax(dim=-1, keepdim=True)
                y_hard = torch.zeros_like(probs).scatter_(-1, hard_idx, 1.0)
                st = y_hard - probs.detach() + probs

                out_lamb = (st * self.classes).sum(dim=-1)

            else:
                out_lamb_probas = torch.softmax(out_lamb_raw_probas, dim=-1)
                out_lamb = torch.sum(out_lamb_probas * self.classes, dim=-1)

            if self.args.get("dual_regularization", "NA") == "S3L" and mu is not None:
                out_mu = self.complete_duals_smooth(out_lamb, x_raw, mu_barrier=mu)
            else:
                out_mu = self.complete_duals(out_lamb)
            return out_mu, out_lamb

        else:
            predicted_class = out_lamb_raw_probas.argmax(dim=-1)
            out_lamb = self.classes[predicted_class]
            if self.args.get("dual_regularization", "NA") == "S3L" and mu is not None:
                out_mu = self.complete_duals_smooth(out_lamb, x_raw, mu_barrier=mu)
            else:
                out_mu = self.complete_duals(out_lamb)
            return out_mu, out_lamb
    
         
def load(args, data, save_dir):
    primal_net = PrimalNetEndToEnd(args, data=data)
    primal_net.load_state_dict(torch.load(save_dir + '/primal_weights.pth', weights_only=True))
    dual_net = DualNetEndToEnd(args, data=data)
    dual_net.load_state_dict(torch.load(save_dir + '/dual_weights.pth', weights_only=True))

    return primal_net, dual_net



if __name__ == "__main__":
    import json
    import os
    import numpy as np
    import pickle
    from gep_config_parser import parse_config

    def compute_node_power_balance_debug(data, X, Y):
        """
        Compute net production, net inflow, combined flow, unmet demand and mask
        for a batch X, Y (here you use batch size 1).
        """
        # Split equality RHS (to get demand D_nt)
        eq_rhs, _ = data.split_X(X)          # [B, neq]
        UI_g, D_nt = data.split_eq_constraints(eq_rhs)  # D_nt: [B, N]

        # Split decision vars from Y (already repaired output of PrimalNetEndToEnd)
        p_gt, f_lt, md_nt = data.split_dec_vars_from_Y(Y)  # [B, G], [B, L], [B, N]

        # Net production per node: p_g * node_to_gen_mask^T
        net_prod = torch.matmul(p_gt, data.node_to_gen_mask.T)   # [B, N]

        # Net flow per node: f_l * lineflow_mask^T
        net_flow = torch.matmul(f_lt, data.lineflow_mask.T)      # [B, N]

        # Combined flow = generation + net inflow (this matches your EstimateSlackLayer)
        combined_flow = net_prod + net_flow                      # [B, N]

        # Unmet demand according to the same formula as EstimateSlackLayer
        md_est = D_nt - combined_flow                            # [B, N]

        # Mask: where unmet demand > demand (your current condition)
        mask = md_est > -0.00001                                     # [B, N]

        return {
            "net_prod": net_prod.detach().cpu(),
            "net_flow": net_flow.detach().cpu(),
            "combined_flow": combined_flow.detach().cpu(),
            "D_nt": D_nt.detach().cpu(),
            "md_est": md_est.detach().cpu(),
            "mask": mask.detach().cpu(),
        }


    def compute_power_balance_debug(data, X, Y):
        """
        Compute net production, net inflow, combined flow, unmet demand and mask
        for a batch X, Y (here you use batch size 1).
        """
        # Split equality RHS (to get demand D_nt)
        eq_rhs, _ = data.split_X(X)          # [B, neq]
        UI_g, D_nt = data.split_eq_constraints(eq_rhs)  # D_nt: [B, N]

        # Split decision vars from Y (already repaired output of PrimalNetEndToEnd)
        p_gt, f_lt, md_nt = data.split_dec_vars_from_Y(Y)  # [B, G], [B, L], [B, N]

        # Net production per node: p_g * node_to_gen_mask^T
        net_prod = torch.matmul(p_gt, data.node_to_gen_mask.T)   # [B, N]

        # Net flow per node: f_l * lineflow_mask^T
        net_flow = torch.matmul(f_lt, data.lineflow_mask.T)      # [B, N]


        # Combined flow = generation + net inflow (this matches your EstimateSlackLayer)
        combined_flow = net_prod + net_flow                      # [B, N]

        # Unmet demand according to the same formula as EstimateSlackLayer
        md_est = D_nt - combined_flow                            # [B, N]

        # Mask: where unmet demand > demand (your current condition)
        mask = md_est > D_nt                                     # [B, N]


        
        B = p_gt.shape[0]
        assert B == 1, "Printing per-node flows only makes sense for batch size 1 in debugging."

        p_gt_np = p_gt[0].detach().cpu().numpy()
        f_lt_np = f_lt[0].detach().cpu().numpy()

        node_to_gen_mask = data.node_to_gen_mask.cpu().numpy()
        lineflow_mask = data.lineflow_mask.cpu().numpy()

        print("\n===== Detailed per-node breakdown =====")
        for n in range(node_to_gen_mask.shape[0]):
            print(f"\n--- Node {n} ---")

            # Generator production at this node
            gens_at_node = node_to_gen_mask[n] == 1
            node_gen_prod = p_gt_np[gens_at_node]
            
            print("Generators at node:", node_gen_prod)

            # Flows involving this node
            flows_mask = lineflow_mask[n]  # +1 inflow, -1 outflow, 0 no connection
            
            inflow_indices  = np.where(flows_mask ==  1)[0]
            outflow_indices = np.where(flows_mask == -1)[0]

            print("Incoming flows (line index → flow value):")
            for li in inflow_indices:
                print(f"  Line {li}: +{f_lt_np[li]:.2f}")

            print("Outgoing flows (line index → flow value):")
            for li in outflow_indices:
                print(f"  Line {li}: {f_lt_np[li]:.2f}")

            # Net flow = sum(flows * +1/-1)
            net_flow_n = np.sum(f_lt_np * flows_mask)
            print(f"Net inflow/outflow for node {n}: {net_flow_n:.2f}")
            print(f"Combined flow: {combined_flow[0,n].item():.2f}")
            print(f"Unmet Demand at node {n}: {md_est[0,n].item():.2f} Mask: {mask[0,n].item()}")
        return {
            "net_prod": net_prod.detach().cpu(),
            "net_flow": net_flow.detach().cpu(),
            "combined_flow": combined_flow.detach().cpu(),
            "D_nt": D_nt.detach().cpu(),
            "md_est": md_est.detach().cpu(),
            "mask": mask.detach().cpu(),
        }


    ARGS_FILE_NAME = "config.json"
    CONFIG_FILE_NAME = "config.toml"

    with open(ARGS_FILE_NAME, "r") as file:
            args = json.load(file)

    ED_args = args["ED_args"]

    input_data = parse_config(CONFIG_FILE_NAME) # Reads the input data using config.toml's experiment.inputs.data path.

    gep_ed_data = input_data["experiment"]["experiments"][0] # Take first experiment, we don't change the inputs here.

    if args["problem_type"] == "ED":
        #! TODO: not all configs are correctly parsed here. E.g. when first running BEL and GER with both coal generators, is the same as with both gas generators.
        # For nodes, just use first letters: ['BEL', 'GER', 'NED'] → 'B-G-N'
        nodes_str = "-".join([n[0] for n in ED_args['N']])
        
        # For generators, count per node: [['BEL', 'WindOn'], ['BEL', 'Gas'],...] = 'B3-G2-N2'
        gen_counts = {}
        for g in ED_args['G']:
            node = g[0]
            gen_counts[node] = gen_counts.get(node, 0) + 1
        gens_str = "-".join([f"{node[0]}{count}" for node, count in gen_counts.items()])
        
        # For lines, just count: [['BEL', 'GER'], ['BEL', 'NED'], ['GER', 'NED']] → 'L3'
        lines_str = f"L{len(ED_args['L'])}"
        
        # Create a shortened filename
        data_save_path = (f"data/ED_data/ED_N{nodes_str}_G{gens_str}_{lines_str}"
                        f"_c{int(ED_args['benders_compact'])}"
                        f"_s{int(ED_args['scale_problem'])}"
                        f"_p{int(ED_args['perturb_operating_costs'])}"
                        f"_smp{ED_args['2n_synthetic_samples']}.pkl")

    def evaluate_individual(data, primal_net, test_indices, index):      

        X = data.X[test_indices]
        print(f"X is {X.shape}")
        Y_target = data.opt_targets["y_operational"][test_indices]
        

        X = X[index,:].unsqueeze(0)
        Y_target = Y_target[index,:].unsqueeze(0)

    
        # Forward pass through networks
        Y = primal_net(X)

        ineq_dist = data.ineq_dist(X, Y)
        eq_resid = data.eq_resid(X, Y)


        relative_ineq_dist = data.relative_ineq_dist(X, Y)
        relative_eq_resid = data.relative_eq_resid(X, Y)

        # Convert lists to arrays for easier handling
        obj_values = data.obj_fn(X, Y).detach().numpy()
        ineq_max_vals = torch.max(ineq_dist, dim=1)[0].detach().numpy() # First element is the max, second is the index
        ineq_mean_vals = torch.mean(ineq_dist, dim=1).detach().numpy()
        eq_max_vals = torch.max(torch.abs(eq_resid), dim=1)[0].detach().numpy() # First element is the max, second is the index
        eq_mean_vals = torch.mean(torch.abs(eq_resid), dim=1).detach().numpy()
        known_obj = data.obj_fn(X, Y_target).detach().numpy()
        # obj_values is negative
        opt_gap = (obj_values - known_obj)/np.abs(known_obj) * 100

        return np.mean(obj_values), np.mean(known_obj), np.mean(opt_gap), np.mean(ineq_max_vals), np.mean(ineq_mean_vals), np.mean(eq_max_vals), np.mean(eq_mean_vals), ineq_max_vals, ineq_mean_vals, ineq_dist, X, Y, Y_target


    repeats = 1
    stats_dict = {}
    const_vio_dict = {}
    data_ineq_list = []

    # args = json.load(open('config.json'))
    data_path = f"experiment-output/ch5/ED_NB-G-F_GB2-G2-F2_L3_c0_s0_p0_smp15.pkl"
    data = pickle.load(open(data_path, 'rb'))

    indices = torch.arange(data.X.shape[0])



    exp_paths = ["outputs/PDL/ED/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-NoNormRepairAll"]
    name_list = ["Baseline"]

    exp_paths = ["outputs/PDL/ED/3Nodes-FraBelGer/learn_primal:True_train:0.8_rho:0.5_rhomax:5000_alpha:10_L:10-OriginalScaledSig100"]
    name_list = ["Baseline"]

    ARGS_FILE_NAME = "config.json"


    for run, (path, name) in enumerate(zip(exp_paths, name_list)):
        stats_dict[name] = {"predicted_obj": [], "known_obj": [], "opt_gap": [], "ineq_max": [], "ineq_mean": [], "eq_max": [], "eq_mean": []}
        const_vio_dict[name] = {"ineq_max_list": [], "ineq_mean_list": []}
        with open(os.path.join(path, 'args.json'), 'r') as f:
            args = json.load(f)
        # with open(ARGS_FILE_NAME, "r") as file:
        #     args = json.load(file)
        # Compute sizes for each set
        train_size = int(args["train"] * data.X.shape[0])
        valid_size = int(args["valid"] * data.X.shape[0])
        # print(f"Train size: {train_size}, Valid size: {valid_size}, Test size: {data.X.shape[0] - train_size - valid_size}")


        # Split the indices
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size+valid_size]
        test_indices = indices[train_size+valid_size:]

        args["hidden_size_factor"] = 28 # Set as this for some reason
        for i, repeat in enumerate(range(repeats)):
            
            directory = os.path.join(path, f"repeat:{repeat}")
            # directory = f"experiment-output/ch5-reproduction-nonconvex/{experiment}/repeat:{repeat}"
            # dual_net = DualNet(args, data=data)
            # dual_net.load_state_dict(torch.load(os.path.join(directory, 'dual_weights.pth'), weights_only=True))
            
            primal_net = PrimalNetEndToEnd(args, data=data)
            
            primal_net.load_state_dict(torch.load(os.path.join(directory, 'primal_weights.pth'), weights_only=True))
            
            index = 2957
            obj_val, known_obj, opt_gap, ineq_max, ineq_mean, eq_max, eq_mean, ineq_max_list, ineq_mean_list, ineq_dist, X_out, Y_out, Y_target = evaluate_individual(data, primal_net, test_indices, index)
            debug_stats = compute_node_power_balance_debug(data, X_out, Y_out)
            debug_opt_obj = data.obj_fn(X_out, Y_target).detach().numpy()
            # print(ineq_max_list)
            data_ineq_list.append(data.ineq_cm.numpy())
            stats_dict[name]["predicted_obj"].append(obj_val)
            stats_dict[name]["known_obj"].append(known_obj)
            stats_dict[name]["opt_gap"].append(opt_gap)
            stats_dict[name]["ineq_max"].append(ineq_max)
            stats_dict[name]["ineq_mean"].append(ineq_mean)
            stats_dict[name]["eq_max"].append(eq_max)
            stats_dict[name]["eq_mean"].append(eq_mean)
            const_vio_dict[name]["opt_gap"] = opt_gap
            const_vio_dict[name]["ineq_max_list"] = ineq_max_list
            const_vio_dict[name]["ineq_mean_list"] = ineq_mean_list
            const_vio_dict[name]["ineq_dist"] = ineq_dist
            const_vio_dict[name]["X"] = X_out
            const_vio_dict[name]["Y"] = Y_out
            const_vio_dict[name]["Y_target"] = Y_target
            const_vio_dict[name]["debug"] = debug_stats
            
            print(f"opt gap predicted vs known: {obj_val} vs {known_obj}, opt gap: {opt_gap}%")

    num_nodes = data.node_to_gen_mask.shape[0]  # e.g. 3 nodes
    lineflow_mask_np = data.lineflow_mask.cpu().numpy()  # [N, L]
    print(f"Ineq Dist: {ineq_max_list}")
    for name, d in const_vio_dict.items():
        dbg = d["debug"]

        # Reconstruct line flows f_lt for this method (batch size 1)
        Y = d["Y"]                       # [1, ydim]
        
        p_gt_node, f_lt_node, md_nt_node = data.split_dec_vars_from_Y(Y)
        f_lt_np = f_lt_node[0].detach().cpu().numpy()   # [L]

        print(f"\n========== {name} ==========")
        print(f"opt gap: {d['opt_gap']:.4f}")
        
        ''' 
        Gen prediction  first 6 features,
        Flow prediction next 3 feature
        Unmet demand last 3 features
        '''


        import torch

        def print_instance_diffs(Y, Y_out, name="instance"):
            # Y, Y_out: shape [1, 12] or [12]
            y = Y.squeeze(0).detach().cpu()
            yhat = Y_out.squeeze(0).detach().cpu()

            diff = yhat - y

            groups = [
                ("Generators", slice(0, 6)),
                ("Flows",      slice(6, 9)),
                ("Unmet",      slice(9, 12)),
            ]

            print(f"\n==== {name} ====")
            for gname, sl in groups:
                print(f"\n-- {gname} --")
                for i, (optv, predv, dv) in enumerate(zip(y[sl], yhat[sl], diff[sl])):
                    # i is within-group index; global index is sl.start + i
                    gi = (sl.start or 0) + i
                    print(i)
                    print(f"idx {gi:02d}: opt={optv.item(): .6f}  pred={predv.item(): .6f}  diff(pred-opt)={dv.item(): .6f} cost={data.cost_vec[i].item()}")

        # usage:
        Y_gt = const_vio_dict["Baseline"]["Y_target"]
        Y_out = const_vio_dict["Baseline"]["Y"]
        X_out = const_vio_dict["Baseline"]["X"]
        # print_instance_diffs(Y_gt, Y_out, name="Baseline (single sample)")

        
        

        pred_node_balance = compute_node_power_balance_debug(data, X_out, Y_out)
        gt_node_balance = compute_node_power_balance_debug(data, X_out, Y_gt)

        # === Extract raw decision variables ===
        p_gt_pred, f_gt_pred, md_pred = data.split_dec_vars_from_Y(Y_out)
        p_gt_gt,   f_gt_gt,   md_gt   = data.split_dec_vars_from_Y(Y_gt)

        # === Recompute balance residuals explicitly ===
        eq_rhs, _ = data.split_X(X_out)
        _, D_nt = data.split_eq_constraints(eq_rhs)

        net_prod_pred = p_gt_pred @ data.node_to_gen_mask.T
        net_flow_pred = f_gt_pred @ data.lineflow_mask.T
        balance_resid_pred = D_nt - (net_prod_pred + net_flow_pred + md_pred)

        net_prod_gt = p_gt_gt @ data.node_to_gen_mask.T
        net_flow_gt = f_gt_gt @ data.lineflow_mask.T
        balance_resid_gt = D_nt - (net_prod_gt + net_flow_gt + md_gt)

        combined_pred = net_prod_pred + net_flow_pred
        combined_gt   = net_prod_gt   + net_flow_gt

        print(f"X is {X_out}")


        # print("\n--- Baseline Node-wise Power Balance Debug Info ---")  
        # print("GT combined_flow vs Pred combined_flow:")
        # for n in range(num_nodes):
        #     gt_cf = gt_node_balance["combined_flow"][0,n].item()
        #     pred_cf = pred_node_balance["combined_flow"][0,n].item()

        #     gt_md = gt_node_balance["md_est"][0,n].item()
        #     pred_md = pred_node_balance["md_est"][0,n].item()
        #     print(f"Node {n}: GT combined flow = {gt_cf:.6f}, Pred combined flow = {pred_cf:.6f}, Diff = {pred_cf - gt_cf:.6f}")
        #     print(f"         GT unmet demand = {gt_md:.6f}, Pred unmet demand = {pred_md:.6f}, Diff = {pred_md - gt_md:.6f}")


        # print(X_out.shape)
        # print(data.cost_vec.shape)

        # print(data.node_to_gen_mask)

        # print(const_vio_dict["Baseline"]["ineq_dist"])