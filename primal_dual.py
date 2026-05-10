import copy
import os
import time
import torch
from torch.utils.data import TensorDataset, DataLoader
from logger import TensorBoardLogger
import numpy as np
import pandas as pd
from mtadam import MTAdam
from networks import DualClassificationNetEndToEnd, DualNet, DualNetEndToEnd, PrimalNet, PrimalNetEndToEnd
import optuna
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(False)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

class EarlyStopping():
    def __init__(self, patience=1000):
        self.patience = patience        # epochs to wait after last improvement
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
class PrimalDualTrainer():

    def __init__(self, data, args, save_dir):
        """_summary_

        Args:
            data (_type_): _description_
            args (_type_): _description_
            save_dir (_type_): _description_
            problem_type (str, optional): Either "GEP" "ED" or "QP". Defaults to "ED".
            optimal_objective_train (_type_, optional): _description_. Defaults to None.
            optimal_objective_val (_type_, optional): _description_. Defaults to None.
            log (bool, optional): _description_. Defaults to True.
            optuna (bool, optional): Whether to use Optuna. Defaults to False.
        """

        print(f"X dim: {data.xdim}")
        print(f"Y dim: {data.ydim}")

        print(f"Size of mu: {data.nineq}")
        print(f"Size of lambda: {data.neq}")

        self.data = data
        self.args = args
        self.save_dir = save_dir
        self.problem_type = args["problem_type"]
        self.log = args["log"]
        self.log_frequency = args["log_frequency"]
        
        if self.args["device"] == "mps":
            self.DTYPE = torch.float32
            self.DEVICE = torch.device("mps")
            self.data.to_mps()
        else:
            self.DTYPE = torch.float64
            self.DEVICE = torch.device("cpu")

        torch.set_default_dtype(self.DTYPE)
        torch.set_default_device(self.DEVICE)

        print(f"DTYPE: {self.DTYPE}, DEVICE: {self.DEVICE}")

        self.train = args["train"]
        self.valid = args["valid"]
        self.test = args["test"]
        self.outer_iterations = args["outer_iterations"]
        self.inner_iterations = args["inner_iterations"]
        self.tau = args["tau"] # Tolerance scalar, determine how much violation improvement is enough to not increase rho
        self.rho_init = float(args["rho"]) # Lagrangian penalty parameter, updated during training, increase if violation does not decrease sufficiently
        self.rho_max = float(args["rho_max"]) # Maximum rho, to prevent it from becoming too large when increasing rho 
        self.rho = self.rho_init
        self.rho_ineq = None
        self.rho_eq = None

        ### Difficulty-based sample weighting
        self.use_difficulty_weighting = args.get("use_difficulty_weighting", False)
        self.difficulty_weight_scale = float(args.get("difficulty_weight_scale", 1.0))
        self.difficulty_weight_clamp_min = float(args.get("difficulty_weight_clamp_min", 0.3))
        self.difficulty_weight_clamp_max = float(args.get("difficulty_weight_clamp_max", 3.0))

        self.prev_v_ineq = None
        self.prev_v_eq = None
        if args["penalty"] == "Single":
            # Initialize a single rho for all constraints
            self.rho = args["rho"] 
            self.rho_max = args["rho_max"] 
        elif args["penalty"] == "Per_Const":
            # Initialize a rho for each constraint
            self.rho_ineq = torch.full((self.data.nineq,), self.rho_init, dtype=self.DTYPE, device=self.DEVICE)
            self.rho_eq   = torch.full((self.data.neq,),   self.rho_init, dtype=self.DTYPE, device=self.DEVICE)
        else:
            # Raise error
            raise ValueError(f"Penalty option {args['penalty']} not supported")

        self.alpha = args["alpha"] # Growth factor for rho, when it needs to be increased
        self.batch_size = args["batch_size"]
        self.primal_lr = args["primal_lr"]
        self.dual_lr = args["dual_lr"]
        self.decay = args["decay"]
        self.patience = args["patience"]
        self.clip_gradients_norm = args["clip_gradients_norm"]
        self.max_violation_save_thresholds = args["max_violation_save_thresholds"]  
        self.early_stopping_patience = args["early_stopping_patience"]
        self.X = data.X.to(self.DTYPE).to(self.DEVICE)

        if self.args.get("use_topology_features", False):
            topo_features = self.build_topology_features(self.X)
            self.X = torch.cat([self.X, topo_features], dim=1) #  [BS, N+G+2*N]
            self.data.xdim += topo_features.shape[1]           #   X dim = N+G+2*N = 3N+G 
            print(f"Added topology features: {topo_features.shape[1]}")
            print(f"Topo features sample 0: {topo_features[10]}")

            topo_min = topo_features.min(dim=0).values
            topo_max = topo_features.max(dim=0).values
            topo_mean = topo_features.mean(dim=0)
            topo_std = topo_features.std(dim=0)

            print("\n=== Topology feature statistics ===")
            for i in range(topo_features.shape[1]):
                print(
                    f"topo_feature_{i:02d}: "
                    f"min={topo_min[i].item():.6f}, "
                    f"max={topo_max[i].item():.6f}, "
                    f"mean={topo_mean[i].item():.6f}, "
                    f"std={topo_std[i].item():.6f}"
                )
            print(f"New X dim: {self.data.xdim}")



        self.loss_option = args.get("loss_option", "Original")
       
        self.dual_regularization = args.get("dual_regularization", "NA")
        self.mu_barrier = float(args.get("mu_barrier_init", 1.0))       # Initial barrier weight (paper: mu^(0)=1)
        self.mu_barrier_decay = float(args.get("mu_barrier_decay", 0.99))  # Per-outer-iter decay  (paper: 0.99)
        self.mu_barrier_min = float(args.get("mu_barrier_min", 1e-4))    # Floor to avoid zero

        self.old_mu = self.mu_barrier

        if args.get("use_prod_surplus_input", False):
            surplus = torch.zeros(self.X.shape[0], self.data.num_n)
            d = self.X[:, :self.data.num_n]
            p = self.X[:, self.data.num_n:self.data.num_n + self.data.num_g]
            for i, node in enumerate(self.data.N):
                local_gen_mask = [j for j, g in enumerate(self.data.G) if g[0] == node]
                surplus[:, i] = d[:, i] - p[:, local_gen_mask].sum(dim=1)
            self.X = torch.cat([self.X, surplus], dim=1)
            self.data.xdim += self.data.num_n

                


        if self.problem_type == "ED":
            self.total_demands = data.total_demands.to(self.DTYPE).to(self.DEVICE)
        else:
            self.total_demands = torch.ones((self.X.shape[0], 1))
        
        # for logging
        self.step = 0
        indices = torch.arange(self.X.shape[0])
        # Compute sizes for each set
        train_size = int(self.train * self.X.shape[0])
        valid_size = int(self.valid * self.X.shape[0])
        print(f"Train size: {train_size}, Valid size: {valid_size}, Test size: {self.X.shape[0] - train_size - valid_size}")

        # Split the indices
        self.train_indices = indices[:train_size]
        self.valid_indices = indices[train_size:train_size+valid_size]
        self.test_indices = indices[train_size+valid_size:]

        self.X_train = self.X[self.train_indices]
        self.X_valid = self.X[self.valid_indices]
        self.total_demands_train = self.total_demands[self.train_indices]
        self.total_demands_valid = self.total_demands[self.valid_indices]


        if self.args.get("use_heuristic_lambda_loss", False):
            # Using Heuristic Soft Label in computing Loss

            self.heur_soft_all = self.data.heuristic_lambda_soft_labels.to(self.DTYPE).to(self.DEVICE)
            self.heur_conf_all = self.data.heuristic_lambda_confidence.to(self.DTYPE).to(self.DEVICE)
            self.heur_tier_all = self.data.heuristic_lambda_tier.to(self.DEVICE)

            self.heur_soft_train = self.heur_soft_all[self.train_indices]
            self.heur_conf_train = self.heur_conf_all[self.train_indices]
            self.heur_tier_train = self.heur_tier_all[self.train_indices]

            self.heur_soft_valid = self.heur_soft_all[self.valid_indices]
            self.heur_conf_valid = self.heur_conf_all[self.valid_indices]
            self.heur_tier_valid = self.heur_tier_all[self.valid_indices]

        else:
            self.heur_soft_train = None
            self.heur_conf_train = None
            self.heur_tier_train = None
            self.heur_soft_valid = None
            self.heur_conf_valid = None
            self.heur_tier_valid = None

        self.mu_targets_all = self.data.opt_targets["mu_operational"].to(self.DTYPE).to(self.DEVICE)
        self.lamb_targets_all = self.data.opt_targets["lamb_operational"].to(self.DTYPE).to(self.DEVICE)

        self.mu_targets_train = self.mu_targets_all[self.train_indices]
        self.lamb_targets_train = self.lamb_targets_all[self.train_indices]

        self.mu_targets_valid = self.mu_targets_all[self.valid_indices]
        self.lamb_targets_valid = self.lamb_targets_all[self.valid_indices]

        if self.log == True:
            self.logger = TensorBoardLogger(args, data, self.X, self.total_demands, self.train_indices, self.valid_indices, save_dir, args["opt_targets"])
        else:
            self.logger = None

        self.opt_targets_train = self.data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.train_indices]
        self.target_obj_train = self.data.obj_fn(self.X_train, self.opt_targets_train)

        self.opt_target_val = self.data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
        self.target_obj_val  = self.data.opt_targets["obj"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
        
        if self.args.get("use_heuristic_lambda_loss", False):
            self.train_dataset = TensorDataset(
                self.X_train,
                self.total_demands_train,
                self.target_obj_train,
                self.mu_targets_train,
                self.lamb_targets_train,
                self.heur_soft_train,
                self.heur_conf_train,
                self.heur_tier_train,
            )

            self.valid_dataset = TensorDataset(
                self.X_valid,
                self.total_demands_valid,
                self.target_obj_val,
                self.mu_targets_valid,
                self.lamb_targets_valid,
                self.heur_soft_valid,
                self.heur_conf_valid,
                self.heur_tier_valid,
            )
        else:
            self.train_dataset = TensorDataset(
                self.X_train,
                self.total_demands_train,
                self.target_obj_train,
                self.mu_targets_train,
                self.lamb_targets_train,
            )

            self.valid_dataset = TensorDataset(
                self.X_valid,
                self.total_demands_valid,
                self.target_obj_val,
                self.mu_targets_valid,
                self.lamb_targets_valid,
            )
            
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, generator=torch.Generator(device=self.DEVICE))
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=len(self.valid_dataset), generator=torch.Generator(device=self.DEVICE))
        # self.test_loader = DataLoader(self.test_dataset, batch_size=len(self.test_dataset))


        if self.problem_type == "QP":
            self.primal_loss_fn = self.primal_loss_QP
            self.dual_loss_fn = self.dual_loss
            self.primal_net = PrimalNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
            self.dual_net = DualNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
        elif self.problem_type == "ED":
            self.primal_loss_fn = self.primal_loss
            #! PrimalNetEndToEnd takes into account whether repairs are used or not.
            self.primal_net = PrimalNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
            
            if self.args["dual_alternate_loss"]:
                if self.args["dual_classification"]:
                    self.dual_net = DualClassificationNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                elif self.args["dual_completion"]:
                    self.dual_net = DualNetEndToEnd(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                else:
                    self.dual_net = DualNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                if self.args.get("oracle_supervised_dual", False):
                    self.dual_loss_fn = self.oracle_supervised_dual_loss
                else:
                    self.dual_loss_fn = self.alternate_dual_loss
                # self.dual_loss_fn = self.alternate_dual_loss
            else:
                self.dual_net = DualNet(self.args, self.data).to(dtype=self.DTYPE, device=self.DEVICE)
                self.dual_loss_fn = self.dual_loss



        elif self.problem_type == "GEP":
            # TODO: Implement GEP networks
            pass

        print(f"DUal Net Architecture ===== ")
        print(self.dual_net)

        self.primal_net.to(self.DTYPE).to(self.DEVICE)
        self.primal_optim = torch.optim.Adam(self.primal_net.parameters(), lr=self.primal_lr)
        self.dual_optim = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

        #! For MTAdam
        # self.primal_optim = MTAdam(self.primal_net.parameters(), lr=self.primal_lr)
        # self.dual_optim = MTAdam(self.dual_net.parameters(), lr=self.dual_lr)


        if self.args.get("use_heuristic_lambda_loss", False):
            data_classes = self.data.heuristic_lambda_classes.to(
                device=self.DEVICE,
                dtype=self.DTYPE,
            )
            model_classes = self.dual_net.classes.to(
                device=self.DEVICE,
                dtype=self.DTYPE,
            )

            print("data heuristic classes:", data_classes)
            print("model lambda classes:", model_classes)

            if data_classes.shape != model_classes.shape or not torch.allclose(data_classes, model_classes, atol=1e-10):
                raise ValueError(
                    "Heuristic lambda class order does not match dual_net.classes. "
                    "Soft labels would supervise the wrong class indices."
                )
        # Add schedulers
        self.primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.primal_optim, mode='min', factor=self.decay, patience=self.patience
        )
        self.dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dual_optim, mode='min', factor=self.decay, patience=self.patience
        )

        # For saving best models:
        self.best_primal_objs = [float('inf') for _ in range(len(self.max_violation_save_thresholds))]
        self.best_dual_obj = -1*float('inf')


        ### GAMBEL ANNELING FOR DUALS
        self.gumbel_tau_decay = args.get("gumbel_tau_decay", 0.95)
        self.gumbel_tau_min = args.get("gumbel_tau_min", 0.1)
        if args.get("dual_gumbel", False):
            self.dual_net.tau = float(args.get("gumbel_tau_init", 1.0))


        pred_y = self.primal_net(self.X_train, self.total_demands_train)
        pred_obj_train = self.data.obj_fn(self.X_train, pred_y)
        # print((pred_obj_train - self.target_obj_train) / self.target_obj_train)
        print(f"Initial known; pred; gap: {self.target_obj_train.mean()}, {pred_obj_train.mean()}, {((pred_obj_train - self.target_obj_train) / self.target_obj_train).mean()}")

        if not self.args["learn_primal"]:
            assert self.args["dual_alternate_loss"] == True, "Cannot disable primal learning without alternate dual loss."
        
        self.primal_early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        self.dual_early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        self.primal_net_best = None
        self.dual_net_best = None

        self.train_time = 0

        self.best_time = 0

        self.primal_obj_list = []
        self.dual_obj_list = []

        self.duality_gap_list =[]

        # Fit the scalers:
        if args.get("normalize") == "z_score":
            with torch.no_grad():
                Xtr = self.X_train

            n = self.data.num_n
            g = self.data.num_g

            d = Xtr[:, :n]
            p = Xtr[:, n:n + g]
            cover = Xtr[:, n + g:n + g + n]
            export = Xtr[:, n + g + n:n + g + 2 * n]

            d_mean = d.mean(dim=0)
            d_std  = d.std(dim=0).clamp_min(1e-8)

            p_mean = p.mean(dim=0)
            p_std  = p.std(dim=0).clamp_min(1e-8)

            self.primal_net.d_mean.copy_(d_mean)
            self.primal_net.d_std.copy_(d_std)
            self.primal_net.p_mean.copy_(p_mean)
            self.primal_net.p_std.copy_(p_std)

            self.dual_net.d_mean.copy_(d_mean)
            self.dual_net.d_std.copy_(d_std)
            self.dual_net.p_mean.copy_(p_mean)
            self.dual_net.p_std.copy_(p_std)

            if cover.shape[1] == n:
                cover_mean = cover.mean(dim=0)
                cover_std  = cover.std(dim=0).clamp_min(1e-8)

                if hasattr(self.dual_net, "cover_mean"):
                    self.dual_net.cover_mean.copy_(cover_mean)
                    self.dual_net.cover_std.copy_(cover_std)

            if export.shape[1] == n:
                export_mean = export.mean(dim=0)
                export_std  = export.std(dim=0).clamp_min(1e-8)

                if hasattr(self.dual_net, "export_mean"):
                    self.dual_net.export_mean.copy_(export_mean)
                    self.dual_net.export_std.copy_(export_std)

            print("Computed and set normalization stats.")
            print(f"D mean: {d_mean}, D std: {d_std}")
            print(f"P mean: {p_mean}, P std: {p_std}")
            if cover.shape[1] == n:
                print(f"Cover mean: {cover_mean}, Cover std: {cover_std}")
            if export.shape[1] == n:
                print(f"Export mean: {export_mean}, Export std: {export_std}")

    def build_topology_features(self, X):
        """
        local surplus = local capacity - demand 
        Add two topology-aware features per node:
        1. cover ratio:
        (local capacity + import capacity) / demand

        2. bounded export pressure:
        local surplus / (local surplus + export capacity)

        Returns:
            topo_features: [B, 2 * num_n]
        """
        D = X[:, :self.data.num_n]  # [B, N]
        Pmax = X[:, self.data.num_n:self.data.num_n + self.data.num_g]  # [B, G]

        node_to_gen = self.data.node_to_gen_mask.to(X.device).to(X.dtype)  # [N, G]
        line_mask = self.data.lineflow_mask.to(X.device).to(X.dtype)       # [N, L]

        # Local generation capacity at each node
        local_capacity = Pmax @ node_to_gen.T  # [B, N]

        # Directed transmission masks
        import_mask = (line_mask == 1).to(X.dtype)    # [N, L]
        export_mask = (line_mask == -1).to(X.dtype)   # [N, L]

        F_imp = torch.tensor(
            [self.data.pImpCap[l] for l in self.data.L],
            dtype=X.dtype,
            device=X.device,
        )

        F_exp = torch.tensor(
            [self.data.pExpCap[l] for l in self.data.L],
            dtype=X.dtype,
            device=X.device,
        )

        import_capacity = (import_mask @ F_imp).unsqueeze(0)  # [1, N]
        export_capacity = (export_mask @ F_exp).unsqueeze(0)  # [1, N]

        eps = float(self.args.get("topology_feature_eps", 1e-8))

        # Feature 1: theoretical local + import supply coverage
        cover_ratio = (local_capacity + import_capacity) / (D + eps)

        # Feature 2: bounded export pressure, avoids explosion when export_capacity = 0
        surplus = torch.relu(local_capacity - D)
        export_ratio = surplus / (surplus + export_capacity + eps)

        topo_features = torch.cat(
            [
                cover_ratio,
                export_ratio,
            ],
            dim=1,
        )  # [B, 2N]

        return topo_features

    def freeze(self, network):
        """
        Create a frozen copy of a network
        """
        if isinstance(network, PrimalNetEndToEnd):
            frozen_net = PrimalNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualNetEndToEnd):
            frozen_net = DualNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualClassificationNetEndToEnd):
            frozen_net = DualClassificationNetEndToEnd(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, PrimalNet):
            frozen_net = PrimalNet(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        elif isinstance(network, DualNet):
            frozen_net = DualNet(self.args, self.data).to(device=self.DEVICE, dtype=self.DTYPE)
        else:
            raise TypeError(f"Unsupported network type: {type(network)}")
        
        # Load a deep copy of the state dictionary
        frozen_net.load_state_dict(copy.deepcopy(network.state_dict()))
    
        # Set to evaluation mode
        frozen_net.eval()
        
        return frozen_net
    
    def compute_input_difficulty(self, X):
        """
        Heuristic per-sample difficulty score from input alone.
        Higher score = harder sample = larger expected dual gap.
        
        Uses max local imbalance max of (Demand - local supper / Demand)
  
        """
   
        D_n = X[:, :self.data.num_n]                              # [B, N]
        p_g_max = X[:, self.data.num_n:self.data.num_n + self.data.num_g]   # [B, G]
        
        total_demand = D_n.sum(dim=1)                         # [B]
        total_capacity = p_g_max.sum(dim=1)                   # [B]
        
        return total_capacity / (total_demand + 1e-8)
    
    # def compute_input_difficulty(self, X):

    #     X = X
    #     D_n = X[:, :self.data.num_n]
    #     p_g_max = X[:, self.data.num_n:self.data.num_n + self.data.num_g]
        
    #     # Identify renewable generator indices from data.G
    #     renewable_techs = {"WindOff", "WindOn", "SunPV"}
    #     renewable_idx = [
    #         i for i, (_, tech) in enumerate(self.data.G)
    #         if tech in renewable_techs
    #     ]
        
    #     total_demand = D_n.sum(dim=1)
    #     renewable_capacity = p_g_max[:, renewable_idx].sum(dim=1)
        
    #     return renewable_capacity / (total_demand + 1e-8)
        

    def snap_lambda_to_class_indices(self, lamb_true):
        """
        Convert true lambda values [B, neq] to nearest class index [B, neq].
        """
        class_values = self.dual_net.classes.to(lamb_true.device).to(lamb_true.dtype)   # [C]
        flat = lamb_true.reshape(-1, 1)                                                 # [B*neq, 1]
        dists = torch.abs(flat - class_values.view(1, -1))                              # [B*neq, C]
        idx = torch.argmin(dists, dim=1)                                                # [B*neq]
        return idx.view_as(lamb_true).long()                                            # [B, neq]


    def oracle_supervised_dual_loss(self, X, mu, lamb, lamb_true):
        """
        Supervised classification loss on lambda classes.

        Uses the classification network logits directly.
        """
        if not isinstance(self.dual_net, DualClassificationNetEndToEnd):
            raise TypeError("oracle_supervised_dual requires DualClassificationNetEndToEnd.")

        x_in = X
        if self.dual_net.normalize:
            x_in = self.dual_net.scale(X)

        logits = self.dual_net.feed_forward(x_in)   # [B, neq * n_classes] or separate-head equivalent output
        logits = logits.view(-1, self.data.neq, self.dual_net.n_classes)  # [B, neq, C]

        target_idx = self.snap_lambda_to_class_indices(lamb_true)         # [B, neq]

        # Cross-entropy over all node-wise lambda variables
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.dual_net.n_classes),
            target_idx.reshape(-1),
            reduction="none",
        ).view(X.shape[0], self.data.neq).mean(dim=1)   # [B]

        return ce, target_idx
        


    def train_PDL(self, optuna_trial=None):
        print("Starting Primal-Dual Learning inside the train_PDL function")
        prev_v_k = 0
        for k in range(self.outer_iterations):
            print("Starting outer iteration:", k)
            begin_time = time.time()
            frozen_dual_net = self.freeze(self.dual_net)
            if self.logger:
                with torch.no_grad():
                    self.logger.log_rho_vk(self.rho, prev_v_k, self.step)
            if self.args["learn_primal"]:
                for l1 in range(self.inner_iterations):
                    self.step += 1
                    # Update primal net using primal loss
                    self.primal_net.train()
                    frozen_dual_net.train() # TODO: Why train the frozen dual net?

                    # Accumulate training loss over all batches
                    # For logging
                    total_train_loss = 0.0
                    total_obj = 0.0
                    total_lagrange_eq = 0.0
                    total_lagrange_ineq = 0.0
                    total_penalty = 0.0

                    num_batches = 0

                    for batch in self.train_loader:
                        if self.args.get("use_heuristic_lambda_loss", False):
                            (
                                Xtrain,
                                total_demands,
                                X_opt,
                                mu_true_batch,
                                lamb_true_batch,
                                heur_soft_batch,
                                heur_conf_batch,
                                heur_tier_batch,
                            ) = batch
                        else:
                            (
                                Xtrain,
                                total_demands,
                                X_opt,
                                mu_true_batch,
                                lamb_true_batch,
                            ) = batch

                            heur_soft_batch = None
                            heur_conf_batch = None
                            heur_tier_batch = None
                        
                        compute_begin_time = time.time()

                        self.primal_optim.zero_grad()
                        
                        y = self.primal_net(Xtrain, total_demands)
                        # y.requires_grad_(True) # If logging gradients

                        with torch.no_grad():
                            if k == 0 and self.problem_type != "QP":
                                mu, lamb = torch.zeros((Xtrain.shape[0], self.data.nineq)), torch.zeros((Xtrain.shape[0], self.data.neq))
                            else:
                                mu, lamb = frozen_dual_net(Xtrain) # Use the frozen dual net to provide stable targets
                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = self.primal_loss_fn(Xtrain, y, mu.detach(), lamb.detach(), X_opt)
                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = batch_loss.mean(), obj.mean(), lagrange_eq.mean(), lagrange_ineq.mean(), penalty.mean()
                        total_train_loss += batch_loss.item()
                        total_obj += obj.item()
                        total_lagrange_eq += lagrange_eq.item()
                        total_lagrange_ineq += lagrange_ineq.item()
                        total_penalty += penalty.item()
                        if isinstance(self.primal_optim, MTAdam):
                            self.primal_optim.step(loss_array=[obj, lagrange_eq, lagrange_ineq, penalty], ranks=[1, 1, 1, 1], feature_map=None)
                        else:
                            # y.retain_grad()
                            batch_loss.backward()
                            # Log the gradients of each decision variable
                            # p_gt, f_lt, md_nt = self.data.split_dec_vars_from_Y(y.grad)
                            # print("Gradients of p_gt:", p_gt.mean())
                            # print("Gradients of f_lt:", f_lt)
                            # print("Gradients of md_nt:", md_nt.mean())
                            self.primal_optim.step()
                        
                        compute_end_time = time.time()
                        self.train_time += compute_end_time - compute_begin_time
                        num_batches += 1

                    # Compute average loss for the epoch
                    avg_train_loss = total_train_loss / num_batches
                    avg_obj = total_obj / num_batches
                    avg_lagrange_eq = total_lagrange_eq / num_batches
                    avg_lagrange_ineq = total_lagrange_ineq / num_batches
                    avg_penalty = total_penalty / num_batches
                    # print(f"Outer iter {k}, inner iter {l1}: Train Loss {avg_train_loss}. Eq {avg_lagrange_eq} Ineq {avg_lagrange_ineq} Penalty {avg_penalty}")
                    # Log training loss:
                    if self.logger and self.log_frequency > 0 and self.step % self.log_frequency == 0:
                        with torch.no_grad():
                            self.logger.log_primal_loss(avg_train_loss, avg_obj, avg_lagrange_eq, avg_lagrange_ineq, avg_penalty, self.step)
                            self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=frozen_dual_net, rho=self.rho, step=self.step)
                    
                    with torch.no_grad():
                        self.primal_net.eval()
                        frozen_dual_net.eval()
                        # print(f"In eval the primal net p and d are: {self.primal_net.p_mean}, {self.primal_net.d_mean}")
                        
                        obj_val_mean, primal_obj_val_mean ,val_loss_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean, dual_loss_mean = self.evaluate(
                            self.valid_dataset.tensors[0],
                            self.valid_dataset.tensors[1],
                            self.primal_net,
                            self.dual_net,
                            self.valid_dataset.tensors[2],
                            lamb_true=self.valid_dataset.tensors[4],
                            print_diff_weighting= False,
                        ) 
                        if k > 0:
                            self.save_if_best(obj_val_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean)
                        # Normalize by rho, so that the scheduler still works correctly if rho is increased

                        if self.primal_early_stopping.step(val_loss_mean):
                            self.primal_net_best = self.freeze(self.primal_net)
                            self.best_time = self.train_time
                        if self.early_stopping_patience > 0 and self.primal_early_stopping.early_stop:
                            print(f"Early stopping at step {self.step}")
                            # Return the best primal net, and the best loss.
                            self.save(self.save_dir)
                            with open(os.path.join(self.save_dir, "train_time.txt"), "w") as f:
                                f.write(f"Train time: {self.train_time}")
                            return self.primal_net_best, self.dual_net, self.primal_early_stopping.best_loss, dual_loss_mean, self.train_time

                        if optuna_trial:
                            optuna_trial.report(val_loss_mean.item(), self.step)
                            if optuna_trial.should_prune():
                                raise optuna.TrialPruned()
                        
                        if self.rho > 0:
                            if self.args["penalty"] == "Single":
                                rho_scale = self.rho
                            elif self.args["penalty"] == "Per_Const":
                                rho_scale = float(torch.mean(self.rho_ineq).item()) 
                            self.primal_scheduler.step(torch.sign(val_loss_mean) * (torch.abs(val_loss_mean) / rho_scale))
                        else:
                            self.primal_scheduler.step(val_loss_mean)


                with torch.no_grad():
                    # Copy primal net into frozen primal net
                    self.primal_net.train() # Otherwise, we are still on eval, and inverse normalize.
                    frozen_primal_net = self.freeze(self.primal_net)

                    # Calculate v_k
                    y = frozen_primal_net(self.X_train, self.total_demands_train)
                    mu_k, lamb_k = frozen_dual_net(self.X_train)
                    v_k = self.violation(self.X_train, y, mu_k)
            if self.args["learn_dual"]:
                for l in range(self.inner_iterations):
                    self.step += 1
                    # Update dual net using dual loss
                    self.dual_net.train()
                    if self.args["learn_primal"]:
                        frozen_primal_net.eval()
                    # For logging
                    total_train_loss = 0.0
                    total_obj = 0.0
                    total_lagrange_eq = 0.0
                    total_lagrange_ineq = 0.0
                    total_penalty = 0.0

                    num_batches = 0
                    for batch in self.train_loader:
                        if self.args.get("use_heuristic_lambda_loss", False):
                            (
                                Xtrain,
                                total_demands,
                                X_opt,
                                mu_true_batch,
                                lamb_true_batch,
                                heur_soft_batch,
                                heur_conf_batch,
                                heur_tier_batch,
                            ) = batch
                        else:
                            (
                                Xtrain,
                                total_demands,
                                X_opt,
                                mu_true_batch,
                                lamb_true_batch,
                            ) = batch

                            heur_soft_batch = None
                            heur_conf_batch = None
                            heur_tier_batch = None
                        compute_begin_time = time.time()
                        self.dual_optim.zero_grad()
                        if self.dual_regularization == "S3L":
                            mu, lamb = self.dual_net(Xtrain, mu=self.mu_barrier)
                        else:
                            mu, lamb = self.dual_net(Xtrain)
                        # print(lamb.mean(), lamb.max(), lamb.min())
                        unique_vals = torch.unique(lamb)
                        # print("[DEBUG] Unique lambda values:", unique_vals)
                        # print("[DEBUG] Expected classes:    ", self.dual_net.classes)
                        if self.args["learn_primal"]:
                            with torch.no_grad():
                                if self.dual_regularization == "S3L":
                                    mu_k, lamb_k = frozen_dual_net(Xtrain, mu=self.mu_barrier)
                                else:
                                    mu_k, lamb_k = frozen_dual_net(Xtrain)

                                y = frozen_primal_net(Xtrain, total_demands).detach()
                        else:
                            mu_k, lamb_k = None, None
                            y = None

                        if self.args.get("oracle_supervised_dual", False):
                            batch_loss, target_idx = self.dual_loss_fn(
                                X=Xtrain,
                                mu=mu,
                                lamb=lamb,
                                lamb_true=lamb_true_batch,
                            )
                            obj = self.data.dual_obj_fn(Xtrain, mu, lamb).detach()
                            lagrange_eq = torch.zeros_like(batch_loss)
                            lagrange_ineq = torch.zeros_like(batch_loss)
                            penalty = torch.zeros_like(batch_loss)
                        else:
            
                            batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = self.dual_loss_fn(
                                X=Xtrain,
                                y=y,
                                mu=mu,
                                lamb=lamb,
                                mu_k=mu_k,
                                lamb_k=lamb_k,
                                X_opt=X_opt,
                                lamb_true=lamb_true_batch if self.args.get("oracle_loss_weight", False) else None,
                                heur_soft=heur_soft_batch,
                                heur_conf=heur_conf_batch,
                                heur_tier=heur_tier_batch,
                            )


                        # batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = self.dual_loss_fn(X = Xtrain, y = y, mu = mu, lamb = lamb, mu_k = mu_k, 
                        #                                                                          lamb_k= lamb_k, X_opt = X_opt,
                        #                                                                         lamb_true=lamb_true_batch if self.args.get("oracle_loss_weight", False) else None,)
                        batch_loss, obj, lagrange_eq, lagrange_ineq, penalty = batch_loss.mean(), obj.mean(), lagrange_eq.mean(), lagrange_ineq.mean(), penalty.mean()
                        total_train_loss += batch_loss.item()
                        total_obj += obj.item()
                        total_lagrange_eq += lagrange_eq.item()
                        total_lagrange_ineq += lagrange_ineq.item()
                        total_penalty += penalty.item()

          

                        batch_loss.backward()
                        # output_layer = self.dual_net.feed_forward.net[-1]  # last Linear
                        #print("[DEBUG] Output layer grad norm:", output_layer.weight.grad.norm().item())

                        self.dual_optim.step()
                        compute_end_time = time.time()
                        self.train_time += compute_end_time - compute_begin_time

                        num_batches += 1
                    
                if self.logger and self.log_frequency > 0 and self.step % self.log_frequency == 0:
                    with torch.no_grad():
                        # Logg training loss:
                        # Compute average loss for the epoch
                        avg_train_loss = total_train_loss / num_batches
                        avg_obj = total_obj / num_batches
                        avg_lagrange_eq = total_lagrange_eq / num_batches
                        avg_lagrange_ineq = total_lagrange_ineq / num_batches
                        avg_penalty = total_penalty / num_batches

                        if self.args.get("entropy_in_loss", False):
                            print(
                                f"[Dual entropy train] "
                                f"outer={k}, inner={l}, step={self.step} | "
                                f"total_loss={avg_train_loss:.6f} | "
                                f"dual_obj={avg_obj:.6f} | "
                                f"raw_dual_loss={avg_lagrange_eq:.6f} | "
                                f"entropy={avg_lagrange_ineq:.6f} | "
                                f"entropy_term={avg_penalty:.6f}"
                            )

                        else:
                            print(
                                f"[Dual train] "
                                f"outer={k}, inner={l}, step={self.step} | "
                                f"total_loss={avg_train_loss:.6f} | "
                                f"dual_obj={avg_obj:.6f}"
                            )

                        self.logger.log_dual_loss(avg_train_loss, self.step, avg_obj, avg_lagrange_eq, avg_lagrange_ineq, avg_penalty)
                        self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=self.dual_net, rho=self.rho, step=self.step)
                    
                    # Evaluate validation loss every epoch, and update learning rate
                    with torch.no_grad():
                        self.primal_net.eval()
                        self.dual_net.eval()
                        

                        obj_val_mean, primal_obj_val_mean ,val_loss_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean, dual_loss_mean = self.evaluate(
                            self.valid_dataset.tensors[0],
                            self.valid_dataset.tensors[1],
                            self.primal_net,
                            self.dual_net,
                            self.valid_dataset.tensors[2],
                            lamb_true=self.valid_dataset.tensors[4],
                            print_diff_weighting= True,
                        ) 
                        # Early stopper also checks whether the dual loss is better than the best seen so far.
                        if self.dual_early_stopping.step(dual_loss_mean):
                            self.dual_net_best = self.freeze(self.dual_net)
                            self.best_time = self.train_time
                        if self.early_stopping_patience > 0 and self.dual_early_stopping.early_stop:
                            print(f"Early stopping at step {self.step}")
                            self.save(self.save_dir)
                            with open(os.path.join(self.save_dir, "train_time.txt"), "w") as f:
                                f.write(f"Train time: {self.train_time}")
                            return self.primal_net, self.dual_net_best, val_loss_mean, self.dual_early_stopping.best_loss, self.train_time
                        
                        if optuna_trial:
                            optuna_trial.report(dual_loss_mean.item(), self.step)
                            if optuna_trial.should_prune():
                                raise optuna.TrialPruned()
                        
                        # Normalize by rho, so that the schedular still works correctly if rho is increased
                        if self.rho > 0:
                            self.dual_scheduler.step(torch.sign(dual_loss_mean) * (torch.abs(dual_loss_mean) / self.rho))
                        else:
                            self.dual_scheduler.step(dual_loss_mean)
                        
                        if self.dual_regularization == "S3L":
                            old_mu = self.mu_barrier
                            self.mu_barrier = max(self.mu_barrier * self.mu_barrier_decay, self.mu_barrier_min)

            
    
            if self.logger:
                with torch.no_grad():
                    self.logger.log_train(self.data, primal_net=self.primal_net, dual_net=self.dual_net, rho=self.rho, step=self.step)
                    self.logger.log_val(self.data, self.primal_net, self.dual_net, self.step)
            
            # EVALUATE primal and dual net after each outer iteration
            if self.args["learn_primal"] and self.args["learn_dual"]:
                obj_value, primal_obj, dual_obj_target, dual_obj= self.compute_primal_dual_metric(self.valid_dataset.tensors[0], self.valid_dataset.tensors[1], self.primal_net, self.dual_net, self.valid_dataset.tensors[2])   


                if self.args.get("use_heuristic_lambda_loss", False):
                    valid_heur_loss = self.evaluate_heuristic_lambda_loss(
                        X=self.valid_dataset.tensors[0],
                        heur_soft=self.valid_dataset.tensors[5],
                        heur_conf=self.valid_dataset.tensors[6],
                        heur_tier=self.valid_dataset.tensors[7],
                        prefix="outer_valid",
                        outer_iter=k,
                    )
                else:
                    valid_heur_loss = None


                primal_opt_gap, primal_opt_gap_full = self.compute_opt_gap(primal_obj, obj_value, if_primal=True)
                dual_opt_gap,dual_opt_gap_full = self.compute_opt_gap(dual_obj, dual_obj_target, if_primal = False)
                duality_gap,_ = self.compute_dual_gap(primal_obj, dual_obj, obj_value)
                self.primal_obj_list.append(primal_opt_gap)
                self.dual_obj_list.append(dual_opt_gap)
                self.duality_gap_list.append(duality_gap)

                '''
                Eval Metric Here
                '''
                eval_df = pd.DataFrame({
                    "outer_iter": k,
                    "inner_iter": l1,
                    "objective": obj_value.cpu().numpy(),
                    "primal_obj": primal_obj.cpu().numpy(),
                    "dual_obj_target": dual_obj_target.cpu().numpy(),
                    "dual_obj": dual_obj.cpu().numpy(),
                    "opt_gap_primal": primal_opt_gap_full.cpu().numpy(),
                    "opt_gap_dual": dual_opt_gap_full.cpu().numpy(),
                })
                print(f"Primal Obj: {primal_obj.cpu().numpy()}, Optimal Obj: {obj_value.cpu().numpy()}")
                csv_name = f"eval_metrics_Norm{self.loss_option}.csv"
                csv_path = os.path.join(self.save_dir, csv_name)
                write_header = not os.path.exists(csv_path)
                eval_df.to_csv(csv_path, mode="a", index=False, header=write_header)
                print(f" --- Primal opt gap: {primal_opt_gap:.4f}, Dual opt gap: {dual_opt_gap:.4f}, Duality gap: {duality_gap:.4f} --- ")
            else:
                print(f" ---  Dual opt gap: {dual_opt_gap:.4f} --- ")

            end_time = time.time()
            if self.args["learn_primal"]:
                print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Rho: {self.rho}. Violation: {v_k}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")
            else:
                print(f"Epoch {k} done. Time taken: {end_time - begin_time}. Primal LR: {self.primal_optim.param_groups[0]['lr']}, Dual LR: {self.dual_optim.param_groups[0]['lr']}")
            print("-----------------------------------------")

            if self.args.get("dual_gumbel", False) and hasattr(self.dual_net, 'tau'):
                old_tau = self.dual_net.tau
                self.dual_net.tau = max(
                    self.gumbel_tau_min,
                    self.dual_net.tau * self.gumbel_tau_decay,
                )
                print(f"[Gumbel] tau: {old_tau:.3f} -> {self.dual_net.tau:.3f}")


            if self.dual_regularization == "S3L":
                print(
                    f"[S3L] mu_barrier: {self.old_mu:.6g} -> {self.mu_barrier:.6g} "
                    f"| decay={self.mu_barrier_decay} | min={self.mu_barrier_min}"
                )

                self.old_mu = self.mu_barrier

            if self.args["learn_primal"]:
                # Update rho from the second iteration onward.
                if self.args["penalty"] == "Single":
                    if k > 0 and v_k > self.tau * prev_v_k:

                        print(f"Updating single rho. v_k: {v_k}, prev_v_k: {prev_v_k}, tau: {self.tau}")
                        self.rho = np.min([self.alpha * self.rho, self.rho_max])
                    prev_v_k = v_k
                elif self.args["penalty"] == "Per_Const":

                    # Compute the violation per ineq constraint and the eq constarint
                    with torch.no_grad():
                        y_full = frozen_primal_net(self.X_train, self.total_demands_train)  # [26214, ydim]
                        
                        v_ineq, v_eq = self.constraint_violation_stats(self.X_train, y_full)
                    
                    if k > 0 and (self.prev_v_ineq is not None):
                        # Find the constraints that have not improved sufficiently, and increase their corresponding rho
                        eps = 1e-10
                        bad_ineq = v_ineq > (self.tau * self.prev_v_ineq + eps)
                        bad_eq   = v_eq   > (self.tau * self.prev_v_eq+ eps)

                        if bad_ineq.any() or bad_eq.any():
                            v_eq_gen_ub = v_ineq[len(self.data.G):len(self.data.G)*2]
                            print(f"v_ineq for generator constraints: {v_eq_gen_ub}")
                            print(f"Updating per-constraint rho: bad_ineq={bad_ineq.sum().item()}, bad_eq={bad_eq.sum().item()}")
                            print(f"Updating rhos nieq min:{self.rho_ineq.min().item()} max: {self.rho_ineq.max().item()}")

                        # Multiply only those constraints
                        self.rho_ineq[bad_ineq] = torch.clamp(self.alpha * self.rho_ineq[bad_ineq], max=self.rho_max)
                        self.rho_eq[bad_eq]     = torch.clamp(self.alpha * self.rho_eq[bad_eq],     max=self.rho_max)

                        pass
                    self.prev_v_ineq = v_ineq
                    self.prev_v_eq = v_eq
                    self.log_rho_snapshot(outer_iter=k, v_k=v_k)
   
        
        self.save(self.save_dir)
        with open(os.path.join(self.save_dir, "train_time.txt"), "w") as f:
            f.write(f"Train time: {self.best_time}")
        self.save_metric_plot(self.primal_obj_list, self.dual_obj_list, self.duality_gap_list, self.save_dir)

        return self.primal_net_best, self.dual_net_best, val_loss_mean, dual_loss_mean, self.best_time

    def save_if_best(self, obj_val_mean, ineq_max, ineq_mean, eq_max, eq_mean, dual_obj_val_mean):
        """
        Saves the primal and/or dual model if they meet the criteria for improvement.
        - Primal model is saved if objective is the best for a given violation threshold.
        - Dual model is saved if its objective is the best seen so far.
        """
        # Primal Model Saving Logic
        for i in range(len(self.max_violation_save_thresholds)):
            threshold = self.max_violation_save_thresholds[i]
            # Check if validation metrics meet the threshold and objective is improved
            if ineq_max < threshold and eq_max < threshold and obj_val_mean < self.best_primal_objs[i]:
                print(f"Saving new best primal model for threshold {threshold}: "
                    f"Obj: {obj_val_mean:.4f}, Eq Max: {eq_max:.4f}, Ineq Max: {ineq_max:.4f}")
                
                primal_save_path = os.path.join(self.save_dir, f'{threshold}_best_primal_net.pth')
                torch.save(self.primal_net.state_dict(), primal_save_path)
                
                # Update the best objective for this threshold
                self.best_primal_objs[i] = obj_val_mean

    @torch.no_grad()
    def evaluate_heuristic_lambda_loss(
        self,
        X,
        heur_soft,
        heur_conf,
        heur_tier,
        prefix="valid",
        outer_iter=None,
    ):
        """
        Evaluate current dual model against precomputed heuristic soft labels.

        Returns:
            mean heuristic CE loss after tier/confidence weighting.
        """
        if not isinstance(self.dual_net, DualClassificationNetEndToEnd):
            return None

        if heur_soft is None:
            return None

        self.dual_net.eval()

        heuristic_term = self.heuristic_lambda_regularization_loss_from_batch(
            X=X,
            soft_labels=heur_soft,
            confidence=heur_conf,
            tier=heur_tier,
        )  # [B]

        mean_heur_loss = heuristic_term.mean().item()

        row = {
            "outer_iter": -1 if outer_iter is None else int(outer_iter),
            "step": int(self.step),
            "prefix": prefix,
            "heuristic_lambda_loss": mean_heur_loss,
        }

        print(
            f"[HeurLoss:{prefix}] "
            f"outer={row['outer_iter']} step={row['step']} | "
            f"heur_lambda_loss={mean_heur_loss:.6f}"
        )

        return mean_heur_loss


    def evaluate(self, X, total_demands, primal_net, dual_net, X_Opt, lamb_true=None, outer_iter=None, inner_iter=None, print_diff_weighting = False):    
        # Forward pass through networks
        Y = primal_net(X, total_demands)

        if self.dual_regularization == "S3L":
            mu, lamb = dual_net(X, mu=self.mu_barrier)
        else:
            mu, lamb = dual_net(X)

        ineq_dist = self.data.ineq_dist(X, Y)
        eq_resid = self.data.eq_resid(X, Y)

        # Convert lists to arrays for easier handling
        obj_values = self.data.obj_fn(X, Y).detach()
        primal_losses, primal_obj, lagrange_eq, lagrange_ineq, penalty = self.primal_loss_fn(X, Y, mu, lamb, X_Opt)

        if self.args.get("oracle_supervised_dual", False):
            dual_losses, target_idx = self.oracle_supervised_dual_loss(
                X=X,
                mu=mu,
                lamb=lamb,
                lamb_true=lamb_true,
            )
            dual_obj = self.data.dual_obj_fn(X, mu, lamb).detach()
        else:
            dual_losses, dual_obj, raw_dual_loss, entropy, entropy_term = self.alternate_dual_loss(
                X, Y, mu, lamb, X_Opt, lamb_true=lamb_true
            )

        primal_losses = primal_losses.detach()
        dual_losses = dual_losses.detach()
        ineq_max_vals = torch.max(ineq_dist, dim=1)[0].detach()
        ineq_mean_vals = torch.mean(ineq_dist, dim=1).detach()
        eq_max_vals = torch.max(torch.abs(eq_resid), dim=1)[0].detach()
        eq_mean_vals = torch.mean(torch.abs(eq_resid), dim=1).detach()

        # === Difficulty-stratified dual gap diagnostic ===
        if self.use_difficulty_weighting and print_diff_weighting:
            with torch.no_grad():
                # Compute target dual obj for per-sample gap calculation
                # X here is validation X. The valid_indices align with valid_dataset.
                # Need target mu/lamb — pull from opt_targets using current indices.
                # If X is the full valid set, this works:
                if X.shape[0] == len(self.valid_indices):
                    target_mu = self.data.opt_targets["mu_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
                    target_lamb = self.data.opt_targets["lamb_operational"].to(self.DTYPE).to(self.DEVICE)[self.valid_indices]
                    dual_obj_target = self.data.dual_obj_fn(X, target_mu, target_lamb).detach()
                    
                    dual_gap_per_sample = (dual_obj_target - dual_obj).abs() / (dual_obj_target.abs() + 1e-8) * 100
                    
                    difficulty = self.compute_input_difficulty(X)
                    top_mask = difficulty > difficulty.quantile(0.9)
                    bot_mask = difficulty < difficulty.quantile(0.1)
                    
                    print(f"[Diff-stratified] Dual gap | "
                        f"top-10% diff: {dual_gap_per_sample[top_mask].mean():.2f}% "
                        f"({top_mask.sum().item()} samples, "
                        f"diff range: [{difficulty[top_mask].min():.2f}, {difficulty[top_mask].max():.2f}]) | "
                        f"bot-10% diff: {dual_gap_per_sample[bot_mask].mean():.2f}% "
                        f"({bot_mask.sum().item()} samples)")

        

        return torch.mean(obj_values), torch.mean(primal_obj), torch.mean(primal_losses), torch.mean(ineq_max_vals), torch.mean(ineq_mean_vals), torch.mean(eq_max_vals), torch.mean(eq_mean_vals), torch.mean(dual_obj), torch.mean(dual_losses)

    def primal_loss_QP(self, X, Y, mu, lamb):
        obj = self.data.obj_fn(X, Y)
        
        # g(Y)
        ineq = self.data.ineq_resid(X, Y)
        # h(Y)
        eq = self.data.eq_resid(X, Y)

        # ! Clamp mu?
        # Element-wise clamping of mu_i when g_i (ineq) is negative
        # mu = torch.where(ineq < 0, torch.zeros_like(mu), mu)
        # ! Clamp ineq_resid?
        # ineq = ineq.clamp(min=0)

        lagrange_ineq = torch.sum(mu * ineq, dim=1)  # Shape (batch_size,)

        lagrange_eq = torch.sum(lamb * eq, dim=1)   # Shape (batch_size,)

        violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
        violation_eq = torch.sum(eq ** 2, dim=1)
        penalty = self.rho/2 * (violation_ineq + violation_eq)

        loss = (obj + (lagrange_ineq + lagrange_eq + penalty))

        return loss, obj, lagrange_eq, lagrange_ineq, penalty
    
    def primal_loss(self, X, Y, mu, lamb, X_opt):
        # if self.args["penalize_md_obj"]:


        obj = self.data.obj_fn(X, Y)
        if self.rho > 0:
            ineq = self.data.ineq_resid(X, Y)
            # ineq = self.data.ineq_resid(X, Y)
            eq = self.data.eq_resid(X, Y)
            lagrange_eq = torch.sum(lamb * eq, dim=1)
            lagrange_ineq = torch.sum(mu * ineq, dim=1).clamp(min=0)  # Shape (batch_size,)
            if self.args["penalty"] == "Single":
                violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
                violation_eq = torch.sum(eq ** 2, dim=1)
                penalty = self.rho * 0.5 * (violation_ineq + violation_eq)
            elif self.args["penalty"] == "Per_Const":
                ineq_pos = torch.relu(ineq)                           # [B, nineq]
                vio_ineq = ineq_pos ** 2                              # [B, nineq]
                vio_eq   = eq ** 2                                    # [B, neq]

                # broadcast rho vectors to batch: [1,n] -> [B,n]
                penalty_ineq = 0.5 * torch.sum(vio_ineq * self.rho_ineq.view(1, -1), dim=1)  # [B]
                penalty_eq   = 0.5 * torch.sum(vio_eq   * self.rho_eq.view(1, -1),   dim=1)  # [B]
                penalty = penalty_ineq + penalty_eq  
            if self.args["use_primal_penalty"]:
                loss = obj + lagrange_ineq + lagrange_eq + penalty
            else:
                penalty = torch.zeros_like(penalty)
                loss = obj + lagrange_ineq + lagrange_eq
            if self.loss_option == "Norm_GT":
          
                loss = loss / X_opt
                lagrange_eq = lagrange_eq / X_opt
                lagrange_ineq = lagrange_ineq / X_opt
                penalty = penalty / X_opt
            elif self.loss_option == "Norm_Obj":
                # scale = obj.detach().abs() + 1e-6
                # loss = loss / scale
                scale = X[:, :self.data.num_n].sum(dim=1).abs() + 1e-6  # total demand
                lagrange_eq = lagrange_eq / scale
                lagrange_ineq = lagrange_ineq / scale
                penalty = penalty / scale

            elif self.loss_option == "Duality_Gap":
                # Comput the primal and dual obj, and use their difference as the loss.
                dual_obj = self.data.dual_obj_fn(X, mu, lamb).detach()
                # loss = (obj + dual_obj) / (obj.detach().abs() + 1e-6)
                # if self.args["dual_alternate_loss"]:
                #     loss = (obj - dual_obj)
                # else:
                loss = (obj + dual_obj) 
            elif self.loss_option == "Log":
                eps = 1e-8
                loss = torch.log(loss + eps)
                pass
            return loss, obj, lagrange_eq, lagrange_ineq, penalty
        else:   
            loss = obj
            return loss, obj, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
        

    def dual_loss(self, X, y, mu, lamb, mu_k, lamb_k, X_opt):
        # mu = [batch, g]
        # lamb = [batch, h]
        # TODO: Update the rho value here, to enable per constraint

        # g(y)
        ineq = self.data.ineq_resid(X, y) # [batch, g]
        # h(y)
        eq = self.data.eq_resid(X, y)   # [batch, h]

        #! From 2nd PDL paper, fix to 1e-1, not rho
        target_mu = torch.maximum(mu_k + self.rho * ineq, torch.zeros_like(ineq))
        # target_mu = torch.maximum(mu_k + 1e-1 * ineq, torch.zeros_like(ineq))

        dual_resid_ineq = mu - target_mu # [batch, g]

        dual_resid_ineq = torch.norm(dual_resid_ineq, dim=1)  # [batch]

        # Compute the dual residuals for equality constraints
        #! From 2nd PDL paper, fix to 1e-1, not rho
        dual_resid_eq = lamb - (lamb_k + self.rho * eq)
        # dual_resid_eq = lamb - (lamb_k + 1e-1 * eq)
        dual_resid_eq = torch.norm(dual_resid_eq, dim=1)  # Norm along constraint dimension

        loss = (dual_resid_ineq + dual_resid_eq)
        if self.loss_option == "Norm_GT":
            loss = loss / X_opt
        elif self.loss_option == "Norm_Obj":
            scale = self.data.obj_fn(X, y).detach().abs() + 1e-6
            loss = loss / scale


        elif self.loss_option == "Duality_Gap":
            # Comput the primal and dual obj, and use their difference as the loss.
            primal_obj = self.data.obj_fn(X, y).detach()
            dual_obj = self.data.dual_obj_fn(X, mu, lamb)
            loss = (primal_obj + dual_obj)
        elif self.loss_option == "Log":
            eps = 1e-8
            loss = torch.log(loss + eps)
        # print(f"Dual loss: {loss.mean().item()} Normalized {((X_opt - loss)/X_opt).mean().item()}")
        return loss, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    
    def lambda_entropy_loss(self, X):
        """
        Entropy regularization for lambda classification.

        Returns:
            entropy_per_sample: Tensor of shape [batch_size]

        Low entropy means sharp / confident class assignment.
        High entropy means uncertain soft mixture.
        """
        if not isinstance(self.dual_net, DualClassificationNetEndToEnd):
            return torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)

        probs = self.dual_net.get_lambda_probs(X)  # [B, neq, C]

        eps = 1e-12
        entropy_per_node = -(probs * torch.log(probs + eps)).sum(dim=-1)  # [B, neq]

        # Average across lambda variables/nodes, giving one entropy value per sample
        entropy_per_sample = entropy_per_node.mean(dim=1)  # [B]

        return entropy_per_sample
    
    def heuristic_lambda_regularization_loss_from_batch(
        self,
        X,
        soft_labels,
        confidence,
        tier,
    ):
        """
        Use precomputed heuristic soft labels.

        Returns:
            loss_per_sample: [B]
        """
        if not isinstance(self.dual_net, DualClassificationNetEndToEnd):
            raise TypeError("Heuristic lambda loss requires DualClassificationNetEndToEnd.")

        logits = self.dual_net.get_lambda_logits(X)  # [B, N, K]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        ce_per_node = -(soft_labels * log_probs).sum(dim=-1)  # [B, N]

        tier1_weight = float(self.args.get("heuristic_lambda_tier1_weight", 1.0))
        tier2_weight = float(self.args.get("heuristic_lambda_tier2_weight", 0.1))
        tier3_weight = float(self.args.get("heuristic_lambda_tier3_weight", 1.0))

        weights = torch.zeros_like(confidence)

        weights = torch.where(
            tier == 1,
            torch.tensor(tier1_weight, device=X.device, dtype=X.dtype),
            weights,
        )
        weights = torch.where(
            tier == 2,
            torch.tensor(tier2_weight, device=X.device, dtype=X.dtype),
            weights,
        )
        weights = torch.where(
            tier == 3,
            torch.tensor(tier3_weight, device=X.device, dtype=X.dtype),
            weights,
        )

        weights = weights * confidence

        loss_per_sample = (weights * ce_per_node).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-12)

        return loss_per_sample
    
    def alternate_dual_loss(self, X, y, mu, lamb, X_opt ,
                            mu_k=None, lamb_k=None, lamb_true = None,
                            heur_soft = None, heur_conf = None, heur_tier = None):
        #! We maximize the dual obj func, so to use it in the loss, take the negation.
        dual_obj = self.data.dual_obj_fn(X, mu, lamb)

        loss = -dual_obj

        if self.use_difficulty_weighting:
            with torch.no_grad():
                difficulty = self.compute_input_difficulty(X)
                # Normalize to mean 1, scale, then clamp
                weights = difficulty / (difficulty.mean() + 1e-8)
                weights = 1.0 + self.difficulty_weight_scale * (weights - 1.0)
                weights = weights.clamp(
                    min=self.difficulty_weight_clamp_min,
                    max=self.difficulty_weight_clamp_max,
                )
            loss = loss * weights


        barrier_term = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
        if self.dual_regularization == "S3L" and self.mu_barrier > 0:
            
            mu_clamped = mu.clamp(min=1e-8)  # Avoid log(0) by clamping mu to a small positive value

            # Log-barrier: sum over all inequality dual variables per sample
            # Shape: [batch, nineq] -> sum -> [batch]
            log_barrier_mu = torch.sum(torch.log(mu_clamped), dim=1)  # [batch]

            # The barrier encourages mu > 0 (interior point).
            # We subtract it from the loss (since loss = -dual_obj,
            # subtracting the barrier = adding it to the objective).
            barrier_term = -self.mu_barrier * log_barrier_mu  # [batch]
            loss = loss + barrier_term
        


        if self.loss_option == "Norm_GT":
            loss = loss / X_opt
        elif self.loss_option == "Norm_Obj":
            scale = X[:, :self.data.num_n].sum(dim=1).abs() + 1e-6  # total demand
            loss = loss / scale


        elif self.loss_option == "Duality_Gap":
            # Comput the primal and dual obj, and use their difference as the loss.
            primal_obj = self.data.obj_fn(X, y).detach()
            dual_obj = self.data.dual_obj_fn(X, mu, lamb)
            # print(f"In alt dual loss, dual objective: {dual_obj.mean().item()}")
            loss = (primal_obj - dual_obj) 

        if self.args.get("oracle_loss_weight", False):
            if lamb_true is None:
                raise ValueError("oracle_loss_weight=True but lamb_true was not provided.")
            weights = self.compute_oracle_loss_weights(lamb_true)
            loss = loss * weights
        #! Dual constraints are never violated, so we do not include penalty and lagrangian terms.


        heuristic_term = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        if self.args.get("use_heuristic_lambda_loss", False) and heur_soft is not None:
            if not isinstance(self.dual_net, DualClassificationNetEndToEnd):
                raise TypeError("use_heuristic_lambda_loss=True requires dual_classification=True.")
            
            heuristic_weight = float(self.args.get("heuristic_lambda_weight", 0.01))
            heuristic_term = self.heuristic_lambda_regularization_loss_from_batch(X,
                                                                                  soft_labels=heur_soft,
                                                                                  confidence=heur_conf,
                                                                                  tier=heur_tier)
            loss = loss + heuristic_weight * heuristic_term
            

        if self.args.get("entropy_in_loss", False):
            # Add entropy to regularize and pushes model to pick one of the discrete classes, instead of a mixture.
            raw_dual_loss = loss
            entropy = self.lambda_entropy_loss(X)  # [B]
            entropy_weight = float(self.args.get("entropy_weight", 0.01))
            entropy_term = entropy_weight * entropy
            loss = loss + entropy_term
            return loss, dual_obj, raw_dual_loss, entropy, entropy_term
        return loss, dual_obj, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)

    def compute_oracle_loss_weights(self, lamb_true):
        """
        Positive per-sample weights.
        Fewer lambda entries equal to scarcity value -> higher weight.

        w_i = 1 + alpha * (1 - count_i / neq)

        Then normalize to mean 1 so the loss scale stays stable.
        """
        scarcity_value = -float(self.data.pVOLL)
        alpha = float(self.args.get("oracle_weight_alpha", 2.0))

        scarcity_tensor = torch.tensor(
            scarcity_value,
            dtype=lamb_true.dtype,
            device=lamb_true.device,
        )

        scarcity_count = torch.isclose(
            lamb_true,
            scarcity_tensor,
            atol=1e-8,
            rtol=0.0,
        ).sum(dim=1).float()   # [batch]

        max_count = float(lamb_true.shape[1])  # neq, here likely 3
        weights = 1.0 + alpha * (1.0 - scarcity_count / max_count)

        # normalize so mean weight is 1
        weights = weights / (weights.mean() + 1e-12)
        return weights

    def violation(self, X, Y, mu_k):
        # Calculate the equality constraint function h_x(y)
        eq = self.data.eq_resid(X, Y)  # Assume shape (num_samples, n_eq)
        
        # Calculate the infinity norm of h_x(y)
        eq_inf_norm = torch.abs(eq).max(dim=1).values  # Shape: (num_samples,)

        # Calculate the inequality constraint function g_x(y)
        ineq = self.data.ineq_resid(X, Y)  # Assume shape (num_samples, n_ineq)
        
        # Calculate sigma_x(y) for each inequality constraint
        if self.args["penalty"] == "Single":
            sigma_y = torch.maximum(ineq, -mu_k / self.rho)  # Element-wise max
        elif self.args["penalty"] == "Per_Const":
            rho_ineq = self.rho_ineq.view(1, -1)          # [1, nineq] for broadcasting
            sigma_y = torch.maximum(ineq, -mu_k / (rho_ineq + 1e-12))
        
        # Calculate the infinity norm of sigma_x(y)
        sigma_y_inf_norm = torch.abs(sigma_y).max(dim=1).values  # Shape: (num_samples,)

        # Compute v_k as the maximum of the two norms
        v_k = torch.maximum(eq_inf_norm, sigma_y_inf_norm)  # Shape: (num_samples,)
        
        return v_k.max().item()

    def save(self, save_dir):
        print("saving")
        if self.primal_net_best is not None:
            torch.save(self.primal_net_best.state_dict(), save_dir + '/primal_weights.pth')
        if self.dual_net_best is not None:
            torch.save(self.dual_net_best.state_dict(), save_dir + '/dual_weights.pth')
    

    @torch.no_grad()
    def compute_primal_dual_metric(self, X, total_demands, primal_net, dual_net, X_Opt):        
        # Forward pass through networks
        Y = primal_net(X, total_demands)

        if self.dual_regularization == "S3L":
            mu, lamb = dual_net(X, mu=self.mu_barrier)
        else:
            mu, lamb = dual_net(X)

        ineq_dist = self.data.ineq_dist(X, Y)
        eq_resid = self.data.eq_resid(X, Y)

        obj_values = self.target_obj_val 
        
        target_mu = self.data.opt_targets["mu_operational"][self.valid_indices]
        target_lamb = self.data.opt_targets["lamb_operational"][self.valid_indices]
        

        dual_obj = self.data.dual_obj_fn(X, mu, lamb).detach()
        dual_obj_target = self.data.dual_obj_fn(X, target_mu, target_lamb).detach()

        primal_losses, primal_obj, lagrange_eq, lagrange_ineq, penalty = self.primal_loss_fn(X, Y, mu, lamb, X_Opt)
        dual_obj = self.data.dual_obj_fn(X, mu, lamb)
        print(f"*** Primal obj mean: {primal_obj.mean().item()} Dual Obj mean {dual_obj.mean().item()}***")
        return obj_values, primal_obj.detach(), dual_obj_target, dual_obj.detach()

    def compute_opt_gap(self, f_pred, f_star, if_primal = True):
        """
        Compute mean optimality gap (%) across samples.
        """
        if if_primal:
            opt_gap = (f_pred - f_star) / (f_star.abs()) * 100.0
        else:
            # For dual, f_pred is lower bound, so reverse the order
            opt_gap = (f_star - f_pred) / (f_star.abs()) * 100.0
        return opt_gap.mean().item(), opt_gap

    def compute_dual_gap(self, primal_pred, dual_pred, obj_val, feas_mask=None):
        """
        Compute mean duality gap (%) using per-sample objectives.
        Returns
        -------
        dual_gap_mean : float
            Mean duality gap (%) across feasible samples.
        dual_feasible_rate : float
            Fraction of samples that were feasible (1.0 if no mask provided).
        """

        # If a feasibility mask is provided, restrict to feasible samples
        if feas_mask is not None:
            if feas_mask.any():
                primal_pred = primal_pred[feas_mask]
                dual_pred = dual_pred[feas_mask]
                obj_val = obj_val[feas_mask]
            else:
                # No feasible samples
                return float("nan"), 0.0

        # Compute per-sample duality gap in percentage
        dual_gap = (primal_pred - dual_pred).abs() / (obj_val.abs() + 1e-12) * 100.0

        # Mean gap and feasible rate
        dual_gap_mean = dual_gap.mean().item()
        dual_feasible_rate = feas_mask.float().mean().item() if feas_mask is not None else 1.0

        return dual_gap_mean, dual_feasible_rate
    
    def log_rho_snapshot(self, outer_iter, v_k=None):
        """
        Save per-constraint rho values (and some summaries) to CSV.
        One row per outer iteration.
        """
        if self.args["penalty"] != "Per_Const":
            return  # nothing to do

        row = {
            "outer_iter": int(outer_iter),
            "step": int(self.step),
        }
        if v_k is not None:
            row["v_k"] = float(v_k)

        # Summaries
        row["rho_ineq_mean"] = float(self.rho_ineq.mean().item())
        row["rho_ineq_max"]  = float(self.rho_ineq.max().item())
        row["rho_eq_mean"]   = float(self.rho_eq.mean().item()) if self.data.neq > 0 else 0.0
        row["rho_eq_max"]    = float(self.rho_eq.max().item())  if self.data.neq > 0 else 0.0

        # Per-constraint entries
        rho_ineq_cpu = self.rho_ineq.detach().cpu().numpy()
        for i, val in enumerate(rho_ineq_cpu):
            row[f"rho_g_{i}"] = float(val)

        if self.data.neq > 0:
            rho_eq_cpu = self.rho_eq.detach().cpu().numpy()
            for j, val in enumerate(rho_eq_cpu):
                row[f"rho_h_{j}"] = float(val)

        csv_path = os.path.join(self.save_dir, "rho_per_constraint.csv")
        df = pd.DataFrame([row])
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", index=False, header=write_header)

    def save_metric_plot(self, primal_obj_list, dual_obj_list, duality_gap_list, save_dir = "./"):
        iters = list(range(len(self.primal_obj_list)))
        # ---------- Plot 1: Primal & Dual Optimality Gaps ----------
        plt.figure(figsize=(7, 5))
        plt.plot(iters, self.primal_obj_list, marker='o', label="Primal Optimality Gap")
        plt.plot(iters, self.dual_obj_list, marker='s', label="Dual Optimality Gap")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Optimality Gap (%)")
        plt.title("Primal vs Dual Optimality Gap per Outer Iteration")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot1_path = os.path.join(save_dir, "opt_gap_primal_dual.png")
        plt.savefig(plot1_path, dpi=300)
        plt.close()
        print(f"Saved: {plot1_path}")

        # ---------- Plot 2: Duality Gap ----------
        plt.figure(figsize=(7, 5))
        plt.plot(iters, self.duality_gap_list, marker='o', color='tab:purple', label="Duality Gap")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Duality Gap (%)")
        plt.title("Duality Gap per Outer Iteration")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot2_path = os.path.join(save_dir, "duality_gap.png")
        plt.savefig(plot2_path, dpi=300)
        plt.close()
        print(f"Saved: {plot2_path}")


        # ----------Plot 3: Primal Gap -----------
        plt.figure(figsize=(7, 5))
        plt.plot(iters, self.primal_obj_list, marker='o', label="Primal Optimality Gap")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Optimality Gap (%)")
        plt.title("Primal Optimality Gap per Outer Iteration")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot3_path = os.path.join(save_dir, "opt_gap_primal.png")
        plt.savefig(plot3_path, dpi=300)
        plt.close()
        print(f"Saved: {plot3_path}")


        # ----------Plot 4: Dual Gap -----------
        plt.figure(figsize=(7, 5))
        plt.plot(iters, self.dual_obj_list, marker='o', label="Dual Optimality Gap")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Optimality Gap (%)")
        plt.title("Dual Optimality Gap per Outer Iteration")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plot4_path = os.path.join(save_dir, "opt_gap_dual.png")
        plt.savefig(plot4_path, dpi=300)
        plt.close()
        print(f"Saved: {plot4_path}")

    @torch.no_grad()
    def constraint_violation_stats(self, X, Y):
        """
        Returns per-constraint violation statistics (mean squared) for ineq and eq.
        - ineq: mean(ReLU(g)^2) over samples -> shape [nineq]
        - eq:   mean(h^2) over samples        -> shape [neq]
        """
        ineq = self.data.ineq_resid(X, Y)  # [N, nineq]
        eq   = self.data.eq_resid(X, Y)    # [N, neq]

        v_ineq = (torch.relu(ineq) ** 2).mean(dim=0)  # [nineq]
        v_eq   = (eq ** 2).mean(dim=0)                # [neq]
        return v_ineq, v_eq