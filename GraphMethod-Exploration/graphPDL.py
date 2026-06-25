# graph_primal_dual.py

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from logger import TensorBoardLogger
import time
from graphNet import DualGNN, PrimalGNN
from graphBuilder import EDGraphBuilder
from matplotlib import pyplot as plt

# from your GNN code:
# from graph_model import PrimalGNN, DualGNN, build_homogeneous_ed_graph

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

class GraphPrimalDualTrainer:
    def __init__(self, data, args, save_dir):
        """
        data: the same ED dataset object used by PrimalDualTrainer
              (with fields like X, opt_targets, obj_fn, ineq_resid, eq_resid, etc.)
        args: config dict (same as before, plus maybe "graph_pdl": True)
        save_dir: output directory
        """
        self.data = data
        self.args = args
        self.save_dir = save_dir
        self.problem_type = args["problem_type"]
        self.log = args["log"]
        self.log_frequency = args["log_frequency"]

        # device / dtype (you can copy the logic from PrimalDualTrainer)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.DTYPE = torch.float64
        torch.set_default_dtype(self.DTYPE)

        self.train = args["train"]
        self.valid = args["valid"]
        self.test = args["test"]
        self.outer_iterations = args["outer_iterations"]
        self.inner_iterations = args["inner_iterations"]
        self.tau = args["tau"] # Tolerance scalar, determine how much violation improvement is enough to not increase rho
        self.rho = args["rho"] # Lagrangian penalty parameter, updated during training, increase if violation does not decrease sufficiently
        self.rho_max = args["rho_max"] # maximum rho, to prevent it from becoming too large when increasing rho 
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
        self.loss_option = args.get("loss_option", "Original")
        self.graphBuilder = EDGraphBuilder(args["ED_args"]) 

        self.num_g = self.data.num_g
        self.num_l = self.data.num_l
        self.num_n = self.data.num_n

        self.nieq = self.num_g*2 + self.num_l*2 + self.num_n*2
        self.neq = self.data.num_n

        if self.problem_type == "ED":
            self.total_demands = data.total_demands.to(self.DTYPE).to(self.DEVICE)
        else:
            self.total_demands = torch.ones((self.X.shape[0], 1))
        

        # ---- Build graph dataset from ED samples (X) ----
        # Here X is something like [num_samples, xdim]
        self.step = 0
        indices = torch.arange(self.X.shape[0])
        self.X = data.X.to(self.DTYPE).to(self.DEVICE)
        train_size = int(self.train * self.X.shape[0])
        valid_size = int(self.valid * self.X.shape[0])
        self.train_indices = indices[:train_size]
        self.valid_indices = indices[train_size:train_size+valid_size]
        self.test_indices = indices[train_size+valid_size:]

        self.total_demands = data.total_demands.to(self.DTYPE).to(self.DEVICE)

        # TODO: write this helper:
        #   build_graph_from_ED_sample(x_i, data) -> Data object
        # It converts one ED sample into a homogeneous graph (bus/gen/line nodes).
        self.full_data = self.build_X_graph(data)
        graph_list = []
        
        for i in range(self.X.shape[0]):
            g_i = self.graphBuilder.build_graph_from_ED_sample(index = i, data = self.full_data)
            g_i.global_feat = self.full_data[i]         
            graph_list.append(g_i)

        for node_id, features in enumerate(graph_list[0].x):
            feat = features.tolist()
            print(f"Node {node_id:2d} | Features = {feat}")
        # Split into train/val the same way as PDL or simpler for now
        num_train = int(args["train"] * len(graph_list))
        num_valid = int(args["valid"] * len(graph_list))
        indices = torch.arange(len(graph_list))

        idx_train = self.train_indices.tolist()
        idx_valid = self.valid_indices.tolist()
        idx_test  = self.test_indices.tolist()

        self.X_train = [graph_list[i] for i in idx_train]
        self.X_valid = [graph_list[i] for i in idx_valid]
        self.X_test  = [graph_list[i] for i in idx_test]

        self.total_demands_train = self.total_demands[self.train_indices]
        self.total_demands_valid = self.total_demands[self.valid_indices]

        # if self.log == True:
        #     self.logger = TensorBoardLogger(args, data, self.X, self.total_demands, self.train_indices, self.valid_indices, save_dir, args["opt_targets"])
        # else:
        #     self.logger = None

        self.train_loader = DataLoader(
            self.X_train,
            batch_size=args["batch_size"],
            shuffle=True
        )
        self.valid_loader = DataLoader(
            self.X_valid,
            batch_size=args["batch_size"],
            shuffle = False
        )
        if args["hidden_size_factor"] is False:
            args["hidden_size_factor"] = 16  # TODO: set as optimal

        self.primal_net = PrimalGNN(
            node_to_gen_mask=data.node_to_gen_mask,
            line_flow_mask=data.lineflow_mask,
            n_gen=len(data.G),
            n_line=len(data.L),
            n_loc=len(data.N),
            in_dim=7,
            hidden_dim=args["hidden_size_factor"],  # example
            num_layers=args["n_layers"]
        ).to(self.DEVICE)

        self.dual_net = DualGNN(
            args=args,
            data=data,
            in_dim=7,
            # hidden_dim=args["hidden_size_factor"],  # example
            num_layers=args["n_layers"]
        ).to(self.DEVICE)

        self.primal_loss_fn = self.primal_loss
        self.dual_loss_fn = self.dual_loss

        self.primal_optim = torch.optim.Adam(self.primal_net.parameters(), lr=self.primal_lr)
        self.dual_optim = torch.optim.Adam(self.dual_net.parameters(), lr=self.dual_lr)

        self.primal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.primal_optim, mode='min', factor=self.decay, patience=self.patience
        )
        self.dual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.dual_optim, mode='min', factor=self.decay, patience=self.patience
        )

        self.best_primal_objs = [float('inf') for _ in range(len(self.max_violation_save_thresholds))]
        self.best_dual_obj = -1*float('inf')


        self.primal_early_stopping = EarlyStopping(patience=self.early_stopping_patience)
        self.dual_early_stopping = EarlyStopping(patience=self.early_stopping_patience)

        self.primal_net_best = None
        self.dual_net_best = None

        self.train_time = 0

        self.best_time = 0

        self.primal_obj_list = []
        self.dual_obj_list = []

        self.duality_gap_list =[]
        self.train_primal_loss_list = []

        self.ineq_max_list = []
        self.ineq_mean_list = []
        self.eq_max_list = []
        self.eq_mean_list = []

        # self.valid_dataset = TensorDataset(self.X_valid, self.total_demands_valid, self.opt_target_val)

        self.opt_y_val, self.opt_obj_valid = self.compute_opt(self.valid_indices)

        self.train_PDL()
    

    def build_X_graph(self, data):
        '''
        Given Data, building feature needed to construct node feature in each graph
        Each instance have size: |Loc| + |Gen| + 2*|Line|
        Order:
          - first |Loc|  : location demands
          - next  |Gen|  : generator max production
          - next  |Line| : flow upper bounds  (UB)
          - next  |Line| : flow lower bounds  (LB, negative)
        '''
        X = []
        print(f"Data T: {data.T}")
        print(f"Data N: {data.N}")
        print(f"Data G: {data.G}")
        print(f"Data L: {data.L}")
        for i in range(data.n_samples):
            t = i % len(data.T) + 1
            Xi = []
            # Demand
            for n in data.N:
                Xi.append(data.pDemand[(n, t)])
            # MaxProd
            for g_idx, g in enumerate(data.G):
                p_gt_ub = data.pUnitInvestment[i, g_idx] * data.pUnitCap[g] * data.pGenAva.get((*g, t), 1.0)
                Xi.append(p_gt_ub)
            for l_idx, l in enumerate(data.L):
                Xi.append(0)
                Xi.append(0)
            # TODO: Add the Flow Capacity, now we set as 0
            X.append(Xi)
        # np.random.shuffle(X)
        X = torch.tensor(X)
        print(f"Built X graph with shape: {X.shape} num samples {data.n_samples}")
        return X



    def train_PDL(self, optuna_trial=None):
        """
        Mirror the interface of PrimalDualTrainer.train_PDL.
        Return:
            primal_net, dual_net, primal_loss, dual_loss, train_time
        """

        # Very simplified training loop sketch:

        start_time = time.time()
        num_batch = len(self.train_loader)
        # for epoch in range(self.args["outer_iterations"]):


        for epoch in range(200):
            # --- train primal (GNN) ---
            total_train_loss = 0.0
            total_obj = 0.0
            total_lagrange_eq = 0.0
            total_lagrange_ineq = 0.0
            total_penalty = 0.0

            for batch in self.train_loader:

                mu = torch.zeros((len(batch), self.nieq), dtype=self.DTYPE, device=self.DEVICE)
                lamb = torch.zeros((len(batch), self.neq), dtype=self.DTYPE, device=self.DEVICE)

                self.primal_optim.zero_grad()
                pred_p, pred_f, pred_md = self.primal_net(batch.x, batch.edge_index, batch.type_masks)



                loss, obj, lagrange_eq, lagrange_ineq, penalty = self.primal_loss_fn(batch.global_feat, pred_p, pred_f, pred_md,mu = mu, lamb = lamb, batch_size=batch.num_graphs)

                loss, obj, lagrange_eq, lagrange_ineq, penalty = loss.mean(), obj.mean(), lagrange_eq.mean(), lagrange_ineq.mean(), penalty.mean()

                total_train_loss += loss.item()
                total_obj += obj.item()
                total_lagrange_eq += lagrange_eq.item()
                total_lagrange_ineq += lagrange_ineq.item()
                total_penalty += penalty.item()

                loss.backward()
                self.primal_optim.step()
                
            avg_train_loss = total_train_loss / num_batch
            avg_obj = total_obj / num_batch
            avg_lagrange_eq = total_lagrange_eq / num_batch
            avg_lagrange_ineq = total_lagrange_ineq / num_batch
            avg_penalty = total_penalty / num_batch

            print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f}, Obj: {avg_obj:.4f}, Lag Eq: {avg_lagrange_eq:.4f}, Lag Ineq: {avg_lagrange_ineq:.4f}, Penalty: {avg_penalty:.4f}")
            # --- train dual (GNN) ---
            # for batch in self.train_loader:
            #     mu, lamb = self.dual_net(batch)
            #     # compute dual loss (alternate_dual_loss-style)
            #     # backprop + step
            pass
            mean_train_loss = total_train_loss / len(self.train_loader)
            # print(f"Epoch {epoch}: Mean train primal loss: {mean_train_loss}")
            mean_opt_gap, opt_gap_list, ineq_max, ineq_mean, eq_max, eq_mean = self.compute_metrics(self.valid_loader)
            self.duality_gap_list.append(mean_opt_gap)
            self.train_primal_loss_list.append(mean_train_loss)
            self.ineq_max_list.append(ineq_max)
            self.ineq_mean_list.append(ineq_mean)
            self.eq_max_list.append(eq_max)
            self.eq_mean_list.append(eq_mean)
            print(f"---- Epoch {epoch}: Final Mean Opt Gap {mean_opt_gap:.4f} %")
            print(f"                     Validation Violations - Ineq Max: {ineq_max:.4f}, Ineq Mean: {ineq_mean:.4f}, Eq Max: {eq_max:.4f}, Eq Mean: {eq_mean:.4f}")
        train_time = time.time() - start_time
        primal_loss = torch.tensor(0.0)
        dual_loss = torch.tensor(0.0)

        self.plot_metrics()

        return None, None, primal_loss, dual_loss, train_time

    def primal_loss(self, X, pred_p, pred_f, pred_md, mu, lamb, batch_size):
        '''
        Given 
        pred_p: [Batch, N_prod]
        pred_f: [Batch, N_line]
        mu_p:   [Batch, N_prod, 2]
        mu_f:   [Batch, N_line, 2]
        mu_e:   [Batch, N_loc, 2]
        lamb:   [Batch, N_loc, 1]

        X is input, Y is output
        THis is how T is built 
        
        eq_rhs = D
        X is [Batch, |D|+|MaxProd|]
        Y is [Batch, |p| + |f| + |e|]

        Group into X and Y, to use the 
        self.data.ineq_resit and self.data.eq_resit function

        Mu:
        - prod: |G|*2
        - flow: |F|*2
        - unmet demand: |N|*2
        = 3*2 + 3*2 + 6*2 = 24

        

        Group into 
        Loss = 
        self.data.opt_fn(X,Y) + 
        lagrange_eq +
        lagrange_ineq +
        penalty_terms

        return:
        - Loss
        - Obj
        - Lag Eq
        - Lag Ineq
        - Penalty
        '''

        X = X.view(batch_size, -1)

        N = pred_md.shape[1] # Number of Location
        G = pred_p.shape[1]  # Number of Generators
        # Get everything from X except for the last 6 columns
        X = X[:,:(N + G)].to(self.DTYPE)
        
        # Merge pred_p, prep_f and pred_md into Y by 1st dimension
        Y = torch.cat([pred_p, pred_f, pred_md], dim=1)

        
        # Print dtype of X and Y

        # Reconstruct X from [Batch, |D| + |MaxProd|]
        obj = self.data.obj_fn(X, Y)
        if self.rho > 0:
            ineq = self.data.ineq_resid(X, Y)
            eq = self.data.eq_resid(X, Y)
            # Eq and Ineq are all negative
            lagrange_eq = torch.sum(lamb * eq, dim=1)
            lagrange_ineq = torch.sum(mu * ineq, dim=1).clamp(min=0)  # Shape (batch_size,)
            violation_ineq = torch.sum(torch.maximum(ineq, torch.zeros_like(ineq)) ** 2, dim=1)
            violation_eq = torch.sum(eq ** 2, dim=1)
            penalty = self.rho/2 * (violation_ineq + violation_eq)
            loss = obj + lagrange_ineq + lagrange_eq + penalty
            # mu and lamb are [B, 24]
            '''
            ineq shape: torch.Size([2000, 24]) eq shape: torch.Size([2000, 3])
            '''
        else:
            loss = obj
            lagrange_eq = torch.tensor(0.0, device=self.DEVICE)
            lagrange_ineq = torch.tensor(0.0, device=self.DEVICE)
            penalty = torch.tensor(0.0, device=self.DEVICE)

        return loss, obj, lagrange_eq, lagrange_ineq, penalty

    def dual_loss(self, pred_p, pred_f,  mu_p, mu_f, mu_e, lamb):
        '''
        
        '''
        pass

    def compute_metrics(self, dataloader):
        """
        dataloader: torch_geometric DataLoader (batch_size can be 1)
        opt_obj_val: tensor/array with the optimal objective value(s)
                    for the whole dataset or per instance
        """
        self.primal_net.eval()

        all_obj_pred = []
        X_loader = []
        Y_loader = []
        

        with torch.no_grad():
            for i, batch in enumerate(dataloader):

                # If you have a GPU:
                # batch = batch.to(self.device)

                # Forward pass
                pred_p, pred_f, pred_md = self.primal_net(
                    batch.x,
                    batch.edge_index,
                    batch.type_masks
                )

                size = self.batch_size
                num_graph_in_batch = batch.num_graphs
                # Assuming "global_feat" is a field of the batch
                # (you had `data.x.global_feat` which looks wrong)

                X = batch.global_feat.view(num_graph_in_batch, -1)
  

                N = pred_md.shape[1]  # Number of locations
                G = pred_p.shape[1]   # Number of generators

                # Take first (N + G) columns
                X = X[:, :(self.num_n + self.num_g)].to(self.DTYPE)

                # Merge predictions along feature dimension
                Y = torch.cat([pred_p, pred_f, pred_md], dim=1)

    

                # Compute objective value for this batch
                obj_pred = self.data.obj_fn(X, Y)   # [batch_size]
                all_obj_pred.append(obj_pred)
                X_loader.append(X)
                Y_loader.append(Y)


            # Concatenate over all batches
            X = torch.cat(X_loader, dim=0)
            Y = torch.cat(Y_loader, dim=0)
            obj_pred_full = torch.cat(all_obj_pred, dim=0)
            


            mean_opt_gap, opt_gap_list = self.compute_opt_gap(
                obj_pred_full,
                self.opt_obj_valid,
                if_primal=True
            )
            
            ineq_max, ineq_mean, eq_max, eq_mean = self.compute_violation(X, Y)
            
        return mean_opt_gap, opt_gap_list, ineq_max, ineq_mean, eq_max, eq_mean 

    def compute_opt(self, indices):
        '''
        Given input:
        indices: indices of samples to compute opt for

        Output:
        - opt_y_targets: optimal y targets for the samples
        - opt_obj_target: optimal objective values for the samples
        '''
        opt_y_targets = self.data.opt_targets["y_operational"].to(self.DTYPE).to(self.DEVICE)[indices]
        X = self.full_data.to(self.DTYPE).to(self.DEVICE)[indices]
        X = X[:, : (self.num_n + self.num_g)]

        opt_obj_target = self.data.obj_fn(X, opt_y_targets)

        return opt_y_targets, opt_obj_target
    

    @torch.no_grad()
    def compute_primal_dual_metric(self, X, total_demands, primal_net, dual_net, X_Opt):     
        pass

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
    
    def compute_violation(self,X, Y):
        '''
        Returns:
        - Inequality Max
        - Inequality Mean
        - Equality Max
        - Equality Mean
        '''
        ineq_dist = data.ineq_dist(X, Y)
        eq_resid = data.eq_resid(X, Y)

        obj_values = data.obj_fn(X, Y).detach().numpy()
        ineq_max_vals = torch.max(ineq_dist, dim=1)[0].detach().numpy() # First element is the max, second is the index
        ineq_mean_vals = torch.mean(ineq_dist, dim=1).detach().numpy()
        eq_max_vals = torch.max(torch.abs(eq_resid), dim=1)[0].detach().numpy() # First element is the max, second is the index
        eq_mean_vals = torch.mean(torch.abs(eq_resid), dim=1).detach().numpy()

        return np.mean(ineq_max_vals), np.mean(ineq_mean_vals), np.mean(eq_max_vals), np.mean(eq_mean_vals)



    def plot_metrics(self):
        """
        Plots the training loss, validation optimality gap, and constraint violations.
        """
        if not self.train_primal_loss_list or not self.duality_gap_list:
            print("No metrics recorded to plot.")
            return

        epochs = range(1, len(self.train_primal_loss_list) + 1)
        
        # Change figure size and layout to accommodate 3 plots
        plt.figure(figsize=(18, 5))
        
        # --- First Plot: Primal Loss (Training) ---
        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_primal_loss_list, label='Primal Training Loss (AUGLoss)', color='blue')
        plt.title('Primal Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.legend()
        plt.grid(True)

        # --- Second Plot: Optimality Gap (Validation) ---
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.duality_gap_list, label='Validation Opt. Gap (%)', color='red')
        plt.title('Validation Optimality Gap per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Optimality Gap (%)')
        plt.legend()
        plt.grid(True)

        # --- Third Plot: Constraint Violations (Validation) ---
        plt.subplot(1, 3, 3)
        
        # Plotting the four violation metrics
        plt.plot(epochs, self.ineq_max_list, label='Inequality Max Violation', color='darkorange', linestyle='-')
        plt.plot(epochs, self.ineq_mean_list, label='Inequality Mean Violation', color='orange', linestyle='--')
        plt.plot(epochs, self.eq_max_list, label='Equality Max Violation', color='darkgreen', linestyle='-')
        plt.plot(epochs, self.eq_mean_list, label='Equality Mean Violation', color='lightgreen', linestyle='--')
        
        plt.title('Validation Constraint Violations per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Violation (Absolute Value)')
        plt.yscale('log') # Use log scale as violations often decrease exponentially
        plt.legend()
        plt.grid(True, which="both", ls="--")
        
        # Display plots
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pass
    import json
    import pickle
    args_path = "config.json"
    with open(args_path, "r") as f:
        args = json.load(f)

    ED_args = args["ED_args"]

    nodes_str = "-".join([n[0] for n in ED_args['N']])
            
    # For generators, count per node: [['BEL', 'WindOn'], ['BEL', 'Gas'],...] = 'B3-G2-N2'
    gen_counts = {}
    for g in ED_args['G']:
        node = g[0]
        gen_counts[node] = gen_counts.get(node, 0) + 1
    gens_str = "-".join([f"{node[0]}{count}" for node, count in gen_counts.items()])
    
    # For lines, just count: [['BEL', 'GER'], ['BEL', 'NED'], ['GER', 'NED']] → 'L3'
    lines_str = f"L{len(ED_args['L'])}"
    data_save_path = (f"data/ED_data/ED_N{nodes_str}_G{gens_str}_{lines_str}"
                    f"_c{int(ED_args['benders_compact'])}"
                    f"_s{int(ED_args['scale_problem'])}"
                    f"_p{int(ED_args['perturb_operating_costs'])}"
                    f"_smp{ED_args['2n_synthetic_samples']}.pkl")

    with open(data_save_path, 'rb') as file:
        data = pickle.load(file)
    GraphPrimalDualTrainer(data=data, args=args, save_dir="./")


    print(f"Data.node_to_gen_mask: {data.node_to_gen_mask}")

    print(f"Line FLow mask: {data.lineflow_mask}")