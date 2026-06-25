import torch
from torch_geometric.data import Data
import json

class EDGraphBuilder:
    def __init__(self, ED_args):
        self.N = ED_args["N"]
        self.G = ED_args["G"]   # list of [loc, tech]
        self.L = ED_args["L"]   # list of [loc_from, loc_to]

        self.n_loc  = len(self.N)
        self.n_gen  = len(self.G)
        self.n_line = len(self.L)

        self.loc_offset  = 0
        self.gen_offset  = self.loc_offset + self.n_loc
        self.line_offset = self.gen_offset  + self.n_gen

        self.num_nodes = self.n_loc + self.n_gen + self.n_line

        # maps for convenience (used for edges if needed)
        self.loc_idx  = {loc: i for i, loc in enumerate(self.N)}
        self.gen_idx  = {tuple(g): self.gen_offset + j for j, g in enumerate(self.G)}
        self.line_idx = {tuple(l): self.line_offset + k for k, l in enumerate(self.L)}

        # You can also prebuild edge_index here if you like:
        src, dst = [], []

        # Gen ↔ Loc
        for (loc, tech) in self.G:
            g_id = self.gen_idx[(loc, tech)]
            n_id = self.loc_idx[loc]
            src += [g_id, n_id]
            dst += [n_id, g_id]

        # Line ↔ its two endpoint locations
        for (u, v) in self.L:
            line_id = self.line_idx[(u, v)]
            u_id = self.loc_idx[u]
            v_id = self.loc_idx[v]

            src += [line_id, u_id, line_id, v_id]
            dst += [u_id,   line_id, v_id,   line_id]

        self.edge_index = torch.tensor([src, dst], dtype=torch.long)

        self.feat_dtype = torch.float64

    def build_graph_from_ED_sample(self, index, data):
        """
        Convert ONE ED instance into a homogeneous graph Data object.

        data[index] has size: |Loc| + |Gen| + 2*|Line|
        Order:
          - first |Loc|  : location demands
          - next  |Gen|  : generator max production
          - next  |Line| : flow upper bounds  (UB)
          - next  |Line| : flow lower bounds  (LB, negative)
        
        Node features: [Demand, MaxProd, FlowUB, FlowLB, LocNode, ProdNode, LineNode]
        """

        # ---- 1) Extract the flat sample vector ----
        # data can be a torch.Tensor or np.ndarray
        if isinstance(data, torch.Tensor):
            sample = data[index].to(self.feat_dtype)
        else:
            # assume numpy
            sample = torch.tensor(data[index], dtype=self.feat_dtype)

        # Sanity check
        expected_len = self.n_loc + self.n_gen + 2 * self.n_line
        assert sample.numel() == expected_len, (
            f"Sample length {sample.numel()} != expected {expected_len} "
            f"(loc={self.n_loc}, gen={self.n_gen}, line={self.n_line})"
        )

        # ---- 2) Slice out each block ----
        offset = 0
        demand_vals = sample[offset : offset + self.n_loc]
        offset += self.n_loc

        maxprod_vals = sample[offset : offset + self.n_gen]
        offset += self.n_gen

        flowUB_vals = sample[offset : offset + self.n_line]
        offset += self.n_line

        flowLB_vals = sample[offset : offset + self.n_line]
        # offset += self.n_line  # not needed afterwards

        # ---- 3) Allocate node feature matrix ----
        # [num_nodes, 7]
        x = torch.zeros(self.num_nodes, 7, dtype=sample.dtype)

        # COLUMN INDICES:
        DEMAND   = 0
        MAXPROD  = 1
        FLOWUB   = 2
        FLOWLB   = 3
        LOCFLAG  = 4
        PRODFLAG = 5
        LINEFLAG = 6

        # ---------- Location nodes ----------
        # node indices: [0 .. n_loc-1]
        for i in range(self.n_loc):
            n_id = self.loc_offset + i
            x[n_id, DEMAND]  = demand_vals[i]
            x[n_id, LOCFLAG] = 1.0

        # ---------- Generator nodes ----------
        # node indices: [gen_offset .. gen_offset + n_gen-1]
        for j in range(self.n_gen):
            g_id = self.gen_offset + j
            x[g_id, MAXPROD]  = maxprod_vals[j]
            x[g_id, PRODFLAG] = 1.0

        # ---------- Line nodes ----------
        # node indices: [line_offset .. line_offset + n_line-1]
        for k in range(self.n_line):
            l_id = self.line_offset + k
            x[l_id, FLOWUB]   = flowUB_vals[k]
            x[l_id, FLOWLB]   = flowLB_vals[k]
            x[l_id, LINEFLAG] = 1.0

        # ---- 4) Build Data object ----
        graph = Data(
            x=x,
            edge_index=self.edge_index.clone(),  # or reuse directly if you prefer
        )

        # Optional masks (useful for heads later)
        graph.loc_mask  = torch.zeros(self.num_nodes, dtype=torch.bool)
        graph.gen_mask  = torch.zeros(self.num_nodes, dtype=torch.bool)
        graph.line_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        graph.loc_mask[self.loc_offset : self.loc_offset + self.n_loc]       = True
        graph.gen_mask[self.gen_offset : self.gen_offset + self.n_gen]       = True
        graph.line_mask[self.line_offset: self.line_offset + self.n_line]    = True

        graph.type_masks = {
            'prod': graph.gen_mask,
            'line': graph.line_mask,
            'loc':  graph.loc_mask,
        }

        return graph



if __name__ == "__main__":
    args_path = "config.json"

    with open(args_path, "r") as f:
        args = json.load(f)

    ED_args = args["ED_args"]

    G = EDGraphBuilder(ED_args)  # Replace `data=None` with your actual ED dataset object
    print(f"Nodes: {G.N}")
    print(f"Generators: {G.G}")
    print(f"Lines: {G.L}")

    print(f"Loc indexes: {G.loc_idx}")
    print(f"Gen indexes: {G.gen_idx}")
    print(f"Line indexes: {G.line_idx}")

    print(f"Total number of nodes in graph: {G.num_nodes}")

    data = [[
        1,2,3,
        100,200,300,1000,10,30,
        50,100,30,-10,-100,-200
    ]]

    g = G.build_graph_from_ED_sample(index=0, data=data)  # Replace `data=None` with your actual ED dataset object
    print("=== Graph Node Features ===")
    for node_id, features in enumerate(g.x):
        feat = features.tolist()
        print(f"Node {node_id:2d} | Features = {feat}")


    print(f"Loc idx: {G.loc_idx}")
    print(f"Gen idx: {G.gen_idx}")
    print(f"Line idx: {G.line_idx}")

    print(f"Gen mask: {g.gen_mask}")
    print(f"Line mask: {g.line_mask}")
    print(f"Loc mask: {g.loc_mask}")


'''
  Loc idx: {'BEL': 0, 'GER': 1, 'FRA': 2}
Gen idx: {('BEL', 'WindOff'): 3, ('BEL', 'Gas'): 4, ('GER', 'Gas'): 5, ('GER', 'SunPV'): 6, ('FRA', 'Nuclear'): 7, ('FRA', 'SunPV'): 8}
Line idx: {('BEL', 'GER'): 9, ('BEL', 'FRA'): 10, ('GER', 'FRA'): 11}  
'''