import torch
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
from torch_geometric.nn import global_mean_pool, global_add_pool


class GNN_model(torch.nn.Module):
    def __init__(self, dims, num_features, training, p, model_type, device):
        super(GNN_model, self).__init__()
        self.device = device
        self.training = False
        self.p = float(p)
        self.layer_num = len(dims)
        self.model_type = model_type

        if model_type == 'gcn':
          self.setup_gcn(num_features, dims)
        if model_type == 'gin':
          self.setup_gin(num_features, dims, True)
        if model_type == 'gat':
          self.setup_gat(num_features, dims, 7, self.p)

    def setup_gcn(self, num_features, dims):
        self.conv1 = GCNConv(num_features, dims[0]).to(self.device)  # Num features = initial embedding dimension
        if len(dims) > 1:
          self.conv1_1 = GCNConv(dims[0], dims[1]).to(self.device)
        if len(dims) > 2:
          self.conv1_2 = GCNConv(dims[1], dims[2])

    def setup_gat(self, num_features, dims, heads, dropout):
        self.conv1 = GATConv(num_features, dims[0], heads=heads, dropout = dropout)  # Num features = initial embedding dimension
        if len(dims) > 1:
          self.conv1_1 = GATConv(heads*dims[0], dims[1], heads=heads, dropout = dropout) #heads
        if len(dims) > 2:
          self.conv1_2 = GATConv(heads*dims[1], dims[2], heads=heads, dropout = dropout)

    def setup_gin(self, num_features, dims, eps): #fix
        self.conv1 = GINConv(
            Sequential(Linear(num_features, dims[0]),
                       BatchNorm1d(dims[0]), ReLU(),
                       Linear(dims[0], dims[0]), ReLU()), train_eps=eps)
        if len(dims) > 1:
            self.conv1_1 = GINConv(
                Sequential(Linear(dims[0], dims[1]),
                       BatchNorm1d(dims[1]), ReLU(),
                       Linear(dims[1], dims[1]), ReLU()), train_eps=eps)
        if len(dims) > 2:
            self.conv1_2 = GINConv(
                Sequential(Linear(dims[1], dims[2]),
                       BatchNorm1d(dims[2]), ReLU(),
                       Linear(dims[2], dims[2]), ReLU()), train_eps=eps)


    def conv_pass(self, features, edge_index, batch):
        ret = []
        h1 = F.dropout(features, p=self.p, training=self.training)
        h1 = self.conv1(h1, edge_index).to(self.device)
        if self.model_type != 'gin':
            h1 = h1.relu()
        ret.append(h1)

        if self.layer_num > 1:
            h1 = F.dropout(h1, p=self.p, training=self.training)
            h1 = self.conv1_1(h1, edge_index)
            if self.model_type != 'gin':
                h1 = h1.relu()
            ret.append(h1)

        if self.layer_num > 2:
            h1 = F.dropout(h1, p=self.p, training=self.training)
            h1 = self.conv1_2(h1, edge_index)
            if self.model_type != 'gin':
                h1 = h1.relu()
            ret.append(h1)

        # concat
        h = torch.concat(ret, dim=-1).to(self.device)

        # Graph-level readout
        if self.model_type == 'gin':
          hG = global_add_pool(h, batch)
        else:
          hG = global_mean_pool(h, batch)

        return hG

    def forward(self, batch1, batch2):
        x1 = batch1.feature.float()
        x2 = batch2.feature.float()

        edge_index1 = batch1.edge_index
        edge_index2 = batch2.edge_index

        # Node embeddings
        hG1 = self.conv_pass(x1, edge_index1, batch1.batch)
        hG2 = self.conv_pass(x2, edge_index2, batch2.batch)

        # norm distance
        dist = torch.linalg.norm(hG1 - hG2, dim=1).to(self.device)

        return dist
