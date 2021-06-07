import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, device):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(6, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)
        self.to(device)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        # x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = global_max_pool(x, batch)
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class DeepLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, act_fun, block='res+'):
        super(DeepLayer, self).__init__()

        # define conv and normalization
        conv = GENConv(in_channels, in_channels, num_layers=3)
        norm = torch.nn.BatchNorm1d(in_channels, momentum=0.08, affine=True)
        self.layer = DeepGCNLayer(conv, norm, act_fun, block, dropout=0.05, ckpt_grad=True)
        
        # define FC layer to out_channels
        self.encoder = torch.nn.Linear(in_channels, out_channels)
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        x = self.layer(x, edge_index)
        return self.encoder(x)

class DeeperGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden_channels, device):
        super(DeeperGCN, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        self.layers = torch.nn.ModuleList()

        # node encoder to hidden_channels
        self.in_encoder = torch.nn.Linear(6, self.hidden_channels)

        # initiate layers list
        activation_functs = [torch.nn.LeakyReLU(negative_slope=0.01)] * (self.num_layers - 2) + [torch.nn.Tanh()] * 2
        for n_layer, act_fun in zip(range(self.num_layers), activation_functs):
            
            # add deep layer
            self.layers.append(DeepLayer(self.hidden_channels // 2**n_layer,
                                         self.hidden_channels // 2**(n_layer + 1),
                                         act_fun, block='res+'))

        # last layer
        self.lin = Linear(self.hidden_channels // 2**(self.num_layers), 2)
        
        self.to(device)

    def forward(self, x, edge_index, batch):
        """ Simple forward through layers"""
        
        # extract attributes and edge_inde
        x = self.in_encoder(x)

        # forward in each layer
        for layer in self.layers:
            x = layer(x, edge_index)
            
        x = global_mean_pool(x, batch)
        # final classifier
        x = self.lin(x)
        
        # forward output layer
        return x
