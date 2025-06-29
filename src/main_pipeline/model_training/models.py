# SPDX-License-Identifier: MIT
# Adapted from Gómez de Lope et al., "Graph Representation Learning Strategies for Omics Data: A Case Study on Parkinson’s Disease", arXiv:2406.14442 (MIT License)

import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv
import torch.nn.functional as F
import torch


class GCNN(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f, p_dropout):
        super(GCNN, self).__init__()

        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out  # returns the embedding x & prediction out


class MLP2(torch.nn.Module):
    def __init__(self, in_f, h1_f, h2_f, out_f, p_dropout):
        super().__init__()
        torch.manual_seed(42)
        self.lin1 = nn.Linear(in_f, h1_f)
        self.lin2 = nn.Linear(h1_f, h2_f)
        self.lin3 = nn.Linear(h2_f, out_f)
        self.bn1 = nn.BatchNorm1d(h1_f)
        self.bn2 = nn.BatchNorm1d(h2_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin3(x)
        return out

class GAT(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GAT, self).__init__()
        # graph CL1
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads, edge_dim=1)
        # graph CL2
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False, edge_dim=1)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout
    def forward(self, x, edge_index, edge_attr):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out
