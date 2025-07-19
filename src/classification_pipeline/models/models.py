# SPDX-License-Identifier: MIT
# Adapted from Gómez de Lope et al., "Graph Representation Learning Strategies for Omics Data: A Case Study on Parkinson’s Disease", arXiv:2406.14442 (MIT License)


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GCNConv, ChebConv, GATv2Conv, GPSConv, GINEConv

from src.classification_pipeline.models.dos_gnn_conv import DOSGNN


class MLP(torch.nn.Module):
    def __init__(self, in_f, h_f, out_f, p_dropout):
        super().__init__()
        torch.manual_seed(42)
        self.lin1 = nn.Linear(in_f, h_f)
        self.lin2 = nn.Linear(h_f, out_f)
        self.bn1 = nn.BatchNorm1d(h_f)
        self.p_dropout = p_dropout

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin2(x)
        return out

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


class Cheb_GCNN(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, K, out_f, p_dropout): #, DL1_F, DL2_F
        super(Cheb_GCNN, self).__init__()

        # graph CL1
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data_processing.x, data_processing.edge_index, data_processing.edge_weight
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out # returns the embedding x & prediction out



class GCNN(nn.Module):
    def __init__(self, in_f ,CL1_F, CL2_F, out_f, p_dropout):
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
        #x, edge_index, edge_weight = data_processing.x, data_processing.edge_index, data_processing.edge_weight

        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out # returns the embedding x & prediction out

def broadcast(src, other, dim):
    # Source: torch_scatter
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

class SparseMaskedLinear_v2(nn.Module):
    """ Masked linear layer with sparse mask AND sparse weight matrix (faster and more memory efficient) """
    def __init__(self, in_features, out_features, sparse_mask, bias=True, device=None, dtype=None):
        """
        in_features: number of input features
        out_features: number of output features
        sparse_mask: torch tensor of shape (n_connections, 2), where indices[:, 0] index the input neurons
                     and indices[:, 1] index the output neurons
        """
        # Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        self.sparse_mask = sparse_mask
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty((sparse_mask.shape[0]), **factory_kwargs)))  # Shape=(n_connections,)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
    def forward(self, input):
        # weight shape: (out_features, in_features)
        x = input[:, self.sparse_mask[:, 0]]  # Shape=(batch_size, n_connections)
        src = x * self.weight[None, :]  # Shape=(batch_size, n_connections)
        # Reduce via scatter sum
        out = torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)  # Shape=(batch_size, out_features)
        index = broadcast(self.sparse_mask[:, 1], src, dim=-1)
        out = out.scatter_add_(dim=-1, index=index, src=src)
        if self.use_bias:
            out = out + self.bias
        return out
    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

class SparseMaskedLinear_v3(nn.Module):
    """ Masked linear layer with sparse mask AND sparse weight matrix (faster and more memory efficient) """
    def __init__(self, in_features, out_features, sparse_mask, K=1, bias=True, device=None, dtype=None):
        """
        in_features: number of input features
        out_features: number of output features
        sparse_mask: torch tensor of shape (n_connections, 2), where indices[:, 0] index the input neurons
                     and indices[:, 1] index the output neurons
        """
        # Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        self.sparse_mask = sparse_mask
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty((sparse_mask.shape[0]), **factory_kwargs)))  # Shape=(n_connections,)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features*K, **factory_kwargs))
    def forward(self, input):
        # weight shape: (out_features, in_features)
        x = input[:, self.sparse_mask[:, 0]]  # Shape=(batch_size, n_connections)
        src = x * self.weight[None, :]  # Shape=(batch_size, n_connections)
        # Reduce via scatter sum
        out = torch.zeros((x.shape[0], self.out_features*self.K), dtype=x.dtype, device=x.device)  # Shape=(batch_size, out_features)
        index = broadcast(self.sparse_mask[:, 1], src, dim=-1)
        out = out.scatter_add_(dim=-1, index=index, src=src)
        if self.use_bias:
            out = out + self.bias
        return out
    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)


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
        #x, edge_index, edge_weight = data_processing.x, data_processing.edge_index, data_processing.edge_weight
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


class GCNN3L(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, CL3_F, out_f):
        super(GCNN3L, self).__init__()

        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # graph CL3
        self.conv3 = GCNConv(in_channels=CL2_F, out_channels=CL3_F)
        # FC1
        self.lin1 = nn.Linear(CL3_F, out_f)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))


        out = self.lin1(x)

        return out  # returns the embedding x & prediction out


class GPST(torch.nn.Module):
    def __init__(self, in_f, heads, out_f, p_dropout, K):
        super(GPST, self).__init__()
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=ChebConv(in_channels=in_f, out_channels=in_f, K=K), dropout=p_dropout)
        self.gps2 = GPSConv(channels=in_f, heads=1, conv=ChebConv(in_channels=in_f, out_channels=in_f, K=K), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, out_f)
        self.bn1 = BatchNorm(in_f)
        self.bn2 = BatchNorm(in_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.gps1(x, edge_index)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.gps2(x, edge_index)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out


class GPST_GINE(torch.nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout, edge_dim):
        super(GPST_GINE, self).__init__()
        mlp1 = torch.nn.Sequential(
            nn.Linear(in_f, CL1_F),
            nn.ReLU(),
            nn.Linear(CL1_F, in_f)
        )
        mlp2 = torch.nn.Sequential(
            nn.Linear(in_f, CL2_F),
            nn.ReLU(),
            nn.Linear(CL2_F, in_f)
        )
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=GINEConv(nn=mlp1, edge_dim=edge_dim), dropout=p_dropout)
        self.gps2 = GPSConv(channels=in_f, heads=1, conv=GINEConv(nn=mlp2, edge_dim=edge_dim), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, out_f)
        self.bn1 = BatchNorm(in_f)
        self.bn2 = BatchNorm(in_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.bn1(self.gps1(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.gps2(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out

class GPST_GINE_lin(torch.nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout, edge_dim):
        super(GPST_GINE_lin, self).__init__()
        mlp1 = torch.nn.Sequential(
            nn.Linear(in_f, CL1_F),
            nn.ReLU(),
            nn.Linear(CL1_F, in_f)
        )
        mlp2 = torch.nn.Sequential(
            nn.Linear(CL1_F, CL2_F),
            nn.ReLU(),
            nn.Linear(CL2_F, CL1_F)
        )
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=GINEConv(nn=mlp1, edge_dim=edge_dim), dropout=p_dropout)
        self.gps2 = GPSConv(channels=CL1_F, heads=1, conv=GINEConv(nn=mlp2, edge_dim=edge_dim), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, CL1_F)
        self.lin2 = nn.Linear(CL1_F, out_f)
        self.bn1 =  BatchNorm(in_f)
        self.bn2 =  BatchNorm(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.bn1(self.gps1(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.lin1(x)))
        x = F.relu(self.bn2(self.gps2(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin2(x)
        return out

class GINE(nn.Module):
    def __init__(self, in_f, HL1_F, CL1_F, HL2_F, CL2_F, out_f, p_dropout, edge_dim):
        super(GINE, self).__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_f, HL1_F),
            nn.ReLU(),
            nn.Linear(HL1_F, CL1_F)
        )
        mlp2 = nn.Sequential(
            nn.Linear(CL1_F, HL2_F),
            nn.ReLU(),
            nn.Linear(HL2_F, CL2_F)
        )
        # graph CL1
        self.conv1 = GINEConv(nn=mlp1, edge_dim=edge_dim)
        # graph CL2
        self.conv2 = GINEConv(nn=mlp2, edge_dim=edge_dim)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 =  BatchNorm(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        # Node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out



class DOS_GNN(nn.Module):
    def __init__(self, in_f ,CL1_F, CL2_F, out_f, p_dropout):
        super(DOS_GNN, self).__init__()

        # dual-channel DOS-GNN convolutions
        self.dos1 = DOSGNN(in_channels=in_f,   out_channels=CL1_F,
                           improved=False, cached=False,
                           add_self_loops=True, normalize=True)
        self.dos2 = DOSGNN(in_channels=CL1_F * 2, out_channels=CL2_F,
                           improved=False, cached=False,
                           add_self_loops=True, normalize=True)
        # FC1
        self.lin1 = nn.Linear(CL2_F * 2, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * 2)
        self.bn2 = nn.BatchNorm1d(CL2_F * 2)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        # first DOS-GNN layer + norm/dropout
        x = F.relu(self.dos1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # second DOS-GNN layer + norm/dropout
        x = F.relu(self.dos2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out # returns the embedding x & prediction out


# GCN layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # for 3_D batch, need a loop!!!

        if self.bias is not None:
            return output + self.bias
        else:
            return output


# Multihead attention layer
class MultiHead(Module):  # currently, allowed for only one sample each time. As no padding mask is required.
    def __init__(
            self,
            input_dim,
            num_heads,
            kdim=None,
            vdim=None,
            embed_dim=128,  # should equal num_heads*head dim
            v_embed_dim=None,
            dropout=0.1,
            bias=True,
    ):
        super(MultiHead, self).__init__()
        self.input_dim = input_dim
        self.kdim = kdim if kdim is not None else input_dim
        self.vdim = vdim if vdim is not None else input_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.v_embed_dim = v_embed_dim if v_embed_dim is not None else embed_dim

        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.bias = bias
        assert (
                self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        assert self.v_embed_dim % num_heads == 0, "v_embed_dim must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.input_dim, self.embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.v_embed_dim, bias=bias)

        self.out_proj = nn.Linear(self.v_embed_dim, self.v_embed_dim // self.num_heads, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        if True:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)
        else:
            nn.init.normal_(self.k_proj.weight)
            nn.init.normal_(self.v_proj.weight)
            nn.init.normal_(self.q_proj.weight)

        nn.init.normal_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

        if self.bias:
            nn.init.constant_(self.k_proj.bias, 0.)
            nn.init.constant_(self.v_proj.bias, 0.)
            nn.init.constant_(self.q_proj.bias, 0.)

    def forward(
            self,
            query,
            key,
            value,
            need_weights: bool = False,
            need_head_weights: bool = False,
    ):
        """Input shape: Time x Batch x Channel
        Args:
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        batch_num, node_num, input_dim = query.size()

        assert key is not None and value is not None

        # project input
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q * self.scaling

        # compute attention
        q = q.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2, -3).contiguous().view(
            batch_num * self.num_heads, node_num, self.head_dim)
        k = k.view(batch_num, node_num, self.num_heads, self.head_dim).transpose(-2, -3).contiguous().view(
            batch_num * self.num_heads, node_num, self.head_dim)
        v = v.view(batch_num, node_num, self.num_heads, self.vdim).transpose(-2, -3).contiguous().view(
            batch_num * self.num_heads, node_num, self.vdim)
        attn_output_weights = torch.bmm(q, k.transpose(-1, -2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        # drop out
        attn_output_weights = F.dropout(attn_output_weights, p=self.dropout, training=self.training)

        # collect output
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = attn_output.view(batch_num, self.num_heads, node_num, self.vdim).transpose(-2,
                                                                                                 -3).contiguous().view(
            batch_num, node_num, self.v_embed_dim)
        attn_output = self.out_proj(attn_output)

        if need_weights:
            attn_output_weights = attn_output_weights  # view: (batch_num, num_heads, node_num, node_num)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output


# Graphsage layer
class SageConv(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, in_features, out_features, bias=False):
        super(SageConv, self).__init__()

        self.proj = nn.Linear(in_features * 2, out_features, bias=bias)

        self.reset_parameters()

        # print("note: for dense graph in graphsage, require it normalized.")

    def reset_parameters(self):

        nn.init.normal_(self.proj.weight)

        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.)

    def forward(self, features, adj):
        """
        Args:
            adj: can be sparse or dense matrix.
        """

        # fuse info from neighbors. to be added:
        if adj.layout != torch.sparse_coo:
            if len(adj.shape) == 3:
                neigh_feature = torch.bmm(adj, features) / (
                            adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                neigh_feature = torch.mm(adj, features) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            # print("spmm not implemented for batch training. Note!")

            neigh_feature = torch.spmm(adj, features) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)

        # perform conv
        data = torch.cat([features, neigh_feature], dim=-1)
        combined = self.proj(data)

        return combined


# GraphAT layers

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        if isinstance(adj, torch.sparse.FloatTensor):
            adj = adj.to_dense()

        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# --------------
### models ###
# --------------

# gcn_encode
class GCN_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class GCN_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


# sage model

class Sage_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En, self).__init__()

        self.sage1 = SageConv(nfeat, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class Sage_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(Sage_En2, self).__init__()

        self.sage1 = SageConv(nfeat, nhid)
        self.sage2 = SageConv(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x


class Sage_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Sage_Classifier, self).__init__()

        self.sage1 = SageConv(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.sage1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x


# GAT model

class GAT_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, alpha=0.2, nheads=8):
        super(GAT_En, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nembed)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))

        return x


class GAT_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout, alpha=0.2, nheads=8):
        super(GAT_En2, self).__init__()

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nembed)
        self.dropout = dropout

        self.attentions_2 = [GraphAttentionLayer(nembed, nembed, dropout=dropout, alpha=alpha, concat=True) for _ in
                             range(nheads)]
        for i, attention in enumerate(self.attentions_2):
            self.add_module('attention2_{}'.format(i), attention)

        self.out_proj_2 = nn.Linear(nembed * nheads, nembed)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.05)
        nn.init.normal_(self.out_proj_2.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = torch.cat([att(x, adj) for att in self.attentions_2], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj_2(x))
        return x


class GAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, alpha=0.2, nheads=8):
        super(GAT_Classifier, self).__init__()

        self.attentions = [GraphAttentionLayer(nembed, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_proj = nn.Linear(nhid * nheads, nhid)

        self.dropout = dropout
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)
        nn.init.normal_(self.out_proj.weight, std=0.05)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = self.mlp(x)

        return x


class Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(Classifier, self).__init__()

        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight, std=0.05)

    def forward(self, x, adj):
        x = self.mlp(x)

        return x


class Decoder(Module):
    """
    Simple Graphsage layer
    """

    def __init__(self, nembed, dropout=0.1):
        super(Decoder, self).__init__()
        self.dropout = dropout

        self.de_weight = Parameter(torch.FloatTensor(nembed, nembed))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.de_weight.size(1))
        self.de_weight.data.uniform_(-stdv, stdv)

    def forward(self, node_embed):
        combine = F.linear(node_embed, self.de_weight)
        adj_out = torch.sigmoid(torch.mm(combine, combine.transpose(-1, -2)))

        return adj_out

