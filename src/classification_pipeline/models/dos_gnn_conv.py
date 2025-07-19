from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> OptPairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(  # noqa: F811
        edge_index, edge_weight, num_nodes, improved, add_self_loops, flow,
        dtype):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(  # noqa: F811
    edge_index: Adj,
    edge_weight: OptTensor = None,
    num_nodes: Optional[int] = None,
    improved: bool = False,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
    dtype: Optional[torch.dtype] = None,
):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)

        adj_t = edge_index

        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = torch_sparse.fill_diag(adj_t, fill_value)

        deg = torch_sparse.sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = torch_sparse.mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    if is_torch_sparse_tensor(edge_index):
        assert edge_index.size(0) == edge_index.size(1)

        if edge_index.layout == torch.sparse_csc:
            raise NotImplementedError("Sparse CSC matrices are not yet "
                                      "supported in 'gcn_norm'")

        adj_t = edge_index
        if add_self_loops:
            adj_t, _ = add_self_loops_fn(adj_t, None, fill_value, num_nodes)

        edge_index, value = to_edge_index(adj_t)
        col, row = edge_index[0], edge_index[1]

        deg = scatter(value, col, 0, dim_size=num_nodes, reduce='sum')
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        value = deg_inv_sqrt[row] * value * deg_inv_sqrt[col]

        return set_sparse_value(adj_t, value), None

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == 'source_to_target' else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    return edge_index, edge_weight


class DOSGNN(MessagePassing):

    r"""The Dual-Channel Oversampling GNN (DOS-GNN) convolutional operator.
    Implements heterophily-aware graph convolution via two parallel channels:
    1) Homophilic channel aggregates similar neighbor features.
    2) Heterophilic channel aggregates dissimilar neighbor features.
    Each channel mixes per-edge messages using
        α_ij = (1 + cosine(h1_i, h1_j)) / 2,
    aggregates by mean over the neighborhood (including self),
    adds a residual connection, and applies ReLU.
    The two channel outputs are concatenated [h1, h2] as final node features.
    """
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = None,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')
        # dual‐channel projection for DOS‐GNN
        self.lin_h2 = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self.lin_h2.reset_parameters()
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    # Compute edge-wise homophily score α_ij = (1 + cosine(h1_i, h1_j)) / 2
    def compute_alpha(self, h1, edge_index):
        src, dst = edge_index
        hi, hj = h1[src], h1[dst]
        cos = F.cosine_similarity(hi, hj, dim=-1)
        return (cos + 1.0) * 0.5

    def dual_layer(self, h1, h2, edge_index, edge_weight):
        # Build and aggregate hetero-/homo-philic messages per edge:
        #   m1 = α * h1_src + (1-α) * h2_src
        #   m2 = (1-α) * h1_src + α * h2_src
        # Then sum over neighbors, divide by degree, add residual, and apply ReLU.
        alpha = self.compute_alpha(h1, edge_index).unsqueeze(1)  # [E,1]
        src, dst = edge_index
        # prepare messages
        m1 = alpha * h1[src] + (1 - alpha) * h2[src]
        m2 = (1 - alpha) * h1[src] + alpha * h2[src]
        # aggregate
        h1_agg = torch.zeros_like(h1).index_add_(0, dst, m1)
        h2_agg = torch.zeros_like(h2).index_add_(0, dst, m2)
        # normalize by degree
        N = h1.size(0)
        deg = torch.zeros(N, device=h1.device).index_add_(0, dst, torch.ones_like(src, dtype=torch.float, device=h1.device))
        deg = deg.clamp(min=1).unsqueeze(1)
        # update with residual + activation
        h1_new = F.relu(h1 + h1_agg / deg)
        h2_new = F.relu(h2 + h2_agg / deg)
        return h1_new, h2_new

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # Initial projections into homophilic (h1) and heterophilic (h2) channels
        x_h1 = self.lin(x)
        x_h2 = self.lin_h2(x)
        # dual aggregation layers
        h1, h2 = x_h1, x_h2
        h1, h2 = self.dual_layer(h1, h2, edge_index, edge_weight)
        h1, h2 = self.dual_layer(h1, h2, edge_index, edge_weight)
        # Concatenate the two channels per node as final output features
        H = torch.cat([h1, h2], dim=1)
        # apply bias if present (expand bias to match double channels)
        if self.bias is not None:
            # concatenate bias for each channel
            bias = torch.cat([self.bias, self.bias], dim=0)
            H = H + bias
        return H

    def get_concat_embeddings(self, x: Tensor, edge_index: Adj,
                              edge_weight: OptTensor = None) -> Tensor:
        r"""Return the concatenated dual-channel embeddings [h1, h2] after two DOS-GNN layers."""
        # (Re)normalize adjacency if needed
        if self.normalize and isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(
                edge_index, edge_weight, x.size(self.node_dim),
                self.improved, self.add_self_loops, self.flow, x.dtype)

        # Initial projections into homophilic and heterophilic channels
        h1 = self.lin(x)
        h2 = self.lin_h2(x)
        # Two dual-aggregation layers
        h1, h2 = self.dual_layer(h1, h2, edge_index, edge_weight)
        h1, h2 = self.dual_layer(h1, h2, edge_index, edge_weight)
        # Concatenate channels
        return torch.cat([h1, h2], dim=1)

    # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
    #     return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    # def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
    #     return spmm(adj_t, x, reduce=self.aggr)
